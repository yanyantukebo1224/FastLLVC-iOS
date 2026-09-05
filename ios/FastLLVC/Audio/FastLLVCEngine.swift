//
//  FastLLVCEngine.swift
//  FastLLVC
//
//  Complete On-the-Fly Real-Time Voice Conversion Pipeline with Core ML & AVAudioEngine.
//  Created by Pop-chan & Antigravity
//

import Foundation
import AVFoundation
import CoreML
import Accelerate

public protocol FastLLVCDelegate: AnyObject {
    func didUpdateMetrics(latencyMs: Double, rtf: Double, inRMS: Float, outRMS: Float, inBands: [Float], outBands: [Float])
    func didChangeAudioRoute(isHeadphonesConnected: Bool)
}

public final class FastLLVCEngine {
    public weak var delegate: FastLLVCDelegate?

    private let audioEngine = AVAudioEngine()
    public private(set) var isRunning = false

    // DSP & Buffers
    private let dsp = AudioDSP()
    private let pitchShifter = PitchShifterDSP()
    private let inRingBuffer = LockFreeRingBuffer(capacity: 32768)
    private let outRingBuffer = LockFreeRingBuffer(capacity: 32768)
    
    // WAV Recorder
    public let recorder = AudioRecorder()

    // Constants
    public let nativeSampleRate: Double = 16000.0
    public let chunkLength: Int = 208 // Native LLVC 13ms chunk (208 samples @ 16kHz)
    private let lookaheadContextLength: Int = 16 // L * 2 = 16 samples

    // Core ML Model & States
    public private(set) var model: MLModel?
    public private(set) var currentModelName: String = "Default FastLLVC (48k)"
    private var encBuf: MLMultiArray?
    private var decBuf: MLMultiArray?
    private var outBuf: MLMultiArray?
    private var convnetPreCtx: MLMultiArray?
    private var prevFrontCtx: [Float]

    // Inference Thread
    private let inferenceQueue = DispatchQueue(label: "com.fastllvc.inference", qos: .userInteractive)
    private var inferenceThreadRunning = false

    // Parameters
    public var inputGain: Float = 1.0
    public var outputGain: Float = 1.0
    public var pitchSemitones: Float = 0.0
    public var thresholdDB: Float = -45.0
    public var lowGainDB: Float = 0.0
    public var midGainDB: Float = 0.0
    public var highGainDB: Float = 0.0
    public var isPassthrough: Bool = false
    public var isMuted: Bool = false

    public init() {
        self.prevFrontCtx = [Float](repeating: 0.0, count: lookaheadContextLength)
        setupAudioRouteNotification()
    }
    
    private func setupAudioRouteNotification() {
        NotificationCenter.default.addObserver(forName: AVAudioSession.routeChangeNotification, object: nil, queue: .main) { [weak self] _ in
            guard let self = self else { return }
            let isHeadphones = self.isHeadphonesConnected()
            self.delegate?.didChangeAudioRoute(isHeadphonesConnected: isHeadphones)
        }
    }
    
    public func isHeadphonesConnected() -> Bool {
        let currentRoute = AVAudioSession.sharedInstance().currentRoute
        for output in currentRoute.outputs {
            let portType = output.portType
            if portType == .headphones || portType == .bluetoothA2DP || portType == .bluetoothHFP || portType == .bluetoothLE {
                return true
            }
        }
        return false
    }

    /// Load compiled Core ML Model (.mlmodelc or .mlpackage)
    public func loadModel(at url: URL, modelName: String? = nil) throws {
        let config = MLModelConfiguration()
        // Prioritize Apple Neural Engine (ANE) & GPU for real-time responsiveness
        config.computeUnits = .all
        self.model = try MLModel(contentsOf: url, configuration: config)
        self.currentModelName = modelName ?? url.deletingPathExtension().lastPathComponent
        resetBuffers()
    }

    public func resetBuffers() {
        dsp.reset()
        pitchShifter.reset()
        inRingBuffer.clear()
        outRingBuffer.clear()
        prevFrontCtx = [Float](repeating: 0.0, count: lookaheadContextLength)

        guard model != nil else { return }
        encBuf = try? MLMultiArray(shape: [1, 8, 20, 256], dataType: .float32)
        decBuf = try? MLMultiArray(shape: [1, 9, 20, 256], dataType: .float32)
        outBuf = try? MLMultiArray(shape: [1, 8, 20, 256], dataType: .float32)
        convnetPreCtx = try? MLMultiArray(shape: [1, 1020, 64], dataType: .float32)
    }

    /// Start Audio Engine and Real-Time On-the-Fly Processing
    public func start() throws {
        guard !isRunning else { return }

        let session = AVAudioSession.sharedInstance()
        try session.setCategory(.playAndRecord, mode: .voiceChat, options: [.defaultToSpeaker, .allowBluetooth, .allowAirPlay])
        try session.setPreferredIOBufferDuration(0.005) // 5ms buffer for lowest hardware latency
        try session.setPreferredSampleRate(48000.0)
        try session.setActive(true)

        let inputNode = audioEngine.inputNode
        let inputFormat = inputNode.outputFormat(forBus: 0)

        let outputNode = audioEngine.outputNode
        let outputFormat = outputNode.inputFormat(forBus: 0)

        // 1. Microphone Input Tap (On-the-Fly capture)
        inputNode.installTap(onBus: 0, bufferSize: 512, format: inputFormat) { [weak self] (buffer, time) in
            guard let self = self, let channelData = buffer.floatChannelData else { return }
            let frameCount = Int(buffer.frameLength)
            let rawInput = Array(UnsafeBufferPointer(start: channelData[0], count: frameCount))

            // 48kHz -> 16kHz Downsample
            var in16k = (inputFormat.sampleRate == 48000.0) ? AudioDSP.downsample3x(input: rawInput) : rawInput
            
            // Gain & DC Blocker
            if self.inputGain != 1.0 {
                var g = self.inputGain
                vDSP_vsmul(in16k, 1, &g, &in16k, 1, vDSP_Length(in16k.count))
            }
            self.dsp.processDCBlocker(&in16k)

            // Push to Input Ring Buffer
            in16k.withUnsafeBufferPointer { ptr in
                if let base = ptr.baseAddress {
                    self.inRingBuffer.write(base, count: in16k.count)
                }
            }
        }

        // 2. Output Source Node (On-the-Fly playback)
        let sourceNode = AVAudioSourceNode { [weak self] (_, _, frameCount, audioBufferList) -> OSStatus in
            guard let self = self else { return noErr }
            let ablPointer = UnsafeMutableAudioBufferListPointer(audioBufferList)
            let frames = Int(frameCount)

            let needed16k = (outputFormat.sampleRate == 48000.0) ? (frames / 3 + 1) : frames
            var out16k = [Float](repeating: 0.0, count: needed16k)

            let readSamples = out16k.withUnsafeMutableBufferPointer { ptr -> Int in
                guard let base = ptr.baseAddress else { return 0 }
                return self.outRingBuffer.read(into: base, count: needed16k)
            }

            var finalAudio: [Float]
            if outputFormat.sampleRate == 48000.0 {
                finalAudio = AudioDSP.upsample3x(input: out16k)
            } else {
                finalAudio = out16k
            }

            for buffer in ablPointer {
                guard let bufPtr = buffer.mData?.assumingMemoryBound(to: Float.self) else { continue }
                for f in 0..<frames {
                    bufPtr[f] = (f < finalAudio.count && readSamples > 0 && !self.isMuted) ? finalAudio[f] : 0.0
                }
            }
            return noErr
        }

        audioEngine.attach(sourceNode)
        audioEngine.connect(sourceNode, to: audioEngine.mainMixerNode, format: outputFormat)

        try audioEngine.start()
        isRunning = true

        // 3. Start On-The-Fly Core ML Worker Loop
        startInferenceWorker()
    }

    public func stop() {
        guard isRunning else { return }
        inferenceThreadRunning = false
        audioEngine.stop()
        audioEngine.inputNode.removeTap(onBus: 0)
        isRunning = false
    }

    private func startInferenceWorker() {
        inferenceThreadRunning = true
        inferenceQueue.async { [weak self] in
            guard let self = self else { return }
            var chunk = [Float](repeating: 0.0, count: self.chunkLength)

            while self.inferenceThreadRunning {
                if self.inRingBuffer.availableRead < self.chunkLength {
                    usleep(1000) // 1ms sleep to yield
                    continue
                }

                chunk.withUnsafeMutableBufferPointer { ptr in
                    guard let base = ptr.baseAddress else { return }
                    _ = self.inRingBuffer.read(into: base, count: self.chunkLength)
                }

                let inRMS = self.dsp.calculateRMS(chunk)
                let inBands = self.dsp.calculateBands(chunk)
                let _ = self.dsp.processNoiseGate(&chunk, thresholdDB: self.thresholdDB)

                let t0 = CACurrentMediaTime()
                var convertedChunk = [Float](repeating: 0.0, count: self.chunkLength)

                if self.isPassthrough || self.model == nil {
                    convertedChunk = chunk
                } else {
                    // Execute Core ML On-The-Fly Chunk Inference
                    self.runCoreMLChunk(input: chunk, output: &convertedChunk)
                }

                // Pitch Shifting DSP
                if self.pitchSemitones != 0.0 {
                    self.pitchShifter.process(&convertedChunk, semitones: self.pitchSemitones)
                }

                // 3-Band Equalizer
                self.dsp.applyEqualizer(&convertedChunk, lowGainDB: self.lowGainDB, midGainDB: self.midGainDB, highGainDB: self.highGainDB)

                // Output Gain & Saturation Limiter
                if self.outputGain != 1.0 {
                    var g = self.outputGain
                    vDSP_vsmul(convertedChunk, 1, &g, &convertedChunk, 1, vDSP_Length(convertedChunk.count))
                }
                self.dsp.applySoftLimiter(&convertedChunk)

                let outRMS = self.dsp.calculateRMS(convertedChunk)
                let outBands = self.dsp.calculateBands(convertedChunk)
                let tInfer = (CACurrentMediaTime() - t0) * 1000.0 // ms
                let rtf = (Double(self.chunkLength) / self.nativeSampleRate * 1000.0) / max(tInfer, 0.001)

                // Append to WAV recorder if active
                if self.recorder.isRecording {
                    self.recorder.appendSamples(convertedChunk)
                }

                // Push Converted Audio to Output Ring Buffer
                convertedChunk.withUnsafeBufferPointer { ptr in
                    if let base = ptr.baseAddress {
                        self.outRingBuffer.write(base, count: self.chunkLength)
                    }
                }

                DispatchQueue.main.async {
                    self.delegate?.didUpdateMetrics(latencyMs: tInfer, rtf: rtf, inRMS: inRMS, outRMS: outRMS, inBands: inBands, outBands: outBands)
                }
            }
        }
    }

    private func runCoreMLChunk(input: [Float], output: inout [Float]) {
        guard let model = model,
              let enc = encBuf, let dec = decBuf, let out = outBuf, let convnet = convnetPreCtx else {
            output = input
            return
        }

        do {
            let inputMLArray = try MLMultiArray(shape: [1, 1, NSNumber(value: chunkLength)], dataType: .float32)
            for i in 0..<chunkLength {
                inputMLArray[i] = NSNumber(value: input[i])
            }

            let prevCtxMLArray = try MLMultiArray(shape: [1, 1, NSNumber(value: lookaheadContextLength)], dataType: .float32)
            for i in 0..<lookaheadContextLength {
                prevCtxMLArray[i] = NSNumber(value: prevFrontCtx[i])
            }

            let featureProvider = try MLDictionaryFeatureProvider(dictionary: [
                "input_chunk": MLFeatureValue(multiArray: inputMLArray),
                "enc_buf": MLFeatureValue(multiArray: enc),
                "dec_buf": MLFeatureValue(multiArray: dec),
                "out_buf": MLFeatureValue(multiArray: out),
                "convnet_pre_ctx": MLFeatureValue(multiArray: convnet),
                "prev_front_ctx": MLFeatureValue(prevCtxMLArray)
            ])

            let prediction = try model.prediction(from: featureProvider)

            // Extract output audio chunk
            if let outVal = prediction.featureValue(for: "var_output")?.multiArrayValue ?? prediction.featureValue(for: "output")?.multiArrayValue {
                for i in 0..<min(chunkLength, outVal.count) {
                    output[i] = outVal[i].floatValue
                }
            }

            // Update state buffers for next streaming chunk
            if let nextEnc = prediction.featureValue(for: "next_enc_buf")?.multiArrayValue { encBuf = nextEnc }
            if let nextDec = prediction.featureValue(for: "next_dec_buf")?.multiArrayValue { decBuf = nextDec }
            if let nextOut = prediction.featureValue(for: "next_out_buf")?.multiArrayValue { outBuf = nextOut }
            if let nextConv = prediction.featureValue(for: "next_convnet_pre_ctx")?.multiArrayValue { convnetPreCtx = nextConv }
            if let nextCtx = prediction.featureValue(for: "next_prev_front_ctx")?.multiArrayValue {
                for i in 0..<lookaheadContextLength {
                    prevFrontCtx[i] = nextCtx[i].floatValue
                }
            }
        } catch {
            output = input
        }
    }
}
