//
//  AudioRecorder.swift
//  FastLLVC
//
//  Lightweight Real-Time WAV Audio Recorder for Voice Conversions
//  Created by Pop-chan & Antigravity
//

import Foundation
import AVFoundation

public final class AudioRecorder: ObservableObject {
    @Published public var isRecording = false
    @Published public var recordedDuration: TimeInterval = 0.0
    @Published public var lastRecordedFileURL: URL?
    
    private var recordedSamples = [Float]()
    private let sampleRate: Double = 16000.0
    private var timer: Timer?
    private let queue = DispatchQueue(label: "com.fastllvc.recorder")
    
    public init() {}
    
    public func startRecording() {
        queue.async {
            self.recordedSamples.removeAll(keepingCapacity: true)
            DispatchQueue.main.async {
                self.isRecording = true
                self.recordedDuration = 0.0
                self.timer = Timer.scheduledTimer(withTimeInterval: 0.1, repeats: true) { [weak self] _ in
                    guard let self = self else { return }
                    self.recordedDuration = Double(self.recordedSamples.count) / self.sampleRate
                }
            }
        }
    }
    
    public func appendSamples(_ samples: [Float]) {
        queue.async {
            guard self.isRecording else { return }
            self.recordedSamples.append(contentsOf: samples)
        }
    }
    
    public func stopRecording(completion: @escaping (URL?) -> Void) {
        queue.async {
            let samplesToSave = self.recordedSamples
            DispatchQueue.main.async {
                self.isRecording = false
                self.timer?.invalidate()
                self.timer = nil
            }
            
            let fileURL = self.exportWAV(samples: samplesToSave)
            DispatchQueue.main.async {
                self.lastRecordedFileURL = fileURL
                completion(fileURL)
            }
        }
    }
    
    private func exportWAV(samples: [Float]) -> URL? {
        guard !samples.isEmpty else { return nil }
        let tempDir = FileManager.default.temporaryDirectory
        let fileName = "FastLLVC_\(Int(Date().timeIntervalSince1970)).wav"
        let fileURL = tempDir.appendingPathComponent(fileName)
        
        let numChannels: Int16 = 1
        let sampleRateInt: Int32 = Int32(self.sampleRate)
        let bitsPerSample: Int16 = 16
        let byteRate = sampleRateInt * Int32(numChannels) * Int32(bitsPerSample / 8)
        let blockAlign = Int16(numChannels * (bitsPerSample / 8))
        let dataSize = Int32(samples.count * 2)
        let totalSize = dataSize + 36
        
        var data = Data()
        // RIFF Header
        data.append("RIFF".data(using: .ascii)!)
        var totalSizeVar = totalSize
        data.append(Data(bytes: &totalSizeVar, count: 4))
        data.append("WAVEfmt ".data(using: .ascii)!)
        
        var subchunk1Size: Int32 = 16
        var audioFormat: Int16 = 1 // PCM
        var numChannelsVar = numChannels
        var sampleRateVar = sampleRateInt
        var byteRateVar = byteRate
        var blockAlignVar = blockAlign
        var bitsPerSampleVar = bitsPerSample
        
        data.append(Data(bytes: &subchunk1Size, count: 4))
        data.append(Data(bytes: &audioFormat, count: 2))
        data.append(Data(bytes: &numChannelsVar, count: 2))
        data.append(Data(bytes: &sampleRateVar, count: 4))
        data.append(Data(bytes: &byteRateVar, count: 4))
        data.append(Data(bytes: &blockAlignVar, count: 2))
        data.append(Data(bytes: &bitsPerSampleVar, count: 2))
        
        // Data Chunk
        data.append("data".data(using: .ascii)!)
        var dataSizeVar = dataSize
        data.append(Data(bytes: &dataSizeVar, count: 4))
        
        // PCM 16-bit Samples
        for s in samples {
            let clamped = max(-1.0, min(1.0, s))
            var int16Val = Int16(clamped * 32767.0)
            data.append(Data(bytes: &int16Val, count: 2))
        }
        
        do {
            try data.write(to: fileURL)
            return fileURL
        } catch {
            print("Failed to save WAV: \(error)")
            return nil
        }
    }
}
