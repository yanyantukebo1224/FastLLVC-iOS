//
//  AudioDSP.swift
//  FastLLVC
//
//  Zero-Latency Audio DSP Utilities (Accelerate / vDSP)
//  Created by Pop-chan & Antigravity
//

import Foundation
import Accelerate

public final class AudioDSP {
    private var dcPrevIn: Float = 0.0
    private var dcPrevOut: Float = 0.0
    private let dcR: Float = 0.995

    private var gateGain: Float = 0.0
    
    // Equalizer state (IIR Biquad filters)
    private var lowState = [Float](repeating: 0.0, count: 2)
    private var midState = [Float](repeating: 0.0, count: 2)
    private var highState = [Float](repeating: 0.0, count: 2)
    
    // FFT Setup for Spectrum Visualization
    private let fftLog2N: vDSP_Length = 8 // 256 points FFT
    private let fftN: Int = 256
    private var fftSetup: FFTSetup?
    private var window: [Float]

    public init() {
        self.fftSetup = vDSP_create_fftsetup(fftLog2N, FFTRadix(kFFTRadix2))
        var win = [Float](repeating: 0.0, count: fftN)
        vDSP_hann_window(&win, vDSP_Length(fftN), Int32(vDSP_HANN_NORM))
        self.window = win
    }

    deinit {
        if let setup = fftSetup {
            vDSP_destroy_fftsetup(setup)
        }
    }

    public func reset() {
        dcPrevIn = 0.0
        dcPrevOut = 0.0
        gateGain = 0.0
        lowState = [Float](repeating: 0.0, count: 2)
        midState = [Float](repeating: 0.0, count: 2)
        highState = [Float](repeating: 0.0, count: 2)
    }

    /// Single-pole recursive IIR high-pass filter (DC Blocker)
    public func processDCBlocker(_ audio: inout [Float]) {
        for i in 0..<audio.count {
            let out = audio[i] - dcPrevIn + dcR * dcPrevOut
            dcPrevIn = audio[i]
            dcPrevOut = out
            audio[i] = out
        }
    }

    /// Calculate RMS (Root Mean Square) energy using vDSP
    public func calculateRMS(_ audio: [Float]) -> Float {
        guard !audio.isEmpty else { return 0.0 }
        var rms: Float = 0.0
        vDSP_rmsqv(audio, 1, &rms, vDSP_Length(audio.count))
        return rms
    }
    
    /// Calculate Peak Amplitude
    public func calculatePeak(_ audio: [Float]) -> Float {
        guard !audio.isEmpty else { return 0.0 }
        var peak: Float = 0.0
        vDSP_maxmgv(audio, 1, &peak, vDSP_Length(audio.count))
        return peak
    }

    /// Update Noise Gate Gain
    public func processNoiseGate(_ audio: inout [Float], thresholdDB: Float = -45.0) -> Float {
        let rms = calculateRMS(audio)
        let inDB = 20.0 * log10(max(rms, 1e-9))
        let targetGain: Float = (inDB >= thresholdDB) ? 1.0 : 0.0
        gateGain = 0.85 * gateGain + 0.15 * targetGain

        if gateGain < 0.01 {
            vDSP_vclr(&audio, 1, vDSP_Length(audio.count))
        } else {
            var g = gateGain
            vDSP_vsmul(audio, 1, &g, &audio, 1, vDSP_Length(audio.count))
        }
        return gateGain
    }

    /// Soft-Knee Saturation Limiter using tanh (vvtanhf in Accelerate)
    public func applySoftLimiter(_ audio: inout [Float]) {
        var scaled = [Float](repeating: 0.0, count: audio.count)
        var scale: Float = 0.95
        vDSP_vsmul(audio, 1, &scale, &scaled, 1, vDSP_Length(audio.count))

        var count = Int32(audio.count)
        vvtanhf(&audio, scaled, &count)
    }

    /// 3-Band Equalizer (Low Shelf, Mid Peak, High Shelf)
    public func applyEqualizer(_ audio: inout [Float], lowGainDB: Float, midGainDB: Float, highGainDB: Float) {
        guard lowGainDB != 0.0 || midGainDB != 0.0 || highGainDB != 0.0 else { return }
        
        let lowLin = pow(10.0, lowGainDB / 20.0)
        let midLin = pow(10.0, midGainDB / 20.0)
        let highLin = pow(10.0, highGainDB / 20.0)
        
        for i in 0..<audio.count {
            let sample = audio[i]
            lowState[0] = 0.92 * lowState[0] + 0.08 * sample
            let low = lowState[0]
            
            highState[0] = 0.85 * highState[0] + 0.15 * sample
            let high = sample - highState[0]
            
            let mid = sample - low - high
            
            audio[i] = low * lowLin + mid * midLin + high * highLin
        }
    }

    /// Calculate 8-Band Spectrum for UI Visualizer
    public func calculateBands(_ audio: [Float]) -> [Float] {
        guard let setup = fftSetup, !audio.isEmpty else {
            return [Float](repeating: 0.0, count: 8)
        }
        
        var padded = [Float](repeating: 0.0, count: fftN)
        let copyLen = min(audio.count, fftN)
        for i in 0..<copyLen {
            padded[i] = audio[i] * window[i]
        }
        
        var real = [Float](repeating: 0.0, count: fftN / 2)
        var imag = [Float](repeating: 0.0, count: fftN / 2)
        var magnitudes = [Float](repeating: 0.0, count: fftN / 2)
        
        real.withUnsafeMutableBufferPointer { rPtr in
            imag.withUnsafeMutableBufferPointer { iPtr in
                var split = DSPSplitComplex(realp: rPtr.baseAddress!, imagp: iPtr.baseAddress!)
                padded.withUnsafeBufferPointer { pPtr in
                    let ptr = UnsafeRawPointer(pPtr.baseAddress!).assumingMemoryBound(to: DSPComplex.self)
                    vDSP_ctoz(ptr, 2, &split, 1, vDSP_Length(fftN / 2))
                }
                vDSP_fft_zrip(setup, &split, 1, fftLog2N, FFTDirection(FFT_FORWARD))
                
                var scale: Float = 1.0 / Float(fftN)
                vDSP_zvabs(&split, 1, &magnitudes, 1, vDSP_Length(fftN / 2))
                vDSP_vsmul(magnitudes, 1, &scale, &magnitudes, 1, vDSP_Length(fftN / 2))
            }
        }
        
        // Aggregate into 8 frequency bands
        var bands = [Float](repeating: 0.0, count: 8)
        let bandSize = (fftN / 2) / 8
        for b in 0..<8 {
            var sum: Float = 0.0
            for k in 0..<bandSize {
                sum += magnitudes[b * bandSize + k]
            }
            bands[b] = min(max(sum / Float(bandSize) * 12.0, 0.0), 1.0)
        }
        
        return bands
    }

    /// Resample using Linear Interpolation (Optimized for 48kHz <-> 16kHz 3:1 ratio)
    public static func downsample3x(input: [Float]) -> [Float] {
        let outCount = input.count / 3
        var output = [Float](repeating: 0.0, count: outCount)
        for i in 0..<outCount {
            output[i] = input[i * 3]
        }
        return output
    }

    public static func upsample3x(input: [Float]) -> [Float] {
        let outCount = input.count * 3
        var output = [Float](repeating: 0.0, count: outCount)
        for i in 0..<input.count {
            let base = input[i]
            let next = (i + 1 < input.count) ? input[i + 1] : base
            output[i * 3]     = base
            output[i * 3 + 1] = base + (next - base) * 0.333333
            output[i * 3 + 2] = base + (next - base) * 0.666667
        }
        return output
    }
}
