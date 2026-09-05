//
//  PitchShifterDSP.swift
//  FastLLVC
//
//  Real-Time Low-Latency Pitch Shifter using Overlap-Add (OLA)
//  Created by Pop-chan & Antigravity
//

import Foundation
import Accelerate

public final class PitchShifterDSP {
    private let maxDelay = 2048
    private var delayBuffer: [Float]
    private var writeIndex: Int = 0
    private var readIndex1: Float = 0.0
    private var readIndex2: Float = 0.0
    private var windowSize: Float = 1024.0
    
    public init() {
        self.delayBuffer = [Float](repeating: 0.0, count: maxDelay)
    }

    public func reset() {
        delayBuffer = [Float](repeating: 0.0, count: maxDelay)
        writeIndex = 0
        readIndex1 = 0.0
        readIndex2 = Float(maxDelay) / 2.0
    }

    /// Process pitch shift in semitones (-12.0 to +12.0)
    public func process(_ audio: inout [Float], semitones: Float) {
        guard semitones != 0.0 else { return }
        
        let pitchRatio = pow(2.0, semitones / 12.0)
        let rate = pitchRatio - 1.0
        let bufLen = Float(maxDelay)
        let halfWindow = windowSize * 0.5
        
        for i in 0..<audio.count {
            let inSample = audio[i]
            delayBuffer[writeIndex] = inSample
            
            // Modulation phase for voice 1
            let pos1 = fmod(Float(writeIndex) - readIndex1 + bufLen, bufLen)
            let gain1 = 0.5 * (1.0 - cos(2.0 * .pi * (pos1 / windowSize)))
            
            // Modulation phase for voice 2 (180 deg shifted)
            let pos2 = fmod(Float(writeIndex) - readIndex2 + bufLen, bufLen)
            let gain2 = 0.5 * (1.0 - cos(2.0 * .pi * (pos2 / windowSize)))
            
            // Linear interpolation for read pointers
            let rIdx1 = Int(readIndex1) % maxDelay
            let rIdx1Next = (rIdx1 + 1) % maxDelay
            let frac1 = readIndex1 - Float(Int(readIndex1))
            let s1 = delayBuffer[rIdx1] * (1.0 - frac1) + delayBuffer[rIdx1Next] * frac1
            
            let rIdx2 = Int(readIndex2) % maxDelay
            let rIdx2Next = (rIdx2 + 1) % maxDelay
            let frac2 = readIndex2 - Float(Int(readIndex2))
            let s2 = delayBuffer[rIdx2] * (1.0 - frac2) + delayBuffer[rIdx2Next] * frac2
            
            let outSample = (s1 * gain1 + s2 * gain2)
            audio[i] = outSample
            
            // Advance indices
            readIndex1 = fmod(readIndex1 + rate + bufLen, bufLen)
            readIndex2 = fmod(readIndex2 + rate + bufLen, bufLen)
            writeIndex = (writeIndex + 1) % maxDelay
        }
    }
}
