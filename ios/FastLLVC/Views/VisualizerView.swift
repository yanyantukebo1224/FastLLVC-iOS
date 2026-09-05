//
//  VisualizerView.swift
//  FastLLVC
//
//  Real-Time Multi-Band Spectrum & Waveform Visualizer
//  Created by Pop-chan & Antigravity
//

import SwiftUI
#if canImport(UIKit)
import UIKit
#endif

struct VisualizerView: View {
    let inBands: [Float]
    let outBands: [Float]
    let inRMS: Float
    let outRMS: Float
    let isRunning: Bool

    var body: some View {
        VStack(spacing: 12) {
            HStack(spacing: 16) {
                // MIC Input Spectrum
                VStack(alignment: .leading, spacing: 6) {
                    HStack {
                        Circle()
                            .fill(isRunning ? Color.blue : Color.gray)
                            .frame(width: 8, height: 8)
                        Text("MIC INPUT")
                            .font(.system(size: 11, weight: .bold, design: .rounded))
                            .foregroundColor(.secondary)
                        Spacer()
                        Text("\(Int(min(max(inRMS * 100, 0), 100)))%")
                            .font(.system(size: 11, weight: .semibold, design: .monospaced))
                            .foregroundColor(.blue)
                    }

                    HStack(alignment: .bottom, spacing: 3) {
                        ForEach(0..<8, id: \.self) { idx in
                            let val = inBands.indices.contains(idx) ? inBands[idx] : 0.0
                            RoundedRectangle(cornerRadius: 2)
                                .fill(
                                    LinearGradient(
                                        colors: [Color.blue.opacity(0.4), Color.cyan],
                                        startPoint: .bottom,
                                        endPoint: .top
                                    )
                                )
                                .frame(maxWidth: .infinity)
                                .frame(height: max(CGFloat(val) * 44.0, 4.0))
                                .animation(.easeOut(duration: 0.08), value: val)
                        }
                    }
                    .frame(height: 48)
                    .padding(6)
                    .background(Color.black.opacity(0.3))
                    .cornerRadius(8)
                }

                // VC Output Spectrum
                VStack(alignment: .leading, spacing: 6) {
                    HStack {
                        Circle()
                            .fill(isRunning ? Color.green : Color.gray)
                            .frame(width: 8, height: 8)
                        Text("VC OUTPUT")
                            .font(.system(size: 11, weight: .bold, design: .rounded))
                            .foregroundColor(.secondary)
                        Spacer()
                        Text("\(Int(min(max(outRMS * 100, 0), 100)))%")
                            .font(.system(size: 11, weight: .semibold, design: .monospaced))
                            .foregroundColor(.green)
                    }

                    HStack(alignment: .bottom, spacing: 3) {
                        ForEach(0..<8, id: \.self) { idx in
                            let val = outBands.indices.contains(idx) ? outBands[idx] : 0.0
                            RoundedRectangle(cornerRadius: 2)
                                .fill(
                                    LinearGradient(
                                        colors: [Color.green.opacity(0.4), Color.mint],
                                        startPoint: .bottom,
                                        endPoint: .top
                                    )
                                )
                                .frame(maxWidth: .infinity)
                                .frame(height: max(CGFloat(val) * 44.0, 4.0))
                                .animation(.easeOut(duration: 0.08), value: val)
                        }
                    }
                    .frame(height: 48)
                    .padding(6)
                    .background(Color.black.opacity(0.3))
                    .cornerRadius(8)
                }
            }
        }
        .padding(14)
        .background(
            RoundedRectangle(cornerRadius: 16)
                .fill(Color(UIColor.secondarySystemBackground).opacity(0.85))
                .overlay(
                    RoundedRectangle(cornerRadius: 16)
                        .stroke(Color.white.opacity(0.08), lineWidth: 1)
                )
        )
    }
}
