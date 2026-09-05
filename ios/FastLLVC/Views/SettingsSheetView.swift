//
//  SettingsSheetView.swift
//  FastLLVC
//
//  Pro Audio FX, Equalizer, Noise Gate & Route Protection Settings
//  Created by Pop-chan & Antigravity
//

import SwiftUI
#if canImport(UIKit)
import UIKit
#endif

struct SettingsSheetView: View {
    @ObservedObject var viewModel: VoiceConversionViewModel
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationView {
            Form {
                // Section 1: Audio Route Safety
                Section(header: Text("Audio Routing & Protection")) {
                    HStack {
                        Image(systemName: viewModel.isHeadphonesConnected ? "headphones" : "speaker.wave.3.fill")
                            .foregroundColor(viewModel.isHeadphonesConnected ? .green : .orange)
                        Text(viewModel.isHeadphonesConnected ? "Headphones Connected (Safe)" : "Speaker Mode (Feedback Risk)")
                            .font(.system(size: 14, weight: .medium))
                    }
                    if !viewModel.isHeadphonesConnected {
                        Text("⚠️ Connect wired or Bluetooth headphones to avoid acoustic feedback howling during real-time voice conversion.")
                            .font(.caption)
                            .foregroundColor(.orange)
                    }
                }

                // Section 2: Levels & Gate
                Section(header: Text("Gain & Noise Gate")) {
                    VStack(alignment: .leading, spacing: 6) {
                        HStack {
                            Text("Input Gain (Mic)")
                            Spacer()
                            Text(String(format: "%.1fx", viewModel.inputGain))
                                .foregroundColor(.secondary)
                        }
                        Slider(value: $viewModel.inputGain, in: 0.0...3.0, step: 0.1)
                            .onChange(of: viewModel.inputGain) { val in
                                viewModel.engine.inputGain = val
                            }
                    }

                    VStack(alignment: .leading, spacing: 6) {
                        HStack {
                            Text("Output Gain (Master)")
                            Spacer()
                            Text(String(format: "%.1fx", viewModel.outputGain))
                                .foregroundColor(.secondary)
                        }
                        Slider(value: $viewModel.outputGain, in: 0.0...3.0, step: 0.1)
                            .onChange(of: viewModel.outputGain) { val in
                                viewModel.engine.outputGain = val
                            }
                    }

                    VStack(alignment: .leading, spacing: 6) {
                        HStack {
                            Text("Noise Gate Threshold")
                            Spacer()
                            Text(String(format: "%.0f dB", viewModel.thresholdDB))
                                .foregroundColor(.secondary)
                        }
                        Slider(value: $viewModel.thresholdDB, in: -60.0...(-20.0), step: 1.0)
                            .onChange(of: viewModel.thresholdDB) { val in
                                viewModel.engine.thresholdDB = val
                            }
                    }
                }

                // Section 3: 3-Band Equalizer
                Section(header: Text("3-Band Equalizer")) {
                    VStack(alignment: .leading, spacing: 6) {
                        HStack {
                            Text("Low (Bass Boost)")
                            Spacer()
                            Text(String(format: "%+.1f dB", viewModel.lowGainDB))
                                .foregroundColor(viewModel.lowGainDB > 0 ? .blue : .secondary)
                        }
                        Slider(value: $viewModel.lowGainDB, in: -12.0...12.0, step: 0.5)
                            .onChange(of: viewModel.lowGainDB) { val in
                                viewModel.engine.lowGainDB = val
                            }
                    }

                    VStack(alignment: .leading, spacing: 6) {
                        HStack {
                            Text("Mid (Clarity)")
                            Spacer()
                            Text(String(format: "%+.1f dB", viewModel.midGainDB))
                                .foregroundColor(viewModel.midGainDB > 0 ? .green : .secondary)
                        }
                        Slider(value: $viewModel.midGainDB, in: -12.0...12.0, step: 0.5)
                            .onChange(of: viewModel.midGainDB) { val in
                                viewModel.engine.midGainDB = val
                            }
                    }

                    VStack(alignment: .leading, spacing: 6) {
                        HStack {
                            Text("High (Air / Treble)")
                            Spacer()
                            Text(String(format: "%+.1f dB", viewModel.highGainDB))
                                .foregroundColor(viewModel.highGainDB > 0 ? .purple : .secondary)
                        }
                        Slider(value: $viewModel.highGainDB, in: -12.0...12.0, step: 0.5)
                            .onChange(of: viewModel.highGainDB) { val in
                                viewModel.engine.highGainDB = val
                            }
                    }

                    Button("Reset EQ to Flat (0 dB)") {
                        viewModel.lowGainDB = 0.0
                        viewModel.midGainDB = 0.0
                        viewModel.highGainDB = 0.0
                        viewModel.engine.lowGainDB = 0.0
                        viewModel.engine.midGainDB = 0.0
                        viewModel.engine.highGainDB = 0.0
                    }
                    .font(.footnote)
                }

                // Section 4: Engine Specs
                Section(header: Text("Engine Specifications")) {
                    HStack {
                        Text("Architecture")
                        Spacer()
                        Text("Fast-LLVC (Streaming ConvNet)")
                            .foregroundColor(.secondary)
                    }
                    HStack {
                        Text("Native Sample Rate")
                        Spacer()
                        Text("16.0 kHz (DSP upsampled to 48k)")
                            .foregroundColor(.secondary)
                    }
                    HStack {
                        Text("Chunk Duration")
                        Spacer()
                        Text("13.0 ms (208 samples)")
                            .foregroundColor(.secondary)
                    }
                    HStack {
                        Text("Core ML Acceleration")
                        Spacer()
                        Text("ANE + Metal GPU")
                            .foregroundColor(.secondary)
                    }
                }
            }
            .navigationTitle("Audio & FX Settings")
            #if os(iOS)
            .navigationBarTitleDisplayMode(.inline)
            #endif
            .toolbar {
                ToolbarItem(placement: .confirmationAction) {
                    Button("Done") {
                        dismiss()
                    }
                }
            }
        }
    }
}
