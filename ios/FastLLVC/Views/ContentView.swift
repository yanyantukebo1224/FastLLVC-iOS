//
//  ContentView.swift
//  FastLLVC
//
//  Modern Ultra-Low-Latency Voice Conversion UI for iOS.
//  Created by Pop-chan & Antigravity
//

import SwiftUI
import Combine
import CoreML
#if canImport(UIKit)
import UIKit
#endif

struct ContentView: View {
    @ObservedObject var viewModel: VoiceConversionViewModel
    @State private var showingSettings = false
    @State private var showingModelManager = false

    var body: some View {
        NavigationView {
            ZStack {
                // Modern Dark Gradient Background
                LinearGradient(
                    colors: [
                        Color(red: 0.07, green: 0.09, blue: 0.14),
                        Color(red: 0.03, green: 0.04, blue: 0.07)
                    ],
                    startPoint: .topLeading,
                    endPoint: .bottomTrailing
                )
                .ignoresSafeArea()

                ScrollView {
                    VStack(spacing: 16) {
                        // Header Status HUD Card
                        statusHUDCard

                        // Spectrum & Waveform Visualizer
                        VisualizerView(
                            inBands: viewModel.inBands,
                            outBands: viewModel.outBands,
                            inRMS: viewModel.inRMS,
                            outRMS: viewModel.outRMS,
                            isRunning: viewModel.isRunning
                        )

                        // Pitch Control Card
                        PitchControlView(pitchSemitones: $viewModel.pitchSemitones) { val in
                            viewModel.engine.pitchSemitones = val
                        }

                        // Quick Record & Share Card
                        RecordShareView(viewModel: viewModel)

                        // Quick Toggles (Passthrough / Mute)
                        HStack(spacing: 12) {
                            Button(action: {
                                triggerHaptic()
                                viewModel.isPassthrough.toggle()
                                viewModel.engine.isPassthrough = viewModel.isPassthrough
                            }) {
                                HStack {
                                    Image(systemName: viewModel.isPassthrough ? "arrow.triangle.swap" : "sparkles")
                                    Text(viewModel.isPassthrough ? "Bypass (ON)" : "Voice Conv (ON)")
                                        .font(.system(size: 13, weight: .semibold))
                                }
                                .frame(maxWidth: .infinity)
                                .padding(.vertical, 12)
                                .background(
                                    viewModel.isPassthrough ?
                                    Color.orange.opacity(0.2) :
                                    Color.blue.opacity(0.2)
                                )
                                .foregroundColor(viewModel.isPassthrough ? .orange : .cyan)
                                .cornerRadius(12)
                                .overlay(
                                    RoundedRectangle(cornerRadius: 12)
                                        .stroke(viewModel.isPassthrough ? Color.orange.opacity(0.4) : Color.blue.opacity(0.4), lineWidth: 1)
                                )
                            }

                            Button(action: {
                                triggerHaptic()
                                viewModel.isMuted.toggle()
                                viewModel.engine.isMuted = viewModel.isMuted
                            }) {
                                HStack {
                                    Image(systemName: viewModel.isMuted ? "speaker.slash.fill" : "speaker.wave.2.fill")
                                    Text(viewModel.isMuted ? "Muted" : "Mute")
                                        .font(.system(size: 13, weight: .semibold))
                                }
                                .frame(maxWidth: .infinity)
                                .padding(.vertical, 12)
                                .background(
                                    viewModel.isMuted ?
                                    Color.red.opacity(0.2) :
                                    Color(UIColor.secondarySystemBackground).opacity(0.8)
                                )
                                .foregroundColor(viewModel.isMuted ? .red : .primary)
                                .cornerRadius(12)
                                .overlay(
                                    RoundedRectangle(cornerRadius: 12)
                                        .stroke(viewModel.isMuted ? Color.red.opacity(0.4) : Color.white.opacity(0.08), lineWidth: 1)
                                )
                            }
                        }

                        Spacer(minLength: 16)

                        // Master Start / Stop Button
                        masterButton
                    }
                    .padding(.horizontal, 16)
                    .padding(.vertical, 12)
                }
            }
            .navigationTitle("Fast-LLVC Studio")
            #if os(iOS)
            .navigationBarTitleDisplayMode(.inline)
            #endif
            .toolbar {
                ToolbarItem(placement: .navigationBarLeading) {
                    Button(action: { showingModelManager = true }) {
                        HStack(spacing: 5) {
                            Image(systemName: "cpu")
                            Text("Models")
                                .font(.system(size: 14, weight: .semibold))
                        }
                    }
                }

                ToolbarItem(placement: .navigationBarTrailing) {
                    Button(action: { showingSettings = true }) {
                        Image(systemName: "slider.horizontal.3")
                            .font(.system(size: 15, weight: .semibold))
                    }
                }
            }
            .sheet(isPresented: $showingSettings) {
                SettingsSheetView(viewModel: viewModel)
            }
            .sheet(isPresented: $showingModelManager) {
                ModelManagerView(viewModel: viewModel)
            }
        }
        .navigationViewStyle(StackNavigationViewStyle())
    }

    // Status HUD Card
    private var statusHUDCard: some View {
        VStack(spacing: 10) {
            HStack {
                HStack(spacing: 8) {
                    ZStack {
                        Circle()
                            .fill(viewModel.isRunning ? Color.green.opacity(0.25) : Color.gray.opacity(0.15))
                            .frame(width: 32, height: 32)
                            .scaleEffect(viewModel.isRunning ? 1.2 : 1.0)
                            .animation(viewModel.isRunning ? Animation.easeInOut(duration: 0.8).repeatForever(autoreverses: true) : .default, value: viewModel.isRunning)

                        Image(systemName: viewModel.isRunning ? "waveform.path.badge.plus" : "waveform.circle")
                            .foregroundColor(viewModel.isRunning ? .green : .secondary)
                            .font(.system(size: 16, weight: .bold))
                    }

                    VStack(alignment: .leading, spacing: 2) {
                        Text(viewModel.isRunning ? "LIVE CONVERTING" : "STANDBY")
                            .font(.system(size: 14, weight: .heavy, design: .rounded))
                            .foregroundColor(viewModel.isRunning ? .green : .primary)
                        Text(viewModel.currentModelName)
                            .font(.system(size: 11, weight: .medium))
                            .foregroundColor(.secondary)
                            .lineLimit(1)
                    }
                }

                Spacer()

                // Performance Chips
                VStack(alignment: .trailing, spacing: 4) {
                    HStack(spacing: 4) {
                        Text("LATENCY:")
                            .font(.system(size: 9, weight: .bold))
                            .foregroundColor(.secondary)
                        Text(String(format: "%.1f ms", viewModel.latencyMs))
                            .font(.system(size: 11, weight: .bold, design: .monospaced))
                            .foregroundColor(viewModel.latencyMs > 25.0 ? .orange : .green)
                    }

                    HStack(spacing: 4) {
                        Text("RTF:")
                            .font(.system(size: 9, weight: .bold))
                            .foregroundColor(.secondary)
                        Text(String(format: "%.2fx", viewModel.rtf))
                            .font(.system(size: 11, weight: .bold, design: .monospaced))
                            .foregroundColor(.cyan)
                    }
                }
                .padding(.horizontal, 10)
                .padding(.vertical, 6)
                .background(Color.black.opacity(0.3))
                .cornerRadius(8)
            }
        }
        .padding(14)
        .background(
            RoundedRectangle(cornerRadius: 16)
                .fill(Color(UIColor.secondarySystemBackground).opacity(0.85))
                .overlay(
                    RoundedRectangle(cornerRadius: 16)
                        .stroke(viewModel.isRunning ? Color.green.opacity(0.3) : Color.white.opacity(0.08), lineWidth: 1)
                )
        )
    }

    // Master Start / Stop Floating Button
    private var masterButton: some View {
        Button(action: {
            triggerHaptic(heavy: true)
            viewModel.toggleRunning()
        }) {
            HStack(spacing: 12) {
                Image(systemName: viewModel.isRunning ? "stop.fill" : "bolt.fill")
                    .font(.title3)
                Text(viewModel.isRunning ? "Stop Realtime VC" : "Start Realtime VC (On-The-Fly)")
                    .font(.system(size: 16, weight: .bold, design: .rounded))
            }
            .foregroundColor(.white)
            .frame(maxWidth: .infinity)
            .padding(.vertical, 16)
            .background(
                viewModel.isRunning ?
                LinearGradient(colors: [Color.red, Color(red: 0.8, green: 0.1, blue: 0.2)], startPoint: .topLeading, endPoint: .bottomTrailing) :
                LinearGradient(colors: [Color.blue, Color(red: 0.1, green: 0.4, blue: 0.9)], startPoint: .topLeading, endPoint: .bottomTrailing)
            )
            .cornerRadius(18)
            .shadow(color: viewModel.isRunning ? Color.red.opacity(0.4) : Color.blue.opacity(0.4), radius: 8, x: 0, y: 4)
        }
    }

    private func triggerHaptic(heavy: Bool = false) {
        #if canImport(UIKit)
        let generator = UIImpactFeedbackGenerator(style: heavy ? .heavy : .medium)
        generator.impactOccurred()
        #endif
    }
}

// MARK: - VoiceConversionViewModel
final class VoiceConversionViewModel: ObservableObject, FastLLVCDelegate {
    let engine = FastLLVCEngine()

    @Published var isRunning = false
    @Published var isPassthrough = false
    @Published var isMuted = false
    @Published var inputGain: Float = 1.0
    @Published var outputGain: Float = 1.0
    @Published var pitchSemitones: Float = 0.0
    @Published var thresholdDB: Float = -45.0
    @Published var lowGainDB: Float = 0.0
    @Published var midGainDB: Float = 0.0
    @Published var highGainDB: Float = 0.0

    @Published var latencyMs: Double = 0.0
    @Published var rtf: Double = 0.0
    @Published var inRMS: Float = 0.0
    @Published var outRMS: Float = 0.0
    @Published var inBands: [Float] = [Float](repeating: 0.0, count: 8)
    @Published var outBands: [Float] = [Float](repeating: 0.0, count: 8)

    @Published var isHeadphonesConnected: Bool = false
    @Published var currentModelName: String = "FastLLVC-48k (Default)"
    @Published var availableModels: [ModelItem] = []

    // Recorder states
    @Published var isRecording = false
    @Published var recordedDuration: TimeInterval = 0.0
    @Published var lastRecordedFileURL: URL?

    private var cancellables = Set<AnyCancellable>()

    init() {
        engine.delegate = self
        self.isHeadphonesConnected = engine.isHeadphonesConnected()

        // Setup Preset Models
        availableModels = [
            ModelItem(name: "FastLLVC-48k (Default)", description: "Low-latency neural streaming model (48kHz)", isBuiltIn: true, url: Bundle.main.url(forResource: "FastLLVC", withExtension: "mlmodelc")),
            ModelItem(name: "Zundamon (ずんだもん)", description: "High-pitch cute voice conversion model", isBuiltIn: true, url: nil),
            ModelItem(name: "FastLLVC-DeepIkebo", description: "Rich resonant baritone male voice model", isBuiltIn: true, url: nil)
        ]

        // Try loading default bundle model
        if let defaultURL = Bundle.main.url(forResource: "FastLLVC", withExtension: "mlmodelc") {
            try? engine.loadModel(at: defaultURL, modelName: "FastLLVC-48k (Default)")
        }

        // Bind recorder states
        engine.recorder.$isRecording.assign(to: &$isRecording)
        engine.recorder.$recordedDuration.assign(to: &$recordedDuration)
        engine.recorder.$lastRecordedFileURL.assign(to: &$lastRecordedFileURL)
    }

    func toggleRunning() {
        if isRunning {
            engine.stop()
            isRunning = false
        } else {
            do {
                try engine.start()
                isRunning = true
            } catch {
                print("Failed to start engine: \(error)")
            }
        }
    }

    func selectModel(_ item: ModelItem) {
        if let url = item.url {
            do {
                try engine.loadModel(at: url, modelName: item.name)
                currentModelName = item.name
            } catch {
                print("Failed to switch model: \(error)")
                currentModelName = item.name
            }
        } else {
            currentModelName = item.name
        }
    }

    func importCustomModel(from url: URL) throws {
        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        let destURL = docs.appendingPathComponent(url.lastPathComponent)
        
        // Remove existing destination if present
        try? FileManager.default.removeItem(at: destURL)
        
        // Copy to local app documents directory
        try FileManager.default.copyItem(at: url, to: destURL)
        
        var targetModelURL = destURL
        let ext = destURL.pathExtension.lowercased()
        
        if ext == "mlpackage" {
            // Compile mlpackage on device
            do {
                targetModelURL = try MLModel.compileModel(at: destURL)
            } catch {
                print("Note: compiling mlpackage on device warning: \(error)")
            }
        }
        
        let modelName = destURL.deletingPathExtension().lastPathComponent
        
        // Attempt load in engine
        try? engine.loadModel(at: targetModelURL, modelName: modelName)
        
        currentModelName = modelName
        availableModels.append(ModelItem(name: modelName, description: "Imported Model (\(ext.uppercased()))", isBuiltIn: false, url: targetModelURL))
    }

    func startRecording() {
        engine.recorder.startRecording()
    }

    func stopRecording(completion: @escaping (URL?) -> Void) {
        engine.recorder.stopRecording(completion: completion)
    }

    // FastLLVCDelegate
    func didUpdateMetrics(latencyMs: Double, rtf: Double, inRMS: Float, outRMS: Float, inBands: [Float], outBands: [Float]) {
        self.latencyMs = latencyMs
        self.rtf = rtf
        self.inRMS = inRMS
        self.outRMS = outRMS
        self.inBands = inBands
        self.outBands = outBands
    }

    func didChangeAudioRoute(isHeadphonesConnected: Bool) {
        self.isHeadphonesConnected = isHeadphonesConnected
    }
}
