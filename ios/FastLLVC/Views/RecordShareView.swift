//
//  RecordShareView.swift
//  FastLLVC
//
//  Quick WAV Recording, Playback Preview & Native Share Sheet
//  Created by Pop-chan & Antigravity
//

import SwiftUI
import AVFoundation
#if canImport(UIKit)
import UIKit
#endif

struct RecordShareView: View {
    @ObservedObject var viewModel: VoiceConversionViewModel
    @State private var audioPlayer: AVAudioPlayer?
    @State private var isPlaying = false
    @State private var showingShareSheet = false
    @State private var recordedURLToShare: URL?

    var body: some View {
        VStack(spacing: 12) {
            HStack {
                Label("Voice Recorder", systemImage: "mic.badge.plus")
                    .font(.system(size: 14, weight: .bold, design: .rounded))
                Spacer()
                if viewModel.isRecording {
                    HStack(spacing: 6) {
                        Circle()
                            .fill(Color.red)
                            .frame(width: 8, height: 8)
                        Text(formatDuration(viewModel.recordedDuration))
                            .font(.system(size: 13, weight: .semibold, design: .monospaced))
                            .foregroundColor(.red)
                    }
                }
            }

            HStack(spacing: 12) {
                // Record Button
                Button(action: {
                    triggerHaptic()
                    if viewModel.isRecording {
                        viewModel.stopRecording { url in
                            self.recordedURLToShare = url
                        }
                    } else {
                        viewModel.startRecording()
                    }
                }) {
                    HStack {
                        Image(systemName: viewModel.isRecording ? "stop.circle.fill" : "record.circle")
                            .font(.title3)
                        Text(viewModel.isRecording ? "Stop Recording" : "Quick Record")
                            .font(.system(size: 14, weight: .bold))
                    }
                    .foregroundColor(.white)
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 10)
                    .background(viewModel.isRecording ? Color.red : Color.indigo)
                    .cornerRadius(12)
                }

                // Play / Share Buttons (if recorded)
                if let url = viewModel.lastRecordedFileURL {
                    Button(action: {
                        togglePlayback(url: url)
                    }) {
                        Image(systemName: isPlaying ? "pause.fill" : "play.fill")
                            .font(.body)
                            .foregroundColor(.primary)
                            .frame(width: 38, height: 38)
                            .background(Color(UIColor.tertiarySystemBackground))
                            .cornerRadius(10)
                    }

                    Button(action: {
                        self.recordedURLToShare = url
                        self.showingShareSheet = true
                    }) {
                        Image(systemName: "square.and.arrow.up.fill")
                            .font(.body)
                            .foregroundColor(.blue)
                            .frame(width: 38, height: 38)
                            .background(Color(UIColor.tertiarySystemBackground))
                            .cornerRadius(10)
                    }
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
        .sheet(isPresented: $showingShareSheet) {
            #if os(iOS)
            if let shareURL = recordedURLToShare {
                ActivityView(activityItems: [shareURL])
            }
            #endif
        }
    }

    private func formatDuration(_ duration: TimeInterval) -> String {
        let mins = Int(duration) / 60
        let secs = Int(duration) % 60
        let ms = Int((duration - Double(Int(duration))) * 10)
        return String(format: "%02d:%02d.%d", mins, secs, ms)
    }

    private func togglePlayback(url: URL) {
        if isPlaying {
            audioPlayer?.stop()
            isPlaying = false
        } else {
            do {
                audioPlayer = try AVAudioPlayer(contentsOf: url)
                audioPlayer?.play()
                isPlaying = true
            } catch {
                print("Failed to play: \(error)")
            }
        }
    }

    private func triggerHaptic() {
        #if canImport(UIKit)
        let generator = UIImpactFeedbackGenerator(style: .medium)
        generator.impactOccurred()
        #endif
    }
}

#if os(iOS)
// iOS Native Share Sheet UIViewControllerRepresentable
struct ActivityView: UIViewControllerRepresentable {
    let activityItems: [Any]
    let applicationActivities: [UIActivity]? = nil

    func makeUIViewController(context: Context) -> UIActivityViewController {
        let controller = UIActivityViewController(
            activityItems: activityItems,
            applicationActivities: applicationActivities
        )
        return controller
    }

    func updateUIViewController(_ uiViewController: UIActivityViewController, context: Context) {}
}
#endif
