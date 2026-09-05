//
//  PitchControlView.swift
//  FastLLVC
//
//  Real-Time Pitch Shifter & Voice Conversion Presets
//  Created by Pop-chan & Antigravity
//

import SwiftUI
#if canImport(UIKit)
import UIKit
#endif

struct PitchControlView: View {
    @Binding var pitchSemitones: Float
    let onChange: (Float) -> Void

    private let presets: [(title: String, semitones: Float, icon: String)] = [
        ("Original", 0.0, "person.fill"),
        ("Female", 12.0, "sparkles"),
        ("Male", -12.0, "waveform"),
        ("Anime", 6.0, "bolt.heart.fill"),
        ("Deep", -4.0, "speaker.wave.3.fill")
    ]

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            HStack {
                Label("Pitch Shift (音高変換)", systemImage: "music.quarternote.3")
                    .font(.system(size: 14, weight: .bold, design: .rounded))
                Spacer()
                Text(pitchText)
                    .font(.system(size: 15, weight: .heavy, design: .monospaced))
                    .foregroundColor(pitchColor)
                    .padding(.horizontal, 10)
                    .padding(.vertical, 4)
                    .background(pitchColor.opacity(0.15))
                    .cornerRadius(8)
            }

            // Slider with Fine-tune Step
            HStack(spacing: 12) {
                Button(action: {
                    triggerHaptic()
                    pitchSemitones = max(-12.0, pitchSemitones - 1.0)
                    onChange(pitchSemitones)
                }) {
                    Image(systemName: "minus.circle.fill")
                        .font(.title3)
                        .foregroundColor(.secondary)
                }

                Slider(
                    value: Binding(
                        get: { pitchSemitones },
                        set: { val in
                            pitchSemitones = val
                            onChange(val)
                        }
                    ),
                    in: -12.0...12.0,
                    step: 0.5
                ) {
                    Text("Pitch")
                } minimumValueLabel: {
                    Text("-12").font(.caption2).foregroundColor(.secondary)
                } maximumValueLabel: {
                    Text("+12").font(.caption2).foregroundColor(.secondary)
                }
                .tint(pitchColor)

                Button(action: {
                    triggerHaptic()
                    pitchSemitones = min(12.0, pitchSemitones + 1.0)
                    onChange(pitchSemitones)
                }) {
                    Image(systemName: "plus.circle.fill")
                        .font(.title3)
                        .foregroundColor(.secondary)
                }
            }

            // Quick Preset Chips
            ScrollView(.horizontal, showsIndicators: false) {
                HStack(spacing: 8) {
                    ForEach(presets, id: \.title) { p in
                        let isSelected = abs(pitchSemitones - p.semitones) < 0.1
                        Button(action: {
                            triggerHaptic()
                            pitchSemitones = p.semitones
                            onChange(pitchSemitones)
                        }) {
                            HStack(spacing: 5) {
                                Image(systemName: p.icon)
                                    .font(.caption2)
                                Text(p.title)
                                    .font(.system(size: 12, weight: .semibold, design: .rounded))
                            }
                            .padding(.horizontal, 12)
                            .padding(.vertical, 7)
                            .background(
                                isSelected ?
                                Color.blue :
                                Color(UIColor.tertiarySystemBackground)
                            )
                            .foregroundColor(isSelected ? .white : .primary)
                            .cornerRadius(12)
                            .overlay(
                                RoundedRectangle(cornerRadius: 12)
                                    .stroke(isSelected ? Color.blue.opacity(0.6) : Color.white.opacity(0.05), lineWidth: 1)
                            )
                        }
                    }
                }
            }
        }
        .padding(16)
        .background(
            RoundedRectangle(cornerRadius: 16)
                .fill(Color(UIColor.secondarySystemBackground).opacity(0.85))
                .overlay(
                    RoundedRectangle(cornerRadius: 16)
                        .stroke(Color.white.opacity(0.08), lineWidth: 1)
                )
        )
    }

    private var pitchText: String {
        if pitchSemitones > 0 {
            return String(format: "+%.1f st", pitchSemitones)
        } else if pitchSemitones < 0 {
            return String(format: "%.1f st", pitchSemitones)
        } else {
            return "0.0 st (Off)"
        }
    }

    private var pitchColor: Color {
        if pitchSemitones > 0.05 {
            return .pink
        } else if pitchSemitones < -0.05 {
            return .indigo
        } else {
            return .secondary
        }
    }

    private func triggerHaptic() {
        #if canImport(UIKit)
        let generator = UIImpactFeedbackGenerator(style: .light)
        generator.impactOccurred()
        #endif
    }
}
