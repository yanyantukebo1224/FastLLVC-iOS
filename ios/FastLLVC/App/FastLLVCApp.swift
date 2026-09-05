//
//  FastLLVCApp.swift
//  FastLLVC
//
//  Created by Pop-chan & Antigravity
//

import SwiftUI
import AVFoundation

@main
struct FastLLVCApp: App {
    @StateObject private var viewModel = VoiceConversionViewModel()
    
    init() {
        // Configure global audio session settings
        do {
            let session = AVAudioSession.sharedInstance()
            try session.setCategory(.playAndRecord, mode: .voiceChat, options: [.defaultToSpeaker, .allowBluetooth, .allowAirPlay])
            try session.setActive(true)
        } catch {
            print("Failed to initialize AVAudioSession: \(error)")
        }
    }
    
    var body: some Scene {
        WindowGroup {
            ContentView(viewModel: viewModel)
                .preferredColorScheme(.dark)
        }
    }
}
