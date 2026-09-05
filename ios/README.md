# 📱 Fast-LLVC for iOS / iPadOS

![iOS Version](https://img.shields.io/badge/iOS-17.0%2B-blue)
![Swift](https://img.shields.io/badge/Swift-5.9%2B-orange)
![CoreML](https://img.shields.io/badge/CoreML-ANE%20%2B%20Metal-purple)
![License](https://img.shields.io/badge/License-MIT-green)

Ultra-Low Latency, On-the-Fly Real-Time Voice Conversion App for iPhone and iPad powered by Apple Neural Engine (ANE) and Core ML.

---

## ✨ Features

- **⚡ Zero Perceptible Latency**: Sub-15ms chunk streaming inference via Core ML & Apple Neural Engine.
- **🎛️ Real-Time Pitch Shifter**: On-the-fly pitch shifting (-12 to +12 semitones) with 1-tap voice presets (Male ↔ Female, Anime, Deep Voice).
- **📊 Multi-Band Spectrum Visualizer**: Live dual-channel audio frequency meters for microphone input & converted output.
- **🛡️ Feedback Loop Protection**: Automatic headphone / Bluetooth route detection to prevent acoustic howling.
- **🎙️ Quick WAV Recorder & Share**: Record converted live voices with 1-tap and export via native iOS Share Sheet (AirDrop, Discord, Files).
- **🎚️ Pro Audio DSP FX**: 3-Band Equalizer (Bass / Mid / Air Treble), Noise Gate (-60dB to -20dB), DC Blocker, and Soft-Knee Saturation Limiter.
- **📦 Custom Model Importer**: Hot-swap custom `.mlmodelc` / `.mlpackage` neural models directly inside the app.

---

## 🏗️ Architecture

```
ios/
├── FastLLVC.xcodeproj/         # Xcode Project Settings
├── Package.swift               # Swift Package Manager Definition
└── FastLLVC/
    ├── App/
    │   ├── FastLLVCApp.swift   # App Entry Point & Audio Session
    │   └── Info.plist          # Microphone & Background Audio Permissions
    ├── Audio/
    │   ├── FastLLVCEngine.swift# Core ML & AVAudioEngine Pipeline
    │   ├── AudioDSP.swift      # vDSP/Accelerate FFT, EQ, Filter & Resampling
    │   ├── RingBuffer.swift    # Lock-Free Ring Buffer
    │   ├── PitchShifterDSP.swift # Overlap-Add (OLA) Real-time Pitch Shifter
    │   └── AudioRecorder.swift # WAV Encoder & Exporter
    ├── Views/
    │   ├── ContentView.swift   # Main Dashboard & HUD
    │   ├── VisualizerView.swift# Spectrum & Waveform Meters
    │   ├── PitchControlView.swift # Pitch Control & Presets
    │   ├── SettingsSheetView.swift# FX, Gain, Noise Gate & Routing
    │   ├── ModelManagerView.swift # Model Manager & File Picker
    │   └── RecordShareView.swift  # Quick Record & Share
    └── Assets.xcassets/        # App Icons & Color Schemes
```

---

## 🚀 Building & Running

### Option 1: Using Xcode
1. Open `ios/FastLLVC.xcodeproj` in Xcode 15+.
2. Select your target device (iPhone / iPad or Simulator).
3. Press `Cmd + R` to build and run.

### Option 2: Swift Package Manager
```bash
cd ios
swift build
```

### Option 3: GitHub Actions CI/CD
Whenever code is pushed to `main` or triggered via `workflow_dispatch`, GitHub Actions automatically builds the Release `.app` and produces a downloadable zip artifact.
