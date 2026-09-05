//
//  ModelManagerView.swift
//  FastLLVC
//
//  Core ML Model Management, Custom Model Importer & Wi-Fi PC AirTransfer
//  Created by Pop-chan & Antigravity
//

import SwiftUI
import CoreML
import UniformTypeIdentifiers
#if canImport(UIKit)
import UIKit
#endif

struct ModelItem: Identifiable, Hashable {
    let id = UUID()
    let name: String
    let description: String
    let isBuiltIn: Bool
    let url: URL?
}

struct ModelManagerView: View {
    @ObservedObject var viewModel: VoiceConversionViewModel
    @Environment(\.dismiss) private var dismiss
    @State private var showingFilePicker = false
    @State private var showingURLPrompt = false
    @State private var serverURLString = "http://192.168.0.10:8080"
    @State private var isDownloading = false
    @State private var importErrorMessage: String?
    @State private var showingErrorAlert = false

    var body: some View {
        NavigationView {
            List {
                Section(header: Text("Active Target Voice Model")) {
                    HStack(spacing: 12) {
                        Image(systemName: "cpu.fill")
                            .font(.title2)
                            .foregroundColor(.blue)
                        VStack(alignment: .leading, spacing: 4) {
                            Text(viewModel.currentModelName)
                                .font(.headline)
                            Text("Accelerated by Apple Neural Engine (ANE)")
                                .font(.caption)
                                .foregroundColor(.green)
                        }
                    }
                    .padding(.vertical, 4)
                }

                Section(header: Text("Preset Models")) {
                    ForEach(viewModel.availableModels) { model in
                        Button(action: {
                            viewModel.selectModel(model)
                        }) {
                            HStack {
                                VStack(alignment: .leading, spacing: 3) {
                                    Text(model.name)
                                        .font(.system(size: 15, weight: .semibold))
                                        .foregroundColor(.primary)
                                    Text(model.description)
                                        .font(.caption)
                                        .foregroundColor(.secondary)
                                }
                                Spacer()
                                if viewModel.currentModelName == model.name {
                                    Image(systemName: "checkmark.circle.fill")
                                        .foregroundColor(.blue)
                                }
                            }
                        }
                    }
                }

                Section(header: Text("Import Model from PC")) {
                    // Option A: Wi-Fi AirTransfer
                    Button(action: {
                        showingURLPrompt = true
                    }) {
                        HStack {
                            Image(systemName: "wifi")
                                .foregroundColor(.green)
                            VStack(alignment: .leading, spacing: 2) {
                                Text("Import via Wi-Fi from PC")
                                    .font(.system(size: 15, weight: .medium))
                                    .foregroundColor(.primary)
                                Text("Download converted .mlpackage directly from PC AirTransfer")
                                    .font(.caption2)
                                    .foregroundColor(.secondary)
                            }
                        }
                    }

                    // Option B: File Picker / AirDrop
                    Button(action: {
                        showingFilePicker = true
                    }) {
                        HStack {
                            Image(systemName: "square.and.arrow.down")
                                .foregroundColor(.blue)
                            VStack(alignment: .leading, spacing: 2) {
                                Text("Import Local File (.mlmodelc / .mlpackage)")
                                    .font(.system(size: 15, weight: .medium))
                                    .foregroundColor(.primary)
                                Text("Select from Files app, iCloud Drive, or AirDrop")
                                    .font(.caption2)
                                    .foregroundColor(.secondary)
                            }
                        }
                    }
                }
            }
            .navigationTitle("Voice Model Manager")
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
            .fileImporter(
                isPresented: $showingFilePicker,
                allowedContentTypes: [.item, .folder],
                allowsMultipleSelection: false
            ) { result in
                switch result {
                case .success(let urls):
                    guard let selectedURL = urls.first else { return }
                    if selectedURL.startAccessingSecurityScopedResource() {
                        defer { selectedURL.stopAccessingSecurityScopedResource() }
                        do {
                            try viewModel.importCustomModel(from: selectedURL)
                        } catch {
                            importErrorMessage = "Failed to load Core ML model: \(error.localizedDescription)"
                            showingErrorAlert = true
                        }
                    }
                case .failure(let error):
                    importErrorMessage = error.localizedDescription
                    showingErrorAlert = true
                }
            }
            .alert("Import from PC (Wi-Fi)", isPresented: $showingURLPrompt) {
                TextField("http://192.168.x.x:8080", text: $serverURLString)
                Button("Download & Import") {
                    downloadModelFromPC(urlString: serverURLString)
                }
                Button("Cancel", role: .cancel) {}
            } message: {
                Text("Enter the Wi-Fi AirTransfer URL displayed on your PC (e.g. from Start_PTH_to_iOS_Converter.bat).")
            }
            .alert("Model Import Error", isPresented: $showingErrorAlert) {
                Button("OK", role: .cancel) {}
            } message: {
                Text(importErrorMessage ?? "Unknown error occurred.")
            }
        }
    }

    private func downloadModelFromPC(urlString: String) {
        guard let url = URL(string: urlString) else {
            importErrorMessage = "Invalid URL format."
            showingErrorAlert = true
            return
        }

        isDownloading = true
        let task = URLSession.shared.downloadTask(with: url) { localTempURL, response, error in
            DispatchQueue.main.async {
                self.isDownloading = false
            }

            if let error = error {
                DispatchQueue.main.async {
                    self.importErrorMessage = "Download failed: \(error.localizedDescription)"
                    self.showingErrorAlert = true
                }
                return
            }

            guard let localTempURL = localTempURL else { return }
            do {
                let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
                let destURL = docs.appendingPathComponent(url.lastPathComponent)
                try? FileManager.default.removeItem(at: destURL)
                try FileManager.default.copyItem(at: localTempURL, to: destURL)

                DispatchQueue.main.async {
                    do {
                        try viewModel.importCustomModel(from: destURL)
                    } catch {
                        self.importErrorMessage = "Model initialization failed: \(error.localizedDescription)"
                        self.showingErrorAlert = true
                    }
                }
            } catch {
                DispatchQueue.main.async {
                    self.importErrorMessage = "Failed to save downloaded model: \(error.localizedDescription)"
                    self.showingErrorAlert = true
                }
            }
        }
        task.resume()
    }
}
