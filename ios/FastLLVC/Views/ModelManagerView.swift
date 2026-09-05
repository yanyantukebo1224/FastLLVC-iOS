//
//  ModelManagerView.swift
//  FastLLVC
//
//  Core ML Model Management & Custom Model Importer
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

                Section(header: Text("Import Custom Model")) {
                    Button(action: {
                        showingFilePicker = true
                    }) {
                        Label("Import .mlmodelc / .mlpackage", systemImage: "square.and.arrow.down")
                            .font(.system(size: 15, weight: .medium))
                    }
                    Text("You can export custom Fast-LLVC voice models from Python using export_coreml.py and transfer via AirDrop or Files.")
                        .font(.caption2)
                        .foregroundColor(.secondary)
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
            .alert("Model Import Error", isPresented: $showingErrorAlert) {
                Button("OK", role: .cancel) {}
            } message: {
                Text(importErrorMessage ?? "Unknown error occurred.")
            }
        }
    }
}
