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
    @State private var serverURLString = "http://192.168.0.10:8080/model.torchscript.pt"
    @State private var isDownloading = false
    @State private var importErrorMessage: String?
    @State private var showingErrorAlert = false
    @State private var showingSuccessAlert = false
    @State private var successMessage = ""

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
                                Text("Download directly from PC AirTransfer URL")
                                    .font(.caption2)
                                    .foregroundColor(.secondary)
                            }
                        }
                    }

                    // Option B: Native iOS Document Picker
                    Button(action: {
                        showingFilePicker = true
                    }) {
                        HStack {
                            Image(systemName: "folder.badge.plus")
                                .foregroundColor(.blue)
                            VStack(alignment: .leading, spacing: 2) {
                                Text("Select File from Files App / iCloud / AirDrop")
                                    .font(.system(size: 15, weight: .medium))
                                    .foregroundColor(.primary)
                                Text("Pick .pt, .pth, .mlmodelc, or .mlpackage")
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
            .sheet(isPresented: $showingFilePicker) {
                #if os(iOS)
                NativeDocumentPicker { selectedURL in
                    guard let url = selectedURL else { return }
                    do {
                        try viewModel.importCustomModel(from: url)
                        successMessage = "Successfully imported: \(url.lastPathComponent)"
                        showingSuccessAlert = true
                    } catch {
                        importErrorMessage = "Failed to import model: \(error.localizedDescription)"
                        showingErrorAlert = true
                    }
                }
                #endif
            }
            .alert("Import from PC (Wi-Fi)", isPresented: $showingURLPrompt) {
                TextField("http://192.168.0.10:8080/model.torchscript.pt", text: $serverURLString)
                Button("Download & Import") {
                    downloadModelFromPC(urlString: serverURLString)
                }
                Button("Cancel", role: .cancel) {}
            } message: {
                Text("Enter the AirTransfer URL from PC (e.g. http://192.168.0.10:8080 or http://192.168.0.10:8080/zundamon.torchscript.pt)")
            }
            .alert("Model Import Error", isPresented: $showingErrorAlert) {
                Button("OK", role: .cancel) {}
            } message: {
                Text(importErrorMessage ?? "Unknown error occurred.")
            }
            .alert("Model Imported!", isPresented: $showingSuccessAlert) {
                Button("OK", role: .cancel) {}
            } message: {
                Text(successMessage)
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
                let filename = url.lastPathComponent.isEmpty ? "imported_model.torchscript.pt" : url.lastPathComponent
                let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
                let destURL = docs.appendingPathComponent(filename)
                
                try? FileManager.default.removeItem(at: destURL)
                try FileManager.default.copyItem(at: localTempURL, to: destURL)

                DispatchQueue.main.async {
                    do {
                        try viewModel.importCustomModel(from: destURL)
                        successMessage = "Successfully downloaded & applied: \(filename)"
                        showingSuccessAlert = true
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

#if os(iOS)
// High-Reliability Native iOS UIDocumentPickerViewController
struct NativeDocumentPicker: UIViewControllerRepresentable {
    let onPick: (URL?) -> Void

    func makeCoordinator() -> Coordinator {
        Coordinator(onPick: onPick)
    }

    func makeUIViewController(context: Context) -> UIDocumentPickerViewController {
        let picker = UIDocumentPickerViewController(forOpeningContentTypes: [.item, .data, .content, .folder], asCopy: true)
        picker.delegate = context.coordinator
        picker.allowsMultipleSelection = false
        return picker
    }

    func updateUIViewController(_ uiViewController: UIDocumentPickerViewController, context: Context) {}

    class Coordinator: NSObject, UIDocumentPickerDelegate {
        let onPick: (URL?) -> Void

        init(onPick: @escaping (URL?) -> Void) {
            self.onPick = onPick
        }

        func documentPicker(_ controller: UIDocumentPickerViewController, didPickDocumentsAt urls: [URL]) {
            guard let url = urls.first else {
                onPick(nil)
                return
            }
            onPick(url)
        }

        func documentPickerWasCancelled(_ controller: UIDocumentPickerViewController) {
            onPick(nil)
        }
    }
}
#endif
