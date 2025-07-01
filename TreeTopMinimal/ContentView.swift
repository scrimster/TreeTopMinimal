//  ContentView.swift
import SwiftUI
import Vision
import CoreML

struct ContentView: View {
    @State private var pickedImage: UIImage?
    @State private var canopyPercentage: Double?
    @State private var showPicker = false

    // error handling
    @State private var errorMessage: String?
    @State private var showErrorAlert = false

    // MARK: Vision model (create once)
    private static let visionModel: VNCoreMLModel = {
        do {
            let mlModel = try Segmentation(configuration: .init()).model // <-- your helper class
            return try VNCoreMLModel(for: mlModel)
        } catch {
            fatalError("❌ couldn’t load model: \(error)")
        }
    }()

    // ---------- UI ----------
    var body: some View {
        NavigationView {
            VStack(spacing: 24) {
                // picked image preview
                Group {
                    if let ui = pickedImage {
                        Image(uiImage: ui)
                            .resizable()
                            .scaledToFit()
                    } else {
                        Rectangle()
                            .fill(.secondary.opacity(0.1))
                            .overlay(Text("Tap below to select image"))
                    }
                }
                .frame(height: 300)
                .cornerRadius(8)

                // result label
                if let pct = canopyPercentage {
                    Text("Canopy cover: \(pct, specifier: "%.1f") %")
                        .font(.title2)
                }

                // buttons
                Button("Pick image")  { showPicker = true }
                Button("Analyze")     { if let img = pickedImage { analyze(img) } }
                    .disabled(pickedImage == nil)
            }
            .padding()
            .navigationTitle("TreeTop")
            .sheet(isPresented: $showPicker) {
                ImagePicker(image: $pickedImage)
            }
            .alert(isPresented: $showErrorAlert) {
                Alert(title: Text("Analysis failed"),
                      message: Text(errorMessage ?? "Unknown error"),
                      dismissButton: .default(Text("OK")))
            }
        }
    }

    // ---------- Analysis ----------
    private func analyze(_ image: UIImage) {
        guard let cg = image.cgImage else { return setError("Couldn’t get CGImage") }

        let request = VNCoreMLRequest(model: Self.visionModel) { req, err in
            if let err { return setError("Vision error: \(err.localizedDescription)") }
            guard let first = req.results?.first else { return setError("Model returned no results") }

            // 1. CLASSIFICATION (single label)
            if let cls = first as? VNClassificationObservation {
                DispatchQueue.main.async { canopyPercentage = Double(cls.confidence) * 100 }
                return
            }

            // 2. SEGMENTATION (binary mask in MLMultiArray)
            if let fv = first as? VNCoreMLFeatureValueObservation,
               let mask = fv.featureValue.multiArrayValue {

                let canopySum = (0..<mask.count).reduce(0.0) { $0 + mask[$1].doubleValue }
                let pct = canopySum / Double(mask.count) * 100

                DispatchQueue.main.async { canopyPercentage = pct }
                return
            }

            setError("Unsupported result type \(type(of: first))")
        }

        let handler = VNImageRequestHandler(cgImage: cg, orientation: .up)
        DispatchQueue.global(qos: .userInitiated).async {
            do    { try handler.perform([request]) }
            catch { setError("Handler failed: \(error)") }
        }
    }

    // ---------- helper ----------
    private func setError(_ msg: String) {
        errorMessage = msg
        showErrorAlert = true
        print("❌", msg)
    }
}
