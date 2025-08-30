import SwiftUI
import AppKit

struct ContentView: View {
    @State private var engine: SuperResolutionEngine? = try? SuperResolutionEngine(modelNameInBundle: "RealESRGAN_x4v3_tensor")
    @State private var inputURL: URL?
    @State private var inputImage: NSImage?
    @State private var outputImage: NSImage?
    @State private var isProcessing = false
    @State private var progress: Double = 0
    @State private var message: String?

    var body: some View {
        VStack(spacing: 12) {
            HStack {
                Button {
                    openImage()
                } label: { Label("開く…", systemImage: "folder") }

                Button {
                    runSR()
                } label: { Label("高画質化", systemImage: "sparkles") }
                .disabled(inputImage == nil || engine == nil || isProcessing)

                Button {
                    saveOutput()
                } label: { Label("保存", systemImage: "square.and.arrow.down") }
                .disabled(outputImage == nil || isProcessing)

                Spacer()
                if isProcessing {
                    ProgressView(value: progress)
                        .frame(width: 180)
                }
            }

            HStack {
                VStack {
                    Text("入力")
                    ZStack {
                        Color(NSColor.underPageBackgroundColor)
                        if let img = inputImage {
                            Image(nsImage: img).resizable().scaledToFit()
                        } else {
                            Text("画像を開いてください").foregroundStyle(.secondary)
                        }
                    }
                }
                VStack {
                    Text("出力")
                    ZStack {
                        Color(NSColor.underPageBackgroundColor)
                        if let img = outputImage {
                            Image(nsImage: img).resizable().scaledToFit()
                        } else {
                            Text("未生成").foregroundStyle(.secondary)
                        }
                    }
                }
            }
        }
        .padding(12)
        .frame(minWidth: 900, minHeight: 560)
        .alert(item: Binding(get: {
            message.map { AlertMessage(text: $0) }
        }, set: { _ in message = nil })) { msg in
            Alert(title: Text(msg.text))
        }
    }

    private func openImage() {
        let panel = NSOpenPanel()
        panel.allowedContentTypes = [.image]
        panel.allowsMultipleSelection = false
        if panel.runModal() == .OK, let url = panel.url, let nsimg = NSImage(contentsOf: url) {
            inputURL = url
            inputImage = nsimg
            outputImage = nil
        }
    }

    private func runSR() {
        guard let cg = inputImage?.cgImage, let engine else { return }
        isProcessing = true
        progress = 0
        outputImage = nil

        DispatchQueue.global(qos: .userInitiated).async {
            do {
                let outCG = try engine.superResolve(
                    cgImage: cg,
                    tileSize: 512,     // 実際はエンジン内で固定入力サイズに自動調整
                    overlap: 32,
                    progress: { p in DispatchQueue.main.async { self.progress = p } },
                    cancellation: nil
                )
                let outNS = NSImage(cgImage: outCG, size: NSSize(width: outCG.width, height: outCG.height))
                DispatchQueue.main.async {
                    self.outputImage = outNS
                    self.isProcessing = false
                    self.progress = 1.0
                }
            } catch {
                DispatchQueue.main.async {
                    self.isProcessing = false
                    self.message = "失敗: \(error)"
                }
            }
        }
    }

    private func saveOutput() {
        guard let out = outputImage,
              let tiff = out.tiffRepresentation,
              let rep = NSBitmapImageRep(data: tiff),
              let data = rep.representation(using: .png, properties: [:]) else {
            message = "保存用データ作成に失敗"
            return
        }
        let panel = NSSavePanel()
        panel.nameFieldStringValue = (inputURL?.deletingPathExtension().lastPathComponent ?? "output") + "_x4.png"
        panel.allowedContentTypes = [.png]
        if panel.runModal() == .OK, let url = panel.url {
            do {
                try data.write(to: url)
                message = "保存しました: \(url.lastPathComponent)"
            } catch {
                message = "保存に失敗: \(error)"
            }
        }
    }
}

private struct AlertMessage: Identifiable { let id = UUID(); let text: String }

private extension NSImage {
    var cgImage: CGImage? {
        var rect = CGRect(origin: .zero, size: size)
        return cgImage(forProposedRect: &rect, context: nil, hints: nil)
    }
}
