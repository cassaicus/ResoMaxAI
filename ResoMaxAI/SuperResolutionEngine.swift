import Foundation
import AppKit
import CoreML
import CoreImage
import Vision

final class SuperResolutionEngine {
    enum SRError: Error {
        case modelLoadFailed
        case firstTileFailed
        case cancelled
    }

    final class CancellationToken {
        private let lock = NSLock()
        private var _isCancelled = false
        var isCancelled: Bool { lock.lock(); defer { lock.unlock() }; return _isCancelled }
        func cancel() { lock.lock(); _isCancelled = true; lock.unlock() }
    }

    private let model: MLModel
    private let ciContext = CIContext(options: [.useSoftwareRenderer: false])
    private let inputName: String
    private let outputName: String
    private let inputConstraint: MLImageConstraint

    init(modelNameInBundle: String, computeUnits: MLComputeUnits = .cpuAndGPU) throws {
        let config = MLModelConfiguration()
        config.computeUnits = computeUnits

        let bundle = Bundle.main
        let candidates: [URL] = [
            bundle.url(forResource: modelNameInBundle, withExtension: "mlpackage"),
            bundle.url(forResource: modelNameInBundle, withExtension: "mlmodelc"),
            bundle.url(forResource: modelNameInBundle, withExtension: "mlmodel")
        ].compactMap { $0 }

        guard let foundURL = candidates.first else { throw SRError.modelLoadFailed }

        let loadURL: URL
        if foundURL.pathExtension == "mlmodel" {
            guard let compiled = try? MLModel.compileModel(at: foundURL) else { throw SRError.modelLoadFailed }
            loadURL = compiled
        } else {
            loadURL = foundURL
        }

        self.model = try MLModel(contentsOf: loadURL, configuration: config)

        guard
            let inEntry = model.modelDescription.inputDescriptionsByName.first,
            let outEntry = model.modelDescription.outputDescriptionsByName.first,
            let inConstraint = model.modelDescription.inputDescriptionsByName[inEntry.key]?.imageConstraint
        else { throw SRError.modelLoadFailed }

        self.inputName = inEntry.key
        self.outputName = outEntry.key
        self.inputConstraint = inConstraint

        print("SR Engine: loaded = \(loadURL.lastPathComponent)")
        print("SR Engine: inputName = \(inputName), outputName = \(outputName)")
        print("SR Engine: inputConstraint \(inConstraint.pixelsWide)x\(inConstraint.pixelsHigh), pixelFormat=\(inConstraint.pixelFormatType)")
    }
    
    
    
    func superResolve(
        cgImage: CGImage,
        tileSize: Int = 1024,
        overlap: Int = 24,
        progress: ((Double) -> Void)? = nil,
        cancellation: CancellationToken? = nil
    ) throws -> CGImage {
        if cancellation?.isCancelled == true { throw SRError.cancelled }

        let fixedW = max(1, inputConstraint.pixelsWide)
        let fixedH = max(1, inputConstraint.pixelsHigh)
        let usedTile = min(tileSize, max(fixedW, fixedH))
        let usedOverlap = min(overlap, usedTile / 2)

        let tiles = Self.splitIntoTiles(image: cgImage, tileSize: usedTile, overlap: usedOverlap)
        guard let firstTile = tiles.first else { return cgImage }

        // 最初のタイルでスケール倍率を確定
        guard let firstTileOut = autoreleasepool(invoking: {
            self.applyModelWithPadding(on: firstTile.image, fixedW: fixedW, fixedH: fixedH)
        }) else {
            throw SRError.firstTileFailed
        }
        let scaleInt = max(1, Int(round(Double(firstTileOut.width) / Double(firstTile.image.width))))
        let overlapUpscaled = usedOverlap * scaleInt
        let halfOverlap = overlapUpscaled / 2

        let outW = cgImage.width * scaleInt
        let outH = cgImage.height * scaleInt
        guard let ctx = CGContext(
            data: nil,
            width: outW,
            height: outH,
            bitsPerComponent: 8,
            bytesPerRow: 0,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        ) else { return cgImage }

        let imgW = cgImage.width
        let imgH = cgImage.height
        let total = tiles.count
        var done = 0

        for tile in tiles {
            if cancellation?.isCancelled == true { throw SRError.cancelled }
            autoreleasepool {
                guard let outTile = self.applyModelWithPadding(on: tile.image, fixedW: fixedW, fixedH: fixedH) else {
                    print("⚠️ タイル推論失敗: \(tile.rect)")
                    return
                }

                let x0In = Int(tile.rect.minX)
                let y0InTop = Int(tile.rect.minY)
                let x1In = Int(tile.rect.maxX)
                let y1InTop = Int(tile.rect.maxY)

                let hasLeft   = x0In > 0
                let hasRight  = x1In < imgW
                let hasTop    = y0InTop > 0
                let hasBottom = y1InTop < imgH

                let insetL = hasLeft   ? halfOverlap : 0
                let insetR = hasRight  ? halfOverlap : 0
                let insetT = hasTop    ? halfOverlap : 0
                let insetB = hasBottom ? halfOverlap : 0

                // 出力タイルからパディング分を除去
                let padLeft   = (fixedW - tile.image.width) / 2 * scaleInt
                let padTop    = (fixedH - tile.image.height) / 2 * scaleInt
                let padRight  = fixedW * scaleInt - tile.image.width * scaleInt - padLeft
                let padBottom = fixedH * scaleInt - tile.image.height * scaleInt - padTop

                let srcX = insetL + padLeft
                let srcYTopLeft = insetT + padTop
                let srcW = max(0, outTile.width  - insetL - insetR - padLeft - padRight)
                let srcH = max(0, outTile.height - insetT - insetB - padTop - padBottom)

                let dstX = x0In * scaleInt + insetL
                let dstY = (imgH * scaleInt) - (y1InTop * scaleInt) + insetB

                if srcW > 0 && srcH > 0 {
                    let cropRect = CGRect(x: srcX, y: srcYTopLeft, width: srcW, height: srcH)
                    if let cropped = outTile.cropping(to: cropRect) {
                        ctx.draw(cropped, in: CGRect(x: dstX, y: dstY, width: srcW, height: srcH))
                    } else {
                        ctx.draw(outTile, in: CGRect(x: dstX, y: dstY, width: outTile.width, height: outTile.height))
                    }
                } else {
                    ctx.draw(outTile, in: CGRect(x: dstX, y: dstY, width: outTile.width, height: outTile.height))
                }

                done += 1
                progress?(Double(done) / Double(total))
            }
        }

        return ctx.makeImage() ?? cgImage
    }

    // MARK: - 内部処理

    private func applyModelWithPadding(on cgImage: CGImage, fixedW: Int, fixedH: Int) -> CGImage? {
        // 元画像をパディングしてモデル入力サイズに合わせる
        guard let padded = Self.padCGImage(cgImage, targetW: fixedW, targetH: fixedH) else {
            print("⚠️ パディング失敗")
            return nil
        }

        guard let inputValue = try? MLFeatureValue(
            cgImage: padded,
            constraint: inputConstraint,
            options: [:]
        ) else {
            print("⚠️ CGImage → MLFeatureValue 変換失敗")
            return nil
        }
        guard let input = try? MLDictionaryFeatureProvider(dictionary: [inputName: inputValue]) else {
            print("⚠️ FeatureProvider 作成失敗")
            return nil
        }

        guard let outFeatures = try? model.prediction(from: input) else {
            print("⚠️ 推論失敗")
            return nil
        }

        if let array = outFeatures.featureValue(for: outputName)?.multiArrayValue {
            return Self.multiArrayToCGImage(array, ciContext: ciContext)
        }
        if let pb = outFeatures.featureValue(for: outputName)?.imageBufferValue {
            return Self.pixelBufferToCGImage(pb, ciContext: ciContext)
        }

        print("⚠️ 出力取得失敗")
        return nil
    }

    
    private static func padCGImage(_ image: CGImage, targetW: Int, targetH: Int) -> CGImage? {
        guard let ctx = CGContext(
            data: nil,
            width: targetW,
            height: targetH,
            bitsPerComponent: 8,
            bytesPerRow: 0,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        ) else { return nil }

        ctx.setFillColor(CGColor(red: 0, green: 0, blue: 0, alpha: 0))
        ctx.fill(CGRect(x: 0, y: 0, width: targetW, height: targetH))

        let offsetX = (targetW - image.width) / 2
        let offsetY = (targetH - image.height) / 2
        ctx.draw(image, in: CGRect(x: offsetX, y: offsetY, width: image.width, height: image.height))

        return ctx.makeImage()
    }

    private static func pixelBufferToCGImage(_ pixelBuffer: CVPixelBuffer, ciContext: CIContext) -> CGImage? {
        let ci = CIImage(cvPixelBuffer: pixelBuffer)
        return ciContext.createCGImage(ci, from: ci.extent)
    }

    private static func multiArrayToCGImage(_ array: MLMultiArray, ciContext: CIContext) -> CGImage? {
        let shape = array.shape.map { $0.intValue }
        let hasBatch = shape.count == 4
        let c = hasBatch ? shape[1] : shape[0]
        let h = hasBatch ? shape[2] : shape[1]
        let w = hasBatch ? shape[3] : shape[2]

        guard c == 3 else { return nil }

        let ptr = UnsafeMutablePointer<Float>(OpaquePointer(array.dataPointer))
        let count = array.count
        let buffer = UnsafeBufferPointer(start: ptr, count: count)

        var pixels = [UInt8](repeating: 0, count: w * h * 4)
        for y in 0..<h {
            for x in 0..<w {
                let r = buffer[(0 * h + y) * w + x]
                let g = buffer[(1 * h + y) * w + x]
                let b = buffer[(2 * h + y) * w + x]
                let offset = (y * w + x) * 4
                pixels[offset]     = UInt8(max(0, min(255, r * 255)))
                pixels[offset + 1] = UInt8(max(0, min(255, g * 255)))
                pixels[offset + 2] = UInt8(max(0, min(255, b * 255)))
                pixels[offset + 3] = 255
            }
        }

        let cs = CGColorSpaceCreateDeviceRGB()
        guard let ctx = CGContext(
            data: &pixels,
            width: w, height: h,
            bitsPerComponent: 8, bytesPerRow: w * 4,
            space: cs,
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        ), let cg = ctx.makeImage() else { return nil }

        return ciContext.createCGImage(CIImage(cgImage: cg), from: CGRect(x: 0, y: 0, width: w, height: h))
    }

    private static func resizeCGImage(_ image: CGImage, width: Int, height: Int) -> CGImage? {
        guard let ctx = CGContext(
            data: nil,
            width: width, height: height,
            bitsPerComponent: 8, bytesPerRow: 0,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        ) else { return nil }
        ctx.interpolationQuality = .high
        ctx.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))
        return ctx.makeImage()
    }

    private static func splitIntoTiles(image: CGImage, tileSize: Int, overlap: Int) -> [ImageTile] {
        var tiles: [ImageTile] = []
        let step = max(32, tileSize - overlap)
        for y in stride(from: 0, to: image.height, by: step) {
            for x in stride(from: 0, to: image.width, by: step) {
                let rect = CGRect(
                    x: x, y: y,
                    width: min(tileSize, image.width - x),
                    height: min(tileSize, image.height - y)
                )
                if let im = image.cropping(to: rect) {
                    tiles.append(ImageTile(rect: rect, image: im))
                }
            }
        }
        return tiles
    }
}

private struct ImageTile {
    let rect: CGRect
    let image: CGImage
}

private extension CGRect {
    func integralNonEmpty() -> CGRect {
        let r = self.integral
        return r.width > 0 && r.height > 0 ? r : .zero
    }
}

