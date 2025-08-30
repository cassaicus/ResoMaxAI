import Foundation
import AppKit
import CoreML
import CoreImage
import Vision

// final classは、これ以上サブクラス化（継承）できないことを示します。パフォーマンス上の利点があります。
// このクラスは、Core MLモデルを使用して画像を高画質化するための主要なロジックをカプセル化します。
final class SuperResolutionEngine {
    // 処理中に発生する可能性のあるカスタムエラーを定義します。
    enum SRError: Error {
        case modelLoadFailed // モデルの読み込みまたはコンパイルに失敗した
        case firstTileFailed // 最初のタイルの処理に失敗し、スケールなどが決定できなかった
        case cancelled       // 処理がユーザーによってキャンセルされた
    }

    // 処理を外部からキャンセルするためのクラス。
    // `isCancelled`プロパティはスレッドセーフにアクセスできるようにNSLockで保護されています。
    final class CancellationToken {
        private let lock = NSLock()
        private var _isCancelled = false
        var isCancelled: Bool { lock.lock(); defer { lock.unlock() }; return _isCancelled }
        func cancel() { lock.lock(); _isCancelled = true; lock.unlock() }
    }

    // 実際に推論を行うCore MLモデル。
    private let model: MLModel
    // Core Imageを使った画像処理を行うためのコンテキスト。GPUレンダリングを優先します。
    private let ciContext = CIContext(options: [.useSoftwareRenderer: false])
    // モデルの入力層の名前。`init`で動的に取得します。
    private let inputName: String
    // モデルの出力層の名前。`init`で動的に取得します。
    private let outputName: String
    // モデルが要求する入力画像の制約（解像度、ピクセルフォーマットなど）。
    private let inputConstraint: MLImageConstraint

    // イニシャライザ。バンドル内のモデル名からエンジンをセットアップします。
    // - Parameter modelNameInBundle: バンドル内のモデル名（拡張子なし）
    // - Parameter computeUnits: 計算に使用するユニット（CPU, GPU, Neural Engine）。デフォルトはCPUとGPU。
    // - Throws: `SRError.modelLoadFailed` モデルが見つからない、または読み込めない場合にスローされます。
    init(modelNameInBundle: String, computeUnits: MLComputeUnits = .cpuAndGPU) throws {
        let config = MLModelConfiguration()
        config.computeUnits = computeUnits

        let bundle = Bundle.main
        // モデルは様々な形式でバンドルに含まれる可能性があるため、候補をリストアップします。
        // .mlpackage (推奨), .mlmodelc (コンパイル済み), .mlmodel (未コンパイル)
        let candidates: [URL] = [
            bundle.url(forResource: modelNameInBundle, withExtension: "mlpackage"),
            bundle.url(forResource: modelNameInBundle, withExtension: "mlmodelc"),
            bundle.url(forResource: modelNameInBundle, withExtension: "mlmodel")
        ].compactMap { $0 } // nilを除去します。

        // 候補の中から最初に見つかったURLを使用します。
        guard let foundURL = candidates.first else { throw SRError.modelLoadFailed }

        let loadURL: URL
        // 未コンパイルの.mlmodelだった場合、その場でコンパイルを試みます。
        if foundURL.pathExtension == "mlmodel" {
            guard let compiled = try? MLModel.compileModel(at: foundURL) else { throw SRError.modelLoadFailed }
            loadURL = compiled
        } else {
            loadURL = foundURL
        }

        // 最終的なURLからモデルを読み込みます。
        self.model = try MLModel(contentsOf: loadURL, configuration: config)

        // モデルの入出力情報を取得します。
        // アドバイス: モデルが複数の入力や出力を持つ場合、`first`に頼るこの実装は脆弱です。
        // より堅牢にするには、特定の名前で入出力を探すか、モデルの仕様を固定するべきです。
        guard
            let inEntry = model.modelDescription.inputDescriptionsByName.first,
            let outEntry = model.modelDescription.outputDescriptionsByName.first,
            let inConstraint = model.modelDescription.inputDescriptionsByName[inEntry.key]?.imageConstraint
        else { throw SRError.modelLoadFailed }

        self.inputName = inEntry.key
        self.outputName = outEntry.key
        self.inputConstraint = inConstraint

        // デバッグ用に読み込んだモデルの情報を出力します。
        print("SR Engine: loaded = \(loadURL.lastPathComponent)")
        print("SR Engine: inputName = \(inputName), outputName = \(outputName)")
        print("SR Engine: inputConstraint \(inConstraint.pixelsWide)x\(inConstraint.pixelsHigh), pixelFormat=\(inConstraint.pixelFormatType)")
    }
    
    
    // 画像を高画質化するメインの関数。
    // VRAMの消費を抑えるため、画像をタイルに分割して処理し、後で結合します。
    // - Parameter cgImage: 高画質化する入力CGImage。
    // - Parameter tileSize: 各タイルの基本サイズ。
    // - Parameter overlap: タイル間の重なり幅。これにより、タイルのつなぎ目が目立たなくなる。
    // - Parameter progress: 0.0から1.0までの進捗を報告するコールバック。
    // - Parameter cancellation: 処理を中断するためのキャンセルトークン。
    // - Throws: `SRError`をスローする可能性があります。
    // - Returns: 高画質化されたCGImage。失敗した場合は元の画像を返します。
    func superResolve(
        cgImage: CGImage,
        tileSize: Int = 1024,
        overlap: Int = 24,
        progress: ((Double) -> Void)? = nil,
        cancellation: CancellationToken? = nil
    ) throws -> CGImage {
        // 処理開始前にキャンセルをチェックします。
        if cancellation?.isCancelled == true { throw SRError.cancelled }

        // モデルが要求する固定入力サイズを取得します。
        let fixedW = max(1, inputConstraint.pixelsWide)
        let fixedH = max(1, inputConstraint.pixelsHigh)
        // 実際に使用するタイルサイズとオーバーラップを計算します。
        let usedTile = min(tileSize, max(fixedW, fixedH))
        let usedOverlap = min(overlap, usedTile / 2)

        // 入力画像をタイルに分割します。
        let tiles = Self.splitIntoTiles(image: cgImage, tileSize: usedTile, overlap: usedOverlap)
        guard let firstTile = tiles.first else { return cgImage } // タイルがなければ元の画像を返します。

        // 最初のタイルを処理して、画像の拡大率を決定します。
        // アドバイス: 拡大率はモデルの仕様から静的に決定できるはずです。
        // 毎回推論して計算するのは非効率であり、モデルによっては失敗する可能性があります。
        guard let firstTileOut = autoreleasepool(invoking: {
            self.applyModelWithPadding(on: firstTile.image, fixedW: fixedW, fixedH: fixedH)
        }) else {
            throw SRError.firstTileFailed
        }
        let scaleInt = max(1, Int(round(Double(firstTileOut.width) / Double(firstTile.image.width))))
        let overlapUpscaled = usedOverlap * scaleInt
        let halfOverlap = overlapUpscaled / 2

        // 最終的な出力画像のサイズを計算し、描画コンテキストを作成します。
        let outW = cgImage.width * scaleInt
        let outH = cgImage.height * scaleInt
        // アドバイス: カラーマネジメントを考慮すると、入力画像のカラースペース(cgImage.colorSpace)を使用する方が良いでしょう。
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

        // 各タイルをループ処理します。
        for tile in tiles {
            if cancellation?.isCancelled == true { throw SRError.cancelled }
            // autoreleasepool内で処理することで、ループごとのメモリ使用量を抑えます。
            autoreleasepool {
                // タイルに対してモデルを適用（推論）します。
                guard let outTile = self.applyModelWithPadding(on: tile.image, fixedW: fixedW, fixedH: fixedH) else {
                    print("⚠️ タイル推論失敗: \(tile.rect)")
                    return // このタイルはスキップします。
                }

                // タイルの位置情報から、オーバーラップを除去するための計算を行います。
                let x0In = Int(tile.rect.minX)
                let y0InTop = Int(tile.rect.minY)
                let x1In = Int(tile.rect.maxX)
                let y1InTop = Int(tile.rect.maxY)

                // 画像の端にあるタイルかどうかを判定します。
                let hasLeft   = x0In > 0
                let hasRight  = x1In < imgW
                let hasTop    = y0InTop > 0
                let hasBottom = y1InTop < imgH

                // 端でない辺のオーバーラップ分を後で切り取るためのインセットを計算します。
                let insetL = hasLeft   ? halfOverlap : 0
                let insetR = hasRight  ? halfOverlap : 0
                let insetT = hasTop    ? halfOverlap : 0
                let insetB = hasBottom ? halfOverlap : 0

                // モデル入力時に追加されたパディングのサイズを計算します。
                let padLeft   = (fixedW - tile.image.width) / 2 * scaleInt
                let padTop    = (fixedH - tile.image.height) / 2 * scaleInt
                let padRight  = fixedW * scaleInt - tile.image.width * scaleInt - padLeft
                let padBottom = fixedH * scaleInt - tile.image.height * scaleInt - padTop

                // 出力タイルから切り取るべきソース矩形(src)を計算します。
                let srcX = insetL + padLeft
                let srcYTopLeft = insetT + padTop // CoreGraphicsのY座標は左下原点だが、ここでは左上原点で計算。
                let srcW = max(0, outTile.width  - insetL - insetR - padLeft - padRight)
                let srcH = max(0, outTile.height - insetT - insetB - padTop - padBottom)

                // 最終的な画像に描画する先の矩形(dst)を計算します。
                // Y座標をCoreGraphicsの座標系（左下原点）に変換します。
                let dstX = x0In * scaleInt + insetL
                let dstY = (imgH * scaleInt) - (y1InTop * scaleInt) + insetB

                // 計算した矩形が有効な場合にのみ、切り取りと描画を行います。
                if srcW > 0 && srcH > 0 {
                    let cropRect = CGRect(x: srcX, y: srcYTopLeft, width: srcW, height: srcH)
                    if let cropped = outTile.cropping(to: cropRect) {
                        ctx.draw(cropped, in: CGRect(x: dstX, y: dstY, width: srcW, height: srcH))
                    } else {
                        // croppingに失敗した場合、デバッグのために元タイルを描画するなどのフォールバック。
                        // バグ: 現在の実装では、cropping失敗時に元のoutTileを描画していますが、
                        // 座標がずれるため、意図しない結果になる可能性があります。
                        ctx.draw(outTile, in: CGRect(x: dstX, y: dstY, width: outTile.width, height: outTile.height))
                    }
                } else {
                    // バグ: srcW/srcHが0以下の場合も、元のoutTileを描画しており、意図しない結果になる可能性があります。
                    ctx.draw(outTile, in: CGRect(x: dstX, y: dstY, width: outTile.width, height: outTile.height))
                }

                // 進捗を更新します。
                done += 1
                progress?(Double(done) / Double(total))
            }
        }

        // 全てのタイルを描画したコンテキストから最終的な画像を生成します。
        return ctx.makeImage() ?? cgImage
    }

    // MARK: - 内部処理

    // 1つの画像（タイル）に対してパディングを追加し、Core MLモデルを適用するヘルパー関数。
    private func applyModelWithPadding(on cgImage: CGImage, fixedW: Int, fixedH: Int) -> CGImage? {
        // 元画像をパディングしてモデル入力サイズに合わせる。
        guard let padded = Self.padCGImage(cgImage, targetW: fixedW, targetH: fixedH) else {
            print("⚠️ パディング失敗")
            return nil
        }

        // パディングされた画像をモデルの入力形式(MLFeatureValue)に変換します。
        guard let inputValue = try? MLFeatureValue(
            cgImage: padded,
            constraint: inputConstraint,
            options: [:]
        ) else {
            print("⚠️ CGImage → MLFeatureValue 変換失敗")
            return nil
        }
        // モデルへの入力をまとめるプロバイダーを作成します。
        guard let input = try? MLDictionaryFeatureProvider(dictionary: [inputName: inputValue]) else {
            print("⚠️ FeatureProvider 作成失敗")
            return nil
        }

        // モデルで推論を実行します。
        guard let outFeatures = try? model.prediction(from: input) else {
            print("⚠️ 推論失敗")
            return nil
        }

        // モデルの出力形式に応じて、結果をCGImageに変換します。
        // 出力はMLMultiArray（ピクセルの生データ配列）かCVPixelBuffer（画像バッファ）のどちらかです。
        if let array = outFeatures.featureValue(for: outputName)?.multiArrayValue {
            return Self.multiArrayToCGImage(array, ciContext: ciContext)
        }
        if let pb = outFeatures.featureValue(for: outputName)?.imageBufferValue {
            return Self.pixelBufferToCGImage(pb, ciContext: ciContext)
        }

        print("⚠️ 出力取得失敗")
        return nil
    }

    
    // 指定された目標サイズになるように、画像の中心に黒い透明なパディングを追加する静的メソッド。
    private static func padCGImage(_ image: CGImage, targetW: Int, targetH: Int) -> CGImage? {
        guard let ctx = CGContext(
            data: nil,
            width: targetW,
            height: targetH,
            bitsPerComponent: 8,
            bytesPerRow: 0,
            space: image.colorSpace ?? CGColorSpaceCreateDeviceRGB(), // 元画像のカラースペースを尊重する
            bitmapInfo: image.bitmapInfo.rawValue
        ) else { return nil }

        // 背景を透明で塗りつぶします。
        ctx.setFillColor(CGColor(red: 0, green: 0, blue: 0, alpha: 0))
        ctx.fill(CGRect(x: 0, y: 0, width: targetW, height: targetH))

        // 中央に元画像を描画します。
        let offsetX = (targetW - image.width) / 2
        let offsetY = (targetH - image.height) / 2
        ctx.draw(image, in: CGRect(x: offsetX, y: offsetY, width: image.width, height: image.height))

        return ctx.makeImage()
    }

    // CVPixelBufferをCGImageに変換する静的メソッド。
    private static func pixelBufferToCGImage(_ pixelBuffer: CVPixelBuffer, ciContext: CIContext) -> CGImage? {
        let ci = CIImage(cvPixelBuffer: pixelBuffer)
        return ciContext.createCGImage(ci, from: ci.extent)
    }

    // MLMultiArrayをCGImageに変換する静的メソッド。
    // モデルの出力が (チャンネル, 高さ, 幅) のようなCHW形式のfloat配列であることを想定しています。
    private static func multiArrayToCGImage(_ array: MLMultiArray, ciContext: CIContext) -> CGImage? {
        let shape = array.shape.map { $0.intValue }
        let hasBatch = shape.count == 4
        // (B, C, H, W) または (C, H, W) の形状に対応します。
        let c = hasBatch ? shape[1] : shape[0]
        let h = hasBatch ? shape[2] : shape[1]
        let w = hasBatch ? shape[3] : shape[2]

        // 3チャンネル(RGB)以外は現在サポートしていません。
        guard c == 3 else { return nil }

        let ptr = UnsafeMutablePointer<Float>(OpaquePointer(array.dataPointer))
        let count = array.count
        let buffer = UnsafeBufferPointer(start: ptr, count: count)

        // RGBAのUInt8配列に変換します。
        var pixels = [UInt8](repeating: 0, count: w * h * 4)
        // アドバイス: このループはパフォーマンスが重要なら、vDSPなどの高速なライブラリを使うと改善できます。
        for y in 0..<h {
            for x in 0..<w {
                // CHWレイアウトからピクセル値を取得します。
                let r = buffer[(0 * h + y) * w + x]
                let g = buffer[(1 * h + y) * w + x]
                let b = buffer[(2 * h + y) * w + x]
                let offset = (y * w + x) * 4
                // [0, 1]のfloat値を[0, 255]のUInt8に変換します。
                pixels[offset]     = UInt8(max(0, min(255, r * 255)))
                pixels[offset + 1] = UInt8(max(0, min(255, g * 255)))
                pixels[offset + 2] = UInt8(max(0, min(255, b * 255)))
                pixels[offset + 3] = 255 // Alphaを不透明(255)に設定します。
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

        // CIContextを経由することで、GPUメモリ上での処理を期待できます。
        return ciContext.createCGImage(CIImage(cgImage: cg), from: CGRect(x: 0, y: 0, width: w, height: h))
    }

    // （現在未使用）画像をリサイズするユーティリティ関数。
    private static func resizeCGImage(_ image: CGImage, width: Int, height: Int) -> CGImage? {
        guard let ctx = CGContext(
            data: nil,
            width: width, height: height,
            bitsPerComponent: 8, bytesPerRow: 0,
            space: image.colorSpace ?? CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: image.bitmapInfo.rawValue
        ) else { return nil }
        ctx.interpolationQuality = .high // 高品質な補間を設定します。
        ctx.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))
        return ctx.makeImage()
    }

    // 画像をオーバーラップを持つタイルに分割する静的メソッド。
    private static func splitIntoTiles(image: CGImage, tileSize: Int, overlap: Int) -> [ImageTile] {
        var tiles: [ImageTile] = []
        let step = max(32, tileSize - overlap) // 各タイルの開始位置の増分。
        for y in stride(from: 0, to: image.height, by: step) {
            for x in stride(from: 0, to: image.width, by: step) {
                // 画像の端でタイルがはみ出さないようにサイズを調整します。
                let rect = CGRect(
                    x: x, y: y,
                    width: min(tileSize, image.width - x),
                    height: min(tileSize, image.height - y)
                )
                // 元画像から矩形領域を切り出します。
                if let im = image.cropping(to: rect) {
                    tiles.append(ImageTile(rect: rect, image: im))
                }
            }
        }
        return tiles
    }
}

// タイルの情報を保持するためのプライベートな構造体。
private struct ImageTile {
    let rect: CGRect   // 元画像におけるタイルの位置とサイズ。
    let image: CGImage // 切り出されたタイル画像データ。
}

// （現在未使用）CGRectを整数座標に丸め、空でないことを保証する拡張。
private extension CGRect {
    func integralNonEmpty() -> CGRect {
        let r = self.integral
        return r.width > 0 && r.height > 0 ? r : .zero
    }
}

