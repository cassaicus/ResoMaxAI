// 必要なフレームワークをインポートします。
import Foundation  //基本的なデータ型やコレクションなど
import AppKit      // NSImageやCGImageなど、macOSアプリのUI関連
import CoreML      // 機械学習モデルの実行
import CoreImage   // CIContextなど、高度な画像処理
import Vision      // 画像解析やモデルの入力形式（今回は直接使っていないが関連が深い）

// final classは、これ以上継承（サブクラス化）できないクラスを定義します。
// パフォーマンス上の利点があり、意図しないオーバーライドを防ぎます。
final class SuperResolutionEngine {

    // このエンジン内で発生しうるカスタムエラーを定義します。
    // Errorプロトコルに準拠させることで、try-catch構文で扱えるようになります。
    enum SRError: Error {
        case modelLoadFailed // モデルの読み込みに失敗した場合
        case firstTileFailed // 最初のタイルの処理に失敗した場合（スケール特定などに使うため重要）
        case cancelled       // 処理がユーザーによってキャンセルされた場合
    }

    // 長時間かかる可能性のある処理をキャンセルするための仕組みです。
    // このクラスのインスタンスをsuperResolveメソッドに渡すことで、外部から処理を中断できます。
    // スレッドセーフ（複数のスレッドから同時にアクセスされても問題ない）に作られています。
    final class CancellationToken {
        // NSLockは、複数のスレッドが同時にisCancelledプロパティにアクセスするのを防ぎます。
        private let lock = NSLock()
        // _isCancelledは実際のキャンセルフラグの値を保持します。privateなのでクラス内からしかアクセスできません。
        private var _isCancelled = false
        // isCancelledは、外部からキャンセルフラグの値を安全に読み取るためのプロパティです。
        // lock()でロックを取得し、defer { lock.unlock() }で処理を抜けるときに必ずロックを解放します。
        var isCancelled: Bool { lock.lock(); defer { lock.unlock() }; return _isCancelled }
        // cancel()メソッドは、外部から処理のキャンセルを要求するために呼び出します。
        // こちらもスレッドセーフにフラグをtrueに設定します。
        func cancel() { lock.lock(); _isCancelled = true; lock.unlock() }
    }

    // --- クラスのプロパティ ---

    // 実際に超解像を行うCore MLモデル。initで読み込まれます。
    private let model: MLModel
    // CIContextは、Core Imageを使った画像処理を効率的に行うためのオブジェクトです。
    // GPUを使って高速に処理できます。一度作成すれば再利用するのが効率的です。
    // useSoftwareRenderer: false はGPUレンダリングを優先することを示します。
    private let ciContext = CIContext(options: [.useSoftwareRenderer: false])
    // モデルの入力層の名前。モデルによって名前が違うため、initで動的に取得します。
    private let inputName: String
    // モデルの出力層の名前。同様にinitで動的に取得します。
    private let outputName: String
    // モデルが受け付ける画像の制約（解像度、ピクセルフォーマットなど）。
    private let inputConstraint: MLImageConstraint

    // --- 初期化 ---

    // このクラスのインスタンスを生成するイニシャライザ（コンストラクタ）です。
    // モデル名を指定して、Core MLモデルを読み込み、使える状態にします。
    // throwsキーワードは、初期化が失敗する可能性があることを示します（例：モデルファイルが見つからない）。
    init(modelNameInBundle: String, computeUnits: MLComputeUnits = .cpuAndGPU) throws {
        // MLModelConfigurationは、モデルを読み込む際の設定を定義します。
        let config = MLModelConfiguration()
        // computeUnitsで、モデルの計算にCPU、GPU、Neural Engineのどれを使うかを指定します。
        // .cpuAndGPUは、CPUとGPUの両方を利用できる設定です。
        config.computeUnits = computeUnits

        // アプリケーションのバンドル（実行ファイルやリソースが含まれるディレクトリ）を取得します。
        let bundle = Bundle.main
        // モデルファイルは拡張子が異なる可能性があるため（mlpackage, mlmodelc, mlmodel）、
        // すべての可能性を試すためのURL候補リストを作成します。
        let candidates: [URL] = [
            bundle.url(forResource: modelNameInBundle, withExtension: "mlpackage"), // 推奨される形式
            bundle.url(forResource: modelNameInBundle, withExtension: "mlmodelc"),  // コンパイル済みモデル
            bundle.url(forResource: modelNameInBundle, withExtension: "mlmodel")     // 未コンパイルのモデル
        ].compactMap { $0 } // compactMapでnilの要素（見つからなかったURL）を取り除きます。

        // 候補リストの最初の有効なURLを取得します。見つからなければモデル読み込み失敗エラーを投げます。
        guard let foundURL = candidates.first else { throw SRError.modelLoadFailed }

        // 実際にモデルを読み込むためのURLを決定します。
        let loadURL: URL
        // もし見つかったのが未コンパイルの .mlmodel ファイルだった場合、
        if foundURL.pathExtension == "mlmodel" {
            // アプリ実行時に動的にコンパイルします。
            // アドバイス：通常はビルド時にコンパイルされる（.mlmodelcが生成される）ため、このパスを通るのは稀です。
            // ユーザーが外部から.mlmodelを読み込むような機能がない限り、このコードは不要かもしれません。
            guard let compiled = try? MLModel.compileModel(at: foundURL) else { throw SRError.modelLoadFailed }
            loadURL = compiled
        } else {
            // .mlpackage や .mlmodelc の場合は、そのままのURLを使います。
            loadURL = foundURL
        }

        // 決定したURLと設定を使って、MLModelインスタンスを生成します。
        // ここで実際にモデルがメモリに読み込まれます。失敗した場合はエラーが投げられます。
        self.model = try MLModel(contentsOf: loadURL, configuration: config)

        // モデルの入出力情報を取得します。
        // これらが取得できない場合（モデルの形式が不正など）、モデル読み込み失敗とみなします。
        guard
            // 最初の入力層の情報を取得
            let inEntry = model.modelDescription.inputDescriptionsByName.first,
            // 最初の出力層の情報を取得
            let outEntry = model.modelDescription.outputDescriptionsByName.first,
            // 入力層の画像制約（サイズなど）を取得
            let inConstraint = model.modelDescription.inputDescriptionsByName[inEntry.key]?.imageConstraint
        else { throw SRError.modelLoadFailed }

        // 取得した情報をプロパティに保存します。
        self.inputName = inEntry.key
        self.outputName = outEntry.key
        self.inputConstraint = inConstraint

        // 読み込みが成功したことを示すログを出力します。デバッグに役立ちます。
        print("SR Engine: loaded = \(loadURL.lastPathComponent)")
        print("SR Engine: inputName = \(inputName), outputName = \(outputName)")
        print("SR Engine: inputConstraint \(inConstraint.pixelsWide)x\(inConstraint.pixelsHigh), pixelFormat=\(inConstraint.pixelFormatType)")
    }
    
    
    // --- メインの超解像処理 ---
    
    /// 画像を超解像するメインの関数。
    /// - Parameters:
    ///   - cgImage: 入力となるCGImage。
    ///   - tileSize: 画像を分割する際のタイルのサイズ。モデルの入力サイズより大きい必要があります。
    ///   - overlap: タイル間の重複領域。重複させることで、タイルのつなぎ目が見えるのを防ぎます。
    ///   - progress: 処理の進捗を通知するクロージャ。0.0から1.0の値が渡されます。
    ///   - cancellation: 処理をキャンセルするためのCancellationToken。
    /// - Throws: SRError型のいずれかのエラーを投げる可能性があります。
    /// - Returns: 超解像されたCGImage。失敗した場合は元の画像を返します。
    func superResolve(
        cgImage: CGImage,
        tileSize: Int = 1024,
        overlap: Int = 24,
        progress: ((Double) -> Void)? = nil,
        cancellation: CancellationToken? = nil
    ) throws -> CGImage {
        // 処理開始前に、キャンセルされていないかチェックします。
        if cancellation?.isCancelled == true { throw SRError.cancelled }

        // モデルが要求する固定の入力サイズを取得します。0の場合は1として扱います。
        let fixedW = max(1, inputConstraint.pixelsWide)
        let fixedH = max(1, inputConstraint.pixelsHigh)
        // 実際に使用するタイルサイズを決定します。指定されたtileSizeとモデルの入力サイズのうち、小さい方に合わせます。
        let usedTile = min(tileSize, max(fixedW, fixedH))
        // 実際に使用する重複量を決定します。タイルサイズの半分を超えることはありません。
        let usedOverlap = min(overlap, usedTile / 2)

        // 入力画像を、指定されたタイルサイズと重複量でタイルに分割します。
        let tiles = Self.splitIntoTiles(image: cgImage, tileSize: usedTile, overlap: usedOverlap)
        // タイルが1つも生成されなかった場合（画像が0x0など）、何もせず元の画像を返します。
        guard let firstTile = tiles.first else { return cgImage }

        // 最初のタイルだけを先に処理して、画像の拡大率（スケール）を確定させます。
        // autoreleasepoolは、ループ内で生成される一時的なオブジェクトを適切に解放し、メモリ使用量を抑えるために使います。
        guard let firstTileOut = autoreleasepool(invoking: {
            self.applyModelWithPadding(on: firstTile.image, fixedW: fixedW, fixedH: fixedH)
        }) else {
            // 最初のタイルの処理に失敗した場合、処理を中断してエラーを投げます。
            throw SRError.firstTileFailed
        }
        // 出力タイルの幅と入力タイルの幅の比率から、整数倍のスケールを計算します（例：4倍）。
        let scaleInt = max(1, Int(round(Double(firstTileOut.width) / Double(firstTile.image.width))))
        // スケール後の重複量と、その半分の値を計算しておきます。
        let overlapUpscaled = usedOverlap * scaleInt
        let halfOverlap = overlapUpscaled / 2

        // 最終的に生成される、超解像後の画像の全体の幅と高さを計算します。
        let outW = cgImage.width * scaleInt
        let outH = cgImage.height * scaleInt
        // このサイズの新しい描画コンテキスト（キャンバスのようなもの）を作成します。
        // ここに各タイルを処理した結果を描画していきます。
        guard let ctx = CGContext(
            data: nil,
            width: outW,
            height: outH,
            bitsPerComponent: 8, // 1チャンネルあたり8ビット（例：RGBで各8ビット）
            bytesPerRow: 0,      // 0を指定すると自動計算される
            space: CGColorSpaceCreateDeviceRGB(), // 標準的なRGBカラースペース
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue // アルファチャンネル（透明度）の扱い
        ) else { return cgImage } // コンテキスト作成に失敗したら元の画像を返す

        // 元画像のサイズと、タイルの総数を取得しておきます。
        let imgW = cgImage.width
        let imgH = cgImage.height
        let total = tiles.count
        var done = 0 // 処理済みのタイル数をカウント

        // 全てのタイルをループで処理します。
        for tile in tiles {
            // 各タイルの処理前にもキャンセルチェックを行います。
            if cancellation?.isCancelled == true { throw SRError.cancelled }

            // autoreleasepoolでメモリを管理しながら、タイルを1枚処理します。
            autoreleasepool {
                // タイル画像を入力し、モデルで推論（超解像）を実行します。
                guard let outTile = self.applyModelWithPadding(on: tile.image, fixedW: fixedW, fixedH: fixedH) else {
                    // 推論に失敗した場合は警告を出し、このタイルはスキップします。
                    print("⚠️ タイル推論失敗: \(tile.rect)")
                    return // autoreleasepoolのクロージャから抜ける
                }

                // --- ここから、重複領域を考慮してタイルを正しくつなぎ合わせるための複雑な計算 ---

                // 元画像におけるタイルの座標を取得します。
                let x0In = Int(tile.rect.minX)
                let y0InTop = Int(tile.rect.minY) // y座標は左上が原点
                let x1In = Int(tile.rect.maxX)
                let y1InTop = Int(tile.rect.maxY)

                // このタイルが画像の端にあるかどうかを判定します。
                let hasLeft   = x0In > 0
                let hasRight  = x1In < imgW
                let hasTop    = y0InTop > 0
                let hasBottom = y1InTop < imgH

                // 端でない場合は、重複領域の半分だけ内側を切り取って使用します。
                // これにより、隣のタイルと滑らかにつながります。
                let insetL = hasLeft   ? halfOverlap : 0
                let insetR = hasRight  ? halfOverlap : 0
                let insetT = hasTop    ? halfOverlap : 0
                let insetB = hasBottom ? halfOverlap : 0

                // モデル入力時に追加したパディング（余白）の分をスケール後に計算します。
                // これも最終的な描画位置から除外する必要があります。
                let padLeft   = (fixedW - tile.image.width) / 2 * scaleInt
                let padTop    = (fixedH - tile.image.height) / 2 * scaleInt
                let padRight  = fixedW * scaleInt - tile.image.width * scaleInt - padLeft
                let padBottom = fixedH * scaleInt - tile.image.height * scaleInt - padTop

                // 出力タイルから、実際に最終画像に描画すべき領域（ソース領域）を計算します。
                // 重複領域(inset)とパディング(pad)を除いた部分です。
                let srcX = insetL + padLeft
                let srcYTopLeft = insetT + padTop // CoreGraphicsのy座標は左上が原点
                let srcW = max(0, outTile.width  - insetL - insetR - padLeft - padRight)
                let srcH = max(0, outTile.height - insetT - insetB - padTop - padBottom)

                // 最終画像キャンバス上のどこに描画するか（デスティネーション領域）を計算します。
                let dstX = x0In * scaleInt + insetL
                // CGContextのy座標は左下が原点なので、元画像のy座標（左上原点）から変換する必要があります。
                let dstY = (imgH * scaleInt) - (y1InTop * scaleInt) + insetB

                // ソース領域の幅か高さが0より大きい場合のみ描画します。
                if srcW > 0 && srcH > 0 {
                    let cropRect = CGRect(x: srcX, y: srcYTopLeft, width: srcW, height: srcH)
                    // 計算したソース領域で出力タイルを切り抜きます。
                    if let cropped = outTile.cropping(to: cropRect) {
                        // 切り抜いた画像を、計算したデスティネーション位置に描画します。
                        ctx.draw(cropped, in: CGRect(x: dstX, y: dstY, width: srcW, height: srcH))
                    } else {
                        // バグ対策：croppingに失敗した場合、とりあえず元のタイルを描画しようと試みます。
                        // 本来はここに来るべきではありません。
                        ctx.draw(outTile, in: CGRect(x: dstX, y: dstY, width: outTile.width, height: outTile.height))
                    }
                } else {
                     // バグ対策：計算ミスでsrcW/srcHが0以下になった場合も、とりあえずタイルを描画します。
                    ctx.draw(outTile, in: CGRect(x: dstX, y: dstY, width: outTile.width, height: outTile.height))
                }

                // 処理済みタイル数を増やし、進捗を通知します。
                done += 1
                progress?(Double(done) / Double(total))
            }
        }

        // 全てのタイルを描画し終わったコンテキストから、最終的なCGImageを生成して返します。
        // 失敗した場合は、元の入力画像をそのまま返します。
        return ctx.makeImage() ?? cgImage
    }

    // MARK: - 内部処理

    // このセクションは、クラスの内部でのみ使用されるヘルパーメソッドです。

    /// 1枚のタイル画像（CGImage）をモデルに入力して、推論結果のCGImageを返します。
    /// モデルの入力サイズに合わせて自動的にパディング（余白追加）を行います。
    private func applyModelWithPadding(on cgImage: CGImage, fixedW: Int, fixedH: Int) -> CGImage? {
        // 元画像をパディングしてモデル入力サイズに合わせます。
        guard let padded = Self.padCGImage(cgImage, targetW: fixedW, targetH: fixedH) else {
            print("⚠️ パディング失敗")
            return nil
        }

        // パディングされたCGImageを、Core MLの入力形式であるMLFeatureValueに変換します。
        guard let inputValue = try? MLFeatureValue(
            cgImage: padded,
            constraint: inputConstraint,
            options: [:]
        ) else {
            print("⚠️ CGImage → MLFeatureValue 変換失敗")
            return nil
        }
        // モデルの入力は辞書形式（[入力名: 値]）なので、MLDictionaryFeatureProviderでラップします。
        guard let input = try? MLDictionaryFeatureProvider(dictionary: [inputName: inputValue]) else {
            print("⚠️ FeatureProvider 作成失敗")
            return nil
        }

        // 実際にモデルの推論を実行します。
        guard let outFeatures = try? model.prediction(from: input) else {
            print("⚠️ 推論失敗")
            return nil
        }

        // モデルの出力形式に応じて、結果をCGImageに変換します。
        // 出力はMLMultiArray（多次元配列）かCVPixelBuffer（ピクセルバッファ）のどちらかのことが多いです。
        if let array = outFeatures.featureValue(for: outputName)?.multiArrayValue {
            return Self.multiArrayToCGImage(array, ciContext: ciContext)
        }
        if let pb = outFeatures.featureValue(for: outputName)?.imageBufferValue {
            return Self.pixelBufferToCGImage(pb, ciContext: ciContext)
        }

        // どちらの形式でもなかった場合、出力取得失敗とします。
        print("⚠️ 出力取得失敗")
        return nil
    }

    
    /// 画像を指定された解像度になるように、中央に配置し、周囲を透明なピクセルで埋めます（パディング）。
    private static func padCGImage(_ image: CGImage, targetW: Int, targetH: Int) -> CGImage? {
        // 指定されたサイズの新しい描画コンテキストを作成します。
        guard let ctx = CGContext(
            data: nil,
            width: targetW,
            height: targetH,
            bitsPerComponent: 8,
            bytesPerRow: 0,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        ) else { return nil }

        // コンテキスト全体を透明色で塗りつぶします。
        ctx.setFillColor(CGColor(red: 0, green: 0, blue: 0, alpha: 0))
        ctx.fill(CGRect(x: 0, y: 0, width: targetW, height: targetH))

        // 元画像を中央に描画するためのオフセットを計算します。
        let offsetX = (targetW - image.width) / 2
        let offsetY = (targetH - image.height) / 2
        // コンテキストの中央に元画像を描画します。
        ctx.draw(image, in: CGRect(x: offsetX, y: offsetY, width: image.width, height: image.height))

        // コンテキストから新しいCGImageを生成して返します。
        return ctx.makeImage()
    }

    /// CVPixelBufferをCGImageに変換します。CIImageを経由するのが簡単です。
    private static func pixelBufferToCGImage(_ pixelBuffer: CVPixelBuffer, ciContext: CIContext) -> CGImage? {
        let ci = CIImage(cvPixelBuffer: pixelBuffer)
        return ciContext.createCGImage(ci, from: ci.extent)
    }

    /// MLMultiArray（モデルの出力でよく使われる多次元配列）をCGImageに変換します。
    /// これはチャンネルの並び（例：RGBかBGRか）や値の範囲（0-1か0-255か）がモデルに依存するため、複雑になりがちです。
    private static func multiArrayToCGImage(_ array: MLMultiArray, ciContext: CIContext) -> CGImage? {
        // 配列の形状（[バッチ, チャンネル, 高さ, 幅]など）を取得します。
        let shape = array.shape.map { $0.intValue }
        // バッチ次元があるかどうかを判定します。
        let hasBatch = shape.count == 4
        // チャンネル数、高さ、幅を取得します。
        let c = hasBatch ? shape[1] : shape[0]
        let h = hasBatch ? shape[2] : shape[1]
        let w = hasBatch ? shape[3] : shape[2]

        // この実装は3チャンネル（RGB）を前提としています。
        guard c == 3 else { return nil }

        // MLMultiArrayのデータポインタを、Floatのポインタとして扱います。
        // アドバイス：モデルの出力がFloat32でない場合、ここを変更する必要があります。
        let ptr = UnsafeMutablePointer<Float>(OpaquePointer(array.dataPointer))
        let count = array.count
        let buffer = UnsafeBufferPointer(start: ptr, count: count)

        // ピクセルデータを格納するためのUInt8（0-255）の配列を用意します（RGBAの4チャンネル分）。
        var pixels = [UInt8](repeating: 0, count: w * h * 4)
        // 全てのピクセルをループで処理します。
        // このループは非常に重い処理なので、パフォーマンスが問題になる場合は最適化が必要です。
        // （例：vDSPフレームワークを使った高速な変換）
        for y in 0..<h {
            for x in 0..<w {
                // モデルの出力が [チャンネル, 高さ, 幅] の順（Planar形式）であることを前提としています。
                let r = buffer[(0 * h + y) * w + x]
                let g = buffer[(1 * h + y) * w + x]
                let b = buffer[(2 * h + y) * w + x]
                let offset = (y * w + x) * 4
                // 値を0-1の範囲から0-255の範囲に変換し、UInt8にキャストします。
                pixels[offset]     = UInt8(max(0, min(255, r * 255)))
                pixels[offset + 1] = UInt8(max(0, min(255, g * 255)))
                pixels[offset + 2] = UInt8(max(0, min(255, b * 255)))
                pixels[offset + 3] = 255 // Alphaチャンネルは不透明（255）に固定
            }
        }

        // ピクセル配列からCGImageを生成します。
        let cs = CGColorSpaceCreateDeviceRGB()
        guard let ctx = CGContext(
            data: &pixels, // 作成したピクセルデータを直接渡す
            width: w, height: h,
            bitsPerComponent: 8, bytesPerRow: w * 4, // 1行あたりのバイト数
            space: cs,
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        ), let cg = ctx.makeImage() else { return nil }

        // CIContextを経由して返すことで、色空間の管理などがより確実になります。
        return ciContext.createCGImage(CIImage(cgImage: cg), from: CGRect(x: 0, y: 0, width: w, height: h))
    }

    /// CGImageを指定された解像度にリサイズします。（このプロジェクトでは現在使われていません）
    /// アドバイス：単純なリサイズは画質が劣化するため、超解像アプリではあまり使いませんが、
    /// サムネイル生成など、別の用途で必要になるかもしれません。
    private static func resizeCGImage(_ image: CGImage, width: Int, height: Int) -> CGImage? {
        guard let ctx = CGContext(
            data: nil,
            width: width, height: height,
            bitsPerComponent: 8, bytesPerRow: 0,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        ) else { return nil }
        // 補間品質を高く設定することで、リサイズ時の画質を向上させます。
        ctx.interpolationQuality = .high
        ctx.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))
        return ctx.makeImage()
    }

    /// 1枚の大きな画像を、重複領域を持つ小さなタイル群に分割します。
    private static func splitIntoTiles(image: CGImage, tileSize: Int, overlap: Int) -> [ImageTile] {
        var tiles: [ImageTile] = []
        // タイルを生成する際の移動ステップを計算します。重複分だけ戻る形です。
        let step = max(32, tileSize - overlap)
        // y方向（縦）にステップごとにループします。
        for y in stride(from: 0, to: image.height, by: step) {
            // x方向（横）にステップごとにループします。
            for x in stride(from: 0, to: image.width, by: step) {
                // 切り出す矩形（CGRect）を計算します。画像の端ではタイルサイズが小さくなります。
                let rect = CGRect(
                    x: x, y: y,
                    width: min(tileSize, image.width - x),
                    height: min(tileSize, image.height - y)
                )
                // 元画像からその矩形部分を切り出します（cropping）。
                if let im = image.cropping(to: rect) {
                    // 切り出した画像(im)と、元画像での位置(rect)をセットで配列に追加します。
                    tiles.append(ImageTile(rect: rect, image: im))
                }
            }
        }
        return tiles
    }
}

// このファイル内でのみ使用するプライベートな構造体。
// 1枚のタイルとその元画像における位置情報を保持します。
private struct ImageTile {
    let rect: CGRect
    let image: CGImage
}

// CGRectに対する便利な拡張機能。
private extension CGRect {
    // 座標が整数で、かつ幅と高さが0より大きいCGRectを返します。
    // 0サイズの画像などを扱う際のクラッシュを防ぎます。
    func integralNonEmpty() -> CGRect {
        let r = self.integral // 座標を整数に丸める
        return r.width > 0 && r.height > 0 ? r : .zero
    }
}

