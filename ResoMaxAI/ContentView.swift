import SwiftUI
import AppKit

// ContentViewは、アプリケーションのメインウィンドウに表示されるUIを定義します。
// SwiftUIのViewプロトコルに準拠しており、UIの構造と状態を宣言的に記述します。
struct ContentView: View {
    // @Stateプロパティラッパーは、Viewの状態を管理するために使用されます。
    // これらの変数の値が変更されると、SwiftUIは自動的にViewの関連部分を再描画します。

    // 高画質化処理を行うエンジン。アプリ起動時に初期化を試みます。
    // 失敗した場合はnilとなり、高画質化機能は無効になります。
    // アドバイス: エラーハンドリングを改善し、ユーザーにモデル読み込み失敗を通知するUIを追加するとより親切です。
    @State private var engine: SuperResolutionEngine? = try? SuperResolutionEngine(modelNameInBundle: "RealESRGAN_x4v3_tensor")

    // ユーザーが選択した入力画像のURL。保存時のデフォルトファイル名などに使用されます。
    @State private var inputURL: URL?

    // ユーザーが選択した入力画像（表示用）。
    @State private var inputImage: NSImage?

    // 高画質化処理によって生成された出力画像。
    @State private var outputImage: NSImage?

    // 高画質化処理が実行中かどうかを示すフラグ。UIの無効化やプログレスバーの表示に使用されます。
    @State private var isProcessing = false

    // 高画質化処理の進捗状況（0.0〜1.0）。プログレスバーにバインドされます。
    @State private var progress: Double = 0

    // ユーザーへの通知メッセージ（エラーや成功メッセージなど）。
    // nilでない場合にアラートが表示されます。
    @State private var message: String?

    // bodyはViewのUI構造を定義するコンピューテッドプロパティです。
    var body: some View {
        // VStackは、子ビューを垂直方向に配置します。
        VStack(spacing: 12) {
            // HStackは、子ビューを水平方向に配置します。主に操作ボタンを配置します。
            HStack {
                // 画像ファイルを選択するための「開く」ボタン。
                Button {
                    openImage()
                } label: { Label("開く…", systemImage: "folder") }

                // 高画質化処理を開始するボタン。
                Button {
                    runSR()
                } label: { Label("高画質化", systemImage: "sparkles") }
                // 入力画像がない、エンジンが初期化されていない、または処理中の場合は無効化します。
                .disabled(inputImage == nil || engine == nil || isProcessing)

                // 生成された画像を保存するボタン。
                Button {
                    saveOutput()
                } label: { Label("保存", systemImage: "square.and.arrow.down") }
                // 出力画像がない、または処理中の場合は無効化します。
                .disabled(outputImage == nil || isProcessing)

                // Spacerは利用可能なスペースを埋めるために使われ、ボタンを左寄せにします。
                Spacer()

                // 処理中の場合にのみプログレスバーを表示します。
                if isProcessing {
                    ProgressView(value: progress)
                        .frame(width: 180) // プログレスバーの幅を固定します。
                }
            }

            // 入力画像と出力画像を並べて表示するためのHStack。
            HStack {
                // 入力画像表示エリア。
                VStack {
                    Text("入力")
                    // ZStackはビューを重ねて配置します。背景色と画像を重ねています。
                    ZStack {
                        // 背景色を設定します。
                        Color(NSColor.underPageBackgroundColor)
                        // inputImageがnilでない場合に画像を表示します。
                        if let img = inputImage {
                            Image(nsImage: img).resizable().scaledToFit()
                        } else {
                            // 画像が選択されていない場合はプレースホルダーテキストを表示します。
                            Text("画像を開いてください").foregroundStyle(.secondary)
                        }
                    }
                }
                // 出力画像表示エリア。
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
        .padding(12) // 全体にパディングを追加します。
        .frame(minWidth: 900, minHeight: 560) // ウィンドウの最小サイズを設定します。
        // messageに文字列が設定されたときにアラートを表示します。
        .alert(item: Binding(get: {
            // messageがnilでなければ、AlertMessageインスタンスを作成して返します。
            message.map { AlertMessage(text: $0) }
        }, set: { _ in
            // アラートが閉じたときにmessageをnilに戻し、アラートを非表示にします。
            message = nil
        })) { msg in
            Alert(title: Text(msg.text))
        }
    }

    // 画像ファイル選択パネルを開き、選択された画像を読み込むプライベートメソッド。
    private func openImage() {
        let panel = NSOpenPanel()
        panel.allowedContentTypes = [.image] // 画像ファイルタイプのみを許可します。
        panel.allowsMultipleSelection = false // 複数選択を不許可にします。

        // runModal()でパネルを表示し、ユーザーが「OK」をクリックした場合に処理を続行します。
        if panel.runModal() == .OK, let url = panel.url, let nsimg = NSImage(contentsOf: url) {
            // 状態変数を更新します。これによりUIが再描画されます。
            inputURL = url
            inputImage = nsimg
            outputImage = nil // 新しい画像を開いたら、前の出力はクリアします。
        }
    }

    // 高画質化処理を実行するプライベートメソッド。
    private func runSR() {
        // 入力画像(のCGImage表現)とエンジンが利用可能であることを確認します。
        guard let cg = inputImage?.cgImage, let engine else { return }

        // 処理中の状態に設定します。
        isProcessing = true
        progress = 0
        outputImage = nil // 処理開始時に前の出力をクリアします。

        // `userInitiated`品質のバックグラウンドスレッドで重い処理を実行し、UIのフリーズを防ぎます。
        DispatchQueue.global(qos: .userInitiated).async {
            do {
                // SuperResolutionEngineのsuperResolveメソッドを呼び出して高画質化処理を実行します。
                let outCG = try engine.superResolve(
                    cgImage: cg,
                    // タイルサイズ。大きいほどVRAMを消費しますが、つなぎ目が減る可能性があります。
                    // アドバイス：この値はユーザーが設定できるようにすると、より柔軟なアプリになります。
                    tileSize: 512,     // 実際はエンジン内で固定入力サイズに自動調整
                    // タイルの重なり幅。つなぎ目の不連続性を減らすために重要です。
                    overlap: 32,
                    // 進捗をUIに反映するためのクロージャ。メインスレッドでprogress状態変数を更新します。
                    progress: { p in DispatchQueue.main.async { self.progress = p } },
                    // キャンセル機能。現在はnilですが、CancellationTokenを渡すことで実装可能です。
                    cancellation: nil
                )
                // 結果のCGImageをNSImageに変換します。
                let outNS = NSImage(cgImage: outCG, size: NSSize(width: outCG.width, height: outCG.height))

                // UIの更新は必ずメインスレッドで行います。
                DispatchQueue.main.async {
                    self.outputImage = outNS
                    self.isProcessing = false
                    self.progress = 1.0 // 完了したのでプログレスを100%にします。
                }
            } catch {
                // エラーが発生した場合も、UI更新はメインスレッドで行います。
                DispatchQueue.main.async {
                    self.isProcessing = false
                    self.message = "失敗: \(error.localizedDescription)" // よりユーザーフレンドリーなエラーメッセージが良いでしょう。
                }
            }
        }
    }

    // 生成された画像をPNG形式で保存するプライベートメソッド。
    private func saveOutput() {
        // 出力画像(out)をPNGデータに変換します。
        // NSImage -> TIFF representation -> NSBitmapImageRep -> PNG data という多段階の変換を行っています。
        // アドバイス: この変換処理は複雑なので、NSImageの拡張メソッドとして切り出すとコードがすっきりします。
        guard let out = outputImage,
              let tiff = out.tiffRepresentation,
              let rep = NSBitmapImageRep(data: tiff),
              let data = rep.representation(using: .png, properties: [:]) else {
            message = "保存用データ作成に失敗"
            return
        }

        let panel = NSSavePanel()
        // デフォルトのファイル名を「(元のファイル名)_x4.png」とします。
        panel.nameFieldStringValue = (inputURL?.deletingPathExtension().lastPathComponent ?? "output") + "_x4.png"
        panel.allowedContentTypes = [.png] // PNG形式のみを許可します。

        if panel.runModal() == .OK, let url = panel.url {
            do {
                // 指定されたURLにファイルとして書き出します。
                try data.write(to: url)
                message = "保存しました: \(url.lastPathComponent)"
            } catch {
                message = "保存に失敗: \(error.localizedDescription)"
            }
        }
    }
}

// .alertモディファイアで使用するためのIdentifiableに準拠した構造体。
// これにより、message文字列が更新されるたびに新しいアラートとして認識させることができます。
private struct AlertMessage: Identifiable {
    let id = UUID() // 一意なIDを自動で生成します。
    let text: String
}

// NSImageをCGImageに変換するための便利な拡張。
// 注意: この方法は常に成功するとは限らず、失敗するとnilを返します。
// より堅牢な実装が必要な場合は、特定の画像表現（例: tiffRepresentation）からCGImageSourceを使う方法もあります。
private extension NSImage {
    var cgImage: CGImage? {
        // NSImageを描画するための矩形を定義します。
        var rect = CGRect(origin: .zero, size: size)
        // cgImage(forProposedRect:context:hints:)メソッドでCGImageを生成します。
        // このメソッドは、複数の表現を持つNSImageから最適なものを選択してCGImageを作成します。
        return cgImage(forProposedRect: &rect, context: nil, hints: nil)
    }
}
