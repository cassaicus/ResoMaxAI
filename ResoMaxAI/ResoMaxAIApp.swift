//
//  ResoMaxAIApp.swift
//  ResoMaxAI
//
//  このファイルはアプリケーションのエントリーポイント（開始点）を定義します。
//  SwiftUIアプリケーションのライフサイクルは、Appプロトコルに準拠した構造体によって管理されます。

import SwiftUI

// @main属性は、このResoMaxAIAppがアプリケーションの起動時に最初に実行されることを示します。
// iOS 14, macOS 11以降で導入された新しい方法で、これによりAppDelegateやSceneDelegateが不要になります。
@main
struct ResoMaxAIApp: App {
    // bodyプロパティは、アプリケーションのUI階層（シーン）を定義します。
    // SceneはアプリのUIのコンテナであり、ここでは単一のウィンドウシーン（WindowGroup）を使用しています。
    var body: some Scene {
        // WindowGroupは、macOSでは新しいウィンドウを開く機能を提供し、
        // iOSやiPadOSでは複数のウィンドウインスタンスを管理するために使用されます。
        WindowGroup {
            // ContentViewは、アプリケーションのメインのUIビューです。
            // アプリケーションが起動すると、このContentViewがウィンドウに表示されます。
            ContentView()
        }
    }
}
