import Foundation
import CoreML

// 計算デバイスの選択肢を表すenum
// CaseIterable: forEachで全ケースをループするため
// Identifiable: Pickerで各項目を区別するため
// CustomStringConvertible: Pickerに表示するテキストを定義するため
enum ComputeUnitOption: CaseIterable, Identifiable, CustomStringConvertible {
    case auto
    case cpuOnly
    case all

    // Identifiableに準拠するためのid
    var id: Self { self }

    // Pickerに表示する文字列
    var description: String {
        switch self {
        case .auto:
            return "自動 (GPU優先)"
        case .cpuOnly:
            return "CPU のみ"
        case .all:
            return "全デバイス (GPU + ANE)"
        }
    }

    // 対応するCore MLの計算ユニット
    var mlComputeUnit: MLComputeUnits {
        switch self {
        case .auto:
            return .cpuAndGPU
        case .cpuOnly:
            return .cpuOnly
        case .all:
            return .all
        }
    }
}
