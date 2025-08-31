import Foundation
import CoreML

// 計算デバイスの選択肢を表すenum
// CaseIterable: forEachで全ケースをループするため
// Identifiable: Pickerで各項目を区別するため
// CustomStringConvertible: Pickerに表示するテキストを定義するため
enum ComputeUnitOption: CaseIterable, Identifiable, CustomStringConvertible {
    case all
    case auto
    case cpuOnly

    // Identifiableに準拠するためのid
    var id: Self { self }

    // Pickerに表示する文字列
    var description: String {
        switch self {
        case .all:
            return "All Devices (GPU + ANE)"
        case .auto:
            return "Automatic (Prefer GPU)"
        case .cpuOnly:
            return "CPU Only"
//        case .all:
//            return "All Devices (GPU + ANE)"
        }
    }

    // 対応するCore MLの計算ユニット
    var mlComputeUnit: MLComputeUnits {
        switch self {
        case .all:
            return .all
        case .auto:
            return .cpuAndGPU
        case .cpuOnly:
            return .cpuOnly

        }
    }
}
