# macOS Super-Resolution App

macOS Super-Resolution App
A 4x super-resolution application for macOS. It uses a Core ML–converted version of Real-ESRGAN_x4v3 internally. The repository includes the pre-converted RealESRGAN_x4v3.mlpackage

macOS 向けの 4x 超解像度アプリケーションです。  
内部で [Real-ESRGAN_x4v3](https://github.com/xinntao/Real-ESRGAN) を Core ML に変換したモデルを利用しています。  
リポジトリには変換済みの **`RealESRGAN_x4v3.mlpackage`** が含まれています。

---

## Conversion Script

Conversion Script
This repository also includes a script scripts/convert_to_coreml2.py for converting the pre-trained PyTorch Real-ESRGAN model (.pth) into Core ML format (.mlpackage).

You can download the official .pth model from the Real-ESRGAN Releases, and regenerate the .mlpackage using the following command:

本リポジトリには、PyTorch 版 Real-ESRGAN の学習済みモデル (`.pth`) を  
Core ML フォーマット (`.mlpackage`) に変換するスクリプト **`scripts/convert_to_coreml2.py`** も含まれています。  

公式の `.pth` モデルを [Real-ESRGAN Releases](https://github.com/xinntao/Real-ESRGAN/releases) から取得し、  
以下のコマンドで `.mlpackage` を再生成できます:

```bash
pip install -r requirements.txt
python scripts/convert_to_coreml2.py \
    --input RealESRGAN_x4v3.pth \
    --output RealESRGAN_x4v3.mlpackage
