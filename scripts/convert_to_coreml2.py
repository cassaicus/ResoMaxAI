import torch
import coremltools as ct
import numpy as np
from realesrgan.archs.srvgg_arch import SRVGGNetCompact

# モデル構造
model = SRVGGNetCompact(
    num_in_ch=3, num_out_ch=3,
    num_feat=64, num_conv=32,
    upscale=4, act_type='prelu'
)

# 重み読み込み
ckpt = torch.load('weights/realesr-general-x4v3.pth', map_location='cpu')
if 'params' in ckpt:
    ckpt = ckpt['params']
model.load_state_dict(ckpt, strict=True)
model.eval()

# RGBWrapper（contiguous付き）
class RGBWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, x):
        x = x[:, [2,1,0], :, :].contiguous()
        out = self.model(x)
        out = out[:, [2,1,0], :, :].contiguous()
        return out

wrapped_model = RGBWrapper(model)

# ダミー入力
example_input = torch.rand(1, 3, 64, 64)

# TorchScript化
traced_model = torch.jit.trace(wrapped_model, example_input)

# Core ML変換（出力をTensorTypeに）
mlmodel = ct.convert(
    traced_model,
    inputs=[ct.ImageType(
        name="input_image",
        shape=example_input.shape,
        scale=1/255.0,
        bias=[0,0,0],
        color_layout="RGB"
    )],
    outputs=[ct.TensorType(
        name="output_image",
        dtype=np.float32
    )],
    compute_units=ct.ComputeUnit.ALL
)

mlmodel.save("RealESRGAN_x4v3_tensor.mlpackage")
print("変換完了: RealESRGAN_x4v3_tensor.mlpackage")
