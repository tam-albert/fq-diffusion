import deploy
import torch


class Quantizer(torch.nn.Module):
    def __init__(self, input_clip_ratio=1.0):
        super().__init__()
        self.input_clip_ratio = input_clip_ratio

    def forward(self, x):
        if not isinstance(x, deploy.PackedQuantizedTensor):
            scales_x = (torch.max(torch.abs(x), dim=-1)[0].unsqueeze(1) / 7).to(
                torch.float16
            ) * self.input_clip_ratio
            quantized_x = deploy.sym_quant(x, scales_x)
            packed_tensor = deploy.PackedQuantizedTensor(quantized_x, scales_x)
            return packed_tensor
        else:
            return x
