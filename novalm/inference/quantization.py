import copy

import torch
from torch import nn
from torch.nn import functional as F


class QuantizedLinear4bit(nn.Module):
    """Symmetric per-output-channel INT4 weight-only Linear reference kernel.

    Weights are nibble-packed. Forward dequantizes them to the activation dtype,
    so this path reduces stored weight memory but is not a fast low-bit kernel.
    """
    def __init__(self, linear: nn.Linear) -> None:
        super().__init__()
        if not isinstance(linear, nn.Linear):
            raise TypeError("QuantizedLinear4bit expects nn.Linear")
        weight = linear.weight.detach().float()
        self.out_features, self.in_features = weight.shape
        scale = weight.abs().amax(dim=1).clamp_min(1e-8) / 7.0
        quantized = torch.round(weight / scale[:, None]).clamp(-7, 7).to(torch.int16) + 8
        if self.in_features % 2:
            quantized = F.pad(quantized, (0, 1), value=8)
        packed = (quantized[:, 0::2].to(torch.uint8)
                  | (quantized[:, 1::2].to(torch.uint8) << 4))
        self.register_buffer("packed_weight", packed.contiguous())
        self.register_buffer("scale", scale.to(torch.float16))
        if linear.bias is None:
            self.register_buffer("bias", None)
        else:
            self.register_buffer("bias", linear.bias.detach().clone())

    def dequantize_weight(self, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        packed = self.packed_weight
        low = (packed & 0x0F).to(torch.int16) - 8
        high = (packed >> 4).to(torch.int16) - 8
        values = torch.stack((low, high), dim=-1).flatten(-2)[..., :self.in_features]
        return values.to(dtype) * self.scale.to(dtype)[:, None]

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        weight = self.dequantize_weight(inputs.dtype)
        bias = None if self.bias is None else self.bias.to(inputs.dtype)
        return F.linear(inputs, weight, bias)


def dynamic_int8(model: nn.Module) -> nn.Module:
    """Return a CPU dynamic-int8 copy; embedding tables remain floating point."""
    return torch.ao.quantization.quantize_dynamic(copy.deepcopy(model).cpu(),
                                                   {nn.Linear}, dtype=torch.qint8)


def quantize_int4(model: nn.Module) -> nn.Module:
    """Deep-copy a model and replace each dense Linear with nibble-packed INT4."""
    quantized_model = copy.deepcopy(model)

    def replace_linears(parent: nn.Module) -> None:
        for name, child in list(parent.named_children()):
            if isinstance(child, nn.Linear):
                setattr(parent, name, QuantizedLinear4bit(child))
            else:
                replace_linears(child)

    replace_linears(quantized_model)
    return quantized_model
