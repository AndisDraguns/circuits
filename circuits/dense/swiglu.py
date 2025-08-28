# from collections.abc import Callable
# from dataclasses import dataclass

import torch as t
import torch.nn as nn
import torch.nn.functional as F


class SwiGLU(nn.Module):
    """SwiGLU (Swish-Gated Linear Unit) activation as used in modern transformers."""
    def __init__(self, in_features: int, out_features: int, dtype: t.dtype = t.bfloat16):
        super().__init__()  # type: ignore
        self.dtype = dtype
        self.in_features = in_features
        self.out_features = out_features
        hidden_features = int(out_features * 2)
        self.w_silu = nn.Linear(in_features, hidden_features, bias=False)
        self.w_gate = nn.Linear(in_features, hidden_features, bias=False)
        self.w_out = nn.Linear(hidden_features, out_features, bias=False)

    def forward(self, x: t.Tensor) -> t.Tensor:
        # x = x.to(dtype=self.w_out.weight.dtype)
        x = x.type(self.dtype)
        return self.w_out(F.silu(self.w_silu(x)) * self.w_gate(x))


def swiglu_from_matrix(w: t.Tensor) -> SwiGLU:
    """
    Prepares SwiGLU weights from Matrices matrix that has biases folded into weights.
    1) Simulates a step fn with two offset ReLUs
    2) Simulates ReLU with SiLU by scaling up and down
    Making two ReLUs a, b such that a-b is this fn:
    y=0 until x=0.5-1/4c, then slope up until x=0.5+1/4c and y=1. Then y=1.
    Demo: https://www.desmos.com/calculator/sk42yz8ami
    """    
    c = 16  # making ReLU-simulated step fn steeper
    q = 16  # scaling before and after SiLU to avoid non-ReLU-like dip

    out_features = w.size(0)

    # constructing w_silu
    w1 = t.cat([
        w,
        w
    ], dim=0)
    w1[1:out_features]  -= 0.5 + 1/(2*c)  # add
    w1[out_features+1:] -= 0.5 - 1/(2*c)  # sub
    w1 *= c * q  # scale up
    w1[0,0] -= q  # to ensure that out vector begins with 1 

    # constructing w_gate
    w2 = t.zeros_like(w1)
    w2[:,0] += 1  # gate = 1

    # constructing w_out
    eye = t.eye(out_features)
    w3 = t.cat((eye, -eye), dim=1)
    w3 /= q  # scale down

    # create swiglu with weights w1, w2, w3
    swiglu = SwiGLU(w.size(1), out_features)
    for wi, param in zip([w1, w2, w3], [swiglu.w_silu, swiglu.w_gate, swiglu.w_out]):
        param.weight.data.zero_()
        param.weight.data[:w.size(0), :w.size(1)] = wi
    return swiglu


class MLP_SwiGLU(nn.Module):
    """MLP with SwiGLU activations"""
    
    def __init__(self, input_size: int, hidden_sizes: list[int], output_size: int, 
                 dtype: t.dtype = t.float32):
        super().__init__()  # type: ignore
        self.dtype = dtype
        
        # Build layers
        layers: list[SwiGLU | nn.Linear] = []
        prev_size = input_size
        
        # Hidden layers with SwiGLU activation
        for hidden_size in hidden_sizes:
            layers.append(SwiGLU(prev_size, hidden_size, dtype=dtype))
            prev_size = hidden_size
        layers.append(nn.Linear(prev_size, output_size, dtype=dtype))
        # layers.append(SwiGLU(prev_size, output_size, dtype=dtype))
        self.layers: nn.Sequential = nn.Sequential(*layers)
    
    def forward(self, x: t.Tensor) -> t.Tensor:
        return self.layers(x.to(self.dtype))

    def predict(self, x: t.Tensor) -> t.Tensor:
        """Binary prediction with sigmoid + threshold."""
        logits = self.forward(x)
        return (t.sigmoid(logits) > 0.5).float()
