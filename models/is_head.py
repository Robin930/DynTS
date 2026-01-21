import torch
import torch.nn as nn
from typing import Literal, Optional

ActivationName = Literal["relu", "gelu", "silu", "tanh", "none"]

def _get_act(name: ActivationName) -> nn.Module:
    name = name.lower()
    if name == "relu": return nn.ReLU(inplace=True)
    if name == "gelu": return nn.GELU()
    if name == "silu": return nn.SiLU()
    if name == "tanh": return nn.Tanh()
    if name == "softplus": return nn.Softplus()
    return nn.Identity()

class ISMLP(nn.Module):
    """
    结构: in -> Linear -> Act -> (Drop/Norm)
          -> Linear -> Act -> (Drop/Norm)
          -> Linear -> out
    """
    def __init__(
        self,
        in_dim: int,
        hidden1: int,
        hidden2: int,
        out_dim: int,
        activation: ActivationName = "silu",
        dropout: float = 0.0,
        layernorm: bool = False,
        bias: bool = True,
    ):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden1, bias=bias)
        self.fc2 = nn.Linear(hidden1, hidden2, bias=bias)
        self.fc3 = nn.Linear(hidden2, out_dim, bias=bias)

        self.act1 = _get_act(activation)
        self.act2 = _get_act(activation)

        self.do1 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.do2 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.ln1 = nn.LayerNorm(hidden1) if layernorm else nn.Identity()
        self.ln2 = nn.LayerNorm(hidden2) if layernorm else nn.Identity()

        # self._init_weights(activation)

    def _init_weights(self, activation: ActivationName):
        # 根据激活函数做合适的初始化
        act = activation.lower()
        for m in [self.fc1, self.fc2]:
            if act in ("relu", "silu"):
                nn.init.kaiming_uniform_(m.weight, a=0.0, nonlinearity="relu")
            elif act == "gelu":
                nn.init.xavier_uniform_(m.weight)  # GELU 常用 Xavier
            elif act == "tanh":
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain("tanh"))
            else:
                nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        # 输出层通常用较小权重
        nn.init.xavier_uniform_(self.fc3.weight)
        if self.fc3.bias is not None:
            nn.init.zeros_(self.fc3.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act1(x)
        x = self.ln1(x)
        x = self.do1(x)

        x = self.fc2(x)
        x = self.act2(x)
        x = self.ln2(x)
        x = self.do2(x)

        x = self.fc3(x)
        return x