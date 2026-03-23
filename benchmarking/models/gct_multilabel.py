"""
GCTMultiLabel — Graph-augmented Causal Transformer for multi-label error detection.

Task: given per-trace node features on a fixed 19-node error-type graph,
predict which error types are present in the trace (binary multi-label).

Architecture:
  - Input: x [N, in_dim]  (N=19 error-type nodes, in_dim=5 features)
  - node_proj: Linear  x → h [N, hid_dim]
  - 2 × GraphTransformerLayer with edge-weight + causal-flag attention bias
  - out: Linear h → logits [N]  (one logit per error type)

Attention bias per head (dense N×N):
  logit_ij = (q_i · k_j) / sqrt(d_head)  +  b_w * w_ij  +  b_c * c_ij
  w_ij = continuous edge weight  (0 for absent edges)
  c_ij = 1 if edge is validated causal, else 0

Loss:
  BCEWithLogitsLoss(logits, y_multi.float(), pos_weight=class_pos_weight)
  pos_weight computed from train split only.

Prediction:
  probs = sigmoid(logits)
  pred  = (probs >= threshold).long()   # threshold tuned on val (default 0.5)
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Graph Transformer Layer
# ---------------------------------------------------------------------------

class GraphTransformerLayer(nn.Module):
    """
    Single transformer layer on a fixed N-node graph.

    Edge biases are dense N×N matrices:
      W_mat [N, N]  — continuous edge weight (0 for absent edges)
      C_mat [N, N]  — causal flag (0/1)

    Attention logit for head h, pair (i,j):
      logit_ij = (q_i · k_j) / sqrt(d_head) + b_w[h] * W_mat[i,j] + b_c[h] * C_mat[i,j]
    """

    def __init__(self, hid_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        assert hid_dim % num_heads == 0, "hid_dim must be divisible by num_heads"
        self.hid_dim   = hid_dim
        self.num_heads = num_heads
        self.d_head    = hid_dim // num_heads

        self.q_proj  = nn.Linear(hid_dim, hid_dim, bias=False)
        self.k_proj  = nn.Linear(hid_dim, hid_dim, bias=False)
        self.v_proj  = nn.Linear(hid_dim, hid_dim, bias=False)
        self.out_proj = nn.Linear(hid_dim, hid_dim)

        # Learnable scalars for edge-weight and causal-flag bias (one per head)
        self.b_w = nn.Parameter(torch.zeros(num_heads))
        self.b_c = nn.Parameter(torch.zeros(num_heads))

        self.attn_drop = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(hid_dim)
        self.norm2 = nn.LayerNorm(hid_dim)

        self.ff = nn.Sequential(
            nn.Linear(hid_dim, hid_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hid_dim * 2, hid_dim),
        )

    def forward(
        self,
        x: torch.Tensor,      # [N, hid_dim]
        W_mat: torch.Tensor,  # [N, N]  edge weights
        C_mat: torch.Tensor,  # [N, N]  causal flags
    ) -> torch.Tensor:        # [N, hid_dim]
        N = x.size(0)
        H = self.num_heads
        d = self.d_head

        Q = self.q_proj(x).view(N, H, d).permute(1, 0, 2)  # [H, N, d]
        K = self.k_proj(x).view(N, H, d).permute(1, 0, 2)
        V = self.v_proj(x).view(N, H, d).permute(1, 0, 2)

        # Scaled dot-product logits: [H, N, N]
        logits = torch.bmm(Q, K.transpose(1, 2)) / math.sqrt(d)

        # Edge bias: b_w[h] * W_mat  +  b_c[h] * C_mat  →  [H, N, N]
        edge_bias = (
            self.b_w.view(H, 1, 1) * W_mat.unsqueeze(0) +
            self.b_c.view(H, 1, 1) * C_mat.unsqueeze(0)
        )
        logits = logits + edge_bias

        attn = F.softmax(logits, dim=-1)   # [H, N, N]
        attn = self.attn_drop(attn)

        out = torch.bmm(attn, V)                              # [H, N, d]
        out = out.permute(1, 0, 2).contiguous().view(N, self.hid_dim)  # [N, hid_dim]
        out = self.out_proj(out)

        x = self.norm1(x + out)
        x = self.norm2(x + self.ff(x))
        return x


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class GCTMultiLabel(nn.Module):
    """
    Graph-augmented Causal Transformer for multi-label error type detection.

    Inputs per forward call:
      x             : [N, in_dim]   per-trace node features
      edge_index    : [2, E]        LongTensor (shared across all traces)
      edge_weight   : [E]           FloatTensor (continuous, 0–1)
      edge_is_causal: [E]           FloatTensor (0 or 1, optional)

    Output:
      logits : [N]   one logit per error type (apply sigmoid for probabilities)
    """

    def __init__(
        self,
        in_dim: int     = 5,
        hid_dim: int    = 64,
        heads: int      = 4,
        num_nodes: int  = 19,
        num_layers: int = 2,
        dropout: float  = 0.1,
    ):
        super().__init__()
        self.num_nodes = num_nodes

        self.node_proj = nn.Linear(in_dim, hid_dim)

        self.layers = nn.ModuleList([
            GraphTransformerLayer(hid_dim, num_heads=heads, dropout=dropout)
            for _ in range(num_layers)
        ])

        self.out = nn.Linear(hid_dim, 1)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _build_dense_mats(
        self,
        edge_index:     torch.Tensor,           # [2, E]
        edge_weight:    torch.Tensor,           # [E]
        edge_is_causal: Optional[torch.Tensor], # [E] or None
        N: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        device = edge_index.device
        W_mat = torch.zeros(N, N, device=device)
        C_mat = torch.zeros(N, N, device=device)
        src, dst = edge_index[0], edge_index[1]
        W_mat[src, dst] = edge_weight
        if edge_is_causal is not None:
            C_mat[src, dst] = edge_is_causal.float()
        return W_mat, C_mat

    def forward(
        self,
        x:              torch.Tensor,                    # [N, in_dim]
        edge_index:     torch.Tensor,                    # [2, E]
        edge_weight:    torch.Tensor,                    # [E]
        edge_is_causal: Optional[torch.Tensor] = None,  # [E]
    ) -> torch.Tensor:                                   # [N]
        N = x.size(0)
        W_mat, C_mat = self._build_dense_mats(edge_index, edge_weight, edge_is_causal, N)

        h = self.node_proj(x)                    # [N, hid_dim]
        for layer in self.layers:
            h = layer(h, W_mat, C_mat)           # [N, hid_dim]

        logits = self.out(h).squeeze(-1)         # [N]
        return logits
