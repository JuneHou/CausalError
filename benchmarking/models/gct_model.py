"""
GCT (Graph-augmented Causal Transformer) error predictor for TRAIL benchmark.

Architecture:
  - Input: x [N, in_dim]  (N=19 error-type nodes, in_dim=5 features)
  - node_proj: 2-layer MLP  x → h [N, hid_dim]
  - 2 × GraphTransformerLayer with edge-weight + causal-flag attention bias
  - root_head:  h [N, hid_dim] → root_logits  [N]  (which node is root cause)
  - multi_head: h [N, hid_dim] → multi_logits [N]  (which nodes are present)

Attention bias per head (dense N×N):
  logit_ij = (q_i · k_j) / sqrt(d_head)  +  b_w * w_ij  +  b_c * c_ij
  w_ij = continuous edge weight  (0 for absent edges)
  c_ij = 1 if edge is validated causal, else 0

Loss (combined during training):
  loss = CrossEntropy(root_logits, y_root) + 0.5 * BCE(multi_logits, y_multi)
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# MLP helper
# ---------------------------------------------------------------------------

class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: Optional[int] = None):
        super().__init__()
        h = hidden_dim or out_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, h),
            nn.LayerNorm(h),
            nn.ReLU(),
            nn.Linear(h, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


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
      logit_ij = (q_i · k_j) / sqrt(d_head) + b_w * W_mat[i,j] + b_c * C_mat[i,j]

    Output shape: same as input [N, hid_dim].
    """

    def __init__(self, hid_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        assert hid_dim % num_heads == 0, "hid_dim must be divisible by num_heads"
        self.hid_dim   = hid_dim
        self.num_heads = num_heads
        self.d_head    = hid_dim // num_heads

        self.q_proj = nn.Linear(hid_dim, hid_dim, bias=False)
        self.k_proj = nn.Linear(hid_dim, hid_dim, bias=False)
        self.v_proj = nn.Linear(hid_dim, hid_dim, bias=False)
        self.out_proj = nn.Linear(hid_dim, hid_dim)

        # Learnable scalars for edge-weight and causal-flag bias
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
        x: torch.Tensor,          # [N, hid_dim]
        W_mat: torch.Tensor,       # [N, N]  edge weights
        C_mat: torch.Tensor,       # [N, N]  causal flags
    ) -> torch.Tensor:             # [N, hid_dim]
        N = x.size(0)
        H = self.num_heads
        d = self.d_head

        # Project Q, K, V  → [N, H, d]
        Q = self.q_proj(x).view(N, H, d)
        K = self.k_proj(x).view(N, H, d)
        V = self.v_proj(x).view(N, H, d)

        # Scaled dot-product attention logits: [H, N, N]
        # Q: [H, N, d], K: [H, N, d]
        Q_ = Q.permute(1, 0, 2)  # [H, N, d]
        K_ = K.permute(1, 0, 2)
        V_ = V.permute(1, 0, 2)

        logits = torch.bmm(Q_, K_.transpose(1, 2)) / math.sqrt(d)  # [H, N, N]

        # Add edge bias:  b_w[h] * W_mat  +  b_c[h] * C_mat
        # W_mat, C_mat: [N, N]  →  broadcast to [H, N, N]
        edge_bias = (
            self.b_w.view(H, 1, 1) * W_mat.unsqueeze(0) +
            self.b_c.view(H, 1, 1) * C_mat.unsqueeze(0)
        )
        logits = logits + edge_bias  # [H, N, N]

        attn = F.softmax(logits, dim=-1)   # [H, N, N]
        attn = self.attn_drop(attn)

        # Aggregate values: [H, N, d] → [N, H*d]
        out = torch.bmm(attn, V_)          # [H, N, d]
        out = out.permute(1, 0, 2).contiguous().view(N, self.hid_dim)  # [N, hid_dim]
        out = self.out_proj(out)

        # Residual + LayerNorm
        x = self.norm1(x + out)
        x = self.norm2(x + self.ff(x))
        return x


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class GCTErrorPredictor(nn.Module):
    """
    Graph-augmented Causal Transformer for TRAIL error prediction.

    Inputs per forward call:
      x          : [N, in_dim]   per-trace node features
      edge_index : [2, E]        LongTensor (shared across all traces)
      edge_weight: [E]           FloatTensor (continuous, 0–1)
      edge_is_causal: [E]        FloatTensor (0 or 1, optional)
      N          : int           number of nodes (default inferred from x)

    Outputs:
      root_logits : [N]          unnormalised score per node (root CE loss)
      multi_logits: [N]          unnormalised score per node (presence BCE loss)
    """

    def __init__(
        self,
        in_dim: int      = 5,
        hid_dim: int     = 64,
        num_nodes: int   = 19,
        num_heads: int   = 4,
        num_layers: int  = 2,
        dropout: float   = 0.1,
    ):
        super().__init__()
        self.num_nodes = num_nodes

        self.node_proj = MLP(in_dim, hid_dim, hidden_dim=hid_dim)

        self.layers = nn.ModuleList([
            GraphTransformerLayer(hid_dim, num_heads=num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])

        self.root_head  = nn.Linear(hid_dim, 1)
        self.multi_head = nn.Linear(hid_dim, 1)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _build_dense_mats(
        self,
        edge_index:    torch.Tensor,    # [2, E]
        edge_weight:   torch.Tensor,    # [E]
        edge_is_causal: Optional[torch.Tensor],  # [E] or None
        N: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build dense [N, N] adjacency matrices from sparse edge representation."""
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
        x:             torch.Tensor,                    # [N, in_dim]
        edge_index:    torch.Tensor,                    # [2, E]
        edge_weight:   torch.Tensor,                    # [E]
        edge_is_causal: Optional[torch.Tensor] = None,  # [E]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        N = x.size(0)

        # Build dense edge matrices (shared structure, per-batch call)
        W_mat, C_mat = self._build_dense_mats(edge_index, edge_weight, edge_is_causal, N)

        # Node projection
        h = self.node_proj(x)                           # [N, hid_dim]

        # Graph transformer layers
        for layer in self.layers:
            h = layer(h, W_mat, C_mat)                  # [N, hid_dim]

        root_logits  = self.root_head(h).squeeze(-1)    # [N]
        multi_logits = self.multi_head(h).squeeze(-1)   # [N]

        return root_logits, multi_logits


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def gct_loss(
    root_logits:  torch.Tensor,     # [N]
    multi_logits: torch.Tensor,     # [N]
    y_root:       torch.Tensor,     # scalar int
    y_multi:      torch.Tensor,     # [N] float
    root_weight:  Optional[torch.Tensor] = None,   # [N] class weights
    multi_pos_weight: Optional[torch.Tensor] = None,  # [N]
    multi_lambda: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns (total_loss, loss_root, loss_multi).

    total_loss = loss_root + multi_lambda * loss_multi
    """
    loss_root = F.cross_entropy(
        root_logits.unsqueeze(0),
        y_root.unsqueeze(0),
        weight=root_weight,
    )
    loss_multi = F.binary_cross_entropy_with_logits(
        multi_logits,
        y_multi.float(),
        pos_weight=multi_pos_weight,
    )
    total = loss_root + multi_lambda * loss_multi
    return total, loss_root, loss_multi
