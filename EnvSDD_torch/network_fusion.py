# network_fusion.py
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_add
from gat_conv import GATConv  # 与第1部分同目录
from torch_geometric.utils import softmax

@torch.no_grad()
def normalize_beta_by_dst(edge_index: torch.Tensor,
                          beta: torch.Tensor,
                          num_nodes: int,
                          eps: float = 1e-8) -> torch.Tensor:
    """把先验 β 按目标节点的入边归一：sum_{u->v} β_uv = 1。"""
    _, dst = edge_index
    sums = scatter_add(beta, dst, dim=0, dim_size=num_nodes) + eps
    return beta / sums[dst]


class FusionGATConv(nn.Module):
    def __init__(self, in_dim, out_dim, heads=1, concat=False,
                 dropout=0.0, add_self_loops=False, bias=False):
        super().__init__()
        # 与原 GATConv 保持形状/参数命名一致（参考你的 gat_conv.py）
        self.heads = heads
        self.out_channels = out_dim
        self.concat = concat
        self.dropout = dropout
        self.add_self_loops = add_self_loops

        # 线性映射（与原 GATConv 相同的共享权重写法）
        self.lin_src = nn.Parameter(torch.zeros(size=(in_dim, out_dim)))
        nn.init.xavier_normal_(self.lin_src.data, gain=1.414)
        self.lin_dst = self.lin_src

        # 节点级注意力向量
        self.att_src = nn.Parameter(torch.empty(1, heads, out_dim))
        self.att_dst = nn.Parameter(torch.empty(1, heads, out_dim))
        nn.init.xavier_normal_(self.att_src.data, gain=1.414)
        nn.init.xavier_normal_(self.att_dst.data, gain=1.414)

        self._alpha = None          # 存融合后的 γ（按 STAGATE 存放方式）
        self.attentions = None      # 存节点级 (alpha_src, alpha_dst)

    def forward(self, x, edge_index, *,
                beta, lam01,
                return_attention_weights=False,
                tied_attention=None):
        """
        x: (N, Fin)
        edge_index: (2, E) long
        beta: (E,) 已按 dst 归一
        lam01: 标量 in [0,1]
        tied_attention: None 或 (alpha_src, alpha_dst) 的节点级注意力分量
        """
        H, C = self.heads, self.out_channels
        assert x.dim()==2
        # 线性映射到多头形状（与原 GATConv 前向一致）
        x_src = x_dst = torch.mm(x, self.lin_src).view(-1, H, C)

        if tied_attention is None:
            alpha_src = (x_src * self.att_src).sum(dim=-1)               # (N, H)
            alpha_dst = (x_dst * self.att_dst).sum(dim=-1) if x_dst is not None else None
            alpha_nodes = (alpha_src, alpha_dst)
            self.attentions = alpha_nodes                                # 保存节点级注意力分量
        else:
            alpha_nodes = tied_attention
            alpha_src, alpha_dst = alpha_nodes

        src, dst = edge_index
        # 边级（未归一）打分 = 节点分量相加
        e = alpha_src[src] + (alpha_dst[dst] if alpha_dst is not None else 0.0)  # (E, H)
        # 按原 STAGATE：sigmoid → softmax(按 dst)
        e = torch.sigmoid(e)                                              # (E, H)
        # softmax: 逐头按 dst 归一
        # 兼容 torch_geometric 的 softmax 接口：index=dst, num_nodes=N
        alpha = softmax(e, dst, num_nodes=x.size(0))                      # (E, H)

        # --- α–β 融合（保持归一性：beta 已按 dst 归一；逐头共享同一 beta）---
        if beta.dim() == 1:
            beta_h = beta.unsqueeze(-1).expand(-1, H)                    # (E, H)
        else:
            beta_h = beta
        gamma = lam01 * alpha + (1.0 - lam01) * beta_h                    # (E, H)
        # dropout 施加在融合后的 γ 上（原始 STAGATE 对 α 做 dropout）
        if self.dropout and self.training:
            gamma = F.dropout(gamma, p=self.dropout, training=True)

        # 聚合：x_j * γ
        msg = x_src[src] * gamma.unsqueeze(-1)                            # (E, H, C)
        out = torch.zeros_like(x_src)                                     # (N, H, C)
        out = out.index_add(0, dst, msg)                                  # 聚合到 dst

        # 兼容 concat=False 的输出约定
        out = out.mean(dim=1) if not self.concat else out.view(-1, H*C)

        # 保存融合后的 γ，供外部可视化/对齐（模仿 gat_conv.GATConv）
        self._alpha = gamma if H==1 else gamma.mean(dim=1)                # (E,) 或 (E,H)

        if isinstance(return_attention_weights, bool) and return_attention_weights:
            return out, (edge_index, self._alpha)
        return out


class STAGATE_Fusion(nn.Module):
    """
    与 STAGATE 同构的 4 层：
      conv1: FusionGATConv  (ELU)
      conv2: GATConv(attention=False)  线性
      conv3: FusionGATConv  (ELU, tied_attention=conv1.attentions)
      conv4: GATConv(attention=False)  线性
    权重绑定：conv3.lin = conv2.lin^T；conv4.lin = conv1.lin^T
    """
    def __init__(self, hidden_dims, lambda_init=0.5, learnable_lambda=True):
        super().__init__()
        in_dim, num_hidden, out_dim = hidden_dims

        self.fconv1 = FusionGATConv(in_dim, num_hidden, heads=1, concat=False,
                                    dropout=0.0, add_self_loops=False, bias=False)
        # 线性层保持为你原始 GATConv(attention=False) 的调用
        self.conv2  = GATConv(num_hidden, out_dim, heads=1, concat=False,
                              dropout=0.0, add_self_loops=False, bias=False)
        self.fconv3 = FusionGATConv(out_dim, num_hidden, heads=1, concat=False,
                                    dropout=0.0, add_self_loops=False, bias=False)
        self.conv4  = GATConv(num_hidden, in_dim, heads=1, concat=False,
                              dropout=0.0, add_self_loops=False, bias=False)

        # λ 参数
        lam = torch.tensor(lambda_init, dtype=torch.float32)
        self.lmbd = nn.Parameter(lam) if learnable_lambda else lam

    def _lambda01(self, device):
        if isinstance(self.lmbd, nn.Parameter):
            return torch.sigmoid(self.lmbd)     # (0,1)
        return self.lmbd.to(device).clamp(0.0, 1.0)

    def forward(self, x, edge_index, beta_norm, return_attention=True):
        lam01 = self._lambda01(x.device)

        # conv1: 注意力融合 + ELU（取 α 用于对齐）
        h1, (ei1, alpha1) = self.fconv1(x, edge_index,
                                        beta=beta_norm, lam01=lam01,
                                        return_attention_weights=True,
                                        tied_attention=None)
        h1 = F.elu(h1)

        # conv2: 线性（attention=False）
        h2 = self.conv2(h1, edge_index, attention=False)

        # 权重绑定（严格等同 STAGATE）
        self.fconv3.lin_src.data = self.conv2.lin_src.transpose(0, 1)
        self.fconv3.lin_dst.data = self.conv2.lin_dst.transpose(0, 1)
        self.conv4.lin_src.data  = self.fconv1.lin_src.transpose(0, 1)
        self.conv4.lin_dst.data  = self.fconv1.lin_dst.transpose(0, 1)

        # conv3: 注意力融合 + ELU（复用 conv1 的节点级注意力分量）
        h3 = self.fconv3(h2, edge_index,
                         beta=beta_norm, lam01=lam01,
                         return_attention_weights=False,
                         tied_attention=self.fconv1.attentions)
        h3 = F.elu(h3)

        # conv4: 线性（attention=False）
        h4 = self.conv4(h3, edge_index, attention=False)

        # 返回编码端 h2，重构 h4，以及 conv1 的融合后 α（用于 KL/诊断）
        return h2, h4, alpha1
