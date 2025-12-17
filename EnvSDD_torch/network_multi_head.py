import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_add


# ========================= 1) STAGATE-style Attention =========================
class STAGATEConv(nn.Module):
    """
    STAGATE-style single-head attention:
      - Linear: h = X W
      - Edge score: e_ij = LeakyReLU( a_src^T h_i + a_dst^T h_j )
      - Attention: alpha_ij = sigmoid(e_ij)
      - Normalize per-dst: alpha_ij <- alpha_ij / sum_{k->j} alpha_kj
      - Aggregate: out_j = sum_{i->j} alpha_ij * h_i
    与标准 GAT 的差异：不用 softmax，而是 sigmoid + 按 dst 归一。
    """
    def __init__(self, in_dim, out_dim, negative_slope=0.2, dropout=0.0, bias=True):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim, bias=False)
        self.att_src = nn.Parameter(torch.empty(out_dim))
        self.att_dst = nn.Parameter(torch.empty(out_dim))
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        self.dropout = nn.Dropout(dropout) if dropout and dropout > 0 else None
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin.weight)
        # 初始化注意力向量为 0（更平滑的起点；也可用 xavier）
        nn.init.zeros_(self.att_src)
        nn.init.zeros_(self.att_dst)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x, edge_index, return_attention_weights=False, eps=1e-8):
        """
        x: (N, Fin); edge_index: (2, E) long
        return_attention_weights: True -> return (out, (edge_index, alpha))
        """
        N = x.size(0)
        src, dst = edge_index  # (E,), (E,)

        h = self.lin(x)                   # (N, Fout)
        h_src = h[src]                    # (E, Fout)
        h_dst = h[dst]                    # (E, Fout)

        # e_ij
        e = (h_src * self.att_src).sum(-1) + (h_dst * self.att_dst).sum(-1)  # (E,)
        e = self.leaky_relu(e)

        # alpha_ij = sigmoid(e_ij)
        alpha = torch.sigmoid(e)          # (E,)
        if self.dropout is not None and self.training:
            alpha = self.dropout(alpha)

        # per-dst normalize
        denom = scatter_add(alpha, dst, dim=0, dim_size=N) + eps
        alpha = alpha / denom[dst]

        # aggregate
        msg = h_src * alpha.unsqueeze(-1) # (E, Fout)
        out = scatter_add(msg, dst, dim=0, dim_size=N)  # (N, Fout)
        if self.bias is not None:
            out = out + self.bias

        if return_attention_weights:
            return out, (edge_index, alpha)
        return out


# ======================= 2) Dual-Head Layer (STAGATE + β) =====================
class DualHeadSTAGATELayer(nn.Module):
    """
    L：STAGATEConv（sigmoid 注意力 + dst 归一）
    F：固定 beta（微环境相似性）消息传递
    Fuse：h = λ * hL + (1-λ) * hF，且 λ ∈ [0,1]
    """
    def __init__(self, in_dim, out_dim, lambda_init=0.5, learnable_lambda=False,
                 act="elu", dropout=0.0):
        super().__init__()
        self.gat = STAGATEConv(in_dim, out_dim, negative_slope=0.2, dropout=dropout, bias=True)
        self.lin_fixed = nn.Linear(in_dim, out_dim, bias=False)
        self.learnable_lambda = learnable_lambda
        lam = torch.tensor(lambda_init, dtype=torch.float32)
        self.lmbd = nn.Parameter(lam) if learnable_lambda else lam
        if act == "elu":
            self.act = nn.ELU()
        elif act == "relu":
            self.act = nn.ReLU()
        elif act == "none":
            self.act = nn.Identity()
        else:
            raise ValueError("act must be one of: 'elu','relu','none'")

    def forward(self, x, edge_index, beta=None, return_attn=False):
        assert beta is not None, "beta (fixed weights) must be provided."

        # L: STAGATE attention
        if return_attn:
            hL, (ei_out, alpha) = self.gat(x, edge_index, return_attention_weights=True)
        else:
            hL = self.gat(x, edge_index)
            alpha = None

        # F: fixed-beta aggregation
        src, dst = edge_index
        xW = self.lin_fixed(x)                     # (N, F_out)
        msg = xW[src] * beta.unsqueeze(-1)         # (E, F_out)
        hF  = scatter_add(msg, dst, dim=0, dim_size=x.size(0))

        # Fuse with λ ∈ [0,1]
        if isinstance(self.lmbd, torch.nn.Parameter):
            lam = torch.sigmoid(self.lmbd)                # learnable scalar -> (0,1)
        else:
            lam = self.lmbd.to(x.device).clamp(0.0, 1.0)  # fixed scalar in [0,1]

        h = lam * hL + (1.0 - lam) * hF
        h = self.act(h)

        if return_attn:
            return h, alpha
        return h


# ======================= 3) AE: F->256->64->256->F (图解码) ====================
class DualHeadSTAGATE_AE(nn.Module):
    """
    与现有 AE 结构一致：F -> 256 -> 64 -> 256 -> F
    仅把 L 头替换为 STAGATE 风格；F 头与接口保持一致
    """
    def __init__(self, in_dim, hidden1=256, z_dim=64, hidden2=256,
                 lambda_init=0.5, learnable_lambda=True, dropout=0.0,
                 act="elu", last_act="none"):
        super().__init__()
        dims = [in_dim, hidden1, z_dim, hidden2, in_dim]
        self.layers = nn.ModuleList([
            DualHeadSTAGATELayer(dims[0], dims[1], lambda_init, learnable_lambda, act, dropout),
            DualHeadSTAGATELayer(dims[1], dims[2], lambda_init, learnable_lambda, act, dropout),  # 这里取 alpha
            DualHeadSTAGATELayer(dims[2], dims[3], lambda_init, learnable_lambda, act, dropout),
            DualHeadSTAGATELayer(dims[3], dims[4], lambda_init, learnable_lambda, last_act, dropout),
        ])
        self.z_dim = z_dim
        self.in_dim = in_dim

    def forward(self, x, edge_index, beta, return_attn=True):
        h = self.layers[0](x, edge_index, beta=beta, return_attn=False)
        if return_attn:
            h, alpha = self.layers[1](h, edge_index, beta=beta, return_attn=True)
        else:
            h = self.layers[1](h, edge_index, beta=beta, return_attn=False)
            alpha = None
        Z = h
        h = self.layers[2](h, edge_index, beta=beta, return_attn=False)
        X_hat = self.layers[3](h, edge_index, beta=beta, return_attn=False)
        return Z, X_hat, alpha


# ============================= 4) Utils & Losses ==============================
@torch.no_grad()
def normalize_beta_by_dst(edge_index: torch.Tensor,
                          beta: torch.Tensor,
                          num_nodes: int,
                          eps: float = 1e-8) -> torch.Tensor:
    """
    将 beta 按每个目标节点 dst 的入边求和归一，使 sum_{u->dst} beta_{u,dst} = 1
    """
    _, dst = edge_index
    sums = scatter_add(beta, dst, dim=0, dim_size=num_nodes) + eps
    beta_norm = beta / sums[dst]
    return beta_norm

def smoothness_loss(Z: torch.Tensor,
                    edge_index: torch.Tensor,
                    beta_norm: torch.Tensor) -> torch.Tensor:
    """
    L_smooth = sum_{(u->v)} beta_uv * || z_u - z_v ||^2 / N
    """
    src, dst = edge_index
    diff = Z[src] - Z[dst]                # (E, z_dim)
    loss = (beta_norm.unsqueeze(1) * (diff ** 2)).sum() / Z.size(0)
    return loss

def align_loss_kl(alpha: torch.Tensor,
                  beta_norm: torch.Tensor,
                  edge_index: torch.Tensor,
                  eps: float = 1e-8) -> torch.Tensor:
    """
    对齐损失（每个 dst 的入边分布）：KL( alpha || beta )
    注意：alpha 再按 dst 归一；beta 使用已按 dst 归一的 beta_norm
    """
    if alpha is None:
        raise ValueError("alpha is None but align loss is requested.")
    alpha = alpha.view(-1)  # (E,)
    _, dst = edge_index

    # 归一 alpha（按 dst）
    alpha = alpha.clamp_min(eps)
    a_sum = scatter_add(alpha, dst, dim=0, dim_size=int(dst.max()) + 1) + eps
    a = alpha / a_sum[dst]

    # beta 已归一
    b = beta_norm.clamp_min(eps)

    # KL = sum a * (log a - log b) / E
    kl = (a * (a.add(eps).log() - b.add(eps).log())).sum() / edge_index.size(1)
    return kl


# ================================ 5) Training =================================
def train_model(
    model: nn.Module,
    x: torch.Tensor,                 # (N,F)
    edge_index: torch.Tensor,        # (2,E) long
    beta: torch.Tensor,              # (E,) float
    epochs: int = 200,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    lambda_rec: float = 1.0,
    lambda_smooth: float = 0.5,
    lambda_align: float = 0.1,
    log_every: int = 10,
    device: str = None,
    # ---- 早停相关参数 ----
    early_stop: bool = True,
    tol: float = 1e-4,       # 相对改进阈值（如 1e-4 表示 <0.01% 的改进视为无改进）
    patience: int = 20,      # 连续多少个 epoch 无显著改进则停止
    min_epochs: int = 50     # 至少训练这么多 epoch 才考虑早停
):
    """
    返回：训练后的 (Z, X_hat) 以及每个 epoch 的日志
    早停准则：若在最近 `patience` 个 epoch 内，相对最佳 loss 的改进均 < tol，则停止。
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # 基本断言（避免隐性错误）
    assert edge_index.dtype == torch.long, "edge_index must be torch.long"
    assert beta.dtype in (torch.float32, torch.float64), "beta must be float tensor"
    assert x.ndim == 2 and edge_index.ndim == 2 and beta.ndim == 1, "shapes mismatch"

    model = model.to(device)
    x = x.to(device)
    edge_index = edge_index.to(device)
    beta = beta.to(device)

    N = x.size(0)

    # 归一化 beta（按 dst）, 固定一份
    with torch.no_grad():
        beta_norm = normalize_beta_by_dst(edge_index, beta, num_nodes=N).to(device)

    optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    logs = []

    # 早停跟踪
    best_loss = float("inf")
    epochs_no_improve = 0

    for ep in range(1, epochs + 1):
        model.train()
        optim.zero_grad()

        Z, X_hat, alpha = model(x, edge_index, beta=beta_norm, return_attn=True)

        # (1) 重构（MSE）
        loss_rec = F.mse_loss(X_hat, x)

        # (2) 平滑
        loss_smooth = smoothness_loss(Z, edge_index, beta_norm)

        # (3) 对齐（KL）
        loss_align = align_loss_kl(alpha, beta_norm, edge_index)

        loss = lambda_rec * loss_rec + lambda_smooth * loss_smooth + lambda_align * loss_align
        loss.backward()
        optim.step()

        # 记录日志
        loss_val = float(loss.item())
        logi = {
            'epoch': ep,
            'loss': loss_val,
            'rec': float(loss_rec.item()),
            'smooth': float(loss_smooth.item()),
            'align': float(loss_align.item()),
        }
        logs.append(logi)

        if (ep % log_every) == 0 or ep == 1:
            print(f"[{ep:04d}] loss={logi['loss']:.6f} | rec={logi['rec']:.6f} | "
                  f"smooth={logi['smooth']:.6f} | align={logi['align']:.6f}")

        # ===== 早停逻辑 =====
        if early_stop:
            if best_loss == float("inf"):
                best_loss = loss_val
                epochs_no_improve = 0
            else:
                # 相对改进（越小越好）
                rel_improve = (best_loss - loss_val) / (abs(best_loss) + 1e-12)
                if rel_improve > tol:
                    best_loss = loss_val
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1

            if ep >= min_epochs and epochs_no_improve >= patience:
                print(f"Early stopping at epoch {ep}: no relative improvement > {tol} "
                      f"for {patience} epochs (best_loss={best_loss:.6f}).")
                break

    # 返回最终表示与重构（不需要注意力）
    model.eval()
    with torch.no_grad():
        Z, X_hat, _ = model(x, edge_index, beta=beta_norm, return_attn=False)
    return Z, X_hat, logs
