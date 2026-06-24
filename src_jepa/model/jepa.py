import copy, math
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from timm.models.vision_transformer import Block

from .pos_embed import get_2d_sincos_pos_embed
from .target import RouteATarget


# ---------------------------------------------------------------- front end
class PatchEmbed(nn.Module):
    """Plain per-patch Conv2D tokenizer, 8ch -> D. Non-overlapping (kernel==stride)
    so no token sees a neighbour's input -> no JEPA copy-leak through the tokenizer."""
    def __init__(self, in_ch, dim, fshape, tshape):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, dim, kernel_size=(fshape, tshape),
                              stride=(fshape, tshape))

    def forward(self, x):                       # x: (B, C, M, T)
        x = self.proj(x)                        # (B, D, Nf, Nt)
        B, D, Nf, Nt = x.shape
        return x.permute(0, 2, 3, 1).reshape(B, Nf * Nt, D)   # freq-major


class Encoder(nn.Module):
    """patch_embed + 2D-sincos pos + ViT. forward(feat, keep_idx)."""
    def __init__(self, in_ch, dim, depth, heads, grid, fshape, tshape, mlp_ratio=4.0):
        super().__init__()
        self.grid = grid
        self.patch_embed = PatchEmbed(in_ch, dim, fshape, tshape)
        pe = get_2d_sincos_pos_embed(dim, grid, cls_token_num=0)
        self.register_buffer("pos", torch.from_numpy(pe).float().unsqueeze(0))
        self.blocks = nn.ModuleList([
            Block(dim=dim, num_heads=heads, mlp_ratio=mlp_ratio, qkv_bias=True,
                  norm_layer=nn.LayerNorm) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)

    def forward(self, feat, keep_idx=None):
        x = self.patch_embed(feat) + self.pos          # (B,P,D)
        if keep_idx is not None:
            x = x[:, keep_idx, :]
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x)


class Predictor(nn.Module):
    """I-JEPA predictor: context tokens + mask tokens at target positions -> reps."""
    def __init__(self, dim, pred_dim, depth, heads, grid, mlp_ratio=4.0):
        super().__init__()
        self.embed = nn.Linear(dim, pred_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, pred_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        pe = get_2d_sincos_pos_embed(pred_dim, grid, cls_token_num=0)
        self.register_buffer("pos", torch.from_numpy(pe).float().unsqueeze(0))
        self.blocks = nn.ModuleList([
            Block(dim=pred_dim, num_heads=heads, mlp_ratio=mlp_ratio, qkv_bias=True,
                  norm_layer=nn.LayerNorm) for _ in range(depth)])
        self.norm = nn.LayerNorm(pred_dim)
        self.proj = nn.Linear(pred_dim, dim)

    def forward(self, z_ctx, context_idx, target_idx):
        B = z_ctx.shape[0]
        x = self.embed(z_ctx) + self.pos[:, context_idx, :]          # (B,N_ctx,P_dim)
        n_t = target_idx.numel()
        m = self.mask_token.expand(B, n_t, -1) + self.pos[:, target_idx, :]
        h = torch.cat([x, m], dim=1)
        for blk in self.blocks:
            h = blk(h)
        h = self.norm(h)[:, -n_t:, :]                                # mask-token outputs
        return self.proj(h)                                          # (B,N_tgt,D)


class DOAHead(nn.Module):
    def __init__(self, dim, n_grid, hidden=None):
        super().__init__()
        hidden = hidden or dim
        self.net = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, hidden),
                                 nn.GELU(), nn.Linear(hidden, n_grid))
    def forward(self, x):                       # (B,N,D) -> (B,N,|G|) logits
        return self.net(x)


# ---------------------------------------------------------------- full module
class SpatialJEPA(nn.Module):
    """I-JEPA on 8ch FOA features (logmel4 + AIV3 + diffuseness1) with Route A
    angular-distribution grounding on the PREDICTOR's masked outputs.

    lam is a fixed scalar loss weight: L = L_jepa + lam * L_iv.
    """
    def __init__(self, in_ch=8, dim=384, enc_depth=6, enc_heads=6,
                 pred_dim=192, pred_depth=4, pred_heads=6,
                 grid=(8, 25), fshape=16, tshape=8,
                 n_grid=256, kappa=40.0, ema=0.996,
                 lam=1.0):
        super().__init__()
        self.grid, self.fshape, self.tshape = grid, fshape, tshape
        self.ema = ema
        self.lam = float(lam)                       # plain fixed scalar, NOT a buffer

        self.context_encoder = Encoder(in_ch, dim, enc_depth, enc_heads, grid, fshape, tshape)
        self.target_encoder = copy.deepcopy(self.context_encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad_(False)

        self.predictor = Predictor(dim, pred_dim, pred_depth, pred_heads, grid)
        self.doa_head = DOAHead(dim, n_grid)
        self.route_a = RouteATarget(n_grid=n_grid, kappa=kappa,
                                    fshape=fshape, tshape=tshape)

    # ---- EMA ----
    @torch.no_grad()
    def update_target(self):
        m = self.ema
        for pt, pc in zip(self.target_encoder.parameters(),
                          self.context_encoder.parameters()):
            pt.mul_(m).add_(pc.detach(), alpha=1 - m)
        for bt, bc in zip(self.target_encoder.buffers(),
                          self.context_encoder.buffers()):
            bt.copy_(bc)

    @staticmethod
    def _weighted_ce(logits, q, w):
        logp = F.log_softmax(logits.float(), dim=-1)
        ce = -(q * logp).sum(dim=-1)                 # (B,N)
        return (w * ce).sum() / w.sum().clamp_min(1e-6)

    def compute_losses(self, feat, intensity_mel, context_idx, target_blocks):
        # Route A targets from the FULL signal (pseudo-labels)
        q_all, Wp_all = self.route_a(intensity_mel)              # (B,P,|G|),(B,P)

        z_ctx = self.context_encoder(feat, keep_idx=context_idx)  # (B,N_ctx,D)
        with torch.no_grad():
            z_tgt_all = self.target_encoder(feat)                 # (B,P,D) stop-grad

        L_jepa, L_iv, Wsum = 0.0, 0.0, 0.0
        for tb in target_blocks:
            zhat = self.predictor(z_ctx, context_idx, tb)         # (B,N_tgt,D)
            L_jepa = L_jepa + F.smooth_l1_loss(zhat, z_tgt_all[:, tb, :])
            logits = self.doa_head(zhat)                          # (B,N_tgt,|G|)
            q, w = q_all[:, tb, :], Wp_all[:, tb]
            L_iv = L_iv + self._weighted_ce(logits, q, w) * w.sum()
            Wsum = Wsum + w.sum()
        L_jepa = L_jepa / len(target_blocks)
        L_iv = L_iv / Wsum.clamp_min(1e-6)
        return L_jepa, L_iv, z_ctx, q_all, Wp_all