"""Mixed masking policy for the spatial MAE.

Three regimes, sampled per clip:

  'span'   : contiguous full-band time spans (every frequency patch of the
             masked time columns is hidden).  Kills the interpolation
             shortcut where the model copies a static DOA from an unmasked
             frequency band; forces trajectory modelling.
  'censor' : everything after a random cut point is masked (pure future
             extrapolation -> velocity, not just position), topped up with
             random patches inside the visible prefix to hit the exact count.
  'random' : plain uniform patch masking (keeps the local TF prediction task
             alive, regularizes).

All regimes mask exactly K = round(mask_ratio * Nt) * Nf patches, so every
sample in a batch has the same number of visible patches -- required by the
encoder's batched boolean gather (`embedded[visible_mask].view(B, L, -1)`).

Token order is freq-major: p = f * Nt + t, matching SpatialFrontEnd,
SphereV5._patchify and RouteATarget.
"""
import torch
from torch import nn


class MixedSpatialMaskMaker(nn.Module):

    REGIMES = ("span", "random", "censor")

    def __init__(self, n_freq_patches: int, n_time_patches: int,
                 mask_ratio: float = 0.8,
                 p_span: float = 0.6, p_random: float = 0.2,
                 p_censor: float = 0.2,
                 span_min: int = 2, span_max: int = None,
                 censor_extra_max: int = None):
        """
        Args:
            n_freq_patches, n_time_patches: patch grid (Nf, Nt).
            mask_ratio: fraction of patches to mask.  Internally quantized to
                whole time columns: K = round(mask_ratio * Nt) * Nf.
            span_min / span_max: span lengths in *time tokens*.  Default
                span_max = max(span_min + 1, Nt // 4) -- sized for short
                (~2 s) crops; raise it if you move to longer crops.
            censor_extra_max: how far past the minimal cut the censor point
                may move (deficit is filled with random prefix patches).
                Default = Nt - K_cols, i.e. the visible prefix is at most
                doubled and at most half of it gets randomly masked.
        """
        super().__init__()
        Nf, Nt = n_freq_patches, n_time_patches
        self.Nf, self.Nt = Nf, Nt
        self.num_patches = Nf * Nt
        self.K_cols = int(round(mask_ratio * Nt))
        assert 0 < self.K_cols < Nt, (
            f"mask_ratio={mask_ratio} with Nt={Nt} must leave at least one "
            f"masked and one visible time column"
        )
        self.K = self.K_cols * Nf                     # exact masked patches
        self.num_visible = self.num_patches - self.K

        self.span_min = span_min
        self.span_max = span_max if span_max is not None \
            else max(span_min + 1, Nt // 4)
        assert 1 <= self.span_min <= self.span_max <= Nt

        self.c_pure = Nt - self.K_cols                # visible prefix, pure censor
        self.censor_extra_max = censor_extra_max if censor_extra_max is not None \
            else self.c_pure

        probs = torch.tensor([p_span, p_random, p_censor], dtype=torch.float)
        assert (probs >= 0).all() and probs.sum() > 0
        # plain attribute (not a buffer): stays on CPU when the model moves to GPU
        self._regime_probs = probs / probs.sum()

    # ---- regimes: each returns a (Nf, Nt) bool grid, True = masked --------
    def _span_cols(self) -> torch.Tensor:
        Nt = self.Nt
        cols = torch.zeros(Nt, dtype=torch.bool)
        need = self.K_cols
        tries = 0
        while need > 0 and tries < 200:
            tries += 1
            L = int(torch.randint(self.span_min, self.span_max + 1, (1,)))
            L = min(L, need)      # a span adds <= L new columns: never overshoots
            s = int(torch.randint(0, Nt - L + 1, (1,)))
            if not (~cols[s:s + L]).any():
                continue          # fully overlaps existing spans; retry
            cols[s:s + L] = True
            need = self.K_cols - int(cols.sum())
        if need > 0:              # rare fragmentation fallback: exact top-up
            vis = (~cols).nonzero(as_tuple=False).flatten()
            cols[vis[torch.randperm(vis.numel())[:need]]] = True
        return cols

    def _mask_span(self) -> torch.Tensor:
        return self._span_cols().unsqueeze(0).expand(self.Nf, self.Nt).clone()

    def _mask_random(self) -> torch.Tensor:
        flat = torch.zeros(self.num_patches, dtype=torch.bool)
        flat[torch.randperm(self.num_patches)[:self.K]] = True
        return flat.view(self.Nf, self.Nt)

    def _mask_censor(self) -> torch.Tensor:
        Nf, Nt = self.Nf, self.Nt
        extra = int(torch.randint(0, self.censor_extra_max + 1, (1,))) \
            if self.censor_extra_max > 0 else 0
        c = min(self.c_pure + extra, Nt - 1)          # visible prefix length
        m = torch.zeros(Nf, Nt, dtype=torch.bool)
        m[:, c:] = True                               # mask the whole future
        deficit = self.K - Nf * (Nt - c)
        if deficit > 0:                               # top up inside the prefix
            idx = torch.randperm(Nf * c)[:deficit]
            m[idx // c, idx % c] = True
        return m

    # -----------------------------------------------------------------------
    def forward(self, batch_size: int) -> torch.Tensor:
        """Returns visible_mask (B, P) bool on CPU, True = VISIBLE.

        Every row has exactly `self.num_visible` True entries.
        """
        regimes = torch.multinomial(self._regime_probs, batch_size,
                                    replacement=True)
        fns = (self._mask_span, self._mask_random, self._mask_censor)
        out = torch.empty(batch_size, self.num_patches, dtype=torch.bool)
        for b in range(batch_size):
            out[b] = ~fns[int(regimes[b])]().reshape(-1)
        return out


if __name__ == "__main__":
    torch.manual_seed(0)
    Nf, Nt = 8, 25                                  # e.g. 128 mel/16, 200 fr/8
    mm = MixedSpatialMaskMaker(Nf, Nt, mask_ratio=0.8)
    B = 512
    vis = mm(B)
    counts = vis.sum(dim=1)
    assert (counts == mm.num_visible).all(), "visible count must be constant"
    print(f"grid {Nf}x{Nt}, K={mm.K} masked, {mm.num_visible} visible "
          f"({mm.num_visible / mm.num_patches:.2%}) -- constant across batch: OK")

    # regime-specific structure checks
    for name, fn in (("span", mm._mask_span), ("censor", mm._mask_censor),
                     ("random", mm._mask_random)):
        for _ in range(200):
            m = fn()
            assert int(m.sum()) == mm.K, f"{name}: wrong mask count"
        if name == "span":
            cols = m.all(dim=0) | (~m).all(dim=0)
            assert cols.all(), "span mask must be full-band per column"
        if name == "censor":
            fully_masked = m.all(dim=0)
            last_vis = (~fully_masked).nonzero().max()
            assert fully_masked[last_vis + 1:].all(), "censor tail must be solid"
    print("span is full-band, censor tail is solid, counts exact: OK")

    # ascii peek at one sample of each regime
    for name, fn in (("span", mm._mask_span), ("censor", mm._mask_censor)):
        m = fn()
        print(f"\n{name} (rows=freq, cols=time, #=masked):")
        for f in range(Nf):
            print("".join("#" if m[f, t] else "." for t in range(Nt)))
    print("\nall spatial_masking tests passed")