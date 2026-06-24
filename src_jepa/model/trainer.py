"""Lightning wrapper for Spatial I-JEPA (constant lambda, automatic optimization).

Plug in your ViSageDataModuleRAM (yields audio (B,4,T) ACN/SN3D ambiX).
Masks are sampled in training_step; the datamodule yields plain audio.
lam is a fixed hyperparameter: L = L_jepa + lam * L_iv.
"""
import torch
import pytorch_lightning as pl
import transformers

from .extractor import FeatureExtractor
from .jepa import SpatialJEPA
from .masking import sample_ijepa_masks


class SpatialJEPALit(pl.LightningModule):
    def __init__(self,
                 sr=32000,
                 n_mels=128,
                 target_frames=200,
                 fshape=16,
                 tshape=8,
                 dim=384,
                 enc_depth=6,
                 enc_heads=6,
                 pred_dim=192,
                 pred_depth=4,
                 pred_heads=6,
                 n_grid=256,
                 kappa=40.0,
                 ema=0.996,
                 lr=2e-4,
                 weight_decay=0.05,
                 warmup_steps=10000,
                 max_steps=300000,
                 lam: float = 1.0,
                 n_targets=4,
                 log_every=200):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = True
        self.grid = (n_mels // fshape, target_frames // tshape)
        self.extractor = FeatureExtractor(sample_rate=sr, n_mels=n_mels)
        self.model = SpatialJEPA(
            in_ch=8, dim=dim, enc_depth=enc_depth, enc_heads=enc_heads,
            pred_dim=pred_dim, pred_depth=pred_depth, pred_heads=pred_heads,
            grid=self.grid, fshape=fshape, tshape=tshape,
            n_grid=n_grid, kappa=kappa, ema=ema, lam=lam)
        self.target_frames = target_frames

    @torch.no_grad()
    def _features(self, audio):
        audio = audio.float()
        rms = torch.sqrt(audio[:, :1].pow(2).mean(-1, keepdim=True) + 1e-5)  # W-RMS norm
        audio = audio / rms
        d = self.extractor.extract_all(audio)
        feat = torch.cat([d["logmel"], d["aiv"], d["diffuseness"]], dim=1)   # (B,8,M,T)
        I = d["intensity_mel"]                                                # (B,3,M,T)

        # per-sample random temporal crop to target_frames
        B, C, _, Tn = feat.shape
        if Tn > self.target_frames:
            starts = torch.randint(0, Tn - self.target_frames + 1, (B,), device=feat.device)
            ar = torch.arange(self.target_frames, device=feat.device)
            idx = (starts[:, None] + ar[None, :])                            # (B, Tf)
            idx = idx[:, None, None, :].expand(B, C, feat.shape[2], self.target_frames)
            feat = feat.gather(-1, idx)
            I = I.gather(-1, idx[:, :3])                                      # I has 3 channels
        return feat.contiguous(), I.contiguous()

    def on_fit_start(self):
        # target encoder has no grad -> not broadcast by DDP at init; sync once
        if self.trainer.world_size > 1:
            for p in self.model.target_encoder.state_dict().values():
                torch.distributed.broadcast(p, src=0)

    def training_step(self, batch, batch_idx):
        audio = batch if torch.is_tensor(batch) else batch[0]
        feat, intensity_mel = self._features(audio)
        seed = self.global_step * self.trainer.world_size + self.global_rank
        ctx, tgts = sample_ijepa_masks(self.grid, n_targets=self.hparams.n_targets, seed=seed)
        ctx = ctx.to(self.device); tgts = [t.to(self.device) for t in tgts]

        L_jepa, L_iv, z_ctx, q, Wp = self.model.compute_losses(feat, intensity_mel, ctx, tgts)
        loss = L_jepa + self.model.lam * L_iv     # lam is a fixed scalar

        with torch.no_grad():
            conf = Wp > Wp.mean()
            ent = -(q.clamp_min(1e-9).log() * q).sum(-1)[conf].mean() if conf.any() \
                else torch.zeros((), device=self.device)
        self.log_dict({"loss": loss, "L_jepa": L_jepa, "L_iv": L_iv,
                       "std_z": z_ctx.std(), "q_entropy_conf": ent}, prog_bar=True)
        return loss

    def on_train_batch_end(self, *args, **kwargs):
        self.model.update_target()        # EMA after the optimizer step

    def configure_optimizers(self):
        params = [p for p in self.model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=self.hparams.lr,
                                weight_decay=self.hparams.weight_decay,
                                betas=(0.9, 0.95))
        sched = transformers.get_cosine_schedule_with_warmup(
            opt, self.hparams.warmup_steps, self.hparams.max_steps)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "step"}}