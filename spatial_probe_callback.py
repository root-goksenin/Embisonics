import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import soundfile as sf
import pytorch_lightning as pl


def _azel_to_xyz(az_deg, el_deg):
    az, el = np.radians(az_deg), np.radians(el_deg)
    return np.stack([np.cos(el) * np.cos(az),
                     np.cos(el) * np.sin(az),
                     np.sin(el)], -1)


class SpatialProbeCallback(pl.Callback):
    def __init__(self, dataset_dir, every_n_steps=5000, max_clips=None,
                 lam=1e-2, batch_size=64, log_prefix="probe",
                 log_hand_baseline=False):
        super().__init__()
        self.dir = dataset_dir
        self.every = every_n_steps
        self.max_clips = max_clips
        self.lam = lam
        self.bs = batch_size
        self.prefix = log_prefix
        self.log_hand_baseline = log_hand_baseline
        self._feats = None          # fp16 CPU cache: (N, 7, T, F)
        self._hand = None           # hand-feature matrix for the ceiling probe
        self._did_baselines = False

    # ------------------------------------------------------------------ data
    def setup(self, trainer, pl_module, stage=None):
        if self._feats is not None or not trainer.is_global_zero:
            return
        if hasattr(self, "meta"):
            return
        meta = pd.read_parquet(os.path.join(self.dir, "metadata.parquet"))
        meta = meta[np.isfinite(meta.rt60_s) & (meta.dist_m > 0)]
        if self.max_clips:
            meta = meta.sample(min(self.max_clips, len(meta)), random_state=0)
        meta = meta.reset_index(drop=True)
        self.meta = meta
        self.tr = torch.from_numpy((meta.split == "train").values.copy())
        self.te = torch.from_numpy((meta.split == "test").values.copy())
        assert self.tr.any() and self.te.any(), "need both splits in metadata"

        self.y_doa = torch.from_numpy(
            _azel_to_xyz(meta.az.values, meta.el.values)).float()
        self.y_dist = torch.log(torch.from_numpy(meta.dist_m.values).float())
        self.y_rt60 = torch.log(torch.from_numpy(meta.rt60_s.values).float())
        self.y_drr = torch.from_numpy(meta.drr_db.values).float()  # linear dB

        wavs = []
        for cid in meta.clip_id:
            x, sr = sf.read(os.path.join(self.dir, "wavs", cid + ".wav"),
                            always_2d=True)
            wavs.append(torch.from_numpy(x.T).float())
        self.wavs = torch.stack(wavs)                       # (N, 4, T) CPU
        print(f"[probe] {len(meta)} clips "
              f"({int(self.tr.sum())} train / {int(self.te.sum())} test; "
              f"{meta[meta.split == 'test'].house.nunique()} test houses)")

    @torch.no_grad()
    def _precompute_features(self, model):
        """7-ch model inputs + hand features. Weight-independent -> once."""
        T = model.target_length
        feats, hand = [], []
        for wb in self.wavs.split(self.bs):
            f = model._extract_features(wb.to(model.device))
            fb7 = torch.cat([f["logmel"], f["aiv"]], dim=1)[:, :, :T]  # (B,7,T,F)
            feats.append(fb7.half().cpu())
            # hand features for the ceiling probe
            aiv = f["aiv"][:, :, :T]
            I = f["intensity_mel"][:, :, :T]
            E = f["energy_mel"][:, :, :T]

            def ma(x, kt):
                pad = (1, 1, kt // 2, kt - 1 - kt // 2)
                return F.avg_pool2d(F.pad(x, pad, mode="replicate"),
                                    (kt, 3), stride=1)

            psi_s = (1 - ma(I, 5).norm(dim=1, keepdim=True)
                     / (ma(E, 5) + 1e-8)).clamp(0, 1)
            psi_l = (1 - ma(I, 35).norm(dim=1, keepdim=True)
                     / (ma(E, 35) + 1e-8)).clamp(0, 1)
            h = torch.cat(
                [aiv.mean(dim=(2, 3)), aiv.std(dim=(2, 3)),
                 psi_s.mean(dim=(1, 2, 3))[:, None],
                 psi_l.mean(dim=(1, 2, 3))[:, None],
                 (psi_l - psi_s).mean(dim=(1, 2, 3))[:, None],
                 f["logmel"][:, 0, :T].mean(dim=(1, 2))[:, None],
                 f["logmel"][:, 0, :T].std(dim=(1, 2))[:, None]], -1)
            hand.append(h.float().cpu())
        self._feats = torch.cat(feats)
        self._hand = torch.cat(hand)

    # ----------------------------------------------------------------- probe
    @torch.no_grad()
    def _embed(self, model):
        Z = []
        for fb in self._feats.split(self.bs):
            fb = fb.float().to(model.device)
            Z.append(model.get_audio_representation(
                fb[:, :4], fb[:, 4:], strategy="mean").float().cpu())
        return torch.cat(Z)

    # @staticmethod
    # def _ridge(Xtr, ytr, Xte, lam):
    #     mu, sd = Xtr.mean(0), Xtr.std(0).clamp_min(1e-6)
    #     Xtr = (Xtr - mu) / sd
    #     Xte = (Xte - mu) / sd
    #     X = torch.cat([Xtr, torch.ones(len(Xtr), 1)], 1).double()
    #     A = X.T @ X + lam * len(Xtr) * torch.eye(X.shape[1]).double()
    #     Wm = torch.linalg.solve(A, X.T @ ytr.double())
    #     Xe = torch.cat([Xte, torch.ones(len(Xte), 1)], 1).double()
    #     return (Xe @ Wm).float()

    @staticmethod
    def _ridge(Xtr, ytr, Xte, *args, k=5):
        """
        Non-linear probe using k-Nearest Neighbors.
        Xtr, Xte: Embeddings (N, D)
        ytr: Targets (N, ...)
        """
        # 1. Normalize embeddings to unit sphere (Cosine Similarity = L2 Distance)
        Xtr = F.normalize(Xtr, p=2, dim=-1)
        Xte = F.normalize(Xte, p=2, dim=-1)
        
        sim = torch.mm(Xte.double(), Xtr.double().T)
        
        topk_sim, topk_idx = sim.topk(k, dim=-1)
        
        if ytr.ndim == 1:
            ytr = ytr.unsqueeze(-1)
        y_neighbors = ytr[topk_idx]
        
        preds = y_neighbors.mean(dim=1)
        
        return preds.float()
    
    def _scores(self, Z):
        tr, te = self.tr, self.te
        out = {}
        v = F.normalize(self._ridge(Z[tr], self.y_doa[tr], Z[te], self.lam),
                        dim=-1)
        cos = (v * self.y_doa[te]).sum(-1).clamp(-1, 1)
        out["doa_median_deg"] = torch.rad2deg(torch.acos(cos)).median()
        for name, y in (("dist", self.y_dist), ("rt60", self.y_rt60)):
            p = self._ridge(Z[tr], y[tr, None], Z[te], self.lam).squeeze(-1)
            out[f"{name}_log_mae"] = (p - y[te]).abs().mean()
            out[f"{name}_pearson"] = torch.corrcoef(
                torch.stack([p, y[te]]))[0, 1]
            
        p = self._ridge(Z[tr], self.y_drr[tr, None], Z[te], self.lam).squeeze(-1)
        out["drr_mae_db"] = (p - self.y_drr[te]).abs().mean()
        out["drr_pearson"] = torch.corrcoef(
            torch.stack([p, self.y_drr[te]]))[0, 1]
        return out

    # -------------------------------------------------------------- schedule
    def _run(self, trainer, model, prefix):
        was = model.training
        model.eval()
        if self._feats is None:
            self._precompute_features(model)
        scores = self._scores(self._embed(model))
        # trainer.logger.log_metrics works with WandbLogger/TensorBoard in ANY
        # hook (model.log is not supported in on_train_start) and pins the
        # wandb step explicitly, so the step-0 random-init point and the
        # periodic evals form one continuous probe/* curve.
        if trainer.logger is not None:
            trainer.logger.log_metrics(
                {f"{prefix}/{k}": float(v) for k, v in scores.items()},
                step=trainer.global_step)
        if was:
            model.train()
        return scores

    def on_train_start(self, trainer, model):
        if not trainer.is_global_zero or self._did_baselines:
            return
        # random-init scores logged under the SAME probe/* keys at step 0, so
        # every wandb chart starts at the untrained floor and continues with
        # the periodic evals as one curve.
        s0 = self._run(trainer, model, self.prefix)
        sh = self._scores(self._hand)                # console reference only
        if self.log_hand_baseline and trainer.logger is not None:
            trainer.logger.log_metrics(
                {f"{self.prefix}_hand/{k}": float(v) for k, v in sh.items()},
                step=trainer.global_step)
        self._did_baselines = True
        print(f"[probe] floor(random-init): "
              f"doa {s0['doa_median_deg']:.1f} deg, "
              f"dist r {s0['dist_pearson']:.2f}, "
              f"drr r {s0['drr_pearson']:.2f}, rt60 r {s0['rt60_pearson']:.2f}")
        print(f"[probe] ceiling(hand feats): "
              f"doa {sh['doa_median_deg']:.1f} deg, "
              f"dist r {sh['dist_pearson']:.2f}, "
              f"drr r {sh['drr_pearson']:.2f}, rt60 r {sh['rt60_pearson']:.2f}")

    def on_train_batch_end(self, trainer, model, *a, **kw):
        if not trainer.is_global_zero:
            return
        step = trainer.global_step
        if step > 0 and step % self.every == 0:
            s = self._run(trainer, model, self.prefix)
            print(f"[probe @ {step}] doa {s['doa_median_deg']:.1f} deg | "
                  f"dist r {s['dist_pearson']:.2f} "
                  f"(logMAE {s['dist_log_mae']:.3f}) | "
                  f"drr r {s['drr_pearson']:.2f} "
                  f"(MAE {s['drr_mae_db']:.1f} dB) | "
                  f"rt60 r {s['rt60_pearson']:.2f}")