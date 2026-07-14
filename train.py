import gc

import hydra
import pytorch_lightning as pl
import torch
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from src.data_modules import ViSageDataModule        # visage_datamodule.py
from src.model import SphereV5                          # sphere_v5.py
from src.patching import PatchStrategy

from spatial_probe_callback import SpatialProbeCallback
from utils import get_identity_from_cfg

torch.set_float32_matmul_precision("high")
torch.backends.cudnn.benchmark = True


class RankDecorrelatedRNG(pl.Callback):
    """Give each DDP rank its own augmentation RNG stream (see module docstring)."""

    def on_fit_start(self, trainer, pl_module):
        base = torch.initial_seed() % (2 ** 31)
        torch.manual_seed(base + trainer.global_rank)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(base + trainer.global_rank)


@hydra.main(version_base=None, config_path="./configs", config_name="sphere")
def main(cfg):
    # before model construction: reproducible weights AND masks/rotations
    seed_everything(cfg.seed, workers=True)

    identity = get_identity_from_cfg(cfg)
    log_dir = f"{cfg.save_dir}/sphere_logs"
    save_dir = f"{cfg.save_dir}/sphere/{identity.replace('_', '/')}"
    logger = WandbLogger(
        project="ViSage", 
        name=identity,
        save_dir=log_dir,
        config=cfg,
        log_model=False,                    # we use ModelCheckpoint instead
        save_code=True,                     # uploads script & git diff
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=save_dir,
        filename="{step}",
        verbose=True,
        every_n_train_steps=cfg.trainer.ckpt_every_n_steps,
        save_last=True,
        enable_version_counter=True,
        save_top_k=-1,
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")
    trainer = pl.Trainer(
        logger=logger,
        accelerator=cfg.trainer.accelerator,
        max_epochs=cfg.trainer.epochs,
        max_steps=cfg.trainer.steps // cfg.trainer.num_gpus,
        precision=cfg.trainer.precision,
        deterministic=False,
        callbacks=[checkpoint_callback, 
                   lr_monitor,
                   SpatialProbeCallback("/projects/0/prjs1261/probe_dataset", every_n_steps=1000)],
        log_every_n_steps=1,
        num_nodes=1,
        use_distributed_sampler=False,
        devices=int(cfg.trainer.num_gpus),
        strategy="ddp_find_unused_parameters_true"
        if int(cfg.trainer.num_gpus) > 1
        else "auto",
    )

    network_instance = SphereV5(
        # ---- architecture ----
        encoder_embedding_dim=cfg.model.encoder_embedding_dim,
        encoder_depth=cfg.model.encoder_depth,
        num_heads=cfg.model.num_heads,
        mlp_ratio=cfg.model.mlp_ratio,
        decoder_depth=cfg.model.decoder_depth,
        decoder_num_heads=cfg.model.decoder_num_heads,
        decoder_embedding_dim=cfg.model.decoder_embedding_dim,
        inter_channel_heads=cfg.model.inter_channel_heads,
        inter_channel_layers=cfg.model.inter_channel_layers,
        gramt_model_id=cfg.model.gramt_model_id,
        gramt_mask_context=cfg.model.gramt_mask_context,
        # ---- losses ----
        q_loss_weight=cfg.loss.q,
        diffuseness_loss_weight=cfg.loss.diffuseness,
        leveldiff_loss_weight=cfg.loss.leveldiff,
        level_loss_weight=cfg.loss.level,
        # ---- optimizer ----
        lr=cfg.optimizer.lr,
        b1=cfg.optimizer.b1,
        b2=cfg.optimizer.b2,
        weight_decay=cfg.optimizer.weight_decay,
        warmup_steps=cfg.optimizer.warmup_steps,
        # ---- patching / data geometry ----
        patch_strategy=PatchStrategy(
            input_tdim=cfg.data.target_length,
            input_fdim=cfg.data.num_mel_bins,
            tstride=cfg.patching.tstride,
            tshape=cfg.patching.tshape,
            fstride=cfg.patching.fstride,
            fshape=cfg.patching.fshape,
        ),
        in_channels=cfg.data.in_channels,
        sr=cfg.data.sr,
        num_mel_bins=cfg.data.num_mel_bins,
        input_length=cfg.data.input_length,
        target_length=cfg.data.target_length,
        f_max=cfg.data.f_max,
        # ---- RouteA direction target ----
        n_grid=cfg.route_a.n_grid,
        vmf_kappa=cfg.route_a.vmf_kappa,
        grid_chunk=cfg.route_a.grid_chunk,
        # ---- coherence windows (short shared with RouteA tile) ----
        coh_tile_t=cfg.coherence.tile_t,
        coh_tile_f=cfg.coherence.tile_f,
        coh_long_t=cfg.coherence.long_t,
        coh_long_f=cfg.coherence.long_f,
        # ---- masking ----
        mask_ratio=cfg.masking.ratio,
        mask_p_span=cfg.masking.p_span,
        mask_p_random=cfg.masking.p_random,
        mask_p_censor=cfg.masking.p_censor,
        span_min_tokens=cfg.masking.span_min_tokens,
        span_max_tokens=cfg.masking.span_max_tokens,
        # ---- rotation augmentation ----
        rotation_mode=cfg.augmentation.rotation_mode,
        rotation_prob=cfg.augmentation.rotation_prob,
        # ---- logging ----
        log_every_n_steps=cfg.trainer.image_log_every_n_steps,

        samples_per_clip=cfg.trainer.samples_per_clip,
        native_sr=cfg.data.native_sr

    )


    data = ViSageDataModule(
        base_data_dir=cfg.data.glob,
        batch_size=cfg.trainer.batch_size,
        sr=cfg.data.sr,
        num_workers=cfg.trainer.num_workers,
        native_sr=cfg.data.native_sr
    )

    trainer.fit(network_instance, data, ckpt_path=cfg.model.get("pretrained_ckpt", None))


if __name__ == "__main__":
    gc.collect()
    torch.cuda.empty_cache()
    main()
    gc.collect()
    torch.cuda.empty_cache()