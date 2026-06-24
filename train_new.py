import gc
import hydra
import pytorch_lightning as pl
import torch
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from src_jepa.model.trainer import SpatialJEPALit
from src_jepa.data_modules import VisageDataModuleFromDisk
from utils import get_identity_from_cfg

torch.set_float32_matmul_precision("medium")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
 

@hydra.main(version_base=None, config_path="./configs", config_name="base")
def main(cfg):
    identity = get_identity_from_cfg(cfg)
    logger = TensorBoardLogger(f"{cfg.save_dir}/sjepa_logs", name=identity.replace("_", "/"))
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"{cfg.save_dir}/sjepa/{identity.replace('_', '/')}",
        filename="{step}", every_n_train_steps=1000, save_last=True,
        save_top_k=-1, enable_version_counter=True, verbose=True,
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")

    max_steps = cfg.trainer.steps // cfg.trainer.num_gpus
    trainer = pl.Trainer(
        logger=logger,
        accelerator=cfg.trainer.accelerator,
        max_steps=max_steps,
        precision=cfg.trainer.precision,
        deterministic=False,
        callbacks=[checkpoint_callback, lr_monitor],
        log_every_n_steps=1,
        num_nodes=1,
        use_distributed_sampler=False,
        devices=int(cfg.trainer.num_gpus),
        # target encoder has no grad + predictor mask-token participates unevenly
        strategy="ddp_find_unused_parameters_true" if int(cfg.trainer.num_gpus) > 1 else "auto",
    )

    model = SpatialJEPALit(
        sr=cfg.data.sr,
        n_mels=cfg.data.num_mel_bins,            # 128
        target_frames=cfg.data.target_length,    # 200
        
        fshape=cfg.patching.fshape,              # 16
        tshape=cfg.patching.tshape,              # 8
       
        dim=cfg.ssl.dim, 
        enc_depth=cfg.ssl.enc_depth, 
        enc_heads=cfg.ssl.enc_heads,
        pred_dim=cfg.ssl.pred_dim, 
        pred_depth=cfg.ssl.pred_depth, 
        pred_heads=cfg.ssl.pred_heads,
        lam_warmup=cfg.ssl.lam, 
        n_targets=cfg.ssl.n_targets,
        ema=cfg.ssl.ema,
        n_grid=cfg.routea.n_grid, 
        kappa=cfg.routea.kappa,

        lr=cfg.optimizer.lr, 
        weight_decay=cfg.optimizer.weight_decay,
        warmup_steps=cfg.optimizer.warmup_steps, 
        max_steps=max_steps,

    )

    data = VisageDataModuleFromDisk(                  # yields plain audio (B,4,T); masks made in training_step
        base_data_dir=cfg.data.base_data_dir,
        batch_size=cfg.trainer.batch_size,
        sr=cfg.data.sr,
    )

    seed_everything(cfg.seed, workers=True)
    trainer.fit(model, data, ckpt_path=cfg.get("resume_from", None))


if __name__ == "__main__":
    gc.collect()
    torch.cuda.empty_cache()
    main()
    gc.collect()
    torch.cuda.empty_cache()