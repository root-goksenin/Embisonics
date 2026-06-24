import gc

import hydra
import pytorch_lightning as pl
import torch
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from src.data_modules import VisageDataModuleMAE
from src.model import SphereV4 as SphereV4
from src.patching import PatchStrategy
from src.masking import SpatialMaskMaker
from utils import get_identity_from_cfg


torch.set_float32_matmul_precision("medium")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True


torch.backends.cudnn.benchmark = True

@hydra.main(version_base=None, config_path="./configs", config_name="base")
def main(cfg):
    identity = get_identity_from_cfg(cfg)
    log_dir = f"{cfg.save_dir}/spherev3_logs" 
    save_dir = f"{cfg.save_dir}/spherev3/{identity.replace('_', '/')}"
    logger = TensorBoardLogger(
        log_dir,
        name=identity.replace("_", "/"),
    )
    checkpoint_callback = ModelCheckpoint(
        dirpath=save_dir,
        filename="{step}",
        verbose=True,
        every_n_train_steps=1000,
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
        callbacks=[checkpoint_callback, lr_monitor],
        log_every_n_steps=1,
        num_nodes=1,
        use_distributed_sampler=False,
        devices=int(cfg.trainer.num_gpus),
        strategy="ddp_find_unused_parameters_true"
        if int(cfg.trainer.num_gpus) > 1
        else "auto",
    )
    network_instance = SphereV4(
        model_size=cfg.model_size,
        lr=cfg.optimizer.lr,
        trainer=cfg.optimizer.name,
        b1=cfg.optimizer.b1,
        b2=cfg.optimizer.b2,
        weight_decay=cfg.optimizer.weight_decay,
        patch_strategy=PatchStrategy(
            input_tdim=cfg.data.target_length,
            input_fdim=cfg.data.num_mel_bins,
            tstride=cfg.patching.tstride,
            tshape=cfg.patching.tshape,
            fstride=cfg.patching.fstride,
            fshape=cfg.patching.fshape,
        ),
        in_channels=cfg.data.in_channels,
        num_mel_bins=cfg.data.num_mel_bins,
        target_length=cfg.data.target_length,
        input_length=cfg.data.input_length,
        sr=cfg.data.sr,
        compile_modules = cfg.trainer.compile_modules,
    )


    masker = SpatialMaskMaker(
        mask_patch=cfg.data.mask_patches, 
        context_cluster=False
    )

    data = VisageDataModuleMAE(
        batch_size=cfg.trainer.batch_size,
        nr_patches = network_instance.num_patches,
        sr = cfg.data.sr,
        masker=masker,
    )
    seed_everything(cfg.seed, workers=True)
    trainer.fit(network_instance, data)


if __name__ == "__main__":
    gc.collect()
    torch.cuda.empty_cache()
    main()
    gc.collect()
    torch.cuda.empty_cache()
