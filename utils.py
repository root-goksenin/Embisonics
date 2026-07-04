def get_identity_from_cfg(cfg):
    identity = "InChannels={}_Fraction={}_UseMSEIV={}".format(
        cfg.data.get("in_channels"),
        cfg.data.get("data_ratio"),
        cfg.optimizer.get("use_mse_loss_on_iv"),
    )
    identity += "Model={}_".format(
        cfg.model, 
    )
    identity += "LR={}_BatchSize={}_NrSamples={}_".format(
        cfg.optimizer.get("lr"),
        cfg.trainer.get("batch_size"),
        cfg.data.get("samples_per_audio"),
    )
    identity += "Patching={}_InputL={}".format(
        cfg.patching.get("name"),
        cfg.data.input_length,
    )
    return identity
