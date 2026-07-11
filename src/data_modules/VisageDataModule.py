from torch.utils.data import DataLoader
import pytorch_lightning as pl
import webdataset as wds

from .dataset_functions import pre_process_audio_visage


class ViSageDataModule(pl.LightningDataModule):
    def __init__(
        self,
        base_data_dir: str,
        batch_size: int = 32,
        sr: int = 32000 ,
        num_workers : int = 4,
        **kwargs,
    ):
        super().__init__()
        self.datapath = base_data_dir
        self.batch_size = batch_size
        self.sr = sr
        self.num_workers = num_workers

    def _augment_sample(self, sample):
        audio, audio_sr = sample[0]
        # NO resample here anymore. Just fix layout/length so default
        # collate in .batched() can stack.
        audio = to_float_foa(audio)                    # whatever pre_process did minus resample
        T_expect = int(5.0 * self.native_sr)           # native sr of the shards
        if audio.shape[-1] != T_expect:
            audio = F.pad(audio, (0, max(0, T_expect - audio.shape[-1])))[..., :T_expect]
        return (audio,)
    
    def make_web_dataset(self, path: str, shuffle: int):
        return (
            wds.WebDataset(
                path,
                resampled=True,
                nodesplitter=wds.shardlists.split_by_node,
                workersplitter=wds.shardlists.split_by_worker,
                shardshuffle=50,          # shuffle shard order instead of relying on a huge sample buffer
            )
            .repeat()
            .shuffle(300, initial=100)    # much smaller in-memory buffer
            .decode(wds.torch_audio, handler=wds.warn_and_continue)
            .to_tuple("flac")
            .map(self._augment_sample)
            .batched(self.batch_size, partial=False)
        )

    def train_dataloader(self):
        return DataLoader(
            self.audio_train,
            batch_size=None,
            pin_memory=True,
            num_workers=6,
            prefetch_factor=2,
            persistent_workers=True,
        )

    def setup(self, stage: str):
        if stage == "fit":
            self.audio_train = self.make_web_dataset(self.datapath, shuffle=2000)