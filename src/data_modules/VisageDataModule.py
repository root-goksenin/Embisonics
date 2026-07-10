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
        num_workers : int = 16,
        **kwargs,
    ):
        super().__init__()
        self.datapath = base_data_dir
        self.batch_size = batch_size
        self.sr = sr
        self.num_workers = num_workers

    def _augment_sample(self, sample):
        audio, audio_sr = sample[0]
        audio = pre_process_audio_visage(audio, audio_sr, self.sr)
        return (audio,)
    
    def make_web_dataset(self, path: str, shuffle: int):
        return (
            wds.WebDataset(
                path,
                resampled=True,
                nodesplitter=wds.shardlists.split_by_node,
                workersplitter=wds.shardlists.split_by_worker,
                shardshuffle=False,
            )
            .repeat()
            .shuffle(shuffle)
            .decode(wds.torch_audio, handler=wds.warn_and_continue)
            .to_tuple("flac")
            .map(self._augment_sample)
            .batched(self.batch_size, partial=False) # batches into ([B, 4, T],)
        )

    def setup(self, stage: str):
        if stage == "fit":
            self.audio_train = self.make_web_dataset(self.datapath, shuffle=2000)

    def train_dataloader(self):
        return DataLoader(
            self.audio_train,
            batch_size=None,
            pin_memory=True,
            num_workers=self.num_workers,
            prefetch_factor=None,
            persistent_workers=True,
        )