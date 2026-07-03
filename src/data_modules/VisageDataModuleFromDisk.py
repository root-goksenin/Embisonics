"""ViSage data module for SphereV5.

Fixes vs the previous version:
  * InfiniteRAMDataset was called as (audio_data, rotations_per_clip,
    shuffle=True) against a (audio_tensors, shuffle) signature -> TypeError.
  * e3nn / rotation matrices removed entirely: rotation augmentation now
    happens on-GPU inside SphereV5._prepare_batch (waveform-domain, exact),
    and masking also moved into the model, so the dataset yields bare
    (4, S) audio tensors and the default collate produces (B, 4, S).

Requirement: pre_process_audio_visage must return equal-length clips
(pad/truncate) or the default collate will fail.
"""
import glob
import random

import torch
import torchaudio
import pytorch_lightning as pl
from torch.utils.data import DataLoader, IterableDataset, get_worker_info
from tqdm import tqdm

from .dataset_functions import pre_process_audio_visage


class InfiniteRAMDataset(IterableDataset):
    """Infinite stream over pre-loaded clips; yields (4, S) audio only."""

    def __init__(self, audio_tensors, shuffle: bool = True):
        super().__init__()
        self.audio_tensors = audio_tensors
        self.shuffle = shuffle

    def __iter__(self):
        # per-worker seeding (each worker streams its own shuffled permutation;
        # for infinite with-replacement sampling this is fine)
        worker_info = get_worker_info()
        seed = torch.initial_seed() if worker_info is None \
            else torch.initial_seed() + worker_info.id
        rng = random.Random(seed)

        indices = list(range(len(self.audio_tensors)))
        while True:
            if self.shuffle:
                rng.shuffle(indices)
            for idx in indices:
                yield self.audio_tensors[idx]


class ViSageDataModuleRAM(pl.LightningDataModule):
    def __init__(
        self,
        base_data_dir: str = "/projects/0/prjs1338/visage/audios/audio/*.flac",
        batch_size: int = 32,
        sr: int = 32000,
        num_workers: int = 4,
        **kwargs,
    ):
        super().__init__()
        self.datapath = base_data_dir
        self.batch_size = batch_size
        self.sr = sr
        self.num_workers = num_workers
        self.audio_data = []

    def setup(self, stage: str = None):
        if stage == "fit" or stage is None:
            files = glob.glob(self.datapath)
            if not files:
                raise FileNotFoundError(f"No files found at {self.datapath}")

            print(f"Loading {len(files)} files into RAM...")
            loaded_data = []
            for f in tqdm(files):
                wav, original_sr = torchaudio.load(f)
                wav = pre_process_audio_visage(wav, original_sr, self.sr)
                assert wav.shape[0] == 4, \
                    f"{f}: expected 4-channel FOA, got {wav.shape[0]}"
                loaded_data.append(wav.cpu())

            self.audio_data = loaded_data
            print(f"Successfully loaded {len(self.audio_data)} samples into memory.")

    def train_dataloader(self):
        dataset = InfiniteRAMDataset(self.audio_data, shuffle=True)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            pin_memory=True,
            num_workers=self.num_workers,
            prefetch_factor=2,
            persistent_workers=True,
        )