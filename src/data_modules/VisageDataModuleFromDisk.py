"""ViSage data module for SphereV5 — disk-streaming version.

Changes vs the RAM version:
  * No longer loads 100k clips into memory. `setup()` only globs the file
    list; decoding + pre-processing happen lazily inside the DataLoader
    workers, so throughput scales with `num_workers` instead of RAM.
  * Files are sharded across (DDP rank x dataloader worker) so no two
    workers decode the same file in the same pass — balanced epochs and
    no intra-batch duplicates.
  * Each pass over a worker's shard is reshuffled with a fresh seed.
  * Corrupt/unreadable files are skipped with a warning instead of
    killing a multi-hour run.

Requirement unchanged: pre_process_audio_visage must return equal-length
clips (pad/truncate) or the default collate will fail.
"""
import glob
import random
import warnings

import torch
import torchaudio
import pytorch_lightning as pl
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from .dataset_functions import pre_process_audio_visage


class InfiniteDiskDataset(IterableDataset):
    """Infinite stream over on-disk clips; yields (4, S) audio only.

    Each worker owns a disjoint shard of the file list (sharded first by
    DDP rank, then by dataloader worker id) and loops over it forever,
    reshuffling every pass.
    """

    def __init__(self, files, sr: int, shuffle: bool = True):
        super().__init__()
        self.files = files
        self.sr = sr
        self.shuffle = shuffle

    def _shard(self):
        """Return this worker's disjoint slice of the file list."""
        rank, world_size = 0, 1
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
            world_size = torch.distributed.get_world_size()

        worker_info = get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        num_workers = worker_info.num_workers if worker_info else 1

        shard_id = rank * num_workers + worker_id
        num_shards = world_size * num_workers
        return self.files[shard_id::num_shards]

    def _load(self, path):
        wav, original_sr = torchaudio.load(path)
        wav = pre_process_audio_visage(wav, original_sr, self.sr)
        if wav.shape[0] != 4:
            raise ValueError(f"expected 4-channel FOA, got {wav.shape[0]}")
        return wav

    def __iter__(self):
        files = self._shard()
        if not files:
            raise RuntimeError(
                "Empty shard: more (ranks x workers) than files. "
                "Reduce num_workers or check the glob pattern."
            )

        worker_info = get_worker_info()
        seed = torch.initial_seed() if worker_info is None \
            else torch.initial_seed() + worker_info.id
        rng = random.Random(seed)

        while True:
            if self.shuffle:
                rng.shuffle(files)
            for path in files:
                try:
                    yield self._load(path)
                except Exception as e:  # noqa: BLE001 — skip bad files, keep training
                    warnings.warn(f"Skipping {path}: {e}")


class ViSageDataModule(pl.LightningDataModule):
    def __init__(
        self,
        base_data_dir: str = "/projects/0/prjs1261/visage/audios/audio/*.flac",
        batch_size: int = 32,
        sr: int = 32000,
        num_workers: int = 8,   # decoding is now on the hot path; scale this up
        prefetch_factor: int = 4,
        **kwargs,
    ):
        super().__init__()
        self.datapath = base_data_dir
        self.batch_size = batch_size
        self.sr = sr
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self.files = []

    def setup(self, stage: str = None):
        if stage == "fit" or stage is None:
            # Sort for a deterministic order -> identical sharding on every
            # DDP rank (glob order is filesystem-dependent).
            self.files = sorted(glob.glob(self.datapath))
            if not self.files:
                raise FileNotFoundError(f"No files found at {self.datapath}")
            print(f"Indexed {len(self.files)} files (streaming from disk).")

    def train_dataloader(self):
        dataset = InfiniteDiskDataset(self.files, sr=self.sr, shuffle=True)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            pin_memory=True,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
            persistent_workers=self.num_workers > 0,
        )