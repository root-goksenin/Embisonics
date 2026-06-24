import os
import glob
import random
import torch
import torchaudio
import pytorch_lightning as pl
from torch.utils.data import DataLoader, IterableDataset, get_worker_info
from e3nn import o3
from tqdm import tqdm


from .dataset_functions import pre_process_audio_visage

class InfiniteRAMDataset(IterableDataset):
    def __init__(self, audio_tensors, shuffle=True):
        super().__init__()
        self.audio_tensors = audio_tensors
        self.shuffle = shuffle

    def __iter__(self):
        # Handle multi-processing seeding
        worker_info = get_worker_info()
        seed = torch.initial_seed() if worker_info is None else torch.initial_seed() + worker_info.id
        random.seed(seed)
        torch.manual_seed(seed)
        
        indices = list(range(len(self.audio_tensors)))
        
        while True:
            if self.shuffle:
                random.shuffle(indices)
            
            for idx in indices:
                audio = self.audio_tensors[idx]
                yield audio

class ViSageDataModuleRAM(pl.LightningDataModule):
    def __init__(
        self,
        base_data_dir: str = "/projects/0/prjs1338/visage/audios/audio/*.flac",
        batch_size: int = 32,
        sr: int = 32000,
        rotations_per_clip: int = 4,
        **kwargs,
    ):
        super().__init__()
        self.datapath = base_data_dir
        self.batch_size = batch_size
        self.sr = sr
        self.rotations_per_clip = rotations_per_clip
        self.audio_data = []

    def setup(self, stage: str = None):
        if stage == "fit" or stage is None:
            files = glob.glob(self.datapath)
            if not files:
                raise FileNotFoundError(f"No files found at {self.datapath}")
            
            print(f"Loading {len(files)} files into RAM...")
            
            # Load and pre-process everything once
            loaded_data = []
            for f in tqdm(files):
                wav, original_sr = torchaudio.load(f)
                # Ensure it's FOA (4 channels) before processing if needed
                wav = pre_process_audio_visage(wav, original_sr, self.sr)
                loaded_data.append(wav.cpu())
            
            self.audio_data = loaded_data
            print(f"Successfully loaded {len(self.audio_data)} samples into memory.")

    def train_dataloader(self):
        dataset = InfiniteRAMDataset(
            self.audio_data, 
            self.rotations_per_clip,
            shuffle=True
        )
        
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            pin_memory=True,
            num_workers=4,
            prefetch_factor=2,
            # persistent_workers is useful for infinite datasets to avoid 
            # re-initializing workers between "epochs" (if any)
            persistent_workers=True, 
        )