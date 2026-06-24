import os
import glob
import random
import torch
import torchaudio
import pytorch_lightning as pl
from torch.utils.data import DataLoader, IterableDataset, get_worker_info
from tqdm import tqdm
from e3nn import o3


from .dataset_functions import pre_process_audio_visage

class InfiniteRAMDataset(IterableDataset):
    def __init__(self, 
                 audio_tensors, 
                 masker, 
                 nr_patches : int = 100,
                 rotations_per_clip : int = 8,
                 shuffle=True):
        super().__init__()
        self.audio_tensors = audio_tensors
        self.masker = masker
        self.shuffle = shuffle
        self.nr_patches = nr_patches
        self.rotations_per_clip = rotations_per_clip

    def __iter__(self):
        worker_info = get_worker_info()
        seed = torch.initial_seed()        # already worker-unique; see below
        rng = random.Random(seed)

        n = len(self.audio_tensors)
        while True:
            idx = rng.randrange(n)
            audio = self.audio_tensors[idx]
            R = torch.stack(
                [o3.rand_matrix() for _ in range(self.rotations_per_clip)], dim=0,
            )
            context_idx = self.masker(
            local_features=None, batch_size=1, n_patches=self.nr_patches,
            )
            yield audio, context_idx[0], R

class ViSageDataModuleMAE(pl.LightningDataModule):
    def __init__(
        self,
        masker, 
        base_data_dir: str = "/projects/0/prjs1338/visage/audios/audio/*.flac",
        rotations_per_clip: int = 8,
        nr_patches: int = 100,
        batch_size: int = 32,
        sr: int = 32000,
        **kwargs,
    ):
        super().__init__()
        self.datapath = base_data_dir
        self.batch_size = batch_size
        self.sr = sr
        self.audio_data = []
        self.masker = masker
        self.rotations_per_clip = rotations_per_clip
        self.nr_patches = nr_patches

    def setup(self, stage: str = None):
        if stage == "fit" or stage is None:
            files = glob.glob(self.datapath)
            if not files:
                raise FileNotFoundError(f"No files found at {self.datapath}")
            
            print(f"Loading {len(files)} files into RAM...")
            
            loaded_data = []
            for f in tqdm(files, desc="Loading Audio"):
                wav, original_sr = torchaudio.load(f)
                wav = pre_process_audio_visage(wav, original_sr, self.sr)
                wav = wav.cpu().share_memory_()
                loaded_data.append(wav)
            
            self.audio_data = loaded_data
            print(f"Successfully loaded {len(self.audio_data)} samples into memory.")

    def train_dataloader(self):
        dataset = InfiniteRAMDataset(
            self.audio_data, 
            shuffle=True,
            masker=self.masker,
            nr_patches=self.nr_patches,
            rotations_per_clip=self.rotations_per_clip
        )
        
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            pin_memory=True,          
            num_workers=4,
            prefetch_factor=2,
            persistent_workers=True, 
            drop_last=True 
        )