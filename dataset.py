import torch
from torch.utils.data import Dataset
import os
from typing import Optional, Tuple
from tokenizer import SMILESTokenizer
from globals import Global


class SMILESDataset(Dataset):
    def __init__(self, path: str, tokenizer: SMILESTokenizer, max_length: Optional[int] = None):
        self.path = path
        self.tokenizer = tokenizer
        self.max_length = max_length or Global.MAX_LEN
        self.sequences = []
        self.tensors = []
        
        self.__fetch_sequences()
        self.__process_sequences()

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Return input and target tensors for training
        tensor = self.tensors[idx]
        # Input is all tokens except last, target is all tokens except first
        return tensor[:-1], tensor[1:]

    def __len__(self) -> int:
        return len(self.tensors)

    def __fetch_sequences(self):
        """Load SMILES strings from files."""
        for fp in os.listdir(self.path):
            with open(os.path.join(self.path, fp)) as f:
                for line in f.readlines():
                    smiles = line.strip()
                    if 3 <= len(smiles) <= self.max_length:
                        self.sequences.append(smiles)

    def __process_sequences(self):
        """Convert SMILES strings to tensors."""
        for smiles in self.sequences:
            tensor = self.tokenizer.encode(smiles, self.max_length)
            self.tensors.append(tensor)

    def get_vocab_size(self) -> int:
        """Return vocabulary size from tokenizer."""
        return self.tokenizer.vocab_size