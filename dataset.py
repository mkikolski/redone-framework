import torch
from torch.utils.data import Dataset
import os
from typing import Optional, Tuple, List
from tokenizer import SMILESTokenizer
from globals import Global


class SMILESDataset(Dataset):
    def __init__(self, path: str, tokenizer: SMILESTokenizer, max_length: Optional[int] = None):
        self.path = path
        self.tokenizer = tokenizer
        self.max_length = max_length or Global.MAX_LEN
        self.file_paths = []
        self.sequence_offsets = []  # Store file offsets for each sequence
        self.total_sequences = 0
        
        self.__index_files()

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Find which file contains this index
        file_idx = 0
        while file_idx < len(self.sequence_offsets) and idx >= self.sequence_offsets[file_idx][1]:
            file_idx += 1
        
        if file_idx >= len(self.sequence_offsets):
            raise IndexError("Index out of range")
            
        # Calculate offset within the file
        file_offset = self.sequence_offsets[file_idx][0] + (idx - self.sequence_offsets[file_idx][1])
        
        # Read the specific line
        with open(self.file_paths[file_idx], 'r') as f:
            f.seek(file_offset)
            smiles = f.readline().strip()
            
        # Process the sequence
        tensor = self.tokenizer.encode(smiles, self.max_length)
        return tensor[:-1], tensor[1:]

    def __len__(self) -> int:
        return self.total_sequences

    def __index_files(self):
        """Index all files and their sequences without loading them into memory."""
        current_offset = 0
        for fp in os.listdir(self.path):
            file_path = os.path.join(self.path, fp)
            if not os.path.isfile(file_path):
                continue
                
            # Count sequences in this file
            with open(file_path, 'r') as f:
                file_sequences = 0
                for line in f:
                    smiles = line.strip()
                    if 3 <= len(smiles) <= self.max_length:
                        file_sequences += 1
                        
            if file_sequences > 0:
                self.file_paths.append(file_path)
                self.sequence_offsets.append((current_offset, self.total_sequences))
                self.total_sequences += file_sequences
                current_offset += os.path.getsize(file_path)

    def get_vocab_size(self) -> int:
        """Return vocabulary size from tokenizer."""
        return self.tokenizer.vocab_size