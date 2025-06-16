import torch
from torch.utils.data import Dataset
import os
from typing import Optional, Tuple, List, Dict
from tokenizer import SMILESTokenizer
from globals import Global


class SMILESDataset(Dataset):
    def __init__(self, path: str, tokenizer: SMILESTokenizer, max_length: Optional[int] = None):
        self.path = path
        self.tokenizer = tokenizer
        self.max_length = max_length or Global.MAX_LEN
        self.file_paths = []
        self.sequence_offsets = []  # List of (file_idx, start_idx, end_idx) tuples
        self.total_sequences = 0
        self.sequence_cache = {}  # Cache for file sequences
        
        self.__index_files()
        print(f"Dataset initialized with {self.total_sequences} total sequences across {len(self.file_paths)} files")

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if idx < 0 or idx >= self.total_sequences:
            raise IndexError(f"Index {idx} out of range [0, {self.total_sequences})")
            
        # Find which file contains this index
        file_idx = 0
        while file_idx < len(self.sequence_offsets) and idx >= self.sequence_offsets[file_idx][2]:
            file_idx += 1
            
        if file_idx >= len(self.sequence_offsets):
            raise IndexError(f"Index {idx} not found in any file")
            
        # Get file info
        file_path = self.file_paths[file_idx]
        start_idx = self.sequence_offsets[file_idx][1]
        target_idx = idx - start_idx
        
        # Get sequences for this file (from cache or load)
        if file_path not in self.sequence_cache:
            self.sequence_cache[file_path] = self.__load_file_sequences(file_path)
        
        sequences = self.sequence_cache[file_path]
        if target_idx >= len(sequences):
            raise IndexError(f"Target index {target_idx} out of range for file {file_path} (has {len(sequences)} sequences)")
            
        smiles = sequences[target_idx]
        tensor = self.tokenizer.encode(smiles, self.max_length)
        return tensor[:-1], tensor[1:]

    def __len__(self) -> int:
        return self.total_sequences

    def __load_file_sequences(self, file_path: str) -> List[str]:
        """Load all valid sequences from a file."""
        sequences = []
        with open(file_path, 'r') as f:
            for line in f:
                smiles = line.strip()
                if 3 <= len(smiles) <= self.max_length:
                    sequences.append(smiles)
        return sequences

    def __index_files(self):
        """Index all files and their sequences without loading them into memory."""
        current_idx = 0
        for fp in os.listdir(self.path):
            file_path = os.path.join(self.path, fp)
            if not os.path.isfile(file_path):
                continue
                
            # Load and count sequences in this file
            sequences = self.__load_file_sequences(file_path)
            valid_sequences = len(sequences)
                        
            if valid_sequences > 0:
                self.file_paths.append(file_path)
                self.sequence_offsets.append((len(self.file_paths) - 1, current_idx, current_idx + valid_sequences))
                current_idx += valid_sequences
                self.total_sequences += valid_sequences
                print(f"Indexed file {fp}: {valid_sequences} sequences (total: {self.total_sequences})")

    def get_vocab_size(self) -> int:
        """Return vocabulary size from tokenizer."""
        return self.tokenizer.vocab_size