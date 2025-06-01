import re
from typing import List, Dict, Optional
import torch
from torch import Tensor


class SMILESTokenizer:
    def __init__(self):
        # Special tokens
        self.PAD_token = "<PAD>"
        self.START_token = "<START>"
        self.END_token = "<END>"
        self.UNK_token = "<UNK>"
        
        # Basic SMILES tokens
        self.base_tokens = [
            # Atoms
            'C', 'N', 'O', 'S', 'F', 'Si', 'Cl', 'Br', 'I', 'H', 'P', 'B',
            # Structural tokens
            '(', ')', '[', ']', '=', '#', '@', '*', '%', '0', '1', '2', '3', '4', '5',
            '6', '7', '8', '9', '.', '+', '-', '/', '\\',
            # Special tokens
            self.PAD_token, self.START_token, self.END_token, self.UNK_token
        ]
        
        # Create token to index mapping
        self.token2idx = {token: idx for idx, token in enumerate(self.base_tokens)}
        self.idx2token = {idx: token for token, idx in self.token2idx.items()}
        self.vocab_size = len(self.base_tokens)

    def tokenize(self, smiles: str) -> List[str]:
        """Convert SMILES string to tokens."""
        pattern = r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|B|Si|\.|\(|\)|\|=|#|-|\+|\\\\|\/|:|\d+|\*)"
        tokens = re.findall(pattern, smiles)
        return [self.START_token] + tokens + [self.END_token]

    def convert_tokens_to_ids(self, tokens: List[str]) -> List[int]:
        """Convert tokens to indices."""
        return [self.token2idx.get(token, self.token2idx[self.UNK_token]) for token in tokens]

    def convert_ids_to_tokens(self, ids: List[int]) -> List[str]:
        """Convert indices back to tokens."""
        return [self.idx2token[idx] for idx in ids]

    def encode(self, smiles: str, max_length: Optional[int] = None) -> Tensor:
        """Encode SMILES string to tensor of indices with optional padding."""
        tokens = self.tokenize(smiles)
        if max_length is not None:
            tokens = tokens[:max_length]
            padding = [self.PAD_token] * (max_length - len(tokens))
            tokens.extend(padding)
        ids = self.convert_tokens_to_ids(tokens)
        return torch.tensor(ids)

    def decode(self, tensor: Tensor, skip_special_tokens: bool = True) -> str:
        """Decode tensor of indices back to SMILES string."""
        ids = tensor.tolist()
        tokens = self.convert_ids_to_tokens(ids)
        if skip_special_tokens:
            tokens = [t for t in tokens if t not in {self.PAD_token, self.START_token, self.END_token, self.UNK_token}]
        return ''.join(tokens)

    def batch_encode(self, smiles_list: List[str], max_length: Optional[int] = None) -> Tensor:
        """Encode a batch of SMILES strings to a padded tensor."""
        if max_length is None:
            max_length = max(len(self.tokenize(s)) for s in smiles_list)
        tensors = [self.encode(s, max_length) for s in smiles_list]
        return torch.stack(tensors) 