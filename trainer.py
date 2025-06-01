import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader
from typing import List, Optional, Dict
import numpy as np
from tqdm import tqdm
import os

from tnn import TNN
from tokenizer import SMILESTokenizer
from dataset import SMILESDataset


class SMILESTrainer:
    def __init__(
        self,
        model: TNN,
        tokenizer: SMILESTokenizer,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        self.criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.token2idx[tokenizer.PAD_token])
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
        self.current_epoch = 0
        self.best_loss = float('inf')

    def save_checkpoint(self, save_path: str, is_best: bool = False) -> None:
        """Save model checkpoint including optimizer state and training progress."""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_loss': self.best_loss,
            'tokenizer_state': {
                'token2idx': self.tokenizer.token2idx,
                'idx2token': self.tokenizer.idx2token,
                'vocab_size': self.tokenizer.vocab_size
            }
        }
        
        # Save regular checkpoint
        torch.save(checkpoint, save_path)
        
        # If this is the best model so far, save a separate best checkpoint
        if is_best:
            best_path = os.path.join(os.path.dirname(save_path), 'best_model.pt')
            torch.save(checkpoint, best_path)
            
    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load model checkpoint including optimizer state and training progress."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_loss = checkpoint['best_loss']
        
        # Restore tokenizer state if needed
        tokenizer_state = checkpoint['tokenizer_state']
        self.tokenizer.token2idx = tokenizer_state['token2idx']
        self.tokenizer.idx2token = tokenizer_state['idx2token']
        self.tokenizer.vocab_size = tokenizer_state['vocab_size']

    def train_epoch(self, dataloader: DataLoader) -> float:
        """Train for one epoch and return average loss."""
        self.model.train()
        total_loss = 0
        
        for batch_idx, (input_ids, target_ids) in enumerate(tqdm(dataloader)):
            input_ids = input_ids.to(self.device)
            target_ids = target_ids.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(input_ids)
            
            # Reshape for cross entropy
            loss = self.criterion(
                outputs.view(-1, outputs.size(-1)),
                target_ids.view(-1)
            )
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            
        return total_loss / len(dataloader)

    @torch.no_grad()
    def generate(
        self,
        max_length: int = 100,
        temperature: float = 0.7,
        top_k: Optional[int] = 50,
        num_samples: int = 1
    ) -> List[str]:
        """Generate SMILES strings using the model."""
        self.model.eval()
        generated = []
        
        for _ in range(num_samples):
            # Start with START token
            current_ids = torch.tensor([[self.tokenizer.token2idx[self.tokenizer.START_token]]], device=self.device)
            
            for _ in range(max_length):
                outputs = self.model(current_ids)
                next_token_logits = outputs[:, -1, :] / temperature
                
                if top_k is not None:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = float('-inf')
                
                probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                if next_token.item() == self.tokenizer.token2idx[self.tokenizer.END_token]:
                    break
                    
                current_ids = torch.cat([current_ids, next_token], dim=1)
            
            generated_smiles = self.tokenizer.decode(current_ids[0], skip_special_tokens=True)
            generated.append(generated_smiles)
        
        return generated

    def train(
        self,
        train_dataloader: DataLoader,
        num_epochs: int,
        save_dir: str = "checkpoints",
        save_frequency: int = 10,
        eval_dataloader: Optional[DataLoader] = None,
        resume_from: Optional[str] = None
    ):
        """Train the model for multiple epochs with checkpointing."""
        os.makedirs(save_dir, exist_ok=True)
        
        # Resume training if checkpoint provided
        if resume_from and os.path.exists(resume_from):
            print(f"Resuming training from checkpoint: {resume_from}")
            self.load_checkpoint(resume_from)
            
        start_epoch = self.current_epoch
        
        for epoch in range(start_epoch, num_epochs):
            self.current_epoch = epoch
            train_loss = self.train_epoch(train_dataloader)
            print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}")
            
            # Evaluation step
            if eval_dataloader is not None:
                eval_loss = self.evaluate(eval_dataloader)
                print(f"Eval Loss: {eval_loss:.4f}")
                is_best = eval_loss < self.best_loss
                if is_best:
                    self.best_loss = eval_loss
            else:
                is_best = train_loss < self.best_loss
                if is_best:
                    self.best_loss = train_loss
            
            # Regular checkpoint saving
            if (epoch + 1) % save_frequency == 0:
                checkpoint_path = os.path.join(save_dir, f"checkpoint_epoch_{epoch+1}.pt")
                self.save_checkpoint(checkpoint_path, is_best=is_best)
                print(f"Saved checkpoint at epoch {epoch+1}")
            
            # Generate a sample every few epochs
            if (epoch + 1) % 5 == 0:
                sample = self.generate(num_samples=1)[0]
                print(f"Sample SMILES: {sample}")

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> float:
        """Evaluate the model on a dataset."""
        self.model.eval()
        total_loss = 0
        
        for input_ids, target_ids in dataloader:
            input_ids = input_ids.to(self.device)
            target_ids = target_ids.to(self.device)
            
            outputs = self.model(input_ids)
            loss = self.criterion(
                outputs.view(-1, outputs.size(-1)),
                target_ids.view(-1)
            )
            total_loss += loss.item()
            
        return total_loss / len(dataloader) 