import torch
from torch.utils.data import DataLoader, random_split
import os
import argparse
from globals import Global
from tokenizer import SMILESTokenizer
from dataset import SMILESDataset
from tnn import TNN
from trainer import SMILESTrainer


def main(args):
    print(torch.cuda.is_available())

    # Initialize tokenizer
    tokenizer = SMILESTokenizer()
    
    # Load and split dataset
    dataset = SMILESDataset(Global.DATA_PATH, tokenizer, max_length=Global.MAX_LEN)
    
    # Split into train/val
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=Global.BATCH_SIZE,
        shuffle=True,
        num_workers=8
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=Global.BATCH_SIZE,
        shuffle=False,
        num_workers=8
    )
    
    # Initialize model
    model = TNN(
        n_embeddings=tokenizer.vocab_size,
        hidden_size=256,
        n_layers=6,
        nheads=8
    )
    
    # Initialize trainer
    trainer = SMILESTrainer(
        model=model,
        tokenizer=tokenizer,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    
    # Create checkpoint directory
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Train model
    trainer.train(
        train_dataloader=train_dataloader,
        eval_dataloader=val_dataloader,
        num_epochs=Global.PRETRAINING_EPOCHS,
        save_dir=args.checkpoint_dir,
        save_frequency=args.save_frequency,
        resume_from=args.resume_from
    )
    
    # Generate some example molecules
    print("\nGenerating example molecules:")
    molecules = trainer.generate(num_samples=5, temperature=0.7)
    for i, mol in enumerate(molecules, 1):
        print(f"Molecule {i}: {mol}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train SMILES generator model')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                      help='Directory to save checkpoints')
    parser.add_argument('--save_frequency', type=int, default=10,
                      help='Save checkpoint every N epochs')
    parser.add_argument('--resume_from', type=str, default=None,
                      help='Path to checkpoint to resume training from')
    
    args = parser.parse_args()
    main(args) 
