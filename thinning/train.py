import os
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset import get_dataloaders
from unet import get_model
from utils import dice_loss, dice_coeff, save_predictions


def train(args):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create directories for checkpoints and logs
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Create TensorBoard writer
    writer = SummaryWriter(args.log_dir)
    
    # Get dataloaders
    dataloaders = get_dataloaders(
        args.data_dir, 
        batch_size=args.batch_size, 
        num_workers=args.num_workers
    )
    
    # Get model
    model = get_model(args.pretrained)
    model = model.to(device)
    
    # Optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', patience=5, factor=0.5
    )
    
    # Binary cross entropy loss
    criterion = nn.BCELoss()
    
    # Track best validation loss
    best_val_loss = float('inf')
    
    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")
        
        # Training phase
        model.train()
        train_loss = 0.0
        train_dice = 0.0
        
        train_loop = tqdm(dataloaders['train'], desc="Training")
        for batch in train_loop:
            # Move data to device
            inputs = batch['input'].to(device)
            targets = batch['target'].to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            
            # Calculate losses
            bce_loss = criterion(outputs, targets)
            dl_loss = dice_loss(outputs, targets)
            loss = bce_loss + dl_loss
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Update metrics
            train_loss += loss.item() * inputs.size(0)
            train_dice += (1 - dl_loss.item()) * inputs.size(0)
            
            # Update progress bar
            train_loop.set_postfix(loss=loss.item(), dice=1-dl_loss.item())
        
        # Calculate average training metrics
        train_loss /= len(dataloaders['train'].dataset)
        train_dice /= len(dataloaders['train'].dataset)
        
        # Log training metrics
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Dice/train', train_dice, epoch)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_dice = 0.0
        
        with torch.no_grad():
            val_loop = tqdm(dataloaders['val'], desc="Validation")
            for batch in val_loop:
                # Move data to device
                inputs = batch['input'].to(device)
                targets = batch['target'].to(device)
                
                # Forward pass
                outputs = model(inputs)
                
                # Calculate losses
                bce_loss = criterion(outputs, targets)
                dl_loss = dice_loss(outputs, targets)
                loss = bce_loss + dl_loss
                
                # Update metrics
                val_loss += loss.item() * inputs.size(0)
                val_dice += (1 - dl_loss.item()) * inputs.size(0)
                
                # Update progress bar
                val_loop.set_postfix(loss=loss.item(), dice=1-dl_loss.item())
        
        # Calculate average validation metrics
        val_loss /= len(dataloaders['val'].dataset)
        val_dice /= len(dataloaders['val'].dataset)
        
        # Log validation metrics
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Dice/val', val_dice, epoch)
        
        # Update learning rate scheduler
        scheduler.step(val_loss)
        
        # Save model if validation loss improved
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 
                       os.path.join(args.checkpoint_dir, 'best_model.pth'))
            print(f"Saved new best model with validation loss: {val_loss:.4f}")
        
        # Save checkpoint every few epochs
        if (epoch + 1) % args.save_every == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': val_loss,
            }, os.path.join(args.checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pth'))
        
        # Save some validation predictions for visualization
        if (epoch + 1) % args.vis_every == 0:
            save_predictions(
                model, dataloaders['val'], device, 
                os.path.join(args.vis_dir, f'epoch_{epoch+1}'), 
                max_samples=8
            )
        
        # Print epoch summary
        print(f"Epoch {epoch+1} - Train Loss: {train_loss:.4f}, "
              f"Train Dice: {train_dice:.4f}, Val Loss: {val_loss:.4f}, "
              f"Val Dice: {val_dice:.4f}")
    
    # Save final model
    torch.save(model.state_dict(), 
               os.path.join(args.checkpoint_dir, 'final_model.pth'))
    
    # Close TensorBoard writer
    writer.close()
    
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train U-Net for road skeletonization")
    
    # Data parameters
    parser.add_argument('--data_dir', type=str, default='../data/thinning',
                        help='Directory containing the dataset')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs to train')
    parser.add_argument('--learning_rate', type=float, default=0.01,
                        help='Learning rate')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of worker threads for data loading')
    
    # Model parameters
    parser.add_argument('--pretrained', type=str, default=None,
                        help='Path to pretrained model weights')
    
    # Output parameters
    parser.add_argument('--checkpoint_dir', type=str, default='../checkpoints',
                        help='Directory to save model checkpoints')
    parser.add_argument('--log_dir', type=str, default='../logs',
                        help='Directory to save TensorBoard logs')
    parser.add_argument('--vis_dir', type=str, default='../visualizations',
                        help='Directory to save visualizations')
    parser.add_argument('--save_every', type=int, default=5,
                        help='Save checkpoint every N epochs')
    parser.add_argument('--vis_every', type=int, default=5,
                        help='Save visualizations every N epochs')
    
    args = parser.parse_args()
    
    # Create output directories
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.vis_dir, exist_ok=True)
    
    # Train model
    train(args)
