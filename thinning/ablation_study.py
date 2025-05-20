"""
Ablation study for road skeletonization model
This script runs multiple model configurations with abbreviated training
to identify the impact of different parameters on performance.
"""
import os
import argparse
import json
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset import get_dataloaders
from utils import dice_coeff, dice_loss, save_metrics_report
from evaluation import evaluate
from metrics import compute_all_metrics

# Different model configurations for ablation study
CONFIGURATIONS = {
    # Vary learning rate
    "lr_0.0001": {"learning_rate": 0.0001, "loss": "combined", "architecture": "standard"},
    "lr_0.001": {"learning_rate": 0.001, "loss": "combined", "architecture": "standard"},
    "lr_0.01": {"learning_rate": 0.01, "loss": "combined", "architecture": "standard"},

    # Vary loss function
    "loss_bce": {"learning_rate": 0.01, "loss": "bce", "architecture": "standard"},
    "loss_dice": {"learning_rate": 0.01, "loss": "dice", "architecture": "standard"},
    "loss_combined": {"learning_rate": 0.01, "loss": "combined", "architecture": "standard"},
    "loss_weighted": {"learning_rate": 0.01, "loss": "weighted", "architecture": "standard"},

    # Vary architecture
    # "arch_small": {"learning_rate": 0.001, "loss": "combined", "architecture": "small"},
    # "arch_standard": {"learning_rate": 0.001, "loss": "combined", "architecture": "standard"},
    # "arch_large": {"learning_rate": 0.001, "loss": "combined", "architecture": "large"},
    # "arch_bilinear": {"learning_rate": 0.001, "loss": "combined", "architecture": "bilinear"},
    # "arch_transpose": {"learning_rate": 0.001, "loss": "combined", "architecture": "transpose"},
}


class DoubleConv(nn.Module):
    """Double convolution block for U-Net"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling with either bilinear upsampling or transpose convolution"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # Handle size differences
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = torch.nn.functional.pad(x1, [diffX // 2, diffX - diffX // 2,
                                          diffY // 2, diffY - diffY // 2])

        # Concatenate along channel dimension
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """Final convolution layer"""

    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNetModel(nn.Module):
    """U-Net architecture with configurable parameters for ablation study"""

    def __init__(self, n_channels=1, n_classes=1, architecture="standard", bilinear_override=None):
        super(UNetModel, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.architecture = architecture
    
        # Set architecture parameters
        if architecture == "small":
            # Smaller model with fewer channels
            channels = [32, 64, 128, 256, 512]
            bilinear = True
        elif architecture == "standard":
            # Standard model
            channels = [64, 128, 256, 512, 1024]
            bilinear = True
        elif architecture == "large":
            # Larger model with more channels
            channels = [64, 128, 256, 512, 1024, 2048]
            bilinear = True
        elif architecture == "bilinear":
            # Standard model with bilinear upsampling
            channels = [64, 128, 256, 512, 1024]
            bilinear = True
        elif architecture == "transpose":
            # Standard model with transpose convolution upsampling
            channels = [64, 128, 256, 512, 1024]
            bilinear = False
        else:
            raise ValueError(f"Unknown architecture: {architecture}")
        
        # Allow bilinear mode to be overridden (for compatibility)
        if bilinear_override is not None:
            bilinear = bilinear_override
            print(f"Overriding bilinear mode to: {bilinear}")

        self.bilinear = bilinear
        self.channels = channels

        # Initialize layers based on the architecture configuration

        # First layer (input -> first hidden)
        self.inc = DoubleConv(n_channels, channels[0])

        # Downsampling path
        self.down_layers = nn.ModuleList()
        for i in range(len(channels) - 1):
            self.down_layers.append(Down(channels[i], channels[i + 1]))

        # Upsampling path
        self.up_layers = nn.ModuleList()
        for i in range(len(channels) - 1, 0, -1):
            self.up_layers.append(Up(channels[i] + channels[i - 1], channels[i - 1], bilinear))

        # Final convolution and activation
        self.outc = OutConv(channels[0], n_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Store intermediate results for skip connections
        features = [self.inc(x)]

        # Contracting path
        for i, down in enumerate(self.down_layers):
            features.append(down(features[-1]))

        # Expanding path with skip connections
        x = features[-1]
        for i, up in enumerate(self.up_layers):
            skip_idx = len(features) - 2 - i
            x = up(x, features[skip_idx])

        # Final convolution and activation
        logits = self.outc(x)
        return self.sigmoid(logits)


def get_model(architecture="standard", pretrained_path=None):
    """
    Get a model with the specified architecture configuration

    Args:
        architecture (str): Architecture configuration ('small', 'standard', 'large', etc.)
        pretrained_path (str, optional): Path to pretrained model weights

    Returns:
        nn.Module: Configured model
    """
    model = UNetModel(n_channels=1, n_classes=1, architecture=architecture)

    if pretrained_path and os.path.exists(pretrained_path):
        model.load_state_dict(torch.load(pretrained_path))
        print(f"Loaded pretrained model from {pretrained_path}")

    return model


def get_loss_function(loss_type="combined"):
    """
    Get the loss function based on configuration

    Args:
        loss_type (str): Type of loss function ('bce', 'dice', 'combined', 'weighted')

    Returns:
        callable: Loss function that takes predictions and targets
    """
    bce_loss = nn.BCELoss()

    if loss_type == "bce":
        return lambda pred, target: bce_loss(pred, target)
    elif loss_type == "dice":
        return lambda pred, target: dice_loss(pred, target)
    elif loss_type == "combined":
        return lambda pred, target: bce_loss(pred, target) + dice_loss(pred, target)
    elif loss_type == "weighted":
        # BCE with higher weight (2:1 ratio)
        return lambda pred, target: 2 * bce_loss(pred, target) + dice_loss(pred, target)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


def train_and_evaluate(config_name, config, args):
    """
    Train and evaluate a model with the specified configuration

    Args:
        config_name (str): Name of the configuration
        config (dict): Configuration parameters
        args (Namespace): Command-line arguments

    Returns:
        dict: Results of the training and evaluation
    """
    print(f"\n=== Running configuration: {config_name} ===")
    print(f"Parameters: {config}")

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create directories for this configuration
    config_dir = os.path.join(args.output_dir, config_name)
    checkpoint_dir = os.path.join(config_dir, 'checkpoints')
    log_dir = os.path.join(config_dir, 'logs')
    eval_dir = os.path.join(config_dir, 'evaluation')

    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(eval_dir, exist_ok=True)

    # Get dataloaders
    dataloaders = get_dataloaders(
        args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    # Get model and send to device
    if hasattr(args, 'compatibility_mode') and args.compatibility_mode:
        print("Using compatibility mode for model architecture")
        # Use UNetModel from this module directly for consistency
        model = UNetModel(n_channels=1, n_classes=1, architecture=config["architecture"])
    else:
        model = get_model(architecture=config["architecture"])
    model = model.to(device)

    # Get loss function
    criterion = get_loss_function(loss_type=config["loss"])

    # Set optimizer
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])

    # Set learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', patience=2, factor=0.5
    )

    # TensorBoard writer
    writer = SummaryWriter(log_dir)

    # Track best validation metrics
    best_val_loss = float('inf')
    best_val_dice = 0.0

    # Training loop
    start_time = time.time()
    train_losses = []
    val_losses = []
    train_dices = []
    val_dices = []

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")

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

            # Calculate loss
            loss = criterion(outputs, targets)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # Update metrics
            train_loss += loss.item() * inputs.size(0)
            train_dice += dice_coeff(outputs, targets).item() * inputs.size(0)

            # Update progress bar
            train_loop.set_postfix(loss=loss.item(), dice=dice_coeff(outputs, targets).item())

        # Calculate average training metrics
        train_loss /= len(dataloaders['train'].dataset)
        train_dice /= len(dataloaders['train'].dataset)

        # Log training metrics
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Dice/train', train_dice, epoch)
        train_losses.append(train_loss)
        train_dices.append(train_dice)

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

                # Calculate loss
                loss = criterion(outputs, targets)

                # Update metrics
                val_loss += loss.item() * inputs.size(0)
                val_dice += dice_coeff(outputs, targets).item() * inputs.size(0)

                # Update progress bar
                val_loop.set_postfix(loss=loss.item(), dice=dice_coeff(outputs, targets).item())

        # Calculate average validation metrics
        val_loss /= len(dataloaders['val'].dataset)
        val_dice /= len(dataloaders['val'].dataset)

        # Log validation metrics
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Dice/val', val_dice, epoch)
        val_losses.append(val_loss)
        val_dices.append(val_dice)

        # Update learning rate scheduler
        scheduler.step(val_loss)

        # Save best model
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'best_model.pth'))
            print(f"Saved new best model with validation dice: {val_dice:.4f}")

        # Print epoch summary
        print(f"Epoch {epoch + 1} - Train Loss: {train_loss:.4f}, Train Dice: {train_dice:.4f}")
        print(f"Epoch {epoch + 1} - Val Loss: {val_loss:.4f}, Val Dice: {val_dice:.4f}")

    # Calculate training time
    training_time = time.time() - start_time

    # Close TensorBoard writer
    writer.close()

    # Save training curves
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss Curves')

    plt.subplot(1, 2, 2)
    plt.plot(train_dices, label='Train Dice')
    plt.plot(val_dices, label='Val Dice')
    plt.xlabel('Epoch')
    plt.ylabel('Dice Coefficient')
    plt.legend()
    plt.title('Dice Curves')

    plt.tight_layout()
    plt.savefig(os.path.join(config_dir, 'training_curves.png'))
    plt.close()

    # Load best model for evaluation
    model.load_state_dict(torch.load(os.path.join(checkpoint_dir, 'best_model.pth')))

    # Create mock args for evaluation
    eval_args = argparse.Namespace(
        data_dir=args.data_dir,
        model_path=os.path.join(checkpoint_dir, 'best_model.pth'),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        output_dir=eval_dir,
        save_vis=True,
        vis_samples=min(10, args.batch_size),
        threshold=0.5,
        threshold_analysis=False,
        compatibility_mode=hasattr(args, 'compatibility_mode') and args.compatibility_mode
    )

    # Evaluate model on test set
    eval_results = evaluate(eval_args)

    # Compile results
    results = {
        "config_name": config_name,
        "parameters": config,
        "training_time": training_time,
        "epochs": args.epochs,
        "best_val_loss": best_val_loss,
        "best_val_dice": best_val_dice,
        "test_metrics": eval_results
    }

    # Save results
    with open(os.path.join(config_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=4)

    return results


def run_ablation_study(args):
    """
    Run ablation study with all configurations

    Args:
        args (Namespace): Command-line arguments

    Returns:
        list: Results for all configurations
    """
    all_results = []

    # Run each configuration
    for config_name, config in CONFIGURATIONS.items():
        if args.configs and config_name not in args.configs.split(','):
            print(f"Skipping configuration: {config_name}")
            continue

        results = train_and_evaluate(config_name, config, args)
        all_results.append(results)

    # Compile and save comparative results
    comparison_table = []
    for result in all_results:
        row = {
            "Configuration": result["config_name"],
            "Learning Rate": result["parameters"]["learning_rate"],
            "Loss Function": result["parameters"]["loss"],
            "Architecture": result["parameters"]["architecture"],
            "Training Time (s)": round(result["training_time"], 2),
            "Best Val Loss": round(result["best_val_loss"], 4),
            "Best Val Dice": round(result["best_val_dice"], 4),
            "Test Loss": round(result["test_metrics"]["test_loss"], 4),
            "Test Dice": round(result["test_metrics"]["dice"], 4),
            "Test IoU": round(result["test_metrics"]["iou"], 4),
            "MSE (Dist. Transform)": round(result["test_metrics"]["mse_dt"], 4),
            "Node Precision": round(result["test_metrics"]["node_precision"], 4),
            "Node Recall": round(result["test_metrics"]["node_recall"], 4),
            "Node F1": round(result["test_metrics"]["node_f1"], 4)
        }
        comparison_table.append(row)

    # Create comparative results directory
    comparative_dir = os.path.join(args.output_dir, 'comparative_results')
    os.makedirs(comparative_dir, exist_ok=True)

    # Save as CSV
    df = pd.DataFrame(comparison_table)
    df.to_csv(os.path.join(comparative_dir, 'ablation_results.csv'), index=False)

    # Save as Markdown
    markdown_table = df.to_markdown(index=False)
    with open(os.path.join(comparative_dir, 'ablation_results.md'), 'w') as f:
        f.write("# Ablation Study Results\n\n")
        f.write(markdown_table)

    # Generate comparative visualizations
    create_comparative_visualizations(comparison_table, comparative_dir)

    return all_results


def create_comparative_visualizations(comparison_table, output_dir):
    """
    Create visualizations comparing different configurations

    Args:
        comparison_table (list): List of dictionaries with results
        output_dir (str): Directory to save visualizations
    """
    df = pd.DataFrame(comparison_table)

    # 1. Learning rate comparison
    lr_df = df[df['Configuration'].str.startswith('lr_')].copy()
    if not lr_df.empty:
        plt.figure(figsize=(12, 8))
        metrics = ['Test Dice', 'Test IoU', 'Node F1', 'MSE (Dist. Transform)']

        for i, metric in enumerate(metrics):
            plt.subplot(2, 2, i + 1)
            plt.bar(lr_df['Learning Rate'].astype(str), lr_df[metric], color='skyblue')
            plt.xlabel('Learning Rate')
            plt.ylabel(metric)
            plt.title(f'Effect of Learning Rate on {metric}')

            # Add values on top of bars
            for j, v in enumerate(lr_df[metric]):
                plt.text(j, v + 0.01, f'{v:.4f}', ha='center')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'learning_rate_comparison.png'))
        plt.close()

    # 2. Loss function comparison
    loss_df = df[df['Configuration'].str.startswith('loss_')].copy()
    if not loss_df.empty:
        plt.figure(figsize=(12, 8))
        metrics = ['Test Dice', 'Test IoU', 'Node F1', 'MSE (Dist. Transform)']

        for i, metric in enumerate(metrics):
            plt.subplot(2, 2, i + 1)
            plt.bar(loss_df['Loss Function'], loss_df[metric], color='lightcoral')
            plt.xlabel('Loss Function')
            plt.ylabel(metric)
            plt.title(f'Effect of Loss Function on {metric}')

            # Add values on top of bars
            for j, v in enumerate(loss_df[metric]):
                plt.text(j, v + 0.01, f'{v:.4f}', ha='center')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'loss_function_comparison.png'))
        plt.close()

    # 3. Architecture comparison
    arch_df = df[df['Configuration'].str.startswith('arch_')].copy()
    if not arch_df.empty:
        plt.figure(figsize=(12, 8))
        metrics = ['Test Dice', 'Test IoU', 'Node F1', 'MSE (Dist. Transform)']

        for i, metric in enumerate(metrics):
            plt.subplot(2, 2, i + 1)
            plt.bar(arch_df['Architecture'], arch_df[metric], color='mediumseagreen')
            plt.xlabel('Architecture')
            plt.ylabel(metric)
            plt.title(f'Effect of Architecture on {metric}')

            # Add values on top of bars
            for j, v in enumerate(arch_df[metric]):
                plt.text(j, v + 0.01, f'{v:.4f}', ha='center')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'architecture_comparison.png'))
        plt.close()

    # 4. Radar chart for overall comparison
    plt.figure(figsize=(10, 8))

    # Select top configurations based on Dice score
    top_configs = df.sort_values('Test Dice', ascending=False).head(5)

    # Normalize metrics for radar chart
    metrics = ['Test Dice', 'Test IoU', 'Node Precision', 'Node Recall']
    normalized_df = top_configs.copy()

    for metric in metrics:
        min_val = normalized_df[metric].min()
        max_val = normalized_df[metric].max()
        if max_val > min_val:
            normalized_df[metric] = (normalized_df[metric] - min_val) / (max_val - min_val)
        else:
            normalized_df[metric] = 1.0

    # Set up radar chart
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # Close the loop

    ax = plt.subplot(111, polar=True)

    for i, row in normalized_df.iterrows():
        values = row[metrics].tolist()
        values += values[:1]  # Close the loop
        ax.plot(angles, values, linewidth=2, label=row['Configuration'])
        ax.fill(angles, values, alpha=0.1)

    # Set ticks and labels
    plt.xticks(angles[:-1], metrics)
    plt.yticks([0.25, 0.5, 0.75, 1.0], ["0.25", "0.5", "0.75", "1.0"], color="grey", size=8)
    plt.ylim(0, 1)

    # Add legend
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    plt.title("Top 5 Configurations Comparison")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'top_configurations_radar.png'))
    plt.close()

    # 5. Training time comparison
    plt.figure(figsize=(12, 6))

    # Sort by training time
    time_df = df.sort_values('Training Time (s)')

    plt.barh(time_df['Configuration'], time_df['Training Time (s)'], color='plum')
    plt.xlabel('Training Time (seconds)')
    plt.ylabel('Configuration')
    plt.title('Training Time Comparison')

    # Add values at the end of bars
    for i, v in enumerate(time_df['Training Time (s)']):
        plt.text(v + 5, i, f'{v:.1f}s', va='center')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_time_comparison.png'))
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ablation study for road skeletonization model")

    # Data parameters
    parser.add_argument('--data_dir', type=str, default='../data/thinning',
                        help='Directory containing the dataset')

    # Training parameters
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for training and evaluation')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of epochs to train (abbreviated for ablation study)')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of worker threads for data loading')

    # Output parameters
    parser.add_argument('--output_dir', type=str, default='../ablation_study',
                        help='Directory to save ablation study results')

    # Configuration selection
    parser.add_argument('--configs', type=str, default=None,
                        help='Comma-separated list of configurations to run (default: all)')

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Run ablation study
    print("=== Starting Ablation Study ===")
    run_ablation_study(args)
    print("=== Ablation Study Complete ===")
