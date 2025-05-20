import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
from metrics import compute_all_metrics


def dice_coeff(pred, target, epsilon=1e-6):
    """
    Compute the Dice coefficient between predictions and targets
    
    Args:
        pred (torch.Tensor): Predicted masks (B, 1, H, W)
        target (torch.Tensor): Target masks (B, 1, H, W)
        epsilon (float): Small constant to avoid division by zero
        
    Returns:
        float: Dice coefficient
    """
    # Flatten predictions and targets
    pred = pred.view(-1)
    target = target.view(-1)
    
    # Compute intersection and union
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    
    # Compute Dice coefficient
    dice = (2. * intersection + epsilon) / (union + epsilon)
    
    return dice


def dice_loss(pred, target, epsilon=1e-6):
    """
    Compute the Dice loss (1 - Dice coefficient)
    
    Args:
        pred (torch.Tensor): Predicted masks (B, 1, H, W)
        target (torch.Tensor): Target masks (B, 1, H, W)
        epsilon (float): Small constant to avoid division by zero
        
    Returns:
        torch.Tensor: Dice loss
    """
    return 1 - dice_coeff(pred, target, epsilon)


def save_predictions(model, dataloader, device, output_dir, max_samples=8):
    """
    Generate and save prediction visualizations
    
    Args:
        model (torch.nn.Module): Trained model
        dataloader (torch.utils.data.DataLoader): DataLoader for validation/test data
        device (torch.device): Device to run the model on
        output_dir (str): Directory to save visualizations
        max_samples (int): Maximum number of samples to visualize
    """
    os.makedirs(output_dir, exist_ok=True)
    
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= max_samples:
                break
                
            # Move data to device
            inputs = batch['input'].to(device)
            targets = batch['target'].to(device)
            filenames = batch['filename']
            
            # Forward pass
            outputs = model(inputs)
            
            # Convert tensors to numpy arrays
            inputs_np = inputs.cpu().numpy()
            targets_np = targets.cpu().numpy()
            outputs_np = outputs.cpu().numpy()
            
            # Create visualizations for each sample in the batch
            for j in range(inputs.size(0)):
                if i * dataloader.batch_size + j >= max_samples:
                    break
                    
                # Create figure with three subplots
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                
                # Plot input image
                axes[0].imshow(inputs_np[j, 0], cmap='gray')
                axes[0].set_title('Input')
                axes[0].axis('off')
                
                # Plot ground truth
                axes[1].imshow(targets_np[j, 0], cmap='gray')
                axes[1].set_title('Ground Truth')
                axes[1].axis('off')
                
                # Plot prediction
                axes[2].imshow(outputs_np[j, 0], cmap='gray')
                axes[2].set_title('Prediction')
                axes[2].axis('off')
                
                # Save figure
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f'sample_{i * dataloader.batch_size + j}.png'))
                plt.close(fig)
                
                # Create node valence visualization
                visualize_node_valence(
                    outputs_np[j, 0], 
                    targets_np[j, 0], 
                    os.path.join(output_dir, f'nodes_{i * dataloader.batch_size + j}.png')
                )
                
    print(f"Saved {min(max_samples, len(dataloader) * dataloader.batch_size)} sample visualizations to {output_dir}")


def calculate_test_loss(model, inputs, targets, device):
    """
    Calculate the combined loss used in training
    
    Args:
        model (torch.nn.Module): The model
        inputs (torch.Tensor): Input tensor
        targets (torch.Tensor): Target tensor
        device (torch.device): Device to run on
        
    Returns:
        float: Loss value
    """
    criterion = nn.BCELoss()
    model.eval()
    with torch.no_grad():
        outputs = model(inputs)
        bce_loss = criterion(outputs, targets)
        dl_loss = dice_loss(outputs, targets)
        loss = bce_loss + dl_loss
    return loss.item()


def threshold_predictions(pred, threshold=0.5):
    """
    Apply threshold to convert probability maps to binary masks
    
    Args:
        pred (numpy.ndarray): Prediction probability map
        threshold (float): Threshold value
        
    Returns:
        numpy.ndarray: Binary mask
    """
    return (pred > threshold).astype(np.uint8)


def evaluate_model(model, dataloader, device):
    """
    Evaluate model performance on a dataset with extensive metrics
    
    Args:
        model (torch.nn.Module): Trained model
        dataloader (torch.utils.data.DataLoader): DataLoader for validation/test data
        device (torch.device): Device to run the model on
        
    Returns:
        dict: Dictionary with evaluation metrics
    """
    model.eval()
    metrics_sum = {}
    batch_count = 0
    test_loss_sum = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            # Move data to device
            inputs = batch['input'].to(device)
            targets = batch['target'].to(device)
            
            # Forward pass
            outputs = model(inputs)
            
            # Calculate test loss
            test_loss = calculate_test_loss(model, inputs, targets, device)
            test_loss_sum += test_loss
            
            # Process each sample in the batch
            for j in range(inputs.size(0)):
                # Get individual tensors and convert to numpy
                output = outputs[j].cpu().numpy()
                target = targets[j].cpu().numpy()
                
                # Compute all metrics
                sample_metrics = compute_all_metrics(output, target)
                
                # Accumulate metrics
                for key, value in sample_metrics.items():
                    if key not in metrics_sum:
                        metrics_sum[key] = 0.0
                    metrics_sum[key] += value
            
            batch_count += inputs.size(0)
    
    # Calculate averages
    results = {key: value / batch_count for key, value in metrics_sum.items()}
    results['test_loss'] = test_loss_sum / len(dataloader)
    
    return results


def visualize_node_valence(pred_probs, target, save_path):
    """
    Visualize node valence in predictions and targets
    
    Args:
        pred_probs (np.ndarray): Prediction probabilities
        target (np.ndarray): Ground truth binary mask
        save_path (str): Path to save visualization
    """
    # Create binary predictions
    pred_binary = threshold_predictions(pred_probs)
    
    # Create RGB images for visualization (initially black)
    pred_viz = np.zeros((pred_binary.shape[0], pred_binary.shape[1], 3), dtype=np.uint8)
    gt_viz = np.zeros((target.shape[0], target.shape[1], 3), dtype=np.uint8)
    
    # Compute node valence
    from metrics import compute_node_valence
    pred_valence = compute_node_valence(pred_binary)
    target_valence = compute_node_valence(target)
    
    # Color scheme for valence:
    # 0: White (isolated points)
    # 1: Blue (endpoints)
    # 2: Green (regular road segments)
    # 3: Yellow (3-way junctions)
    # 4+: Red (complex junctions)
    colors = {
        0: [255, 255, 255], # White
        1: [0, 0, 255],     # Blue
        2: [0, 255, 0],     # Green
        3: [255, 255, 0],   # Yellow
        4: [255, 0, 0]      # Red
    }
    
    # Color predicted nodes
    for (r, c), valence in pred_valence.items():
        color_idx = min(valence, 4)  # Cap at 4
        pred_viz[r, c] = colors[color_idx]
    
    # Color ground truth nodes
    for (r, c), valence in target_valence.items():
        color_idx = min(valence, 4)  # Cap at 4
        gt_viz[r, c] = colors[color_idx]
    
    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    axes[0].imshow(gt_viz)
    axes[0].set_title('Ground Truth Nodes')
    axes[0].axis('off')
    
    axes[1].imshow(pred_viz)
    axes[1].set_title('Predicted Nodes')
    axes[1].axis('off')
    
    # Add legend
    patches = []
    labels = ['Isolated points (0)', 'Endpoints (1)', 'Road segments (2)', '3-way junctions (3)', 'Complex junctions (4+)']
    colors_list = [[1, 1, 1], [0, 0, 1], [0, 1, 0], [1, 1, 0], [1, 0, 0]]  # Normalized colors for matplotlib
    
    for color, label in zip(colors_list, labels):
        patches.append(plt.Line2D([0], [0], color=color, marker='o', linestyle='', markersize=10))
    
    fig.legend(patches, labels, loc='lower center', ncol=4)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)


def save_metrics_report(metrics, save_path):
    """
    Save evaluation metrics as a formatted report
    
    Args:
        metrics (dict): Dictionary of evaluation metrics
        save_path (str): Path to save the report
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Format metrics into a report
    report = "# Road Network Skeletonization Evaluation Report\n\n"
    
    report += "## Overall Metrics\n\n"
    report += f"- Test Loss: {metrics['test_loss']:.4f}\n"
    report += f"- Dice Coefficient: {metrics['dice']:.4f}\n"
    report += f"- IoU Score: {metrics['iou']:.4f}\n"
    report += f"- MSE with Distance Transform: {metrics['mse_dt']:.4f}\n\n"
    
    report += "## Node-based Metrics\n\n"
    report += "| Valence | Precision | Recall | F1 Score |\n"
    report += "|---------|-----------|--------|----------|\n"
    
    for v in range(1, 5):
        report += f"| {v} | {metrics[f'valence_{v}_precision']:.4f} | {metrics[f'valence_{v}_recall']:.4f} | {metrics[f'valence_{v}_f1']:.4f} |\n"
    
    report += f"\n- Overall Node Precision: {metrics['node_precision']:.4f}\n"
    report += f"- Overall Node Recall: {metrics['node_recall']:.4f}\n"
    report += f"- Overall Node F1 Score: {metrics['node_f1']:.4f}\n"
    
    # Save as markdown report
    with open(save_path, 'w') as f:
        f.write(report)
    
    # Also save as JSON for programmatic access
    json_path = save_path.replace('.md', '.json')
    with open(json_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    
    print(f"Saved metrics report to {save_path} and {json_path}")
