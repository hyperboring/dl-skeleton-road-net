import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd

from dataset import get_dataloaders
from unet import get_model
from utils import (
    evaluate_model, save_predictions, 
    save_metrics_report, visualize_node_valence,
    threshold_predictions
)


def evaluate(args):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Get test dataloader
    dataloaders = get_dataloaders(
        args.data_dir, 
        batch_size=args.batch_size, 
        num_workers=args.num_workers
    )
    test_loader = dataloaders['test']
    
    # Get model and load weights
    if hasattr(args, 'compatibility_mode') and args.compatibility_mode:
        print("Using compatibility mode for model loading")
        try:
            # Dynamically import UNetModel from ablation_study
            import sys
            import os
            sys.path.append(os.path.dirname(os.path.abspath(__file__)))
            from ablation_study import UNetModel
            
            model = UNetModel(n_channels=1, n_classes=1)
            state_dict = torch.load(args.model_path)
            model.load_state_dict(state_dict)
            print("Successfully loaded model in compatibility mode")
        except Exception as e:
            print(f"Error in compatibility mode: {e}")
            print("Falling back to standard model loading with architecture detection")
            model = get_model(args.model_path)
    else:
        model = get_model(args.model_path)
    
    model = model.to(device)
    model.eval()  # Ensure model is in evaluation mode
    
    # Create output directories
    vis_dir = os.path.join(args.output_dir, 'visualizations')
    metrics_dir = os.path.join(args.output_dir, 'metrics')
    os.makedirs(vis_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)
    
    # Evaluate model
    print("Evaluating model with comprehensive metrics...")
    results = evaluate_model(model, test_loader, device)
    
    # Print evaluation results
    print("\n=== Evaluation Results ===")
    print(f"Test Loss: {results['test_loss']:.4f}")
    print(f"Dice Coefficient: {results['dice']:.4f}")
    print(f"IoU Score: {results['iou']:.4f}")
    print(f"MSE with Distance Transform: {results['mse_dt']:.4f}")
    print("\nNode-based Metrics:")
    
    # Create a table for node-based metrics
    node_metrics_table = []
    for v in range(0, 5):
        node_metrics_table.append({
            'Valence': v if v < 4 else '4+',
            'Precision': results[f'valence_{v}_precision'],
            'Recall': results[f'valence_{v}_recall'],
            'F1 Score': results[f'valence_{v}_f1']
        })
    
    df = pd.DataFrame(node_metrics_table)
    print(df.to_string(index=False))
    
    print(f"\nOverall Node Precision: {results['node_precision']:.4f}")
    print(f"Overall Node Recall: {results['node_recall']:.4f}")
    print(f"Overall Node F1 Score: {results['node_f1']:.4f}")
    
    # Save metrics report
    report_path = os.path.join(metrics_dir, 'evaluation_report.md')
    save_metrics_report(results, report_path)
    
    # Save visualizations if requested
    if args.save_vis:
        print(f"\nSaving visualizations to {vis_dir}...")
        save_predictions(model, test_loader, device, vis_dir, max_samples=args.vis_samples)
        
        # Create and save metrics visualization
        create_metrics_visualization(results, os.path.join(metrics_dir, 'metrics_visualization.png'))
    
    # Save threshold analysis if requested
    if args.threshold_analysis:
        print("\nPerforming threshold analysis...")
        thresholds = np.arange(0.1, 1.0, 0.1)
        threshold_results = perform_threshold_analysis(model, test_loader, device, thresholds)
        
        # Plot threshold analysis results
        plot_threshold_analysis(threshold_results, os.path.join(metrics_dir, 'threshold_analysis.png'))
    
    return results


def create_metrics_visualization(metrics, save_path):
    """Create a visualization of the evaluation metrics"""
    # Set up the figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    
    # Overall metrics bar chart
    overall_metrics = ['dice', 'iou', 'node_precision', 'node_recall', 'node_f1']
    overall_values = [metrics[metric] for metric in overall_metrics]
    overall_labels = ['Dice', 'IoU', 'Node Precision', 'Node Recall', 'Node F1']
    
    ax1.bar(overall_labels, overall_values, color='skyblue')
    ax1.set_title('Overall Metrics')
    ax1.set_ylim(0, 1.0)
    ax1.set_ylabel('Score')
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add values on top of bars
    for i, v in enumerate(overall_values):
        ax1.text(i, v + 0.02, f'{v:.3f}', ha='center')
    
    # Node-based metrics by valence
    valence_values = list(range(1, 5))
    precision_values = [metrics[f'valence_{v}_precision'] for v in valence_values]
    recall_values = [metrics[f'valence_{v}_recall'] for v in valence_values]
    f1_values = [metrics[f'valence_{v}_f1'] for v in valence_values]
    
    x = np.arange(len(valence_values))
    width = 0.25
    
    ax2.bar(x - width, precision_values, width, label='Precision', color='steelblue')
    ax2.bar(x, recall_values, width, label='Recall', color='lightcoral')
    ax2.bar(x + width, f1_values, width, label='F1 Score', color='mediumseagreen')
    
    ax2.set_title('Node Metrics by Valence')
    ax2.set_xticks(x)
    ax2.set_xticklabels(['1', '2', '3', '4+'])
    ax2.set_xlabel('Node Valence')
    ax2.set_ylabel('Score')
    ax2.set_ylim(0, 1.0)
    ax2.grid(axis='y', linestyle='--', alpha=0.7)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Metrics visualization saved to {save_path}")


def perform_threshold_analysis(model, dataloader, device, thresholds):
    """
    Analyze the effect of different thresholds on evaluation metrics
    
    Args:
        model (torch.nn.Module): Trained model
        dataloader (torch.utils.data.DataLoader): DataLoader for test data
        device (torch.device): Device to run on
        thresholds (list): List of threshold values to evaluate
        
    Returns:
        dict: Dictionary with metrics at different thresholds
    """
    from metrics import compute_all_metrics
    
    threshold_results = {
        'thresholds': thresholds,
        'dice': [],
        'iou': [],
        'node_precision': [],
        'node_recall': [],
        'node_f1': []
    }
    
    # Get a sample batch for analysis
    model.eval()
    all_outputs = []
    all_targets = []
    
    print("Collecting predictions for threshold analysis...")
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Processing batches"):
            # Move data to device
            inputs = batch['input'].to(device)
            targets = batch['target'].to(device)
            
            # Forward pass
            outputs = model(inputs)
            
            # Add batch to collections
            all_outputs.extend([out.cpu().numpy() for out in outputs])
            all_targets.extend([tgt.cpu().numpy() for tgt in targets])
            
            # Limit to 100 samples for efficiency
            if len(all_outputs) >= 100:
                break
    
    print(f"Analyzing {len(all_outputs)} samples across {len(thresholds)} thresholds...")
    for threshold in tqdm(thresholds, desc="Testing thresholds"):
        dice_scores = []
        iou_scores = []
        node_precision_scores = []
        node_recall_scores = []
        node_f1_scores = []
        
        for pred, target in zip(all_outputs, all_targets):
            # Apply threshold
            metrics = compute_all_metrics(pred, target, binary_threshold=threshold)
            
            # Collect metrics
            dice_scores.append(metrics['dice'])
            iou_scores.append(metrics['iou'])
            node_precision_scores.append(metrics['node_precision'])
            node_recall_scores.append(metrics['node_recall'])
            node_f1_scores.append(metrics['node_f1'])
        
        # Average metrics for this threshold
        threshold_results['dice'].append(np.mean(dice_scores))
        threshold_results['iou'].append(np.mean(iou_scores))
        threshold_results['node_precision'].append(np.mean(node_precision_scores))
        threshold_results['node_recall'].append(np.mean(node_recall_scores))
        threshold_results['node_f1'].append(np.mean(node_f1_scores))
    
    return threshold_results


def plot_threshold_analysis(results, save_path):
    """
    Plot the results of the threshold analysis
    
    Args:
        results (dict): Results from threshold analysis
        save_path (str): Path to save the plot
    """
    plt.figure(figsize=(12, 8))
    
    # Plot all metrics
    plt.plot(results['thresholds'], results['dice'], 'o-', label='Dice', linewidth=2)
    plt.plot(results['thresholds'], results['iou'], 's-', label='IoU', linewidth=2)
    plt.plot(results['thresholds'], results['node_precision'], '^-', label='Node Precision', linewidth=2)
    plt.plot(results['thresholds'], results['node_recall'], 'D-', label='Node Recall', linewidth=2)
    plt.plot(results['thresholds'], results['node_f1'], '*-', label='Node F1', linewidth=2)
    
    # Find optimal threshold for each metric
    best_dice_idx = np.argmax(results['dice'])
    best_iou_idx = np.argmax(results['iou'])
    best_node_f1_idx = np.argmax(results['node_f1'])
    
    # Add best thresholds to the plot
    plt.axvline(x=results['thresholds'][best_dice_idx], color='blue', linestyle='--', alpha=0.5, 
                label=f'Best Dice Threshold: {results["thresholds"][best_dice_idx]:.2f}')
    plt.axvline(x=results['thresholds'][best_node_f1_idx], color='green', linestyle='--', alpha=0.5,
                label=f'Best Node F1 Threshold: {results["thresholds"][best_node_f1_idx]:.2f}')
    
    # Customize the plot
    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.title('Effect of Threshold on Evaluation Metrics')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='best')
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Threshold analysis plot saved to {save_path}")
    
    # Print optimal thresholds
    print("\nOptimal Thresholds:")
    print(f"- Best Dice: {results['thresholds'][best_dice_idx]:.2f} (Score: {results['dice'][best_dice_idx]:.4f})")
    print(f"- Best IoU: {results['thresholds'][best_iou_idx]:.2f} (Score: {results['iou'][best_iou_idx]:.4f})")
    print(f"- Best Node F1: {results['thresholds'][best_node_f1_idx]:.2f} (Score: {results['node_f1'][best_node_f1_idx]:.4f})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate U-Net for road skeletonization with comprehensive metrics")
    
    # Data parameters
    parser.add_argument('--data_dir', type=str, default='../data/thinning',
                        help='Directory containing the dataset')
    
    # Model parameters
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained model weights')
    
    # Evaluation parameters
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for evaluation')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of worker threads for data loading')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Threshold for binary predictions')
    parser.add_argument('--threshold_analysis', action='store_true',
                        help='Perform threshold analysis to find optimal threshold')
    
    # Output parameters
    parser.add_argument('--output_dir', type=str, default='../evaluation',
                        help='Directory to save evaluation results')
    parser.add_argument('--save_vis', action='store_true',
                        help='Save visualization of predictions')
    parser.add_argument('--vis_samples', type=int, default=20,
                        help='Number of samples to visualize')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Evaluate model
    evaluate(args)
