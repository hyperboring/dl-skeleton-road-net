import numpy as np
import torch
import torch.nn as nn
from scipy.ndimage import distance_transform_edt
from scipy.optimize import linear_sum_assignment
import cv2


def dice_coefficient(pred, target, epsilon=1e-6):
    """
    Compute the Dice coefficient (F1 score) between prediction and target
    
    Args:
        pred (np.ndarray): Binary prediction mask
        target (np.ndarray): Binary target mask
        epsilon (float): Small constant to avoid division by zero
        
    Returns:
        float: Dice coefficient
    """
    intersection = np.sum(pred * target)
    union = np.sum(pred) + np.sum(target)
    return (2. * intersection + epsilon) / (union + epsilon)


def iou_score(pred, target, epsilon=1e-6):
    """
    Compute the Intersection over Union (IoU) score
    
    Args:
        pred (np.ndarray): Binary prediction mask
        target (np.ndarray): Binary target mask
        epsilon (float): Small constant to avoid division by zero
        
    Returns:
        float: IoU score
    """
    intersection = np.sum(pred * target)
    union = np.sum(pred) + np.sum(target) - intersection
    return (intersection + epsilon) / (union + epsilon)


def mse_with_distance_transform(pred, target):
    """
    Compute MSE using distance transform from ground truth
    
    Args:
        pred (np.ndarray): Binary prediction mask
        target (np.ndarray): Binary target mask
        
    Returns:
        float: MSE with distance transform
    """
    # Compute distance transform of the ground truth
    dist_transform = distance_transform_edt(1 - target)
    
    # Compute squared differences only at predicted positive pixels
    squared_diff = np.square(dist_transform) * pred
    
    # Normalize by number of predicted positive pixels
    num_pos_pixels = np.sum(pred)
    if num_pos_pixels == 0:
        return 0.0
    
    return np.sum(squared_diff) / num_pos_pixels


def compute_node_valence(binary_image):
    """
    Compute the valence of each positive pixel in a binary skeleton image
    
    Args:
        binary_image (np.ndarray): Binary skeleton image
        
    Returns:
        dict: Dictionary mapping coordinates to valence
    """
    # Ensure binary image is boolean
    binary = binary_image.astype(bool)
    
    # Find all skeleton points
    skeleton_points = np.argwhere(binary)
    
    # Create a dictionary to store {(row, col): valence}
    node_valence = {}
    
    # Define 8-neighborhood offsets: 
    # (top-left, top, top-right, right, bottom-right, bottom, bottom-left, left)
    neighborhood = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, 1), (1, 1), (1, 0),
        (1, -1), (0, -1)
    ]
    
    # Process each skeleton point
    for point in skeleton_points:
        r, c = point
        neighbors = 0
        last_val = binary[r + neighborhood[-1][0], c + neighborhood[-1][1]] if 0 <= r + neighborhood[-1][0] < binary.shape[0] and 0 <= c + neighborhood[-1][1] < binary.shape[1] else False
        
        # Count transitions from 0 to 1 in the 8-neighborhood
        for dr, dc in neighborhood:
            # Check if the neighbor is within the image bounds
            if 0 <= r + dr < binary.shape[0] and 0 <= c + dc < binary.shape[1]:
                current_val = binary[r + dr, c + dc]
                # Count transitions from 0 to 1
                if current_val and not last_val:
                    neighbors += 1
                last_val = current_val
            else:
                last_val = False
        
        # Store the valence
        node_valence[(r, c)] = neighbors
    
    return node_valence


def node_precision_recall(pred_binary, target_binary, max_distance=3):
    """
    Compute precision and recall for each node valence
    
    Args:
        pred_binary (np.ndarray): Binary prediction mask
        target_binary (np.ndarray): Binary target mask
        max_distance (int): Maximum distance for a match (in pixels)
        
    Returns:
        dict: Dictionary containing precision and recall for each valence
    """
    # Compute valence for prediction and target
    pred_valence = compute_node_valence(pred_binary)
    target_valence = compute_node_valence(target_binary)
    
    # Organize nodes by valence (0, 1, 2, 3, 4+)
    pred_by_valence = {0: [], 1: [], 2: [], 3: [], 4: []}
    target_by_valence = {0: [], 1: [], 2: [], 3: [], 4: []}
    
    for coord, valence in pred_valence.items():
        bucket = min(valence, 4)  # Group 4+ valence together
        pred_by_valence[bucket].append(coord)
    
    for coord, valence in target_valence.items():
        bucket = min(valence, 4)  # Group 4+ valence together
        target_by_valence[bucket].append(coord)
    
    # Compute precision and recall for each valence
    results = {}
    
    for valence in range(0, 5):
        pred_nodes = pred_by_valence[valence]
        target_nodes = target_by_valence[valence]
        
        # Handle empty cases
        if len(pred_nodes) == 0 and len(target_nodes) == 0:
            results[f'valence_{valence}_precision'] = 1.0
            results[f'valence_{valence}_recall'] = 1.0
            results[f'valence_{valence}_f1'] = 1.0
            continue
        elif len(pred_nodes) == 0:
            results[f'valence_{valence}_precision'] = 0.0
            results[f'valence_{valence}_recall'] = 0.0
            results[f'valence_{valence}_f1'] = 0.0
            continue
        elif len(target_nodes) == 0:
            results[f'valence_{valence}_precision'] = 0.0
            results[f'valence_{valence}_recall'] = 0.0
            results[f'valence_{valence}_f1'] = 0.0
            continue
        
        # Create distance matrix
        distance_matrix = np.zeros((len(pred_nodes), len(target_nodes)))
        
        for i, pred_node in enumerate(pred_nodes):
            for j, target_node in enumerate(target_nodes):
                # Calculate Euclidean distance
                distance = np.sqrt((pred_node[0] - target_node[0])**2 + 
                                  (pred_node[1] - target_node[1])**2)
                distance_matrix[i, j] = distance
        
        # Use Hungarian algorithm to find optimal matching
        pred_indices, target_indices = linear_sum_assignment(distance_matrix)
        
        # Count matches within max_distance
        matches = 0
        for pred_idx, target_idx in zip(pred_indices, target_indices):
            if distance_matrix[pred_idx, target_idx] <= max_distance:
                matches += 1
        
        # Calculate precision and recall
        precision = matches / len(pred_nodes)
        recall = matches / len(target_nodes)
        
        results[f'valence_{valence}_precision'] = precision
        results[f'valence_{valence}_recall'] = recall
        
        # Calculate F1 score
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0
        results[f'valence_{valence}_f1'] = f1
    
    # Calculate overall node metrics (average across valences)
    # Only include valences that exist in the results
    precision_values = [results[f'valence_{v}_precision'] for v in range(0, 5) if f'valence_{v}_precision' in results]
    recall_values = [results[f'valence_{v}_recall'] for v in range(0, 5) if f'valence_{v}_recall' in results]
    f1_values = [results[f'valence_{v}_f1'] for v in range(0, 5) if f'valence_{v}_f1' in results]
    
    # If any valence type is missing, use available ones for averaging
    results['node_precision'] = np.mean(precision_values) if precision_values else 0.0
    results['node_recall'] = np.mean(recall_values) if recall_values else 0.0
    results['node_f1'] = np.mean(f1_values) if f1_values else 0.0
    
    return results


def calculate_test_loss(model, inputs, targets, device):
    """
    Calculate the loss used during training on test data
    
    Args:
        model (torch.nn.Module): The trained model
        inputs (torch.Tensor): Input tensor
        targets (torch.Tensor): Target tensor
        device (torch.device): Device to run computation on
        
    Returns:
        float: Loss value
    """
    model.eval()
    with torch.no_grad():
        outputs = model(inputs)
        bce_loss = nn.BCELoss()(outputs, targets)
        dl_loss = 1 - dice_coefficient(
            outputs.cpu().numpy().squeeze(), 
            targets.cpu().numpy().squeeze()
        )
        loss = bce_loss.item() + dl_loss
    return loss


def compute_all_metrics(pred, target, binary_threshold=0.5):
    """
    Compute all metrics for a given prediction and target
    
    Args:
        pred (np.ndarray): Prediction array [0, 1]
        target (np.ndarray): Target array [0, 1]
        binary_threshold (float): Threshold to convert predictions to binary
        
    Returns:
        dict: Dictionary with all metrics
    """
    # Ensure inputs are numpy arrays
    if torch.is_tensor(pred):
        pred = pred.cpu().numpy()
    if torch.is_tensor(target):
        target = target.cpu().numpy()
    
    # Ensure proper shape (remove batch and channel dimensions if needed)
    if pred.ndim > 2:
        pred = pred.squeeze()
    if target.ndim > 2:
        target = target.squeeze()
    
    # Convert prediction to binary using threshold
    pred_binary = (pred > binary_threshold).astype(np.uint8)
    target_binary = (target > 0.5).astype(np.uint8)
    
    # Calculate standard segmentation metrics
    dice = dice_coefficient(pred_binary, target_binary)
    iou = iou_score(pred_binary, target_binary)
    
    # Calculate MSE with distance transform
    mse_dt = mse_with_distance_transform(pred_binary, target_binary)
    
    # Calculate node-based metrics
    node_metrics = node_precision_recall(pred_binary, target_binary)
    
    # Combine all metrics
    metrics = {
        'dice': dice,
        'iou': iou,
        'mse_dt': mse_dt
    }
    metrics.update(node_metrics)
    
    return metrics
