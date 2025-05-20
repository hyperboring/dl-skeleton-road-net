import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import cv2
from torchvision import transforms


class RoadSkeletonDataset(Dataset):
    """
    Dataset for road skeletonization task
    """
    def __init__(self, root_dir, split='train', transform=None):
        """
        Args:
            root_dir (str): Directory with all the images.
            split (str): 'train', 'val', or 'test'
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        
        # Get paths to input and ground truth images
        self.input_dir = os.path.join(root_dir, "input")
        self.gt_dir = os.path.join(root_dir, "ground_truth")
        
        # List all files in the input directory
        all_files = sorted(os.listdir(self.input_dir))
        
        # Split data into train, val, test (80/10/10 by default)
        if self.split == 'train':
            self.files = all_files[:int(0.8 * len(all_files))]
        elif self.split == 'val':
            self.files = all_files[int(0.8 * len(all_files)):int(0.9 * len(all_files))]
        else:  # test
            self.files = all_files[int(0.9 * len(all_files)):]
            
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        # Get input image path
        input_file = self.files[idx]
        input_path = os.path.join(self.input_dir, input_file)
        
        # Get corresponding ground truth path (replacing "image" with "target")
        gt_file = input_file.replace("image", "target")
        gt_path = os.path.join(self.gt_dir, gt_file)
        
        # Read images
        input_image = Image.open(input_path).convert('L')  # Convert to grayscale
        gt_image = Image.open(gt_path).convert('L')
        
        # Convert to numpy arrays
        input_array = np.array(input_image) / 255.0  # Normalize to [0, 1]
        gt_array = np.array(gt_image) / 255.0
        
        # Convert to torch tensors
        input_tensor = torch.tensor(input_array, dtype=torch.float32).unsqueeze(0)  # Add channel dimension
        gt_tensor = torch.tensor(gt_array, dtype=torch.float32).unsqueeze(0)
        
        # Apply transformations if provided
        if self.transform:
            input_tensor = self.transform(input_tensor)
            gt_tensor = self.transform(gt_tensor)
            
        return {
            'input': input_tensor,
            'target': gt_tensor,
            'filename': input_file
        }


def get_dataloaders(data_dir, batch_size=8, num_workers=4):
    """
    Create train, validation, and test dataloaders
    
    Args:
        data_dir (str): Directory containing the dataset
        batch_size (int): Batch size for dataloaders
        num_workers (int): Number of workers for dataloaders
        
    Returns:
        dict: Dictionary containing train, val, and test dataloaders
    """
    train_dataset = RoadSkeletonDataset(data_dir, split='train')
    val_dataset = RoadSkeletonDataset(data_dir, split='val')
    test_dataset = RoadSkeletonDataset(data_dir, split='test')
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        num_workers=num_workers, pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader
    }
