import os
import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """
    Double convolution block for U-Net: (Conv2d -> BatchNorm -> ReLU) * 2
    """
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
    """
    Downscaling with maxpool then double conv
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """
    Upscaling then double conv
    """
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # If bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        # Concatenate along the channel dimension
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """
    Final convolution layer
    """
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    """
    U-Net architecture for image-to-image tasks
    """
    def __init__(self, n_channels=1, n_classes=1, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        # Downsampling path
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        
        # Upsampling path
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        
        # Final convolution 
        self.outc = OutConv(64, n_classes)
        
        # Sigmoid activation for binary output
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Contracting path
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        # Expanding path with skip connections
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        
        # Final convolution and activation
        logits = self.outc(x)
        return self.sigmoid(logits)
    
    
def detect_model_architecture(state_dict):
    """
    Detect the architecture parameters of a saved model based on its state dict
    
    Args:
        state_dict (dict): State dict of the saved model
        
    Returns:
        dict: Architecture parameters including bilinear upsampling and factor
    """
    # Check if it's a UNetModel from ablation_study or UNet
    is_ablation_model = any('down_layers' in key for key in state_dict.keys())
    
    # Detect if using bilinear upsampling
    bilinear = True  # Default value
    
    # Detect model size (standard or different channel sizes)
    # Check the dimension of the bottleneck layer
    if is_ablation_model:
        # For UNetModel from ablation_study.py
        bottleneck_key = None
        for key in state_dict.keys():
            if 'down_layers.3.maxpool_conv.1.double_conv.0.weight' in key:
                bottleneck_key = key
                break
    else:
        # For UNet from unet.py
        bottleneck_key = 'down4.maxpool_conv.1.double_conv.0.weight'
    
    if bottleneck_key and bottleneck_key in state_dict:
        bottleneck_size = state_dict[bottleneck_key].shape[0]  # Output channels
        input_size = state_dict[bottleneck_key].shape[1]  # Input channels
        
        # Determine if bilinear mode was used (affects the network structure)
        # In bilinear mode, bottleneck is typically half the size (i.e., 512 instead of 1024)
        factor = 2 if bottleneck_size < 1024 else 1
        
        print(f"Detected bottleneck size: {bottleneck_size}, input size: {input_size}")
        print(f"Detected factor: {factor} (bilinear: {bilinear})")
    else:
        print("Could not detect bottleneck size, using default architecture")
        factor = 2  # Default factor for bilinear upsampling
    
    return {
        'bilinear': bilinear,
        'factor': factor,
        'is_ablation_model': is_ablation_model
    }

def get_model_matching_architecture(arch_params):
    """
    Create a U-Net model with the architecture matching the detected parameters
    
    Args:
        arch_params (dict): Architecture parameters
        
    Returns:
        nn.Module: U-Net model with the specified architecture
    """
    model = UNet(n_channels=1, n_classes=1, bilinear=arch_params['bilinear'])
    
    if arch_params['factor'] != 2:
        # Modify the architecture to match the factor
        print(f"Creating model with non-default factor: {arch_params['factor']}")
        model.down4 = Down(512, 1024)  # Use full size bottleneck
        
        # Update the upsampling path with the corresponding sizes
        model.up1 = Up(1024 + 512, 512, arch_params['bilinear'])
        model.up2 = Up(512 + 256, 256, arch_params['bilinear'])
        model.up3 = Up(256 + 128, 128, arch_params['bilinear'])
        model.up4 = Up(128 + 64, 64, arch_params['bilinear'])
    
    return model

def get_model(pretrained_path=None):
    """
    Get a U-Net model, optionally loading pretrained weights
    
    Args:
        pretrained_path (str, optional): Path to pretrained model weights
        
    Returns:
        torch.nn.Module: U-Net model
    """
    # Create a default model first
    model = UNet(n_channels=1, n_classes=1)
    
    if pretrained_path and os.path.exists(pretrained_path):
        print(f"Loading model from {pretrained_path}...")
        
        # Load the state dict first to analyze architecture
        state_dict = torch.load(pretrained_path)
        
        try:
            # Try direct loading first
            model.load_state_dict(state_dict)
            print(f"Successfully loaded model weights directly")
        except RuntimeError as e:
            print(f"Direct loading failed. Detecting model architecture...")
            
            # Detect architecture from the state dict
            arch_params = detect_model_architecture(state_dict)
            
            # Create a model with matching architecture
            if arch_params['is_ablation_model']:
                # For UNetModel from ablation_study.py
                print("Adapting weights from UNetModel structure...")
                new_state_dict = {}
                
                # Map between UNetModel structure and UNet structure
                key_mapping = {
                    'inc': 'inc',
                    'down_layers.0': 'down1',
                    'down_layers.1': 'down2', 
                    'down_layers.2': 'down3',
                    'down_layers.3': 'down4',
                    'up_layers.0': 'up1',
                    'up_layers.1': 'up2',
                    'up_layers.2': 'up3',
                    'up_layers.3': 'up4',
                    'outc': 'outc',
                    'sigmoid': 'sigmoid'
                }
                
                for key in state_dict:
                    for old_key_prefix, new_key_prefix in key_mapping.items():
                        if key.startswith(old_key_prefix):
                            new_key = key.replace(old_key_prefix, new_key_prefix, 1)
                            new_state_dict[new_key] = state_dict[key]
                            break
                
                # Create a model with the appropriate architecture
                model = get_model_matching_architecture(arch_params)
                
                # Try loading the adapted state dict
                missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
                
                if missing:
                    print(f"Warning: Missing keys: {len(missing)} keys")
                    if len(missing) < 10:  # Print if few keys are missing
                        print(missing)
                
                if unexpected:
                    print(f"Warning: Unexpected keys: {len(unexpected)} keys")
                    if len(unexpected) < 10:  # Print if few unexpected keys
                        print(unexpected)
                
                print("Successfully adapted model weights with architecture matching")
            else:
                # For UNet with different architecture parameters
                print("Recreating model with matching architecture parameters...")
                model = get_model_matching_architecture(arch_params)
                
                try:
                    model.load_state_dict(state_dict)
                    print("Successfully loaded weights with matching architecture")
                except RuntimeError as e2:
                    print(f"Still encountering issues: {e2}")
                    print("Loading with strict=False to get partial weights...")
                    missing, unexpected = model.load_state_dict(state_dict, strict=False)
                    
                    if missing:
                        print(f"Warning: Missing keys: {len(missing)} keys")
                    if unexpected:
                        print(f"Warning: Unexpected keys: {len(unexpected)} keys")
                    
                    print("Loaded partial weights, model may not perform as expected")
        
    return model
