"""
Script to run the training and evaluation on Google Colab
"""
import os
import argparse
import torch
import time
from datetime import datetime

def setup_colab_environment():
    """Install required packages and clone the repository if running on Colab"""
    try:
        import google.colab
        is_colab = True
    except:
        is_colab = False
        
    if is_colab:
        print("Setting up Colab environment...")
        
        # Install required packages using subprocess
        import subprocess
        subprocess.check_call(["pip", "install", "-q", 
                              "opencv-python", "tqdm", "matplotlib", 
                              "tensorboard", "pandas", "scipy"])
        
        # Clone the repository if it doesn't exist
        if not os.path.exists('dl-skeleton-road-net'):
            subprocess.check_call(["git", "clone", 
                                  "https://github.com/hyperboring/dl-skeleton-road-net.git"])
            os.chdir('dl-skeleton-road-net')
        
        print("Colab environment setup complete!")
    else:
        print("Not running on Colab, skipping environment setup.")
    
    return is_colab

def main(args):
    # Setup Colab environment if needed
    is_colab = setup_colab_environment()
    
    # Get current timestamp for run identification
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Set directories
    if is_colab:
        base_dir = '/content/dl-skeleton-road-net'
        data_dir = f"{base_dir}/data/thinning"
        checkpoint_dir = f"{base_dir}/checkpoints/{timestamp}"
        log_dir = f"{base_dir}/logs/{timestamp}"
        vis_dir = f"{base_dir}/visualizations/{timestamp}"
        eval_dir = f"{base_dir}/evaluation/{timestamp}"
    else:
        base_dir = '.'
        data_dir = args.data_dir
        checkpoint_dir = args.checkpoint_dir
        log_dir = args.log_dir
        vis_dir = args.vis_dir
        eval_dir = os.path.join(base_dir, 'evaluation', timestamp)
    
    # Create directories
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)
    os.makedirs(eval_dir, exist_ok=True)
    
    # Check GPU availability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
        print(f"Memory cached: {torch.cuda.memory_reserved(0) / 1e9:.2f} GB")
    
    # Run training
    if args.train:
        print("\n=== Starting Training ===")
        # Get the correct script path - handle both cases of running from project root or thinning directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        train_script = os.path.join(script_dir, "train.py")
        
        train_cmd = (
            f"python {train_script} "
            f"--data_dir {data_dir} "
            f"--batch_size {args.batch_size} "
            f"--epochs {args.epochs} "
            f"--learning_rate {args.learning_rate} "
            f"--checkpoint_dir {checkpoint_dir} "
            f"--log_dir {log_dir} "
            f"--vis_dir {vis_dir} "
            f"--save_every {args.save_every} "
            f"--vis_every {args.vis_every} "
            f"--num_workers {args.num_workers}"
        )
        
        if args.pretrained:
            train_cmd += f" --pretrained {args.pretrained}"
            
        print(f"Running command: {train_cmd}")
        os.system(train_cmd)
        
        # Save the best model path for evaluation
        best_model_path = os.path.join(checkpoint_dir, 'best_model.pth')
    else:
        best_model_path = args.model_path
    
    # Run evaluation
    if args.evaluate:
        print("\n=== Starting Evaluation with Comprehensive Metrics ===")
        
        # Get evaluation script path
        script_dir = os.path.dirname(os.path.abspath(__file__))
        eval_script = os.path.join(script_dir, "evaluation.py")
        
        eval_cmd = (
            f"python {eval_script} "
            f"--data_dir {data_dir} "
            f"--model_path {best_model_path if args.train else args.model_path} "
            f"--batch_size {args.batch_size} "
            f"--output_dir {eval_dir} "
            f"--num_workers {args.num_workers} "
        )
        
        # Add visualization options
        if args.vis_samples > 0:
            eval_cmd += f" --save_vis --vis_samples {args.vis_samples}"
            
        # Add threshold analysis if requested
        if args.threshold_analysis:
            eval_cmd += " --threshold_analysis"
            
        print(f"Running command: {eval_cmd}")
        os.system(eval_cmd)
        
        # Display the evaluation report
        if os.path.exists(os.path.join(eval_dir, 'metrics', 'evaluation_report.md')):
            print("\n=== Evaluation Report ===")
            with open(os.path.join(eval_dir, 'metrics', 'evaluation_report.md'), 'r') as f:
                print(f.read())
    
    print("\n=== All tasks completed! ===")
    
    # Return paths for further use
    return {
        'checkpoint_dir': checkpoint_dir,
        'log_dir': log_dir,
        'vis_dir': vis_dir,
        'eval_dir': eval_dir,
        'best_model_path': best_model_path if args.train else args.model_path
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run training and evaluation on Colab")
    
    # Operation modes
    parser.add_argument('--train', action='store_true',
                        help='Run training')
    parser.add_argument('--evaluate', action='store_true',
                        help='Run evaluation')
    
    # Data parameters
    parser.add_argument('--data_dir', type=str, default='../data/thinning',
                        help='Directory containing the dataset')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for training and evaluation')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs to train')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--num_workers', type=int, default=2,
                        help='Number of worker threads for data loading')
    
    # Model parameters
    parser.add_argument('--pretrained', type=str, default=None,
                        help='Path to pretrained model weights for training')
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to model weights for evaluation (if not training)')
    
    # Evaluation parameters
    parser.add_argument('--threshold_analysis', action='store_true',
                        help='Perform threshold analysis to find optimal threshold')
    
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
    parser.add_argument('--vis_samples', type=int, default=10,
                        help='Number of samples to visualize during evaluation')
    
    args = parser.parse_args()
    
    # If no operation is specified, default to training
    if not args.train and not args.evaluate:
        print("No operation specified. Defaulting to training mode.")
        args.train = True
        
    if not args.train and args.evaluate and args.model_path is None:
        parser.error("--model_path must be specified when running evaluation without training")
    
    # Run the main function
    main(args)
