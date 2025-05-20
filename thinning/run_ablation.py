#!/usr/bin/env python
"""
Script to run the ablation study for road skeletonization model.
This script provides a convenient way to start the ablation study
with default or custom parameters.
"""

import os
import sys
import argparse
import time
from datetime import datetime


def main():
    parser = argparse.ArgumentParser(description="Run ablation study for road skeletonization")

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

    # Configuration parameters
    parser.add_argument('--configs', type=str, default=None,
                        help='Comma-separated list of configurations to run (default: all)')
    parser.add_argument('--category', type=str, default=None,
                        choices=['lr', 'loss', 'arch'],
                        help='Run only configurations in a specific category')

    # Output parameters
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Directory to save ablation study results (default: ../ablation_study_TIMESTAMP)')

    args = parser.parse_args()

    # Set output directory with timestamp if not specified
    if not args.output_dir:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = f"../ablation_study_{timestamp}"

    # Filter configurations by category if specified
    if args.category:
        prefix_map = {
            'lr': 'lr_',
            'loss': 'loss_',
            'arch': 'arch_'
        }

        prefix = prefix_map.get(args.category)
        if prefix and not args.configs:
            # Import the configurations to filter them
            sys.path.append(os.path.dirname(os.path.abspath(__file__)))
            from ablation_study import CONFIGURATIONS

            # Filter configurations by prefix
            filtered_configs = [name for name in CONFIGURATIONS.keys()
                                if name.startswith(prefix)]

            if filtered_configs:
                args.configs = ','.join(filtered_configs)
                print(f"Running configurations in category '{args.category}': {args.configs}")
            else:
                print(f"No configurations found for category: {args.category}")
                return

    # Prepare arguments for ablation study
    ablation_args = argparse.Namespace(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        epochs=args.epochs,
        num_workers=args.num_workers,
        output_dir=args.output_dir,
        configs=args.configs,
        compatibility_mode=True  # Enable compatibility mode for model loading
    )

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Import and run the ablation study
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))

    print("\n========================================================")
    print(f"Starting Ablation Study at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Output directory: {args.output_dir}")
    print(f"Number of epochs: {args.epochs}")
    if args.configs:
        print(f"Selected configurations: {args.configs}")
    else:
        print("Running all configurations")
    print("========================================================\n")

    start_time = time.time()

    # Run the ablation study
    try:
        from ablation_study import run_ablation_study
        run_ablation_study(ablation_args)
    except Exception as e:
        print(f"\nError during ablation study: {e}")
        import traceback
        traceback.print_exc()
        print("\nTrying to continue with remaining configurations...")

    # Calculate and print total execution time
    total_time = time.time() - start_time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)

    print("\n========================================================")
    print(f"Ablation Study Completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total execution time: {int(hours)}h {int(minutes)}m {seconds:.2f}s")
    print(f"Results saved to: {args.output_dir}")
    print("========================================================\n")

    # Print report location
    print(f"Comprehensive comparison report available at:")
    print(f"{os.path.join(args.output_dir, 'comparative_results', 'ablation_results.md')}")
    print(f"Visualizations available in: {os.path.join(args.output_dir, 'comparative_results')}")


if __name__ == "__main__":
    main()
