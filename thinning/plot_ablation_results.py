#!/usr/bin/env python
"""
Script to plot and analyze ablation study results from a previous run.
This is useful for visualizing results without rerunning the entire study.
"""

import os
import argparse
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


def load_results(study_dir):
    """
    Load results from all configurations in the study directory
    
    Args:
        study_dir (str): Path to the ablation study directory
        
    Returns:
        pd.DataFrame: DataFrame with all results
    """
    results = []
    
    # Try loading from comparative results first (faster)
    comparative_csv = os.path.join(study_dir, 'comparative_results', 'ablation_results.csv')
    if os.path.exists(comparative_csv):
        print(f"Loading comparative results from {comparative_csv}")
        return pd.read_csv(comparative_csv)
    
    # Otherwise, load from individual configuration results
    print(f"Loading individual configuration results from {study_dir}")
    for item in os.listdir(study_dir):
        config_dir = os.path.join(study_dir, item)
        if not os.path.isdir(config_dir) or item == 'comparative_results':
            continue
        
        results_file = os.path.join(config_dir, 'results.json')
        if os.path.exists(results_file):
            with open(results_file, 'r') as f:
                result = json.load(f)
                
                # Extract key information
                row = {
                    "Configuration": result["config_name"],
                    "Learning Rate": result["parameters"]["learning_rate"],
                    "Loss Function": result["parameters"]["loss"],
                    "Architecture": result["parameters"]["architecture"],
                    "Training Time (s)": round(result["training_time"], 2),
                    "Best Val Loss": round(result["best_val_loss"], 4),
                    "Best Val Dice": round(result["best_val_dice"], 4)
                }
                
                # Add test metrics
                for key, value in result["test_metrics"].items():
                    if isinstance(value, (int, float)):
                        row[key.replace('_', ' ').title()] = round(value, 4)
                
                results.append(row)
    
    if not results:
        raise ValueError(f"No results found in {study_dir}")
    
    return pd.DataFrame(results)


def category_plots(df, output_dir):
    """
    Create category-specific plots for learning rate, loss function, and architecture
    
    Args:
        df (pd.DataFrame): DataFrame with results
        output_dir (str): Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Set general plotting style
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # 1. Learning rate comparison
    lr_df = df[df['Configuration'].str.startswith('lr_')].copy()
    if not lr_df.empty:
        plt.figure(figsize=(12, 8))
        metrics = ['Dice', 'Iou', 'Node F1', 'Mse Dt']
        
        for i, metric in enumerate(metrics):
            col_name = f'Test {metric}' if f'Test {metric}' in lr_df.columns else metric
            if col_name not in lr_df.columns:
                continue
                
            plt.subplot(2, 2, i+1)
            plt.bar(lr_df['Learning Rate'].astype(str), lr_df[col_name], color='skyblue')
            plt.xlabel('Learning Rate')
            plt.ylabel(metric)
            plt.title(f'Effect of Learning Rate on {metric}')
            
            # Add values on top of bars
            for j, v in enumerate(lr_df[col_name]):
                plt.text(j, v + 0.01, f'{v:.4f}', ha='center')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'learning_rate_comparison.png'), dpi=300)
        plt.close()
    
    # 2. Loss function comparison
    loss_df = df[df['Configuration'].str.startswith('loss_')].copy()
    if not loss_df.empty:
        plt.figure(figsize=(12, 8))
        metrics = ['Dice', 'Iou', 'Node F1', 'Mse Dt'] 
        
        for i, metric in enumerate(metrics):
            col_name = f'Test {metric}' if f'Test {metric}' in loss_df.columns else metric
            if col_name not in loss_df.columns:
                continue
                
            plt.subplot(2, 2, i+1)
            plt.bar(loss_df['Loss Function'], loss_df[col_name], color='lightcoral')
            plt.xlabel('Loss Function')
            plt.ylabel(metric)
            plt.title(f'Effect of Loss Function on {metric}')
            
            # Add values on top of bars
            for j, v in enumerate(loss_df[col_name]):
                plt.text(j, v + 0.01, f'{v:.4f}', ha='center')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'loss_function_comparison.png'), dpi=300)
        plt.close()
    
    # 3. Architecture comparison
    arch_df = df[df['Configuration'].str.startswith('arch_')].copy()
    if not arch_df.empty:
        plt.figure(figsize=(12, 8))
        metrics = ['Dice', 'Iou', 'Node F1', 'Mse Dt']
        
        for i, metric in enumerate(metrics):
            col_name = f'Test {metric}' if f'Test {metric}' in arch_df.columns else metric
            if col_name not in arch_df.columns:
                continue
                
            plt.subplot(2, 2, i+1)
            plt.bar(arch_df['Architecture'], arch_df[col_name], color='mediumseagreen')
            plt.xlabel('Architecture')
            plt.ylabel(metric)
            plt.title(f'Effect of Architecture on {metric}')
            
            # Add values on top of bars
            for j, v in enumerate(arch_df[col_name]):
                plt.text(j, v + 0.01, f'{v:.4f}', ha='center')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'architecture_comparison.png'), dpi=300)
        plt.close()


def create_radar_chart(df, output_dir):
    """
    Create radar chart comparing top configurations
    
    Args:
        df (pd.DataFrame): DataFrame with results
        output_dir (str): Directory to save plots
    """
    # Select metrics for radar chart
    metrics = ['Test Dice', 'Test Iou', 'Node Precision', 'Node Recall']
    available_metrics = [m for m in metrics if m in df.columns]
    
    if len(available_metrics) < 2:
        print("Not enough metrics available for radar chart")
        return
    
    # Select top configurations based on Dice score
    dice_col = 'Test Dice' if 'Test Dice' in df.columns else next((c for c in df.columns if 'dice' in c.lower()), None)
    if dice_col is None:
        print("No Dice score column found for radar chart")
        return
        
    top_configs = df.sort_values(dice_col, ascending=False).head(5)
    
    # Normalize metrics for radar chart
    normalized_df = top_configs.copy()
    
    for metric in available_metrics:
        min_val = normalized_df[metric].min()
        max_val = normalized_df[metric].max()
        if max_val > min_val:
            normalized_df[metric] = (normalized_df[metric] - min_val) / (max_val - min_val)
        else:
            normalized_df[metric] = 1.0
    
    # Set up radar chart
    angles = np.linspace(0, 2*np.pi, len(available_metrics), endpoint=False).tolist()
    angles += angles[:1]  # Close the loop
    
    plt.figure(figsize=(10, 8))
    ax = plt.subplot(111, polar=True)
    
    for i, row in normalized_df.iterrows():
        values = row[available_metrics].tolist()
        values += values[:1]  # Close the loop
        ax.plot(angles, values, linewidth=2, label=row['Configuration'])
        ax.fill(angles, values, alpha=0.1)
    
    # Set ticks and labels
    plt.xticks(angles[:-1], [m.replace('Test ', '') for m in available_metrics])
    plt.yticks([0.25, 0.5, 0.75, 1.0], ["0.25", "0.5", "0.75", "1.0"], color="grey", size=8)
    plt.ylim(0, 1)
    
    # Add legend
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    plt.title("Top 5 Configurations Comparison")
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'top_configurations_radar.png'), dpi=300)
    plt.close()


def create_training_time_plot(df, output_dir):
    """
    Create training time comparison plot
    
    Args:
        df (pd.DataFrame): DataFrame with results
        output_dir (str): Directory to save plots
    """
    if 'Training Time (s)' not in df.columns:
        print("No training time data available")
        return
    
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
    plt.savefig(os.path.join(output_dir, 'training_time_comparison.png'), dpi=300)
    plt.close()


def create_correlation_matrix(df, output_dir):
    """
    Create correlation matrix between different metrics
    
    Args:
        df (pd.DataFrame): DataFrame with results
        output_dir (str): Directory to save plots
    """
    # Select only numeric columns for correlation
    numeric_df = df.select_dtypes(include=[np.number])
    
    if numeric_df.shape[1] < 2:
        print("Not enough numeric columns for correlation matrix")
        return
    
    plt.figure(figsize=(12, 10))
    
    # Calculate correlation matrix
    corr = numeric_df.corr()
    
    # Create heatmap
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title('Metric Correlation Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'metric_correlation.png'), dpi=300)
    plt.close()


def create_parameter_impact_plot(df, output_dir):
    """
    Create a plot showing the impact of different parameters on performance
    
    Args:
        df (pd.DataFrame): DataFrame with results
        output_dir (str): Directory to save plots
    """
    # Make sure we have the necessary columns
    if not all(col in df.columns for col in ['Learning Rate', 'Loss Function', 'Architecture']):
        print("Missing necessary columns for parameter impact plot")
        return
    
    # Choose the metric to analyze
    metric = next((col for col in ['Test Dice', 'Dice'] if col in df.columns), None)
    if metric is None:
        print("No Dice metric found for parameter impact plot")
        return
    
    plt.figure(figsize=(15, 6))
    
    # Create grouped bar chart for each parameter
    plt.subplot(1, 3, 1)
    sns.barplot(x='Learning Rate', y=metric, data=df, errorbar=None)
    plt.title(f'Impact of Learning Rate on {metric}')
    plt.xticks(rotation=45)
    
    plt.subplot(1, 3, 2)
    sns.barplot(x='Loss Function', y=metric, data=df, errorbar=None)
    plt.title(f'Impact of Loss Function on {metric}')
    plt.xticks(rotation=45)
    
    plt.subplot(1, 3, 3)
    sns.barplot(x='Architecture', y=metric, data=df, errorbar=None)
    plt.title(f'Impact of Architecture on {metric}')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'parameter_impact.png'), dpi=300)
    plt.close()


def create_metric_distribution_plot(df, output_dir):
    """
    Create boxplots showing the distribution of key metrics
    
    Args:
        df (pd.DataFrame): DataFrame with results
        output_dir (str): Directory to save plots
    """
    # Identify key metrics
    metric_cols = [col for col in df.columns if any(term in col.lower() for term in 
                                                  ['dice', 'iou', 'precision', 'recall', 'f1'])]
    
    if len(metric_cols) < 2:
        print("Not enough metric columns for distribution plot")
        return
    
    plt.figure(figsize=(12, 6))
    
    # Melt the dataframe to get metrics in long format
    melted_df = pd.melt(df, id_vars=['Configuration'], value_vars=metric_cols, 
                        var_name='Metric', value_name='Value')
    
    # Create boxplot
    sns.boxplot(x='Metric', y='Value', data=melted_df)
    plt.title('Distribution of Metrics Across Configurations')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'metric_distribution.png'), dpi=300)
    plt.close()


def create_reports(df, output_dir, args):
    """
    Create textual reports summarizing the ablation study findings
    
    Args:
        df (pd.DataFrame): DataFrame with results
        output_dir (str): Directory to save reports
        args (Namespace): Command-line arguments
    """
    # 1. Best configurations for each metric
    report = "# Ablation Study Analysis Report\n\n"
    report += f"Generated for study directory: {args.study_dir}\n\n"
    report += "## Best Configurations by Metric\n\n"
    
    for col in df.columns:
        if any(term in col.lower() for term in ['dice', 'iou', 'precision', 'recall', 'f1', 'loss']):
            # Sort ascending for loss, descending for others
            ascending = 'loss' in col.lower()
            best_config = df.sort_values(col, ascending=ascending).iloc[0]
            
            report += f"### Best for {col}\n"
            report += f"- **Configuration:** {best_config['Configuration']}\n"
            report += f"- **Value:** {best_config[col]:.4f}\n"
            report += f"- **Parameters:** Learning Rate={best_config['Learning Rate']}, "
            report += f"Loss Function={best_config['Loss Function']}, "
            report += f"Architecture={best_config['Architecture']}\n\n"
    
    # 2. Parameter importance summary
    report += "## Parameter Importance Summary\n\n"
    
    # Learning Rate analysis
    if len(df['Learning Rate'].unique()) > 1:
        metric = next((col for col in ['Test Dice', 'Dice'] if col in df.columns), None)
        if metric:
            lr_impact = df.groupby('Learning Rate')[metric].mean().sort_values(ascending=False)
            report += "### Learning Rate Impact\n"
            report += "Average Dice score for each learning rate:\n\n"
            for lr, score in lr_impact.items():
                report += f"- **{lr}:** {score:.4f}\n"
            report += f"\n**Best learning rate:** {lr_impact.index[0]}\n\n"
    
    # Loss Function analysis
    if len(df['Loss Function'].unique()) > 1:
        metric = next((col for col in ['Test Dice', 'Dice'] if col in df.columns), None)
        if metric:
            loss_impact = df.groupby('Loss Function')[metric].mean().sort_values(ascending=False)
            report += "### Loss Function Impact\n"
            report += "Average Dice score for each loss function:\n\n"
            for loss, score in loss_impact.items():
                report += f"- **{loss}:** {score:.4f}\n"
            report += f"\n**Best loss function:** {loss_impact.index[0]}\n\n"
    
    # Architecture analysis
    if len(df['Architecture'].unique()) > 1:
        metric = next((col for col in ['Test Dice', 'Dice'] if col in df.columns), None)
        if metric:
            arch_impact = df.groupby('Architecture')[metric].mean().sort_values(ascending=False)
            report += "### Architecture Impact\n"
            report += "Average Dice score for each architecture:\n\n"
            for arch, score in arch_impact.items():
                report += f"- **{arch}:** {score:.4f}\n"
            report += f"\n**Best architecture:** {arch_impact.index[0]}\n\n"
    
    # 3. Key findings
    report += "## Key Findings and Recommendations\n\n"
    
    # Best overall configuration
    metric = next((col for col in ['Test Dice', 'Dice'] if col in df.columns), None)
    if metric:
        best_overall = df.sort_values(metric, ascending=False).iloc[0]
        report += f"### Best Overall Configuration\n"
        report += f"- **Configuration:** {best_overall['Configuration']}\n"
        report += f"- **{metric}:** {best_overall[metric]:.4f}\n"
        report += f"- **Parameters:**\n"
        report += f"  - Learning Rate: {best_overall['Learning Rate']}\n"
        report += f"  - Loss Function: {best_overall['Loss Function']}\n"
        report += f"  - Architecture: {best_overall['Architecture']}\n\n"
    
        # Best trade-off configuration (performance vs. training time)
        if 'Training Time (s)' in df.columns:
            # Normalize Dice and training time
            df_norm = df.copy()
            df_norm[metric + '_norm'] = (df_norm[metric] - df_norm[metric].min()) / (df_norm[metric].max() - df_norm[metric].min())
            df_norm['Training Time_norm'] = 1 - (df_norm['Training Time (s)'] - df_norm['Training Time (s)'].min()) / (df_norm['Training Time (s)'].max() - df_norm['Training Time (s)'].min())
            
            # Calculate combined score (equal weight to performance and speed)
            df_norm['tradeoff_score'] = 0.7 * df_norm[metric + '_norm'] + 0.3 * df_norm['Training Time_norm']
            best_tradeoff = df_norm.sort_values('tradeoff_score', ascending=False).iloc[0]
            
            report += f"### Best Performance-Time Trade-off\n"
            report += f"- **Configuration:** {best_tradeoff['Configuration']}\n"
            report += f"- **{metric}:** {best_tradeoff[metric]:.4f}\n"
            report += f"- **Training Time:** {best_tradeoff['Training Time (s)']:.1f} seconds\n"
            report += f"- **Parameters:**\n"
            report += f"  - Learning Rate: {best_tradeoff['Learning Rate']}\n"
            report += f"  - Loss Function: {best_tradeoff['Loss Function']}\n"
            report += f"  - Architecture: {best_tradeoff['Architecture']}\n\n"
    
    # 4. Save the report
    with open(os.path.join(output_dir, 'ablation_analysis_report.md'), 'w') as f:
        f.write(report)
    
    # 5. Also save a summary table with all results
    df.to_csv(os.path.join(output_dir, 'all_results.csv'), index=False)
    
    print(f"Analysis report saved to {os.path.join(output_dir, 'ablation_analysis_report.md')}")


def create_efficiency_plot(df, output_dir):
    """
    Create a plot showing the trade-off between performance and training time
    
    Args:
        df (pd.DataFrame): DataFrame with results
        output_dir (str): Directory to save plots
    """
    # Check if we have the necessary columns
    if 'Training Time (s)' not in df.columns:
        print("No training time data available for efficiency plot")
        return
    
    # Choose the metric to analyze
    metric = next((col for col in ['Test Dice', 'Dice'] if col in df.columns), None)
    if metric is None:
        print("No Dice metric found for efficiency plot")
        return
    
    plt.figure(figsize=(10, 8))
    
    # Create scatter plot with configuration labels
    plt.scatter(df['Training Time (s)'], df[metric], s=100, alpha=0.7)
    
    # Add labels for each point
    for i, row in df.iterrows():
        plt.annotate(row['Configuration'], 
                     (row['Training Time (s)'], row[metric]),
                     xytext=(5, 5), 
                     textcoords='offset points',
                     fontsize=8)
    
    plt.xlabel('Training Time (seconds)')
    plt.ylabel(metric)
    plt.title(f'Performance vs. Training Time Trade-off')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add a line representing the Pareto frontier
    # (not a true Pareto frontier, but a visual reference)
    x = df['Training Time (s)']
    y = df[metric]
    plt.plot(x, y, 'r--', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'performance_vs_time.png'), dpi=300)
    plt.close()


def create_convergence_plots(study_dir, output_dir):
    """
    Create plots showing convergence behavior across configurations
    
    Args:
        study_dir (str): Path to the ablation study directory
        output_dir (str): Directory to save plots
    """
    # Check for tensorboard logs
    tensorboard_data = {}
    for item in os.listdir(study_dir):
        config_dir = os.path.join(study_dir, item)
        if not os.path.isdir(config_dir) or item == 'comparative_results':
            continue
            
        log_dir = os.path.join(config_dir, 'logs')
        if os.path.exists(log_dir):
            # We would ideally parse tensorboard logs here,
            # but that requires additional dependencies
            # For simplicity, we'll check for training curves images
            curves_file = os.path.join(config_dir, 'training_curves.png')
            if os.path.exists(curves_file):
                # Copy or reference existing training curves
                print(f"Found training curves for {item}")
                
    # If we don't have tensorboard data, create a summary plot
    # Based on final validation metrics
    plt.figure(figsize=(10, 6))
    
    try:
        # Try to load comparative results
        df = pd.DataFrame()
        comparative_dir = os.path.join(study_dir, 'comparative_results')
        if os.path.exists(comparative_dir):
            csv_file = os.path.join(comparative_dir, 'ablation_results.csv')
            if os.path.exists(csv_file):
                df = pd.read_csv(csv_file)
                
        if not df.empty and 'Best Val Dice' in df.columns:
            # Sort by configuration type and plot
            configs = df['Configuration'].tolist()
            lr_configs = sorted([c for c in configs if c.startswith('lr_')])
            loss_configs = sorted([c for c in configs if c.startswith('loss_')])
            arch_configs = sorted([c for c in configs if c.startswith('arch_')])
            
            x_labels = lr_configs + loss_configs + arch_configs
            val_dice = [df[df['Configuration'] == c]['Best Val Dice'].values[0] for c in x_labels]
            
            plt.bar(range(len(x_labels)), val_dice, color=['skyblue'] * len(lr_configs) + 
                                                      ['lightcoral'] * len(loss_configs) + 
                                                      ['mediumseagreen'] * len(arch_configs))
            plt.xticks(range(len(x_labels)), x_labels, rotation=45, ha='right')
            plt.ylabel('Best Validation Dice Score')
            plt.title('Convergence Analysis: Best Validation Performance')
            
            # Add dividers between categories
            if lr_configs and loss_configs:
                plt.axvline(x=len(lr_configs)-0.5, color='gray', linestyle='--', alpha=0.5)
            if loss_configs and arch_configs:
                plt.axvline(x=len(lr_configs)+len(loss_configs)-0.5, color='gray', linestyle='--', alpha=0.5)
                
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'convergence_summary.png'), dpi=300)
            
    except Exception as e:
        print(f"Could not create convergence summary plot: {e}")
        
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Plot and analyze ablation study results")
    parser.add_argument('study_dir', type=str, help='Path to the ablation study directory')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Directory to save analysis results (default: study_dir/analysis)')
    parser.add_argument('--all_plots', action='store_true', 
                        help='Generate all possible plots for comprehensive analysis')
    args = parser.parse_args()
    
    # Set output directory if not specified
    if not args.output_dir:
        args.output_dir = os.path.join(args.study_dir, 'analysis')
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Loading ablation study results from: {args.study_dir}")
    print(f"Analysis outputs will be saved to: {args.output_dir}")
    
    # Load results
    df = load_results(args.study_dir)
    
    print(f"Loaded results for {len(df)} configurations")
    
    # Clean column names for consistency
    df.columns = [col.replace('_', ' ').title() if not col.startswith(('Test', 'Best')) else col for col in df.columns]
    
    # Generate standard plots
    print("Generating category-specific plots...")
    category_plots(df, args.output_dir)
    
    print("Generating radar chart of top configurations...")
    create_radar_chart(df, args.output_dir)
    
    print("Generating training time comparison plot...")
    create_training_time_plot(df, args.output_dir)
    
    # Generate additional plots if requested
    if args.all_plots:
        print("Generating correlation matrix...")
        create_correlation_matrix(df, args.output_dir)
        
        print("Generating parameter impact plots...")
        create_parameter_impact_plot(df, args.output_dir)
        
        print("Generating metric distribution plots...")
        create_metric_distribution_plot(df, args.output_dir)
        
        print("Generating efficiency plot...")
        create_efficiency_plot(df, args.output_dir)
        
        print("Generating convergence plots...")
        create_convergence_plots(args.study_dir, args.output_dir)
    
    # Generate reports
    print("Generating analysis report...")
    create_reports(df, args.output_dir, args)
    
    print("\nAnalysis complete! The following files were generated:")
    for file in os.listdir(args.output_dir):
        print(f" - {file}")


if __name__ == "__main__":
    main()
