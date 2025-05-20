# Road Network Skeletonization Ablation Study

This document explains how to run and interpret the ablation study for the road network skeletonization project. The ablation study systematically varies key parameters to identify their impact on model performance.

## Overview

The ablation study evaluates the impact of three key factors:

1. **Learning Rate**: Different learning rates (0.0001, 0.001, 0.01)
2. **Loss Functions**: Different loss functions (BCE, Dice, Combined, Weighted)
3. **Architecture Choices**: Different model architectures (Small, Standard, Large, Bilinear, Transpose convolution)

For each configuration, a model is trained for a specified number of epochs on the same dataset, and then evaluated on a test set using comprehensive metrics including Dice coefficient, IoU, and specialized node-based metrics.

## Running the Ablation Study

### Prerequisites

Ensure you have the same environment set up as for the main project.

### Basic Usage

To run the ablation study with default parameters:
