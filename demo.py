"""
DeepCore Demonstration Script

This script demonstrates the core functionality of the DeepCore library for coreset selection
in deep learning. It uses synthetic data that mimics the structure of real datasets like CIFAR10.

Usage:
    python demo.py

The script demonstrates:
1. Uniform coreset selection method with different fractions
2. Network architectures (ResNet18, VGG11)
3. Training on a coreset subset
4. Evaluation and results visualization
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from types import SimpleNamespace
import os

# Import deepcore components
import deepcore.nets as nets
import deepcore.methods as methods
from utils import train, test, init_recorder


class SyntheticDataset(torch.utils.data.Dataset):
    """
    Synthetic dataset that mimics the structure of image classification datasets.
    Creates class-separable data for more realistic training behavior.
    """
    def __init__(self, num_samples=1000, num_classes=10, im_size=(32, 32), channels=3, 
                 train=True, seed=None):
        self.num_samples = num_samples
        self.num_classes = num_classes
        self.im_size = im_size
        self.channels = channels
        self.train = train
        
        # Generate random data with class-specific patterns
        seed_value = seed if seed else (42 if train else 43)
        torch.manual_seed(seed_value)
        np.random.seed(seed_value)
        
        # Create data with class-dependent features for realistic training
        self.data = torch.zeros(num_samples, channels, *im_size)
        self.targets = torch.zeros(num_samples, dtype=torch.long)
        
        samples_per_class = num_samples // num_classes
        for c in range(num_classes):
            start_idx = c * samples_per_class
            end_idx = start_idx + samples_per_class if c < num_classes - 1 else num_samples
            n = end_idx - start_idx
            
            # Each class has distinct mean and pattern
            class_mean = (c + 1) / (num_classes + 1)
            noise_scale = 0.2
            self.data[start_idx:end_idx] = torch.randn(n, channels, *im_size) * noise_scale + class_mean
            self.targets[start_idx:end_idx] = c
        
        # Shuffle the data
        perm = torch.randperm(num_samples)
        self.data = self.data[perm]
        self.targets = self.targets[perm]
        
        self.classes = [str(i) for i in range(num_classes)]
        self.transform = None
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]


def run_experiment(selection_method, model_name, args, dst_train, dst_test, 
                   fraction=0.1, epochs=5, lr=0.01, seed=42):
    """Run a complete experiment with the specified selection method and model."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Coreset selection
    method = methods.__dict__[selection_method](
        dst_train, args, fraction=fraction, random_seed=seed
    )
    subset = method.select()
    dst_subset = torch.utils.data.Subset(dst_train, subset['indices'])
    
    # Data loaders
    train_loader = torch.utils.data.DataLoader(
        dst_subset, batch_size=64, shuffle=True, num_workers=0
    )
    test_loader = torch.utils.data.DataLoader(
        dst_test, batch_size=64, shuffle=False, num_workers=0
    )
    
    # Model
    network = nets.__dict__[model_name](args.channel, args.num_classes, args.im_size)
    network.to(args.device)
    
    # Training setup
    criterion = nn.CrossEntropyLoss(reduction='none').to(args.device)
    optimizer = torch.optim.SGD(network.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, len(train_loader) * epochs
    )
    
    rec = init_recorder()
    
    # Training loop
    for epoch in range(epochs):
        train(train_loader, network, criterion, optimizer, scheduler, epoch, args, rec)
        test(test_loader, network, criterion, epoch, args, rec)
    
    return {
        'method': selection_method,
        'model': model_name,
        'fraction': fraction,
        'num_selected': len(subset['indices']),
        'train_loss': rec.train_loss,
        'train_acc': rec.train_acc,
        'test_loss': rec.test_loss,
        'test_acc': rec.test_acc,
        'final_test_acc': rec.test_acc[-1] if rec.test_acc else 0
    }


def plot_results(results, save_path='./result'):
    """Plot training curves and comparison results."""
    os.makedirs(save_path, exist_ok=True)
    
    # Use better colors for visibility
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    # Plot training curves
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    for i, result in enumerate(results):
        label = f"{result['method']}-{result['model']}-{result['fraction']*100:.0f}%"
        epochs = range(len(result['train_loss']))
        color = colors[i % len(colors)]
        
        axes[0, 0].plot(epochs, result['train_loss'], label=label, color=color, linewidth=2)
        axes[0, 1].plot(epochs, result['train_acc'], label=label, color=color, linewidth=2)
        axes[1, 0].plot(epochs, result['test_loss'], label=label, color=color, linewidth=2)
        axes[1, 1].plot(epochs, result['test_acc'], label=label, color=color, linewidth=2)
    
    axes[0, 0].set_title('Training Loss', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend(fontsize=9)
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].set_title('Training Accuracy', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy (%)')
    axes[0, 1].legend(fontsize=9)
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].set_title('Test Loss', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].legend(fontsize=9)
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].set_title('Test Accuracy', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Accuracy (%)')
    axes[1, 1].legend(fontsize=9)
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle('DeepCore Coreset Selection - Training Progress', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'training_curves.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # Bar chart comparing final accuracies
    fig, ax = plt.subplots(figsize=(12, 6))
    methods_labels = [f"{r['method']}\n{r['model']}\n{r['fraction']*100:.0f}%" for r in results]
    accuracies = [r['final_test_acc'] for r in results]
    
    bars = ax.bar(methods_labels, accuracies, color=colors[:len(results)], edgecolor='black', linewidth=1.5)
    ax.set_ylabel('Test Accuracy (%)', fontsize=11)
    ax.set_title('Final Test Accuracy Comparison - Coreset Selection Methods', fontsize=12, fontweight='bold')
    ax.set_ylim(0, min(100, max(accuracies) * 1.2) if accuracies else 100)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax.annotate(f'{acc:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 5),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'accuracy_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nPlots saved to {save_path}/")


def main():
    """Main demonstration function."""
    print("=" * 70)
    print("DeepCore: Coreset Selection Library - Experimental Results")
    print("=" * 70)
    
    # Configuration
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")
    print(f"PyTorch version: {torch.__version__}")
    
    # Create synthetic datasets (mimics CIFAR10 structure)
    print("\n" + "-" * 70)
    print("1. Dataset Preparation - Synthetic Data (CIFAR10-like)")
    print("-" * 70)
    
    num_train = 10000
    num_test = 2000
    num_classes = 10
    im_size = (32, 32)
    channels = 3
    
    dst_train = SyntheticDataset(
        num_samples=num_train, num_classes=num_classes, 
        im_size=im_size, channels=channels, train=True
    )
    dst_test = SyntheticDataset(
        num_samples=num_test, num_classes=num_classes,
        im_size=im_size, channels=channels, train=False
    )
    
    print(f"Training samples: {len(dst_train)}")
    print(f"Test samples: {len(dst_test)}")
    print(f"Number of classes: {num_classes}")
    print(f"Image size: {channels} x {im_size[0]} x {im_size[1]}")
    
    # Arguments namespace
    args = SimpleNamespace(
        channel=channels,
        im_size=im_size,
        num_classes=num_classes,
        class_names=[str(i) for i in range(num_classes)],
        device=device,
        print_freq=500,
        balance=True
    )
    
    # Experiment configurations
    print("\n" + "-" * 70)
    print("2. Running Experiments")
    print("-" * 70)
    
    experiments = [
        {'method': 'Uniform', 'model': 'ResNet18', 'fraction': 0.05},
        {'method': 'Uniform', 'model': 'ResNet18', 'fraction': 0.1},
        {'method': 'Uniform', 'model': 'ResNet18', 'fraction': 0.2},
        {'method': 'Uniform', 'model': 'VGG11', 'fraction': 0.1},
    ]
    
    epochs = 10
    
    print(f"\nTotal training samples: {num_train}")
    print(f"Training epochs: {epochs}")
    print(f"\nExperiments to run:")
    for i, exp in enumerate(experiments):
        samples = int(num_train * exp['fraction'])
        print(f"  {i+1}. {exp['method']} ({exp['fraction']*100:.0f}% = {samples} samples) + {exp['model']}")
    
    # Run experiments
    results = []
    for i, exp in enumerate(experiments):
        print(f"\n{'='*70}")
        print(f">>> Running experiment {i+1}/{len(experiments)}: "
              f"{exp['method']} ({exp['fraction']*100:.0f}%) + {exp['model']}")
        print(f"{'='*70}")
        result = run_experiment(
            selection_method=exp['method'],
            model_name=exp['model'],
            args=args,
            dst_train=dst_train,
            dst_test=dst_test,
            fraction=exp['fraction'],
            epochs=epochs
        )
        results.append(result)
        print(f"\n>>> Experiment {i+1} Complete:")
        print(f"    Samples selected: {result['num_selected']}")
        print(f"    Final test accuracy: {result['final_test_acc']:.2f}%")
    
    # Results summary
    print("\n" + "=" * 70)
    print("3. EXPERIMENTAL RESULTS SUMMARY")
    print("=" * 70)
    
    print(f"\n{'Method':<12} {'Model':<12} {'Fraction':<10} {'Samples':<10} {'Final Test Acc':<15}")
    print("-" * 65)
    for r in results:
        print(f"{r['method']:<12} {r['model']:<12} {r['fraction']*100:.0f}%       "
              f"{r['num_selected']:<10} {r['final_test_acc']:.2f}%")
    
    # Plot results
    print("\n" + "-" * 70)
    print("4. Generating Visualization")
    print("-" * 70)
    
    plot_results(results, save_path='./result')
    
    print("\n" + "=" * 70)
    print("DEMONSTRATION COMPLETE!")
    print("=" * 70)
    
    print("\n📊 KEY FINDINGS:")
    print("-" * 40)
    best_result = max(results, key=lambda x: x['final_test_acc'])
    print(f"✓ Best accuracy: {best_result['final_test_acc']:.2f}% "
          f"({best_result['method']}-{best_result['model']}-{best_result['fraction']*100:.0f}%)")
    
    resnet_results = [r for r in results if r['model'] == 'ResNet18']
    if resnet_results:
        print(f"\n✓ ResNet18 performance at different coreset fractions:")
        for r in sorted(resnet_results, key=lambda x: x['fraction']):
            print(f"   - {r['fraction']*100:.0f}% ({r['num_selected']} samples): {r['final_test_acc']:.2f}%")
    
    print("\n📁 Output files saved to ./result/")
    print("   - training_curves.png: Training and test metrics over epochs")
    print("   - accuracy_comparison.png: Final accuracy comparison bar chart")


if __name__ == '__main__':
    main()
