"""
Visualize the learning rate schedule
Requires: pip install matplotlib
"""
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from cosine_plateau_scheduler import CosinePlateauScheduler


def visualize_schedule(
    total_steps=10000,
    warmup_steps=1000,
    base_lr=0.001,
    min_lr_ratio=0.1,
    plateau_steps=None,
    warmup_type='linear'
):
    """Generate and plot the learning rate schedule."""
    
    # Create dummy model and optimizer
    model = nn.Linear(10, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=base_lr)
    
    # Create scheduler
    scheduler = CosinePlateauScheduler(
        optimizer,
        total_steps=total_steps,
        warmup_steps=warmup_steps,
        min_lr_ratio=min_lr_ratio,
        plateau_steps=plateau_steps,
        warmup_type=warmup_type
    )
    
    # Collect learning rates
    lrs = []
    steps = []
    
    for step in range(total_steps):
        lrs.append(optimizer.param_groups[0]['lr'])
        steps.append(step)
        scheduler.step()
    
    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(steps, lrs, linewidth=2)
    plt.xlabel('Training Step', fontsize=12)
    plt.ylabel('Learning Rate', fontsize=12)
    plt.title('Learning Rate Schedule - Cosine Plateau Scheduler', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # Add annotations
    if warmup_steps > 0:
        plt.axvline(x=warmup_steps, color='red', linestyle='--', alpha=0.7, label=f'Warmup End ({warmup_steps} steps)')
    
    # Mark plateau regions
    if plateau_steps:
        training_steps = total_steps - warmup_steps
        for i, (pos_pct, dur_pct) in enumerate(plateau_steps):
            start = warmup_steps + int(training_steps * pos_pct / 100)
            end = start + int(training_steps * dur_pct / 100)
            plt.axvspan(start, end, alpha=0.2, color='orange', label=f'Plateau {i+1}')
    
    plt.legend(fontsize=10)
    plt.tight_layout()
    
    # Save figure
    output_file = 'lr_schedule.png'
    plt.savefig(output_file, dpi=150)
    print(f"Schedule visualization saved to {output_file}")
    
    plt.show()
    
    return lrs


def main():
    """Run different schedule configurations."""
    
    print("=" * 60)
    print("Example 1: With warmup and two plateaus")
    print("=" * 60)
    visualize_schedule(
        total_steps=10000,
        warmup_steps=1000,
        base_lr=0.001,
        min_lr_ratio=0.1,
        plateau_steps=[(50, 30), (85, 10)],
        warmup_type='cosine'
    )
    
    print("\n" + "=" * 60)
    print("Example 2: Without plateaus (pure cosine with warmup)")
    print("=" * 60)
    visualize_schedule(
        total_steps=10000,
        warmup_steps=1000,
        base_lr=0.001,
        min_lr_ratio=0.0,
        plateau_steps=None,
        warmup_type='cosine'
    )
    
    print("\n" + "=" * 60)
    print("Example 3: Linear warmup with single plateau")
    print("=" * 60)
    visualize_schedule(
        total_steps=10000,
        warmup_steps=500,
        base_lr=0.001,
        min_lr_ratio=0.05,
        plateau_steps=[(60, 20)],
        warmup_type='linear'
    )
    
if __name__ == "__main__":
    main()

