"""
Generate visualizations for the Cosine Plateau Scheduler
This file generates all graphics for debugging and documentation
"""
import sys
sys.path.insert(0, 'src')

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from cosine_plateau_scheduler import CosinePlateauScheduler

plt.switch_backend('Agg')


def test_warmup_only():
    """Test ONLY the warmup phase (no cosine decay after)"""
    print("\n" + "="*70)
    print("TEST 1: Warmup Phase Only")
    print("="*70)
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    model = nn.Linear(10, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
    
    # Total steps = warmup steps to see ONLY warmup
    scheduler = CosinePlateauScheduler(
        optimizer,
        total_steps=100,
        warmup_steps=100,
        warmup_type='linear'
    )
    
    lrs = []
    for step in range(100):
        lrs.append(optimizer.param_groups[0]['lr'])
        scheduler.step()
    
    ax.plot(range(100), lrs, linewidth=3, color='#2563eb', label='Linear Warmup')
    ax.axhline(y=1.0, color='green', linestyle=':', alpha=0.5, linewidth=2, label='Target LR')
    ax.set_xlabel('Warmup Step', fontsize=13, fontweight='bold')
    ax.set_ylabel('Learning Rate', fontsize=13, fontweight='bold')
    ax.set_title('Warmup Phase: Linear Increase from 0 to Base LR', 
                 fontsize=14, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    ax.set_ylim(-0.05, 1.1)
    
    plt.tight_layout()
    plt.savefig('1_warmup_only.png', dpi=150, bbox_inches='tight')
    print("[OK] Generated: 1_warmup_only.png")


def test_cosine_decay_only():
    """Test ONLY cosine annealing (no warmup, no plateaus)"""
    print("\n" + "="*70)
    print("TEST 2: Cosine Annealing Only")
    print("="*70)
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    model = nn.Linear(10, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
    
    scheduler = CosinePlateauScheduler(
        optimizer,
        total_steps=1000,
        warmup_steps=0,  # No warmup
        min_lr_ratio=0.1,
        plateau_steps=None  # No plateaus
    )
    
    lrs = []
    steps = list(range(1000))
    for _ in range(1000):
        lrs.append(optimizer.param_groups[0]['lr'])
        scheduler.step()
    
    ax.plot(steps, lrs, linewidth=2.5, color='#e74c3c', label='Cosine Decay')
    ax.axhline(y=1.0, color='green', linestyle=':', alpha=0.4, linewidth=1.5, label='Base LR')
    ax.axhline(y=0.1, color='orange', linestyle=':', alpha=0.4, linewidth=1.5, label='Min LR (10%)')
    ax.set_xlabel('Training Step', fontsize=13, fontweight='bold')
    ax.set_ylabel('Learning Rate', fontsize=13, fontweight='bold')
    ax.set_title('Cosine Annealing: Smooth Decay from Base LR to Min LR', 
                 fontsize=14, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    
    plt.tight_layout()
    plt.savefig('2_cosine_decay_only.png', dpi=150, bbox_inches='tight')
    print("[OK] Generated: 2_cosine_decay_only.png")


def test_plateaus_only():
    """Test cosine decay WITH plateaus (no warmup)"""
    print("\n" + "="*70)
    print("TEST 3: Cosine Decay with Plateaus")
    print("="*70)
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    
    model = nn.Linear(10, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
    
    scheduler = CosinePlateauScheduler(
        optimizer,
        total_steps=1000,
        warmup_steps=0,
        min_lr_ratio=0.1,
        plateau_steps=[(30, 20), (70, 15)]  # Two plateaus
    )
    
    lrs = []
    steps = list(range(1000))
    for _ in range(1000):
        lrs.append(optimizer.param_groups[0]['lr'])
        scheduler.step()
    
    ax.plot(steps, lrs, linewidth=2.5, color='#9b59b6', label='LR Schedule')
    
    # Mark plateaus
    plateau1_start = int(1000 * 0.30)
    plateau1_end = plateau1_start + int(1000 * 0.20)
    plateau2_start = int(1000 * 0.70)
    plateau2_end = plateau2_start + int(1000 * 0.15)
    
    ax.axvspan(plateau1_start, plateau1_end, alpha=0.2, color='orange', label='Plateau 1 (30%, 20%)')
    ax.axvspan(plateau2_start, plateau2_end, alpha=0.2, color='cyan', label='Plateau 2 (70%, 15%)')
    
    ax.set_xlabel('Training Step', fontsize=13, fontweight='bold')
    ax.set_ylabel('Learning Rate', fontsize=13, fontweight='bold')
    ax.set_title('Cosine Decay with Plateau Steps: Constant LR Periods', 
                 fontsize=14, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11, loc='upper right')
    
    plt.tight_layout()
    plt.savefig('3_plateaus.png', dpi=150, bbox_inches='tight')
    print("[OK] Generated: 3_plateaus.png")


def test_complete_schedule():
    """Test the complete schedule: warmup + cosine decay + plateaus"""
    print("\n" + "="*70)
    print("TEST 4: Complete Schedule (Warmup + Decay + Plateaus)")
    print("="*70)
    
    fig, ax = plt.subplots(1, 1, figsize=(14, 6))
    
    model = nn.Linear(10, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    
    total_steps = 10000
    warmup_steps = 1000
    
    scheduler = CosinePlateauScheduler(
        optimizer,
        total_steps=total_steps,
        warmup_steps=warmup_steps,
        min_lr_ratio=0.1,
        plateau_steps=[(50, 30), (85, 10)]
    )
    
    lrs = []
    steps = list(range(total_steps))
    for _ in range(total_steps):
        lrs.append(optimizer.param_groups[0]['lr'])
        scheduler.step()
    
    ax.plot(steps, lrs, linewidth=2.5, color='#2563eb', label='Learning Rate')
    
    # Mark warmup end
    ax.axvline(x=warmup_steps, color='red', linestyle='--', alpha=0.6, 
               linewidth=2, label=f'Warmup End ({warmup_steps})')
    
    # Mark plateaus
    training_steps = total_steps - warmup_steps
    plateau1_start = warmup_steps + int(training_steps * 0.50)
    plateau1_end = plateau1_start + int(training_steps * 0.30)
    plateau2_start = warmup_steps + int(training_steps * 0.85)
    plateau2_end = plateau2_start + int(training_steps * 0.10)
    
    ax.axvspan(plateau1_start, plateau1_end, alpha=0.25, color='orange', 
               label='Plateau 1 (50%, 30%)')
    ax.axvspan(plateau2_start, plateau2_end, alpha=0.25, color='purple', 
               label='Plateau 2 (85%, 10%)')
    
    # Reference lines
    ax.axhline(y=0.001, color='green', linestyle=':', alpha=0.5, 
               linewidth=1.5, label='Base LR (0.001)')
    ax.axhline(y=0.0001, color='brown', linestyle=':', alpha=0.5, 
               linewidth=1.5, label='Min LR (0.0001)')
    
    ax.set_xlabel('Training Step', fontsize=13, fontweight='bold')
    ax.set_ylabel('Learning Rate', fontsize=13, fontweight='bold')
    ax.set_title('Complete Schedule: Warmup + Cosine Decay + Plateaus\n' +
                 f'(Total: {total_steps:,} steps, Warmup: {warmup_steps:,}, Min LR: 10%)', 
                 fontsize=14, fontweight='bold', pad=15)
    ax.legend(fontsize=10, loc='upper right', framealpha=0.9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('4_complete_schedule.png', dpi=150, bbox_inches='tight')
    print("[OK] Generated: 4_complete_schedule.png")


def test_multiple_configurations():
    """Compare different scheduler configurations side by side"""
    print("\n" + "="*70)
    print("TEST 5: Comparison of Different Configurations")
    print("="*70)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('Cosine Plateau Scheduler - Configuration Comparison', 
                 fontsize=16, fontweight='bold')
    
    configs = [
        {
            'title': 'Warmup + Cosine Decay',
            'total_steps': 1000,
            'warmup_steps': 200,
            'min_lr_ratio': 0.0,
            'plateau_steps': None,
            'color': '#3498db'
        },
        {
            'title': 'Warmup + Single Plateau',
            'total_steps': 1000,
            'warmup_steps': 200,
            'min_lr_ratio': 0.1,
            'plateau_steps': [(60, 25)],
            'color': '#e74c3c'
        },
        {
            'title': 'Warmup + Multiple Plateaus',
            'total_steps': 1000,
            'warmup_steps': 100,
            'min_lr_ratio': 0.1,
            'plateau_steps': [(40, 15), (65, 15), (85, 10)],
            'color': '#2ecc71'
        },
        {
            'title': 'No Warmup + Plateaus',
            'total_steps': 1000,
            'warmup_steps': 0,
            'min_lr_ratio': 0.05,
            'plateau_steps': [(30, 20), (70, 15)],
            'color': '#9b59b6'
        }
    ]
    
    for idx, config in enumerate(configs):
        ax = axes[idx // 2, idx % 2]
        
        model = nn.Linear(10, 2)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
        
        scheduler = CosinePlateauScheduler(
            optimizer,
            total_steps=config['total_steps'],
            warmup_steps=config['warmup_steps'],
            min_lr_ratio=config['min_lr_ratio'],
            plateau_steps=config['plateau_steps']
        )
        
        lrs = []
        for _ in range(config['total_steps']):
            lrs.append(optimizer.param_groups[0]['lr'])
            scheduler.step()
        
        ax.plot(range(config['total_steps']), lrs, linewidth=2.5, color=config['color'])
        
        # Mark warmup if present
        if config['warmup_steps'] > 0:
            ax.axvline(x=config['warmup_steps'], color='red', linestyle='--', 
                      alpha=0.5, linewidth=1.5)
        
        # Mark plateaus if present
        if config['plateau_steps']:
            training_steps = config['total_steps'] - config['warmup_steps']
            colors = ['orange', 'cyan', 'pink']
            for i, (pos_pct, dur_pct) in enumerate(config['plateau_steps']):
                start = config['warmup_steps'] + int(training_steps * pos_pct / 100)
                end = start + int(training_steps * dur_pct / 100)
                ax.axvspan(start, end, alpha=0.2, color=colors[i % len(colors)])
        
        ax.set_xlabel('Step', fontsize=11)
        ax.set_ylabel('Learning Rate', fontsize=11)
        ax.set_title(config['title'], fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('5_comparison.png', dpi=150, bbox_inches='tight')
    print("[OK] Generated: 5_comparison.png")


def main():
    """Run all visualization tests"""
    print("\n" + "="*70)
    print("COSINE PLATEAU SCHEDULER - GRAPHICS GENERATOR")
    print("="*70)
    print("Generating all visualization graphics...")
    
    test_warmup_only()
    test_cosine_decay_only()
    test_plateaus_only()
    test_complete_schedule()
    test_multiple_configurations()
    
    print("\n" + "="*70)
    print("ALL GRAPHICS GENERATED SUCCESSFULLY!")
    print("="*70)
    print("\nGenerated files:")
    print("  1. 1_warmup_only.png         - Linear warmup phase")
    print("  2. 2_cosine_decay_only.png   - Pure cosine annealing")
    print("  3. 3_plateaus.png            - Cosine decay with plateaus")
    print("  4. 4_complete_schedule.png   - Full schedule example")
    print("  5. 5_comparison.png          - Side-by-side comparison")
    print("="*70)


if __name__ == "__main__":
    main()

