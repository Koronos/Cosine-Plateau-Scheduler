"""
Basic usage example of CosinePlateauScheduler
"""
import torch
import torch.nn as nn
from cosine_plateau_scheduler import CosinePlateauScheduler


def main():
    # Create a simple model and optimizer
    model = nn.Linear(10, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Configuration
    total_steps = 10000
    warmup_steps = 1000
    
    # Create scheduler with plateau steps
    # First plateau: starts at 50% of training, lasts 30% of training steps
    # Second plateau: starts at 85% of training, lasts 10% of training steps
    scheduler = CosinePlateauScheduler(
        optimizer,
        total_steps=total_steps,
        warmup_steps=warmup_steps,
        min_lr_ratio=0.1,  # Min LR will be 10% of base LR
        plateau_steps=[(50, 30), (85, 10)]
    )
    
    # Training loop
    print("Training with CosinePlateauScheduler")
    print(f"Total steps: {total_steps}")
    print(f"Warmup steps: {warmup_steps}")
    print(f"Base LR: {optimizer.param_groups[0]['lr']}")
    
    for step in range(total_steps):
        # Simulate training step
        loss = model(torch.randn(32, 10)).sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Update learning rate
        scheduler.step()
        
        # Log some steps
        if step % 1000 == 0 or step < 10:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Step {step:5d}: LR = {current_lr:.6f}")
    
    print("\nTraining completed!")


if __name__ == "__main__":
    main()

