"""
Tests for CosinePlateauScheduler
"""
import pytest
import torch
import torch.nn as nn
from cosine_plateau_scheduler import CosinePlateauScheduler


@pytest.fixture
def simple_optimizer():
    """Create a simple optimizer for testing."""
    model = nn.Linear(10, 2)
    return torch.optim.SGD(model.parameters(), lr=0.1)


def test_scheduler_initialization(simple_optimizer):
    """Test that scheduler initializes correctly."""
    scheduler = CosinePlateauScheduler(
        simple_optimizer,
        total_steps=1000,
        warmup_steps=100,
        min_lr_ratio=0.1
    )
    
    assert scheduler.total_steps == 1000
    assert scheduler.warmup_steps == 100
    assert scheduler.min_lr_ratio == 0.1
    assert scheduler.training_steps == 900


def test_warmup_phase(simple_optimizer):
    """Test that warmup increases learning rate."""
    scheduler = CosinePlateauScheduler(
        simple_optimizer,
        total_steps=1000,
        warmup_steps=100,
        min_lr_ratio=0.1,
        warmup_type='linear'
    )
    
    lrs = []
    for _ in range(100):
        lrs.append(simple_optimizer.param_groups[0]['lr'])
        scheduler.step()
    
    # Check that LR increases during warmup
    assert lrs[0] < lrs[50] < lrs[99]
    # Final warmup LR should be close to base LR
    assert abs(lrs[99] - 0.1) < 0.01


def test_cosine_decay(simple_optimizer):
    """Test that LR follows cosine decay after warmup."""
    scheduler = CosinePlateauScheduler(
        simple_optimizer,
        total_steps=1000,
        warmup_steps=100,
        min_lr_ratio=0.1
    )
    
    # Skip warmup
    for _ in range(100):
        scheduler.step()
    
    lr_after_warmup = simple_optimizer.param_groups[0]['lr']
    
    # Continue for several steps
    for _ in range(400):
        scheduler.step()
    
    lr_mid = simple_optimizer.param_groups[0]['lr']
    
    # LR should decrease
    assert lr_mid < lr_after_warmup


def test_min_lr_respected(simple_optimizer):
    """Test that minimum LR is respected."""
    base_lr = 0.1
    min_lr_ratio = 0.1
    min_lr = base_lr * min_lr_ratio
    
    scheduler = CosinePlateauScheduler(
        simple_optimizer,
        total_steps=1000,
        warmup_steps=0,
        min_lr_ratio=min_lr_ratio
    )
    
    # Run entire schedule
    lrs = []
    for _ in range(1000):
        lrs.append(simple_optimizer.param_groups[0]['lr'])
        scheduler.step()
    
    # Check that no LR goes below minimum
    assert all(lr >= min_lr - 1e-6 for lr in lrs)


def test_plateau_steps(simple_optimizer):
    """Test that plateau steps maintain constant LR."""
    scheduler = CosinePlateauScheduler(
        simple_optimizer,
        total_steps=1000,
        warmup_steps=100,
        min_lr_ratio=0.0,
        plateau_steps=[(50, 20)]  # Plateau at 50% for 20% of training steps
    )
    
    lrs = []
    for _ in range(1000):
        lrs.append(simple_optimizer.param_groups[0]['lr'])
        scheduler.step()
    
    # Calculate plateau region
    training_steps = 900
    plateau_start = 100 + int(training_steps * 0.50)
    plateau_end = plateau_start + int(training_steps * 0.20)
    
    # Check that LRs in plateau are constant
    plateau_lrs = lrs[plateau_start:plateau_end]
    if len(plateau_lrs) > 1:
        # All LRs in plateau should be very similar
        assert max(plateau_lrs) - min(plateau_lrs) < 1e-6


def test_invalid_plateau_values(simple_optimizer):
    """Test that invalid plateau values raise errors."""
    with pytest.raises(ValueError):
        CosinePlateauScheduler(
            simple_optimizer,
            total_steps=1000,
            plateau_steps=[(150, 20)]  # Invalid: position > 100
        )
    
    with pytest.raises(ValueError):
        CosinePlateauScheduler(
            simple_optimizer,
            total_steps=1000,
            plateau_steps=[(50, 150)]  # Invalid: duration > 100
        )


def test_invalid_warmup_type(simple_optimizer):
    """Test that invalid warmup type raises error."""
    with pytest.raises(ValueError):
        CosinePlateauScheduler(
            simple_optimizer,
            total_steps=1000,
            warmup_type='invalid'
        )


def test_multiple_param_groups():
    """Test scheduler with multiple parameter groups."""
    model = nn.Linear(10, 2)
    optimizer = torch.optim.SGD([
        {'params': model.weight, 'lr': 0.1},
        {'params': model.bias, 'lr': 0.01}
    ])
    
    scheduler = CosinePlateauScheduler(
        optimizer,
        total_steps=1000,
        warmup_steps=100
    )
    
    # Both groups should have their LRs scheduled
    initial_lrs = [group['lr'] for group in optimizer.param_groups]
    
    for _ in range(100):
        scheduler.step()
    
    after_warmup_lrs = [group['lr'] for group in optimizer.param_groups]
    
    # Both groups should have different LRs but both should increase
    assert after_warmup_lrs[0] > initial_lrs[0]
    assert after_warmup_lrs[1] > initial_lrs[1]
    assert after_warmup_lrs[0] != after_warmup_lrs[1]


def test_resume_training(simple_optimizer):
    """Test resuming training with last_epoch."""
    scheduler1 = CosinePlateauScheduler(
        simple_optimizer,
        total_steps=1000,
        warmup_steps=100
    )
    
    # Run for 500 steps
    for _ in range(500):
        scheduler1.step()
    
    lr_at_500 = simple_optimizer.param_groups[0]['lr']
    
    # Create new scheduler starting at step 500
    model = nn.Linear(10, 2)
    optimizer2 = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler2 = CosinePlateauScheduler(
        optimizer2,
        total_steps=1000,
        warmup_steps=100,
        last_epoch=499
    )
    
    # Step once to get to step 500
    scheduler2.step()
    lr_resumed = optimizer2.param_groups[0]['lr']
    
    # LRs should match (within floating point precision)
    assert abs(lr_at_500 - lr_resumed) < 1e-6

