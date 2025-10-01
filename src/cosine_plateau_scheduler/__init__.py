"""
Cosine Plateau Scheduler - Advanced Learning Rate Scheduler for PyTorch

A sophisticated learning rate scheduler that combines:
- Warm-up with cosine smoothing
- Cosine annealing decay
- Plateau steps for stable training periods
"""

from .scheduler import CosinePlateauScheduler

__version__ = "0.1.0"
__all__ = ["CosinePlateauScheduler"]
