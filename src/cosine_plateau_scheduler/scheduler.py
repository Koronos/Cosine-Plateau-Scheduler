"""
Cosine Plateau Scheduler - Advanced Learning Rate Scheduler with Warm-up and Plateau Steps
"""
import math
from typing import List, Tuple, Optional, Union
from torch.optim import Optimizer

# Compatibility with PyTorch < 2.0 (uses _LRScheduler) and >= 2.0 (uses LRScheduler)
try:
    from torch.optim.lr_scheduler import LRScheduler
except ImportError:
    from torch.optim.lr_scheduler import _LRScheduler as LRScheduler


class CosinePlateauScheduler(LRScheduler):
    """
    Learning rate scheduler with cosine warm-up and plateau steps.
    
    This scheduler combines three key features:
    1. Warm-up phase: Linear increase from 0 to base learning rate
    2. Cosine annealing: Smooth decay following cosine curve
    3. Plateau steps: Constant learning rate periods at specified intervals
    
    The learning rate follows this pattern:
    - Warm-up: Increases linearly from 0 to base_lr
    - Main phase: Decreases following cosine curve from base_lr to min_lr
    - Plateaus: Flat regions where LR is held constant for specified durations
    
    Args:
        optimizer (Optimizer): PyTorch optimizer to schedule
        total_steps (int): Total number of training steps
        base_lr (float, optional): Base (maximum) learning rate. If None, uses optimizer's LR
        min_lr_ratio (float): Minimum LR as ratio of base_lr (default: 0.0)
        warmup_steps (int): Number of warm-up steps (default: 0)
        warmup_type (str): Type of warm-up: 'linear' (default: 'linear')
        plateau_steps (List[Tuple[float, float]], optional): List of (position%, duration%) tuples
            where position is the % of post-warmup steps where plateau starts,
            and duration is the % of post-warmup steps the plateau lasts.
            Example: [(50, 30)] means plateau starts at 50% and lasts 30% of remaining steps
        last_epoch (int): Index of last epoch for resuming training (default: -1)
        verbose (bool): If True, prints a message for each update (default: False)
    
    Example:
        >>> optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        >>> scheduler = CosinePlateauScheduler(
        ...     optimizer,
        ...     total_steps=10000,
        ...     warmup_steps=1000,
        ...     min_lr_ratio=0.1,
        ...     plateau_steps=[(50, 30), (85, 10)]
        ... )
        >>> for epoch in range(epochs):
        ...     train(...)
        ...     scheduler.step()
    """
    
    def __init__(
        self,
        optimizer: Optimizer,
        total_steps: int,
        base_lr: Optional[float] = None,
        min_lr_ratio: float = 0.0,
        warmup_steps: int = 0,
        warmup_type: str = 'linear',
        plateau_steps: Optional[List[Tuple[float, float]]] = None,
        last_epoch: int = -1,
        verbose: bool = False
    ):
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.min_lr_ratio = min_lr_ratio
        self.warmup_type = warmup_type.lower()
        self.verbose = verbose
        
        # Validate warmup_type
        if self.warmup_type not in ['linear']:
            raise ValueError(f"warmup_type must be 'linear', got '{warmup_type}'")
        
        # Base learning rate
        if base_lr is None:
            self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        else:
            self.base_lrs = [base_lr] * len(optimizer.param_groups)
        
        # Calculate training steps (after warmup)
        self.training_steps = total_steps - warmup_steps
        
        # Pre-compute min_lr for each base_lr (avoid repeated multiplication)
        self.min_lrs = [base_lr * min_lr_ratio for base_lr in self.base_lrs]
        
        # Process and pre-compute plateau information
        self.plateau_regions = []
        self.segments = []  # Pre-computed segments for fast lookup
        
        if plateau_steps:
            # Parse and validate plateaus
            for pos_pct, dur_pct in plateau_steps:
                if not (0 <= pos_pct <= 100):
                    raise ValueError(f"Plateau position must be between 0 and 100, got {pos_pct}")
                if not (0 <= dur_pct <= 100):
                    raise ValueError(f"Plateau duration must be between 0 and 100, got {dur_pct}")
                
                start_step = int(self.training_steps * pos_pct / 100)
                duration = int(self.training_steps * dur_pct / 100)
                
                self.plateau_regions.append({
                    'start': start_step,
                    'end': start_step + duration,
                })
            
            # Sort by start step
            self.plateau_regions.sort(key=lambda x: x['start'])
            
            # Pre-compute total plateau duration (used for LR calculations)
            self.total_plateau_duration = sum(p['end'] - p['start'] for p in self.plateau_regions)
            self.effective_training_steps = self.training_steps - self.total_plateau_duration
            
            # Pre-compute LR values for each plateau and build segments
            self._precompute_segments()
        else:
            self.total_plateau_duration = 0
            self.effective_training_steps = self.training_steps
            # Single segment from 0 to training_steps
            self.segments = [{
                'start': 0,
                'end': self.training_steps,
                'type': 'cosine',
                'start_lrs': self.base_lrs,
                'end_lrs': self.min_lrs,
            }]
        
        super().__init__(optimizer, last_epoch=last_epoch)
    
    def _precompute_segments(self):
        """
        Pre-compute all segments and plateau LR values.
        This eliminates redundant calculations during training.
        """
        # Pre-compute plateau LR values based on global cosine progression
        plateau_lrs = []
        effective_position = 0
        
        for i, plateau in enumerate(self.plateau_regions):
            # Calculate effective position (excluding previous plateau durations)
            effective_position = plateau['start']
            for j in range(i):
                effective_position -= (self.plateau_regions[j]['end'] - self.plateau_regions[j]['start'])
            
            # Calculate LR at this position on the global cosine
            if self.effective_training_steps > 0:
                progress = effective_position / self.effective_training_steps
                progress = max(0.0, min(1.0, progress))
                cosine_factor = 0.5 * (1 + math.cos(math.pi * progress))
                # Store LR values for each param group
                plateau_lr = [
                    min_lr + (base_lr - min_lr) * cosine_factor
                    for base_lr, min_lr in zip(self.base_lrs, self.min_lrs)
                ]
            else:
                plateau_lr = list(self.min_lrs)
            
            plateau_lrs.append(plateau_lr)
        
        # Build segments list for fast lookup
        current_pos = 0
        for i, plateau in enumerate(self.plateau_regions):
            # Cosine segment before plateau
            if current_pos < plateau['start']:
                start_lrs = self.base_lrs if i == 0 else plateau_lrs[i - 1]
                end_lrs = plateau_lrs[i]
                
                self.segments.append({
                    'start': current_pos,
                    'end': plateau['start'],
                    'type': 'cosine',
                    'start_lrs': start_lrs,
                    'end_lrs': end_lrs,
                })
            
            # Plateau segment
            self.segments.append({
                'start': plateau['start'],
                'end': plateau['end'],
                'type': 'plateau',
                'lrs': plateau_lrs[i],
            })
            
            current_pos = plateau['end']
        
        # Final cosine segment after last plateau
        if current_pos < self.training_steps:
            self.segments.append({
                'start': current_pos,
                'end': self.training_steps,
                'type': 'cosine',
                'start_lrs': plateau_lrs[-1] if plateau_lrs else self.base_lrs,
                'end_lrs': self.min_lrs,
            })
    
    def _get_warmup_lr(self, step: int, base_lr: float) -> float:
        """
        Calculate learning rate during warm-up phase.
        Simple linear warm-up from 0 to base_lr.
        """
        if step >= self.warmup_steps:
            return base_lr
        
        if self.warmup_steps == 0:
            return base_lr
        
        # Linear warm-up: straight line from 0 to base_lr
        return base_lr * (step / self.warmup_steps)
    
    def _get_cosine_lr(self, step: int, param_group_idx: int) -> float:
        """
        Optimized LR calculation using pre-computed segments.
        No redundant calculations or searches.
        """
        adjusted_step = step - self.warmup_steps
        
        # Find the segment (typically O(1) or O(log n) with few segments)
        # Most common case: sequential access, so check last segment first
        for segment in self.segments:
            if segment['start'] <= adjusted_step < segment['end']:
                if segment['type'] == 'plateau':
                    # Plateau: return pre-computed LR
                    return segment['lrs'][param_group_idx]
                else:
                    # Cosine segment: calculate using pre-computed start/end LRs
                    segment_length = segment['end'] - segment['start']
                    if segment_length <= 0:
                        return segment['start_lrs'][param_group_idx]
                    
                    step_in_segment = adjusted_step - segment['start']
                    progress = step_in_segment / segment_length
                    
                    # Fast cosine calculation
                    cosine_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
                    start_lr = segment['start_lrs'][param_group_idx]
                    end_lr = segment['end_lrs'][param_group_idx]
                    
                    return end_lr + (start_lr - end_lr) * cosine_factor
        
        # Fallback (should rarely happen)
        return self.min_lrs[param_group_idx]
    
    def get_lr(self) -> List[float]:
        """
        Calculate learning rate for current step.
        Optimized version with minimal overhead.
        """
        if self.last_epoch < 0:
            return self.base_lrs
        
        step = self.last_epoch
        
        # Warm-up phase
        if step < self.warmup_steps:
            return [self._get_warmup_lr(step, base_lr) for base_lr in self.base_lrs]
        
        # Training phase: use pre-computed segments
        return [self._get_cosine_lr(step, i) for i in range(len(self.base_lrs))]
    
    def get_last_lr(self) -> List[float]:
        """Return last computed learning rate."""
        return self._last_lr

