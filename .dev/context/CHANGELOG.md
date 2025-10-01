# Changelog

All notable changes to the Cosine Plateau Scheduler will be documented in this file.

## [0.2.0] - 2025-10-01

### Changed
- **BREAKING**: Removed `warmup_type` parameter
  - Only linear warmup is supported (was the only implemented option)
  - Simplifies API and follows PyTorch scheduler conventions
  - Users should remove `warmup_type='linear'` from their code
- Updated to follow PyTorch naming conventions more closely
  - `warmup_steps` follows standard (not percentage-based)
  - Aligns with HuggingFace Transformers and PyTorch-Ignite conventions

### Improved
- Cleaner API with fewer unnecessary parameters
- Better compatibility with PyTorch ecosystem
- Simplified validation logic

### Documentation
- Updated all examples to remove `warmup_type` parameter
- Updated README with clearer parameter descriptions
- Updated tests to reflect API changes

### Migration Guide (0.1.0 → 0.2.0)
```python
# Before (0.1.0):
scheduler = CosinePlateauScheduler(
    optimizer,
    total_steps=10000,
    warmup_steps=1000,
    warmup_type='linear',  # ❌ Remove this line
    min_lr_ratio=0.1
)

# After (0.2.0):
scheduler = CosinePlateauScheduler(
    optimizer,
    total_steps=10000,
    warmup_steps=1000,  # ✅ That's it!
    min_lr_ratio=0.1
)
```

### Build System
- Changed from `uv_build` to `setuptools` for better pip compatibility
- Enables direct installation from Git: `pip install git+https://github.com/...`
- No longer requires `uv` to be installed

### Package Metadata
- Removed private email from author information
- Broadened Python compatibility: 3.7+ (was 3.10+)
- Broadened PyTorch compatibility: 1.4+ (was 2.0+)
- Added backward compatibility for PyTorch < 2.0 (LRScheduler vs _LRScheduler)

## [0.1.0] - 2025-10-01

### Added
- Initial release of Cosine Plateau Scheduler
- Cosine annealing with configurable minimum learning rate ratio
- Flexible warm-up system with two modes:
  - Linear warm-up: Simple linear increase
  - Cosine warm-up: Straight line with smooth curve at end
- **warmup_smoothing parameter**: Border-radius effect for warm-up (0.0-1.0)
  - Acts like CSS border-radius
  - 0.0 = pure linear (no curve)
  - 0.3 = curve only in last 30% (default)
  - 1.0 = maximum curve throughout warm-up
- Plateau steps: Configurable flat regions during training
  - Specified as (position%, duration%) tuples
  - Multiple plateaus supported
- Full PyTorch LRScheduler compatibility
- Support for multiple parameter groups
- Resume training capability with last_epoch parameter
- Comprehensive test suite
- Example scripts and visualization tools

### Features
- **Smart warm-up**: Combines linear start with optional cosine smoothing at the end
- **Plateau system**: Maintain constant LR at critical training phases
- **Flexible configuration**: All parameters have sensible defaults
- **Professional API**: Standard names compatible with PyTorch schedulers
- **Type hints**: Full type annotation support
- **Well documented**: Comprehensive README and docstrings

### Examples Included
- Basic usage with plateaus
- Pure cosine annealing (no plateaus)
- Linear warm-up configuration
- Custom warm-up smoothing
- Training resumption from checkpoints
- Visualization scripts

### Testing
- Unit tests for all major functionality
- Warm-up phase validation
- Cosine decay verification
- Plateau step testing
- Minimum LR enforcement
- Multiple parameter group support
- Resume training capability

## Future Plans
- Additional warm-up curves (exponential, polynomial)
- Dynamic plateau adjustment based on metrics
- Integration examples with popular frameworks
- More sophisticated decay strategies
- Automatic hyperparameter tuning suggestions

