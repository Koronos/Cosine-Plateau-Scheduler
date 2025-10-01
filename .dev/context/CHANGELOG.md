# Changelog

All notable changes to the Cosine Plateau Scheduler will be documented in this file.

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

