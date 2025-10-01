# Getting Started with Cosine Plateau Scheduler

## Quick Setup

### 1. Install Dependencies

```bash
# Using uv (recommended)
cd cosine-plateau-scheduler
uv pip install -e .

# Or using pip
pip install -e .
```

### 2. Run Examples

```bash
# Basic usage example
python examples/basic_usage.py

# Visualize the schedule (requires matplotlib)
uv pip install matplotlib
python examples/visualize_scheduler.py
```

### 3. Run Tests

```bash
# Install dev dependencies
uv pip install -e ".[dev]"

# Run tests
pytest tests/
```

## Project Structure

```
cosine-plateau-scheduler/
├── src/
│   └── cosine_plateau_scheduler/
│       ├── __init__.py          # Package exports
│       ├── scheduler.py         # Main scheduler implementation
│       └── py.typed            # Type hints marker
├── examples/
│   ├── basic_usage.py          # Basic usage example
│   └── visualize_scheduler.py  # Visualization example
├── tests/
│   ├── __init__.py
│   └── test_scheduler.py       # Test suite
├── pyproject.toml              # Project configuration
├── README.md                   # Main documentation
├── PUBLISHING.md               # PyPI publishing guide
└── LICENSE                     # MIT License

```

## Key Features Implemented

✅ **Cosine Warm-up**: Smooth acceleration phase with linear start and cosine ending
✅ **Cosine Annealing**: Proven decay strategy following cosine curve
✅ **Plateau Steps**: Configurable flat regions for training stability
✅ **Min LR Control**: Set minimum learning rate as ratio of base LR
✅ **Flexible Configuration**: Multiple parameters with sensible defaults
✅ **Resume Support**: Continue training with `last_epoch` parameter

## Usage Example

```python
import torch
from cosine_plateau_scheduler import CosinePlateauScheduler

# Setup
model = torch.nn.Linear(10, 2)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Create scheduler
scheduler = CosinePlateauScheduler(
    optimizer,
    total_steps=10000,
    warmup_steps=1000,          # 10% warm-up
    min_lr_ratio=0.1,           # Min LR = 10% of base
    plateau_steps=[
        (50, 30),               # Plateau at 50%, lasts 30%
        (85, 10)                # Plateau at 85%, lasts 10%
    ],
    warmup_type='cosine'        # or 'linear'
)

# Training loop
for step in range(10000):
    # Forward pass
    loss = model(data)
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Update LR
    scheduler.step()
```

## Publishing to PyPI

When ready to publish:

1. **Update version** in `pyproject.toml`
2. **Build the package**: `uv build`
3. **Test on TestPyPI**: See `PUBLISHING.md`
4. **Publish to PyPI**: `python -m twine upload dist/*`

See `PUBLISHING.md` for detailed instructions.

## Parameters Reference

### Essential Parameters

- `optimizer`: PyTorch optimizer
- `total_steps`: Total training steps
- `base_lr`: Maximum learning rate (optional, uses optimizer's LR if None)
- `min_lr_ratio`: Minimum LR ratio (default: 0.0)

### Warm-up Parameters

- `warmup_steps`: Number of warm-up steps (default: 0)
- `warmup_type`: 'cosine' or 'linear' (default: 'cosine')

### Plateau Parameters

- `plateau_steps`: List of (position%, duration%) tuples (default: None)
  - Example: `[(50, 30)]` = plateau at 50% lasting 30% of training

### Other Parameters

- `last_epoch`: For resuming training (default: -1)
- `verbose`: Print updates (default: False)

## Validation Results

The scheduler has been tested and verified to:

✓ Correctly implement warm-up phase with smooth transition
✓ Follow cosine annealing after warm-up
✓ Maintain constant LR during plateau regions
✓ Respect minimum LR constraints
✓ Handle multiple parameter groups
✓ Support training resumption

## Next Steps

1. **Experiment**: Try different configurations with the visualization example
2. **Integrate**: Add to your training pipeline
3. **Tune**: Find optimal parameters for your use case
4. **Share**: Publish to PyPI when ready (see `PUBLISHING.md`)

## Support

For questions or issues:
- Check the `README.md` for detailed documentation
- Review examples in `examples/` directory
- Run tests in `tests/` to verify functionality
- Open an issue on GitHub (when repository is public)

## License

MIT License - See `LICENSE` file

