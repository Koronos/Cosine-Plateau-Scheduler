# Development Notes - Cosine Plateau Scheduler

## Project Context

This scheduler was developed to provide smooth learning rate scheduling with plateau support for AI training.

### Key Design Decisions

1. **Linear Warmup Only**
   - Initially considered cosine warmup with "border-radius" effect (warmup_smoothing parameter)
   - Decided to simplify to linear warmup only for better user experience
   - Removed complexity that wasn't essential for most use cases

2. **Independent Cosine Segments**
   - Each segment between plateaus is an INDEPENDENT full cosine curve
   - This creates smooth transitions at both entry and exit of each segment
   - Previously tried: global cosine that "paused" - resulted in abrupt transitions
   - Current approach: each segment has its own smooth start and end

3. **Plateau Definition**
   - Plateaus defined as (position%, duration%) of post-warmup training
   - Example: [(50, 30)] means plateau at 50% of training, lasting 30%
   - LR at plateau is determined by global cosine progression (excluding plateau time)

### Performance Optimizations

All optimization work documented in `PERFORMANCE_OPTIMIZATION.md`

Key optimizations:
- Pre-computation of all segments and plateau LRs in `__init__`
- Eliminated O(n) searches per step
- Reduced function call depth
- ~10x speedup achieved

Scalability verified:
- Tested up to 10M steps
- Per-step time: ~0.5 microseconds (constant)
- Initialization: <0.05 ms (independent of step count)
- Memory: O(segments), not O(steps)

### Code Evolution

1. **Initial Implementation**
   - Basic structure with warm-up, cosine decay, plateaus
   - Multiple nested functions with redundant calculations

2. **Smoothing Attempt**
   - Added warmup_smoothing parameter (0-1) for border-radius effect
   - Worked correctly but added complexity
   - User feedback: too complex for most use cases
   - **Removed in favor of simple linear warmup**

3. **Plateau Algorithm Evolution**
   - V1: Single global cosine with "pauses" → discontinuous
   - V2: Independent cosines per segment → smooth but wrong LR values
   - V3: Independent cosines with global LR determination → **FINAL** ✓

4. **Performance Pass**
   - Pre-computed segments structure
   - Eliminated redundant calculations
   - Direct lookups instead of searches
   - Result: Production-ready performance

### Testing Strategy

Tests included:
- Unit tests for all phases (warmup, decay, plateaus)
- Visual verification with matplotlib
- Scalability testing (1K to 10M steps)
- Multiple parameter groups
- Resume training capability

### File Structure

```
src/cosine_plateau_scheduler/
  - scheduler.py          # Main implementation (~260 lines optimized)
  
examples/
  - basic_usage.py        # Simple usage example
  - visualize_scheduler.py # Visualization examples
  - generate_graphics.py  # Comprehensive test suite with plots
  - images/               # Example plots for README
  
tests/
  - test_scheduler.py     # Unit tests
  
.dev/
  - context/              # Internal development docs
  - images/               # All generated test images
```

### Future Improvements (if needed)

1. **Additional Warmup Types**
   - Could add back cosine/exponential if users request
   - Keep as optional to maintain simplicity

2. **Adaptive Plateaus**
   - Dynamic plateau adjustment based on metrics
   - Would require callback mechanism

3. **JIT Compilation**
   - Use @torch.jit.script for additional 2-5x speedup
   - Currently not needed given performance

4. **Plateau Scheduling**
   - Allow plateaus to be triggered by conditions (loss thresholds)
   - Currently step-based only

### Known Limitations

- Plateaus must be specified as percentages (not absolute steps)
  - Design decision: more intuitive for users
  - Easy to calculate from absolute if needed

- Single warmup type (linear only)
  - Simplified from original design (removed `warmup_type` parameter in v0.2.0)
  - Sufficient for vast majority of use cases
  - Can be added back if users request it

- `min_lr_ratio` instead of `eta_min`
  - Uses ratio (0.1 = 10% of base_lr) instead of absolute value
  - More intuitive for most users
  - Different from PyTorch's CosineAnnealingLR convention (which uses eta_min)
  - Considered changing but decided current approach is clearer

### User Feedback Incorporated

1. "Border-radius warmup too complex" → Simplified to linear
2. "Plateaus look like cut cosine" → Changed to independent segments
3. "Need smooth transitions" → Implemented full cosine per segment
4. "Will it work with millions of steps?" → Verified scalability

### Version History

- **v0.2.0** (2025-10-01): API simplification and compatibility improvements
  - Removed `warmup_type` parameter (breaking change)
  - Changed build system from `uv_build` to `setuptools`
  - Broadened Python support: 3.7+ (was 3.10+)
  - Broadened PyTorch support: 1.4+ (was 2.0+)
  - Added PyTorch < 2.0 compatibility layer
  - Removed private email from package metadata
  - Better alignment with PyTorch scheduler conventions

- **v0.1.0** (2025-10-01): Initial release with optimized implementation
  - Linear warmup with optional smoothing (later removed)
  - Independent cosine segments between plateaus
  - Performance optimized implementation (~0.5μs per step)

## Maintenance Notes

### When modifying the scheduler:

1. **Always test with `generate_graphics.py`**
   - Visual verification is crucial
   - Compare before/after plots

2. **Verify scalability if changing segment logic**
   - Run scalability tests for large step counts

3. **Check both single and multiple param groups**
   - Different LRs per layer is common use case

4. **Update examples if API changes**
   - Keep documentation in sync

### Common Issues & Solutions

**Issue:** Discontinuous LR at plateau boundaries  
**Solution:** Ensure segment start/end LRs match plateau LR

**Issue:** Wrong LR values in plateaus  
**Solution:** Check global cosine calculation excludes plateau durations

**Issue:** Performance degradation  
**Solution:** Verify pre-computation happens in `__init__`, not in `get_lr()`

## Contact & Contribution

Original developer: Koronos  
Development period: October 2025  
Status: Production Ready



