# Structure Factor Optimization

This directory contains optimized implementations for computing 2-body and 3-body structure factors.

## Files

### `struct_2b.py` (Original, optimized)
2-body structure factor calculations with L1 pre-computation optimization.

**Optimization:** Pre-computes L1, S1, and eigendecomposition for fixed k1 before the k2 loop.

### `struct_3b.py` (Original, optimized) 
3-body structure factor calculations with support for pre-computed eigendecompositions.

**Optimization:** Accepts optional pre-computed L1, S1, l1, P1, Q1 and L2, S2, l2, P2, Q2 parameters.

### `struct_optimized.py` (New, fully optimized)
Merged and fully optimized implementation with k2 caching.

**Optimizations:**
1. Pre-computes L1-related values (constant for each k)
2. Pre-computes all L2-related values (reusable across different k1 values)
3. Only computes L12-related values per (k1, k2) pair

## Performance Improvements

### Before Optimization
For each call to `RHS_S(k, ...)`:
- 2000 calls to `compute_3bod` (200 k2_abs × 10 angles)
- Each call performs 3 eigendecompositions + 3 inversions + 3 Sylvester solves
- **Total: 6000 eig() + 6000 inv() + 6000 Sylvester solves**

### After Phase 1 (struct_2b.py, struct_3b.py)
For each call to `RHS_S(k, ...)`:
- Pre-compute L1 eigendecomposition once
- 2000 calls to `compute_3bod` with cached L1 values
- Each call performs 2 eigendecompositions + 2 inversions + 2 Sylvester solves
- **Total: 1 + 4000 eig() + 1 + 4000 inv() + 1 + 4000 Sylvester solves**
- **Reduction: ~33% fewer operations**

### After Phase 2 (struct_optimized.py with k2 cache)
When calling `RHS_S` for multiple k values with the same k2 grid:
- Pre-compute all k2 eigendecompositions once (2000 k2 values)
- For each k: Pre-compute L1 once, then 2000 calls with both L1 and L2 cached
- Each call performs 1 eigendecomposition + 1 inversion + 1 Sylvester solve (only L12)
- **Per RHS_S call: 1 + 2000 eig() + 1 + 2000 inv() + 1 + 2000 Sylvester solves**
- **With k2 cache reuse: ~67% fewer operations per call**

## Usage Examples

### Basic Usage (struct_2b.py)
```python
from struct_2b import RHS_S
from struct_aux import Vexp

k = 1.0
lp = 1.0
phi = 0.5
eps = 1e-2

result = RHS_S(k, lp, phi, eps, Vexp)
```

### Advanced Usage with k2 Cache (struct_optimized.py)
```python
import numpy as np
from struct_optimized import RHS_S_with_k2_cache, precompute_k2_values
from struct_aux import Vexp

# Setup
k_arr = np.linspace(0, 10, 100)
k2abs_arr = np.linspace(0, 10, 200)
alpha = np.linspace(0, 2 * np.pi, 10, endpoint=False)
lp, phi, eps = 1.0, 0.5, 1e-2
s = 10

# Pre-compute k2 cache once (reuse for all k values)
k2_cache = precompute_k2_values(k2abs_arr, alpha, lp, phi, eps, Vexp, s=s)
# Cache uses (i, j) index tuples for k2abs_arr[i] * exp(1j * alpha[j])

# Compute RHS_S for many k values efficiently
results = []
for k in k_arr:
    result = RHS_S_with_k2_cache(k, lp, phi, eps, Vexp, k2_cache, k2abs_arr, alpha)
    results.append(result)
```

### Direct 3-body Calculation with Caching
```python
from struct_optimized import compute_3bod_optimized, compute_S, precompute_k2_values
from struct_aux import L, Vexp
from scipy.linalg import eig, inv

k1 = 1.0
k2_values = [1.0, 1.5, 2.0, 2.5]
lp, phi, eps = 1.0, 0.5, 1e-2
s = 10

# Pre-compute k1 values once
L1 = L(k1, lp, phi, eps, Vexp)
S1 = compute_S(L1, lp, s=s)
l1, P1 = eig(L1)
Q1 = inv(P1)

# Pre-compute all k2 values once
k2_cache = {}
for k2 in k2_values:
    L2 = L(k2, lp, phi, eps, Vexp)
    S2 = compute_S(L2, lp, s=s)
    l2, P2 = eig(L2)
    Q2 = inv(P2)
    k2_cache[k2] = {'L2': L2, 'S2': S2, 'l2': l2, 'P2': P2, 'Q2': Q2}

# Note: For production use with grid k2 values, use precompute_k2_values()
# which uses (i,j) index tuples to avoid floating point comparison issues

# Compute 3-body factors efficiently
results = []
for k2 in k2_values:
    k2_data = k2_cache[k2]
    S3 = compute_3bod_optimized(
        k1, k2, lp, phi, eps, Vexp, s=s,
        L1=L1, S1=S1, l1=l1, P1=P1, Q1=Q1,
        L2=k2_data['L2'], S2=k2_data['S2'], 
        l2=k2_data['l2'], P2=k2_data['P2'], Q2=k2_data['Q2']
    )
    results.append(S3)
```

## Backward Compatibility

All optimized versions maintain backward compatibility with the original API:
- `struct_2b.py` and `struct_3b.py` work as drop-in replacements
- `struct_optimized.py` provides `compute_3bod` as an alias for backward compatibility
- Existing code will continue to work unchanged
- Optional parameters enable optimization when provided

## Testing

Run the test suite:
```bash
python /tmp/test_optimized.py
```

## Notes

1. The optimization is most effective when:
   - Calling `RHS_S` multiple times with the same k2 grid
   - Computing many (k1, k2) pairs where k1 or k2 values repeat
   
2. Memory vs Speed tradeoff:
   - k2 cache stores ~2000 × (6 items: k2 + 5 matrices) per entry
   - Cache uses (i, j) index tuples to avoid floating point comparison issues
   - Each matrix is 21×21 complex = ~3.5 KB
   - Total memory: ~40 MB per cache (acceptable for most systems)

3. For single isolated calculations, the overhead of caching may not provide benefits.

4. The original `struct_2b.py` and `struct_3b.py` have been updated to include Phase 1 optimizations while maintaining full API compatibility.
