"""
Test SDM on Unimodal Benchmark Functions
=========================================

Tests SDM (Enhanced config: 150 leaders, 1600 followers) on 6 unimodal functions
to see if there's enhancement from the modified configuration.

Unimodal functions tested:
- Sphere (separable, convex)
- Ellipsoid (separable, ill-conditioned)
- Rosenbrock (non-separable, narrow valley)
- Quartic (noisy)
- Step (discontinuous)
- Zakharov (non-separable)
"""

import numpy as np
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dpasa import DPASAOptimizer
from benchmarks import BENCHMARK_SUITE, get_function_characteristics


# Unimodal functions from the 15 benchmarks
UNIMODAL_FUNCTIONS = ["Sphere", "Ellipsoid", "Rosenbrock", "Quartic", "Step", "Zakharov"]


def test_unimodal_functions(n_trials: int = 5, dim: int = 10, max_iter: int = 300):
    """Test SDM Enhanced config on unimodal benchmark functions."""
    
    print("=" * 70)
    print("SDM UNIMODAL BENCHMARK TEST")
    print(f"Config: 150 leaders, 1600 followers, {max_iter} iterations")
    print(f"Dimension: D={dim}, Trials: {n_trials}")
    print("=" * 70)
    
    all_results = {}
    
    for func_name in UNIMODAL_FUNCTIONS:
        if func_name not in BENCHMARK_SUITE:
            print(f"Warning: {func_name} not found in BENCHMARK_SUITE")
            continue
        
        func, bounds, optimum = BENCHMARK_SUITE[func_name]
        char = get_function_characteristics(func_name)
        
        print(f"\n{'='*70}")
        print(f"FUNCTION: {func_name}")
        print(f"Characteristics: {char}")
        print(f"Optimum: {optimum}")
        print("=" * 70)
        
        trial_results = []
        start_time = time.time()
        
        for trial in range(n_trials):
            np.random.seed(trial * 42)
            
            # Enhanced config: 150 leaders, 1600 followers
            optimizer = DPASAOptimizer(
                objective_func=func,
                dim=dim,
                bounds=bounds,
                n_leaders=150,
                n_followers=1600,
                n_cultures=5,
                max_iterations=max_iter,
                elite_size=25,
                seed=trial * 42
            )
            
            result = optimizer.optimize(verbose=False)
            best_fitness = result['best_fitness']
            trial_results.append(best_fitness)
            
            print(f"  Trial {trial+1}: {best_fitness:12.6e}")
        
        elapsed = time.time() - start_time
        trial_results = np.array(trial_results)
        
        # Statistics
        print(f"\n  --- Statistics ---")
        print(f"  Best:   {np.min(trial_results):12.6e}")
        print(f"  Mean:   {np.mean(trial_results):12.6e}")
        print(f"  Std:    {np.std(trial_results):12.6e}")
        print(f"  Time:   {elapsed:.1f}s ({elapsed/n_trials:.1f}s/trial)")
        
        all_results[func_name] = {
            'best': np.min(trial_results),
            'mean': np.mean(trial_results),
            'std': np.std(trial_results),
            'results': trial_results.tolist()
        }
    
    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY TABLE")
    print("=" * 70)
    print(f"{'Function':<15} {'Best':>15} {'Mean':>15} {'Std':>15}")
    print("-" * 60)
    
    for func_name, stats in all_results.items():
        print(f"{func_name:<15} {stats['best']:>15.6e} {stats['mean']:>15.6e} {stats['std']:>15.6e}")
    
    print("\n" + "=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)
    
    return all_results


if __name__ == "__main__":
    # Fast test: 5 trials, D=10, 300 iterations
    test_unimodal_functions(n_trials=5, dim=10, max_iter=300)
