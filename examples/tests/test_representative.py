"""
Test SDM on Representative Benchmark Functions
===============================================

Tests SDM (Enhanced config: 150/1600) on one representative function
from each category of the 15 benchmark functions.

Categories:
- Unimodal/Separable: Sphere
- Unimodal/Ill-conditioned: Ellipsoid  
- Unimodal/Non-separable: Rosenbrock
- Discontinuous: Step
- Multimodal/Regular: Rastrigin
- Multimodal/Deceptive: Schwefel
- Multimodal/Many local minima: Ackley
- Multimodal/Product-separable: Griewank
"""

import numpy as np
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dpasa import DPASAOptimizer
from benchmarks import BENCHMARK_SUITE, get_function_characteristics


# One representative from each category
REPRESENTATIVE_FUNCTIONS = [
    "Sphere",      # Unimodal, Separable, Convex
    "Ellipsoid",   # Unimodal, Ill-conditioned
    "Rosenbrock",  # Unimodal, Non-separable, Narrow valley
    "Step",        # Discontinuous
    "Rastrigin",   # Multimodal, Regular
    "Schwefel",    # Multimodal, Deceptive
    "Ackley",      # Multimodal, Many local minima
    "Griewank",    # Multimodal, Product-separable
]


def test_representative_functions(n_trials: int = 3, dim: int = 10, max_iter: int = 300):
    """Test SDM Enhanced config on representative benchmark functions."""
    
    print("=" * 70)
    print("SDM REPRESENTATIVE BENCHMARK TEST")
    print(f"Config: 150 leaders, 1600 followers, {max_iter} iterations")
    print(f"Dimension: D={dim}, Trials: {n_trials}")
    print("=" * 70)
    
    all_results = {}
    
    for func_name in REPRESENTATIVE_FUNCTIONS:
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
        print(f"  Time:   {elapsed:.1f}s ({elapsed/n_trials:.1f}s/trial)")
        
        all_results[func_name] = {
            'best': np.min(trial_results),
            'mean': np.mean(trial_results),
            'std': np.std(trial_results),
            'characteristics': char
        }
    
    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY TABLE - SDM (150/1600) Results")
    print("=" * 70)
    print(f"{'Function':<15} {'Characteristics':<35} {'Best':>12}")
    print("-" * 65)
    
    for func_name, stats in all_results.items():
        print(f"{func_name:<15} {stats['characteristics']:<35} {stats['best']:>12.4e}")
    
    print("\n" + "=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)
    
    return all_results


if __name__ == "__main__":
    # Fast test: 3 trials, D=10, 300 iterations
    test_representative_functions(n_trials=3, dim=10, max_iter=300)
