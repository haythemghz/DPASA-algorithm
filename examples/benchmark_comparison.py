"""
Benchmark Comparison Example
=============================

Runs SDM on multiple benchmark functions and displays results.
"""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dpasa import DPASAOptimizer
from benchmarks import BENCHMARK_SUITE, get_function_characteristics


def run_benchmark_suite(n_trials: int = 5, dim: int = 10):
    """Run SDM on the full benchmark suite."""
    
    # Select representative functions
    test_functions = ["Sphere", "Rosenbrock", "Rastrigin", "Griewank", "Ackley"]
    
    print("=" * 70)
    print("SDM Benchmark Comparison")
    print(f"Dimension: {dim}, Trials per function: {n_trials}")
    print("=" * 70)
    
    results = {}
    
    for func_name in test_functions:
        if func_name not in BENCHMARK_SUITE:
            continue
            
        func, bounds, optimum = BENCHMARK_SUITE[func_name]
        characteristics = get_function_characteristics(func_name)
        
        print(f"\n{func_name} ({characteristics})")
        print("-" * 50)
        
        trial_results = []
        
        for trial in range(n_trials):
            optimizer = DPASAOptimizer(
                objective_func=func,
                dim=dim,
                bounds=bounds,
                max_iterations=300,
                seed=trial
            )
            result = optimizer.optimize(verbose=False)
            trial_results.append(result['best_fitness'])
        
        trial_results = np.array(trial_results)
        
        print(f"  Mean ± Std:  {np.mean(trial_results):.4e} ± {np.std(trial_results):.4e}")
        print(f"  Best:        {np.min(trial_results):.4e}")
        print(f"  Median:      {np.median(trial_results):.4e}")
        
        results[func_name] = trial_results.tolist()
    
    print("\n" + "=" * 70)
    print("Summary Table")
    print("=" * 70)
    print(f"{'Function':<15} {'Mean':>12} {'Std':>12} {'Best':>12}")
    print("-" * 51)
    
    for func_name, values in results.items():
        values = np.array(values)
        print(f"{func_name:<15} {np.mean(values):>12.4e} {np.std(values):>12.4e} {np.min(values):>12.4e}")
    
    return results


if __name__ == "__main__":
    run_benchmark_suite(n_trials=5, dim=10)
