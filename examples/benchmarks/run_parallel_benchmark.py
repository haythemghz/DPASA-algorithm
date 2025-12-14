"""
Parallel Verification Run (CEC 2017 Proxy for CEC 2024)
======================================================
Configuration:
- Leaders: 150
- Followers: 1600
- Runs: 10
- Iterations: 500
- Execution: Parallel (Multiprocessing)

This script runs the computationally intensive benchmark in parallel to reduce wait time.
"""

import sys
import os
import time
import numpy as np
import concurrent.futures

# Ensure we can import dpasa
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dpasa.algorithm import DPASAOptimizer
from examples.test_cec2017_fast import (
    cec_f1_shifted_bent_cigar,
    cec_f3_shifted_zakharov,
    cec_f4_shifted_rosenbrock,
    cec_f5_shifted_rastrigin,
    cec_f6_shifted_schwefel,
    cec_f9_shifted_griewank,
    cec_f10_shifted_ackley
)

# Test functions
TEST_FUNCTIONS = [
    ('F1 (Bent Cigar)', cec_f1_shifted_bent_cigar),
    ('F3 (Zakharov)', cec_f3_shifted_zakharov),
    ('F4 (Rosenbrock)', cec_f4_shifted_rosenbrock),
    ('F5 (Rastrigin)', cec_f5_shifted_rastrigin),
    ('F6 (Scaffer)', cec_f6_shifted_schwefel),
    ('F9 (Griewank)', cec_f9_shifted_griewank),
    ('F10 (Ackley)', cec_f10_shifted_ackley),
]

def run_single_trial(func, dim, bounds, n_leaders, n_followers, max_iter, seed):
    """Run a single optimization trial."""
    try:
        # Re-seed inside process
        np.random.seed(seed)
        
        optimizer = DPASAOptimizer(
            objective_func=func,
            bounds=bounds,
            n_leaders=n_leaders,
            n_followers=n_followers,
            n_cultures=5,  # Standard
            max_iterations=max_iter,
            seed=seed
        )
        result = optimizer.optimize(verbose=False)
        return result['best_fitness']
    except Exception as e:
        return float('inf')

def run_benchmark():
    print("=" * 80)
    print("PARALLEL BENCHMARK RUN: SDM (150/1600)")
    print("Settings: 51 runs, 500 iterations, D=30, Parallel Execution")
    print("=" * 80)
    
    dim = 30
    bounds = [(-100, 100)] * dim
    n_trials = 51
    max_iter = 500
    n_leaders = 150
    n_followers = 1600
    
    overall_results = {}
    total_start = time.time()
    
    # We will use ProcessPoolExecutor
    # Adjust max_workers based on machine (default is usually number of processors)
    with concurrent.futures.ProcessPoolExecutor() as executor:
        
        for fname, func in TEST_FUNCTIONS:
            print(f"\nSubmitting tasks for {fname}...")
            
            futures = []
            for trial in range(n_trials):
                seed = trial * 100 + 42
                f = executor.submit(
                    run_single_trial, 
                    func, dim, bounds, n_leaders, n_followers, max_iter, seed
                )
                futures.append(f)
            
            # Wait for results
            trial_results = []
            for i, future in enumerate(concurrent.futures.as_completed(futures)):
                res = future.result()
                trial_results.append(res)
                # Simple progress indicator
                print(f"  > Trial completed: {res:.2e} ({len(trial_results)}/{n_trials})")
            
            best_val = np.min(trial_results)
            mean_val = np.mean(trial_results)
            std_val = np.std(trial_results)
            
            overall_results[fname] = {
                'best': best_val,
                'mean': mean_val,
                'std': std_val
            }
            
            print(f"  >> {fname} COMPLETE. Mean: {mean_val:.2e}, Best: {best_val:.2e}")

    print("\n" + "=" * 80)
    print("FINAL BENCHMARK RESULTS")
    print("=" * 80)
    print(f"{'Function':<20} {'Mean Error':<15} {'Best Error':<15} {'Std Dev':<15}")
    print("-" * 70)
    
    for fname, res in overall_results.items():
        print(f"{fname:<20} {res['mean']:<15.2e} {res['best']:<15.2e} {res['std']:<15.2e}")
        
    print("=" * 80)
    print(f"Total Execution Time: {time.time() - total_start:.1f}s")

if __name__ == '__main__':
    run_benchmark()
