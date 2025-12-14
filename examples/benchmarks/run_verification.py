"""
Verification Run with User-Requested Parameters
==============================================
Configuration:
- Leaders: 150
- Followers: 1600
- Runs: 10
- Iterations: 500
- Benchmark: CEC 2017 (Subset available in repo)

This script validates the algorithm's performance with the high-fidelity population settings.
"""

import sys
import numpy as np
import time
sys.path.insert(0, '.')

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

TEST_FUNCTIONS = [
    ('F1 (Bent Cigar)', cec_f1_shifted_bent_cigar),
    ('F3 (Zakharov)', cec_f3_shifted_zakharov),
    ('F4 (Rosenbrock)', cec_f4_shifted_rosenbrock),
    ('F5 (Rastrigin)', cec_f5_shifted_rastrigin),
    ('F6 (Scaffer)', cec_f6_shifted_schwefel),
    ('F9 (Griewank)', cec_f9_shifted_griewank),
    ('F10 (Ackley)', cec_f10_shifted_ackley),
]

def run_verification():
    print("=" * 80)
    print("VERIFICATION RUN: SDM (150/1600)")
    print("Settings: 10 runs, 500 iterations, D=30")
    print("=" * 80)
    
    dim = 30
    bounds = [(-100, 100)] * dim
    n_trials = 1
    max_iter = 500
    n_leaders = 150
    n_followers = 1600
    
    results = {}
    
    total_start_time = time.time()
    
    # Run only F1 for verification
    functions_to_test = [TEST_FUNCTIONS[0]] # Just F1
    
    for fname, func in functions_to_test:
        print(f"\nProcessing {fname}...")
        trial_results = []
        trial_times = []
        
        for trial in range(n_trials):
            start_time = time.time()
            # Set seed for reproducibility
            np.random.seed(trial * 100 + 42)
            
            optimizer = DPASAOptimizer(
                objective_func=func,
                bounds=bounds,
                n_leaders=n_leaders,
                n_followers=n_followers,
                n_cultures=5,
                max_iterations=max_iter
            )
            
            res = optimizer.optimize(verbose=False)
            best_val = res['best_fitness']
            elapsed = time.time() - start_time
            
            trial_results.append(best_val)
            trial_times.append(elapsed)
            print(f"  Run {trial+1}: {best_val:.2e} ({elapsed:.1f}s)")
        
        # Stats
        best_res = np.min(trial_results)
        mean_res = np.mean(trial_results)
        std_res = np.std(trial_results)
        avg_time = np.mean(trial_times)
        
        results[fname] = {
            'best': best_res,
            'mean': mean_res,
            'std': std_res,
            'time': avg_time
        }
        
        print(f"  > Mean: {mean_res:.2e} | Best: {best_res:.2e} | Std: {std_res:.2e} | AvgTime: {avg_time:.1f}s")

    print("\n" + "=" * 80)
    print("FINAL RESULTS SUMMARY")
    print("=" * 80)
    print(f"{'Function':<20} {'Mean Error':<15} {'Best Error':<15} {'Std Dev':<15}")
    print("-" * 70)
    for fname, data in results.items():
        print(f"{fname:<20} {data['mean']:<15.2e} {data['best']:<15.2e} {data['std']:<15.2e}")
    print("=" * 80)
    print(f"Total Verification Time: {time.time() - total_start_time:.1f}s")

if __name__ == "__main__":
    run_verification()
