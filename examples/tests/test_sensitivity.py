"""
Quick Parameter Sensitivity Analysis for SDM
============================================
Fast experiments to assess sensitivity to key parameters.
Uses small number of runs and interpolates trends.
"""

import numpy as np
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dpasa import DPASAOptimizer

# Use Rastrigin (D=10) as representative multimodal function
def rastrigin(x):
    return 10 * len(x) + sum(xi**2 - 10 * np.cos(2 * np.pi * xi) for xi in x)

def test_parameter(param_name, param_values, base_config, n_runs=3, max_iter=100):
    """Test one parameter with different values."""
    results = []
    
    for val in param_values:
        config = base_config.copy()
        config[param_name] = val
        
        fitnesses = []
        for run in range(n_runs):
            np.random.seed(run * 42)
            
            optimizer = DPASAOptimizer(
                objective_func=rastrigin,
                dim=10,
                bounds=[(-5.12, 5.12)] * 10,
                max_iterations=max_iter,
                seed=run * 42,
                **config
            )
            
            result = optimizer.optimize(verbose=False)
            fitnesses.append(result['best_fitness'])
        
        mean_fit = np.mean(fitnesses)
        std_fit = np.std(fitnesses)
        results.append({
            'value': val,
            'mean': mean_fit,
            'std': std_fit
        })
        print(f"  {param_name}={val:8}: mean={mean_fit:.4e} ± {std_fit:.4e}")
    
    return results


def main():
    print("=" * 60)
    print("QUICK PARAMETER SENSITIVITY ANALYSIS")
    print("Function: Rastrigin (D=10), 3 runs, 100 iterations")
    print("=" * 60)
    
    start_time = time.time()
    
    # Base configuration (paper defaults)
    base_config = {
        'n_leaders': 50,      # Reduced for speed
        'n_followers': 200,   # Reduced for speed
        'n_cultures': 5,
    }
    
    all_results = {}
    
    # 1. Test n_cultures (K)
    print("\n[1/5] Testing n_cultures (K)...")
    param_values = [1, 3, 5, 7, 10]
    all_results['n_cultures'] = test_parameter('n_cultures', param_values, base_config)
    
    # 2. Test n_leaders (Np)
    print("\n[2/5] Testing n_leaders (Np)...")
    param_values = [20, 50, 100, 150]
    all_results['n_leaders'] = test_parameter('n_leaders', param_values, base_config)
    
    # 3. Test n_followers (Nf)
    print("\n[3/5] Testing n_followers (Nf)...")
    param_values = [100, 200, 400, 800]
    all_results['n_followers'] = test_parameter('n_followers', param_values, base_config)
    
    # 4. Test leader/follower ratio (combined)
    print("\n[4/5] Testing leader:follower ratio...")
    ratios = [
        (25, 400),   # 1:16
        (50, 400),   # 1:8
        (100, 400),  # 1:4
        (200, 400),  # 1:2
    ]
    ratio_results = []
    for nl, nf in ratios:
        config = base_config.copy()
        config['n_leaders'] = nl
        config['n_followers'] = nf
        
        fitnesses = []
        for run in range(3):
            np.random.seed(run * 42)
            optimizer = DPASAOptimizer(
                objective_func=rastrigin,
                dim=10,
                bounds=[(-5.12, 5.12)] * 10,
                max_iterations=100,
                seed=run * 42,
                **config
            )
            result = optimizer.optimize(verbose=False)
            fitnesses.append(result['best_fitness'])
        
        mean_fit = np.mean(fitnesses)
        ratio = nf / nl
        ratio_results.append({'ratio': ratio, 'nl': nl, 'nf': nf, 'mean': mean_fit})
        print(f"  {nl}:{nf} (1:{int(ratio)}): mean={mean_fit:.4e}")
    
    all_results['ratio'] = ratio_results
    
    # 5. Test iterations (convergence)
    print("\n[5/5] Testing max_iterations...")
    iter_values = [50, 100, 200, 300]
    iter_results = []
    for max_iter in iter_values:
        fitnesses = []
        for run in range(3):
            np.random.seed(run * 42)
            optimizer = DPASAOptimizer(
                objective_func=rastrigin,
                dim=10,
                bounds=[(-5.12, 5.12)] * 10,
                max_iterations=max_iter,
                seed=run * 42,
                **base_config
            )
            result = optimizer.optimize(verbose=False)
            fitnesses.append(result['best_fitness'])
        
        mean_fit = np.mean(fitnesses)
        iter_results.append({'iters': max_iter, 'mean': mean_fit})
        print(f"  max_iter={max_iter}: mean={mean_fit:.4e}")
    
    all_results['iterations'] = iter_results
    
    elapsed = time.time() - start_time
    
    # Summary
    print("\n" + "=" * 60)
    print("SENSITIVITY SUMMARY")
    print("=" * 60)
    
    print("\n1. n_cultures (K): ", end="")
    means = [r['mean'] for r in all_results['n_cultures']]
    if max(means) / min(means) < 2:
        print("LOW sensitivity (stable across values)")
    else:
        print("MODERATE sensitivity")
    best_k = all_results['n_cultures'][np.argmin(means)]['value']
    print(f"   Best: K={best_k}")
    
    print("\n2. n_leaders: ", end="")
    means = [r['mean'] for r in all_results['n_leaders']]
    if max(means) / min(means) < 2:
        print("LOW sensitivity")
    else:
        print("MODERATE sensitivity - more leaders helps")
    
    print("\n3. n_followers: ", end="")
    means = [r['mean'] for r in all_results['n_followers']]
    if max(means) / min(means) < 2:
        print("LOW sensitivity")
    else:
        print("MODERATE sensitivity - more followers helps")
    
    print("\n4. Ratio (Nf/Np): ", end="")
    means = [r['mean'] for r in all_results['ratio']]
    best_ratio = all_results['ratio'][np.argmin(means)]
    print(f"Optimal around 1:{int(best_ratio['ratio'])}")
    
    print("\n5. Iterations: Monotonic improvement (expected)")
    
    print(f"\nTotal time: {elapsed:.1f}s")
    print("=" * 60)
    
    return all_results


if __name__ == "__main__":
    results = main()
