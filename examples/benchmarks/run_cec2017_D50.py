"""
CEC 2017-Style Benchmark Runner for SDM (D=50)
==============================================

Runs SDM on CEC 2017-style benchmark functions at D=50.
Sequential execution to avoid multiprocessing issues.

Configuration:
- Dimension: 50
- Runs: 51 (competition standard)
- Population: Np=150, Nf=1600
"""

import sys
import os
import numpy as np
from datetime import datetime
import time

# Add parent directories to path
script_dir = os.path.dirname(os.path.abspath(__file__))
examples_dir = os.path.dirname(script_dir)
root_dir = os.path.dirname(examples_dir)
sys.path.insert(0, root_dir)

from dpasa import DPASAOptimizer

# ============================================================================
# CEC 2017-Style Benchmark Functions
# These functions support BOTH single vector and batch (2D array) inputs
# ============================================================================

def ensure_2d(x):
    """Ensure x is 2D for vectorized operations"""
    x = np.atleast_2d(x)
    return x

def f1_sphere(x):
    """F1: Sphere (Unimodal)"""
    x = ensure_2d(x)
    result = np.sum(x**2, axis=1)
    return result[0] if len(result) == 1 else result

def f2_elliptic(x):
    """F2: High Conditioned Elliptic"""
    x = ensure_2d(x)
    D = x.shape[1]
    powers = np.arange(D) / max(D - 1, 1) * 6
    result = np.sum((10**powers) * x**2, axis=1)
    return result[0] if len(result) == 1 else result

def f3_bent_cigar(x):
    """F3: Bent Cigar"""
    x = ensure_2d(x)
    result = x[:, 0]**2 + 1e6 * np.sum(x[:, 1:]**2, axis=1)
    return result[0] if len(result) == 1 else result

def f4_discus(x):
    """F4: Discus"""
    x = ensure_2d(x)
    result = 1e6 * x[:, 0]**2 + np.sum(x[:, 1:]**2, axis=1)
    return result[0] if len(result) == 1 else result

def f5_rosenbrock(x):
    """F5: Rosenbrock"""
    x = ensure_2d(x)
    D = x.shape[1]
    result = np.zeros(x.shape[0])
    for i in range(D - 1):
        result += 100 * (x[:, i+1] - x[:, i]**2)**2 + (x[:, i] - 1)**2
    return result[0] if len(result) == 1 else result

def f6_ackley(x):
    """F6: Ackley"""
    x = ensure_2d(x)
    D = x.shape[1]
    term1 = -20 * np.exp(-0.2 * np.sqrt(np.sum(x**2, axis=1) / D))
    term2 = -np.exp(np.sum(np.cos(2 * np.pi * x), axis=1) / D)
    result = term1 + term2 + 20 + np.e
    return result[0] if len(result) == 1 else result

def f7_griewank(x):
    """F7: Griewank"""
    x = ensure_2d(x)
    D = x.shape[1]
    term1 = np.sum(x**2, axis=1) / 4000
    indices = np.sqrt(np.arange(1, D + 1))
    term2 = np.prod(np.cos(x / indices), axis=1)
    result = term1 - term2 + 1
    return result[0] if len(result) == 1 else result

def f8_rastrigin(x):
    """F8: Rastrigin"""
    x = ensure_2d(x)
    D = x.shape[1]
    result = 10 * D + np.sum(x**2 - 10 * np.cos(2 * np.pi * x), axis=1)
    return result[0] if len(result) == 1 else result

def f9_schwefel(x):
    """F9: Schwefel"""
    x = ensure_2d(x)
    D = x.shape[1]
    z = x + 420.9687462275036
    
    result = np.zeros(x.shape[0])
    for i in range(x.shape[0]):
        inner_sum = 0
        for j in range(D):
            zj = z[i, j]
            if abs(zj) <= 500:
                inner_sum += zj * np.sin(np.sqrt(abs(zj)))
            elif zj > 500:
                inner_sum += (500 - zj % 500) * np.sin(np.sqrt(abs(500 - zj % 500)))
            else:
                inner_sum += (zj % 500 - 500) * np.sin(np.sqrt(abs(zj % 500 - 500)))
        result[i] = 418.9829 * D - inner_sum
    
    return result[0] if len(result) == 1 else result

def f10_levy(x):
    """F10: Levy"""
    x = ensure_2d(x)
    D = x.shape[1]
    w = 1 + (x - 1) / 4
    
    term1 = np.sin(np.pi * w[:, 0])**2
    term2 = np.sum((w[:, :-1] - 1)**2 * (1 + 10 * np.sin(np.pi * w[:, :-1] + 1)**2), axis=1)
    term3 = (w[:, -1] - 1)**2 * (1 + np.sin(2 * np.pi * w[:, -1])**2)
    
    result = term1 + term2 + term3
    return result[0] if len(result) == 1 else result

def f11_happycat(x):
    """F11: HappyCat"""
    x = ensure_2d(x)
    D = x.shape[1]
    alpha = 1/8
    sum_sq = np.sum(x**2, axis=1)
    sum_x = np.sum(x, axis=1)
    result = np.abs(sum_sq - D)**(2*alpha) + (0.5*sum_sq + sum_x)/D + 0.5
    return result[0] if len(result) == 1 else result

def f12_hgbat(x):
    """F12: HGBat"""
    x = ensure_2d(x)
    D = x.shape[1]
    sum_sq = np.sum(x**2, axis=1)
    sum_x = np.sum(x, axis=1)
    result = np.abs(sum_sq**2 - sum_x**2)**0.5 + (0.5*sum_sq + sum_x)/D + 0.5
    return result[0] if len(result) == 1 else result

# Function registry
BENCHMARK_FUNCTIONS = [
    ('F1_Sphere', f1_sphere),
    ('F2_Elliptic', f2_elliptic),
    ('F3_BentCigar', f3_bent_cigar),
    ('F4_Discus', f4_discus),
    ('F5_Rosenbrock', f5_rosenbrock),
    ('F6_Ackley', f6_ackley),
    ('F7_Griewank', f7_griewank),
    ('F8_Rastrigin', f8_rastrigin),
    ('F9_Schwefel', f9_schwefel),
    ('F10_Levy', f10_levy),
    ('F11_HappyCat', f11_happycat),
    ('F12_HGBat', f12_hgbat),
]

# ============================================================================
# Main Execution
# ============================================================================

def run_benchmark(func_name, func, dim, n_runs, max_iter):
    """Run benchmark for a single function"""
    bounds = [(-100, 100)] * dim
    results = []
    
    for trial in range(n_runs):
        seed = trial * 42 + hash(func_name) % 10000
        
        try:
            optimizer = DPASAOptimizer(
                objective_func=func,
                bounds=bounds,
                n_leaders=150,
                n_followers=1600,
                n_cultures=5,
                max_iterations=max_iter,
                seed=seed
            )
            
            result = optimizer.optimize(verbose=False)
            results.append(result['best_fitness'])
            
            if (trial + 1) % 10 == 0:
                print(f"    Trial {trial+1}/{n_runs}: {result['best_fitness']:.4e}")
                
        except Exception as e:
            print(f"    Trial {trial+1} failed: {e}")
            results.append(float('inf'))
    
    return results

def main():
    # Configuration
    DIM = 50
    N_RUNS = 51
    
    # Calculate max iterations from MaxFES
    # MaxFES = 10^4 * D
    max_fes = 10000 * DIM
    evals_per_iter = 150 + 1600
    MAX_ITER = max_fes // evals_per_iter
    
    print("=" * 80)
    print("CEC 2017-STYLE BENCHMARK (D=50)")
    print("=" * 80)
    print(f"Dimension: D = {DIM}")
    print(f"Runs per function: {N_RUNS}")
    print(f"Population: Np=150, Nf=1600")
    print(f"Max Iterations: {MAX_ITER}")
    print(f"MaxFES: {max_fes:,}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    all_results = {}
    start_time = time.time()
    
    for idx, (func_name, func) in enumerate(BENCHMARK_FUNCTIONS):
        print(f"\n[{idx+1}/12] Running {func_name}...")
        
        func_start = time.time()
        results = run_benchmark(func_name, func, DIM, N_RUNS, MAX_ITER)
        func_time = time.time() - func_start
        
        valid_results = [r for r in results if r != float('inf')]
        
        if valid_results:
            best = np.min(valid_results)
            mean = np.mean(valid_results)
            std = np.std(valid_results)
            median = np.median(valid_results)
            
            all_results[func_name] = {
                'best': best, 'mean': mean, 'std': std, 
                'median': median, 'values': valid_results
            }
            
            print(f"  Result: Best={best:.4e} Mean={mean:.4e} Std={std:.4e} ({func_time/60:.1f}min)")
        else:
            print(f"  FAILED - all trials failed")
    
    total_time = time.time() - start_time
    
    # Save results
    output_file = f"cec2017_D{DIM}_results.txt"
    with open(output_file, 'w') as f:
        f.write(f"CEC 2017-Style Benchmark Results (D={DIM}, {N_RUNS} runs)\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total runtime: {total_time/60:.2f} minutes\n")
        f.write(f"Configuration: Np=150, Nf=1600, MaxIter={MAX_ITER}\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"{'Function':<20} {'Best':>15} {'Mean':>15} {'Std':>15} {'Median':>15}\n")
        f.write("-" * 80 + "\n")
        
        for name, data in all_results.items():
            f.write(f"{name:<20} {data['best']:>15.4e} {data['mean']:>15.4e} "
                   f"{data['std']:>15.4e} {data['median']:>15.4e}\n")
        
        f.write("-" * 80 + "\n")
    
    # Print summary
    print("\n" + "=" * 80)
    print("FINAL RESULTS SUMMARY")
    print("=" * 80)
    print(f"\n{'Function':<20} {'Best':>15} {'Mean':>15} {'Std':>15}")
    print("-" * 65)
    
    for name, data in all_results.items():
        print(f"{name:<20} {data['best']:>15.4e} {data['mean']:>15.4e} {data['std']:>15.4e}")
    
    print("-" * 65)
    print(f"\nResults saved to: {output_file}")
    print(f"Total runtime: {total_time/60:.2f} minutes")
    print("=" * 80)

if __name__ == "__main__":
    main()
