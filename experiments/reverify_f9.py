
import sys
import numpy as np
import time

# Add root directory to path to import dpasa
sys.path.insert(0, 'DPASA-algorithm')

from dpasa import DPASAOptimizer

def cec_f9_shifted_griewank(x):
    """F9: Shifted and Rotated Griewank (Multimodal)"""
    x = np.atleast_2d(x)
    z = x - 1
    D = z.shape[1]
    term1 = np.sum(z**2, axis=1) / 4000
    term2 = np.prod(np.cos(z / np.sqrt(np.arange(1, D+1))), axis=1)
    result = term1 - term2 + 1
    return result[0] if result.size == 1 else result

def run_f9_verification():
    with open("verification_f9.log", "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("VERIFICATION: F9 Shifted Griewank (CEC 2017)\n")
        f.write("=" * 80 + "\n")
        
        # Configuration
        dim = 30
        bounds = [(-100, 100)] * dim
        n_trials = 5
        max_iter = 300
        
        # DPASA parameters
        n_leaders = 150
        n_followers = 1600
        n_cultures = 5
        
        f.write(f"Dimension: {dim}\n")
        f.write(f"Population: {n_leaders} leaders + {n_followers} followers\n")
        f.write(f"Iterations: {max_iter}\n")
        f.write("-" * 50 + "\n")
        
        results = []
        
        for trial in range(n_trials):
            start_time = time.time()
            
            optimizer = DPASAOptimizer(
                objective_func=cec_f9_shifted_griewank,
                bounds=bounds,
                n_strategies=n_leaders,
                n_candidates=n_followers,
                n_partitions=n_cultures,
                max_iterations=max_iter,
                seed=42 + trial 
            )
            
            res = optimizer.optimize(verbose=False)
            runs_best = res['best_fitness']
            duration = time.time() - start_time
            
            results.append(runs_best)
            f.write(f"Trial {trial+1}: {runs_best:.4e} (Time: {duration:.2f}s)\n")
            f.flush()

        f.write("-" * 50 + "\n")
        mean_val = np.mean(results)
        std_val = np.std(results)
        min_val = np.min(results)
        
        f.write(f"Mean Error: {mean_val:.4e}\n")
        f.write(f"Std Dev:    {std_val:.4e}\n")
        f.write(f"Best Run:   {min_val:.4e}\n")
        f.write("=" * 80 + "\n")

if __name__ == "__main__":
    run_f9_verification()
