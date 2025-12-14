
import sys
import os
import time
import numpy as np

# Ensure root is in path
sys.path.append(os.getcwd())

from cec2022_impl import cec2022
from dpasa import DPASAOptimizer

# Configuration
RUNS = 1
DIMS = [30, 50]
ITERS = 1000  # Standard
POP_SIZE = 100

def run_test():
    print("Testing D=30/50 Runtime for F1...")
    
    for D in DIMS:
        print(f"\n--- Dimension {D} ---")
        bench = cec2022.F12022(ndim=D)
        
        def objective_func(x):
            if x.ndim == 2:
                return np.array([bench.evaluate(xi) for xi in x])
            else:
                return bench.evaluate(x)

        # Initialize DPASA
        n_leaders = int(POP_SIZE * 0.2)
        n_followers = POP_SIZE - n_leaders
        
        optimizer = DPASAOptimizer(
            objective_func=objective_func,
            dim=D,
            bounds=(-100, 100),
            n_leaders=n_leaders,
            n_followers=n_followers,
            max_iterations=ITERS,
            seed=42
        )
        
        start = time.time()
        res = optimizer.optimize(verbose=False)
        end = time.time()
        
        print(f"D={D} | Time: {end-start:.4f}s | Best: {res['best_fitness']:.4e}")

if __name__ == "__main__":
    run_test()
