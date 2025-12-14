
import time
import sys
import os
sys.path.append(os.getcwd())
import numpy as np
from dpasa import DPASAOptimizer
from benchmarks.classical import BenchmarkFunctions

# Mock baseline implementations for comparison
# (Since we might not have external libraries installed, we implement simple standard versions)

class SimplePSO:
    def __init__(self, obj_func, dim, bounds, pop_size=100, max_iter=300):
        self.func = obj_func
        self.dim = dim
        self.bounds = bounds
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.X = np.random.uniform(bounds[0], bounds[1], (pop_size, dim))
        self.V = np.zeros_like(self.X)
        self.P = self.X.copy()
        self.P_fit = np.array([self.func(x) for x in self.X])
        self.gbest_idx = np.argmin(self.P_fit)
        self.gbest = self.P[self.gbest_idx].copy()
        self.gbest_fit = self.P_fit[self.gbest_idx]
        
    def optimize(self):
        w, c1, c2 = 0.7, 1.5, 1.5
        for t in range(self.max_iter):
            r1, r2 = np.random.rand(self.pop_size, self.dim), np.random.rand(self.pop_size, self.dim)
            self.V = w * self.V + c1 * r1 * (self.P - self.X) + c2 * r2 * (self.gbest - self.X)
            self.X = self.X + self.V
            self.X = np.clip(self.X, self.bounds[0], self.bounds[1])
            fitness = np.array([self.func(x) for x in self.X])
            
            improved = fitness < self.P_fit
            self.P[improved] = self.X[improved]
            self.P_fit[improved] = fitness[improved]
            
            min_fit_idx = np.argmin(self.P_fit)
            if self.P_fit[min_fit_idx] < self.gbest_fit:
                self.gbest_fit = self.P_fit[min_fit_idx]
                self.gbest = self.P[min_fit_idx].copy()
        return self.gbest, self.gbest_fit

class SimpleDE:
    def __init__(self, obj_func, dim, bounds, pop_size=100, max_iter=300):
        self.func = obj_func
        self.dim = dim
        self.bounds = bounds
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.X = np.random.uniform(bounds[0], bounds[1], (pop_size, dim))
        self.fitness = np.array([self.func(x) for x in self.X])
        self.best_idx = np.argmin(self.fitness)
        self.best_fit = self.fitness[self.best_idx]
        
    def optimize(self):
        F, CR = 0.8, 0.9
        for t in range(self.max_iter):
            U = np.zeros_like(self.X)
            for i in range(self.pop_size):
                idxs = [idx for idx in range(self.pop_size) if idx != i]
                a, b, c = self.X[np.random.choice(idxs, 3, replace=False)]
                mutant = a + F * (b - c)
                mutant = np.clip(mutant, self.bounds[0], self.bounds[1])
                cross_points = np.random.rand(self.dim) < CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, self.X[i])
                f_trial = self.func(trial)
                if f_trial <= self.fitness[i]:
                    self.X[i] = trial
                    self.fitness[i] = f_trial
                    if f_trial < self.best_fit:
                        self.best_fit = f_trial
        return self.X[self.best_idx], self.best_fit

# Vectorized benchmark functions
def vec_sphere(x):
    # Handle single (D,) or batch (N, D)
    if x.ndim == 1:
        return np.sum(x ** 2)
    return np.sum(x ** 2, axis=1)

def vec_rastrigin(x):
    if x.ndim == 1:
        return 10 * len(x) + np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x))
    return 10 * x.shape[1] + np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x), axis=1)

def run_comparison():
    D = 30
    funcs = {
        "Sphere": vec_sphere,
        "Rastrigin": vec_rastrigin
    }
    
    print(f"Running Runtime Comparison (500 Iterations, D={D})")
    print(f"Running Runtime Comparison (500 Iterations, D={D})")
    
    with open("runtime_simple.txt", "w") as f:
        f.write(f"{'Algorithm':<10} | {'Func':<10} | {'Time(s)':<10} | {'Best Fitness':<15}\n")
        f.write("-" * 55 + "\n")

        for name, func in funcs.items():
            # 1. SDM (Optimized)
            start = time.time()
            sdm = DPASAOptimizer(func, dim=D, n_strategies=100, n_candidates=800, max_iterations=500, n_partitions=5) 
            res = sdm.optimize(verbose=False)
            sdm_fit = res['best_fitness']
            sdm_time = time.time() - start
            
            line_sdm = f"{'SDM':<10} | {name:<10} | {sdm_time:<10.4f} | {sdm_fit:<15.4e}\n"
            print(line_sdm, end='')
            f.write(line_sdm)
            
            # 2. PSO (Baseline)
            start = time.time()
            pso = SimplePSO(func, dim=D, bounds=(-100, 100), pop_size=900, max_iter=500)
            _, pso_fit = pso.optimize()
            pso_time = time.time() - start
            line_pso = f"{'PSO':<10} | {name:<10} | {pso_time:<10.4f} | {pso_fit:<15.4e}\n"
            print(line_pso, end='')
            f.write(line_pso)

            f.write("-" * 55 + "\n")

if __name__ == "__main__":
    run_comparison()

if __name__ == "__main__":
    run_comparison()
