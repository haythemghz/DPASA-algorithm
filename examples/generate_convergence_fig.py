"""
Generate Convergence Figure Using Actual SDM Optimizer
======================================================
Uses the real DPASAOptimizer with wrapper for history collection.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dpasa import DPASAOptimizer

# Set style
plt.rcParams.update({
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 11,
    'legend.fontsize': 9,
    'font.family': 'serif',
    'figure.dpi': 150
})

# Benchmark functions with history tracking
class FunctionWrapper:
    """Wrapper to track convergence history during optimization."""
    def __init__(self, func):
        self.func = func
        self.best_so_far = float('inf')
        self.history = []
        self.call_count = 0
        self.record_interval = 10  # Record every N evaluations
        
    def __call__(self, x):
        result = self.func(x)
        if result < self.best_so_far:
            self.best_so_far = result
        self.call_count += 1
        if self.call_count % self.record_interval == 0:
            self.history.append(self.best_so_far)
        return result
    
    def reset(self):
        self.best_so_far = float('inf')
        self.history = []
        self.call_count = 0

# Define functions
def sphere(x):
    return np.sum(x**2)

def rastrigin(x):
    return 10 * len(x) + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))

def schwefel(x):
    return 418.9829 * len(x) - np.sum(x * np.sin(np.sqrt(np.abs(x))))

def rosenbrock(x):
    return sum(100 * (x[i+1] - x[i]**2)**2 + (x[i] - 1)**2 for i in range(len(x)-1))

def griewank(x):
    return 1 + np.sum(x**2) / 4000 - np.prod(np.cos(x / np.sqrt(np.arange(1, len(x)+1))))

def step_func(x):
    return np.sum(np.floor(x + 0.5)**2)


def run_with_history(func_class, dim, bounds, n_runs=5, max_iter=300):
    """Run SDM and collect convergence history."""
    all_histories = []
    
    for run in range(n_runs):
        wrapper = FunctionWrapper(func_class)
        
        optimizer = DPASAOptimizer(
            objective_func=wrapper,
            dim=dim,
            bounds=bounds,
            n_leaders=50,
            n_followers=200,
            n_cultures=5,
            max_iterations=max_iter,
            seed=run * 42 + 1
        )
        
        result = optimizer.optimize(verbose=False)
        
        # Ensure we have enough points
        history = wrapper.history
        if len(history) < max_iter:
            # Pad with final value
            history = history + [history[-1] if history else result['best_fitness']] * (max_iter - len(history))
        
        # Trim to max_iter
        history = history[:max_iter]
        
        all_histories.append(history)
    
    all_histories = np.array(all_histories)
    mean_hist = np.mean(all_histories, axis=0)
    std_hist = np.std(all_histories, axis=0)
    
    # Ensure monotonically decreasing
    for i in range(1, len(mean_hist)):
        mean_hist[i] = min(mean_hist[i], mean_hist[i-1])
    
    return mean_hist, std_hist


def main():
    print("=" * 60)
    print("GENERATING CONVERGENCE FIGURE (ACTUAL SDM)")
    print("=" * 60)
    
    dim = 10
    max_iter = 300
    n_runs = 3  # 3 runs for speed
    
    functions = [
        ("Sphere", sphere, [(-5.12, 5.12)] * dim),
        ("Rastrigin", rastrigin, [(-5.12, 5.12)] * dim),
        ("Schwefel", schwefel, [(-500, 500)] * dim),
        ("Rosenbrock", rosenbrock, [(-5, 10)] * dim),
        ("Griewank", griewank, [(-600, 600)] * dim),
        ("Step", step_func, [(-100, 100)] * dim),
    ]
    
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    axes = axes.flatten()
    
    colors = ['#2563EB', '#DC2626', '#059669', '#7C3AED', '#EA580C', '#0891B2']
    
    for idx, (name, func, bounds) in enumerate(functions):
        print(f"Running {name}...")
        
        mean_hist, std_hist = run_with_history(func, dim, bounds, n_runs, max_iter)
        
        iterations = np.arange(len(mean_hist))
        ax = axes[idx]
        
        # Handle log scale
        mean_plot = np.maximum(mean_hist, 1e-12)
        std_lower = np.maximum(mean_hist - std_hist, 1e-12)
        std_upper = np.maximum(mean_hist + std_hist, 1e-12)
        
        # Plot
        ax.semilogy(iterations, mean_plot, color=colors[idx], linewidth=2, label='Mean')
        ax.fill_between(iterations, std_lower, std_upper, 
                       alpha=0.3, color=colors[idx])
        
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Best Fitness')
        ax.set_title(name, fontweight='bold', fontsize=12)
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        ax.set_xlim(0, len(mean_hist)-1)
        
        print(f"  Final: {mean_plot[-1]:.4e}")
    
    plt.tight_layout()
    
    output_path = r'c:\Users\Dell\Desktop\The_Religion_Inspired_Metaheuristic__A_Sociocultural_Framework_for_Global_Optimization\real_fig_convergence_profiles.png'
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    print(f"\nSaved: {output_path}")
    plt.close()


if __name__ == "__main__":
    main()
