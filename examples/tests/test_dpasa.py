"""Test DPASAOptimizer"""
import numpy as np
import sys
sys.path.insert(0, '.')
from dpasa import DPASAOptimizer

print("Testing DPASAOptimizer (unconstrained)")
opt = DPASAOptimizer(
    lambda x: np.sum(x**2),
    dim=5,
    bounds=(-5, 5),
    max_iterations=50
)
r = opt.optimize(verbose=False)
print(f"Unconstrained test: {r['best_fitness']:.6f}")

print("\nTesting DPASAOptimizer (constrained)")
from benchmarks import SpringDesign
opt2 = DPASAOptimizer(
    objective_func=SpringDesign.objective,
    constraint_func=SpringDesign.constraints,
    bounds=SpringDesign.bounds,
    max_iterations=100
)
r2 = opt2.optimize(verbose=False)
print(f"Spring Design: {r2['best_fitness']:.6f}")
