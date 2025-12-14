"""
Basic DPASA Usage Example
=========================

Demonstrates how to use the DPASA optimizer on a simple optimization problem.
"""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dpasa import DPASAOptimizer


def sphere(x):
    """Sphere function - unimodal, separable."""
    return np.sum(x ** 2)


def rastrigin(x):
    """Rastrigin function - highly multimodal."""
    return 10 * len(x) + np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x))


def main():
    print("=" * 60)
    print("DPASA (Dual-Population Adversarial Strategy Adaptation) - Basic Example")
    print("=" * 60)
    
    # Example 1: Simple Sphere function
    print("\n1. Optimizing Sphere function (D=10)...")
    optimizer = DPASAOptimizer(
        objective_func=sphere,
        dim=10,
        bounds=(-5, 5),
        max_iterations=200,
        seed=42
    )
    
    result = optimizer.optimize(verbose=True)
    
    print(f"\nResults:")
    print(f"  Best fitness: {result['best_fitness']:.6e}")
    print(f"  Function evaluations: {result['history']['evaluation_count']}")
    
    # Example 2: Multimodal Rastrigin function
    print("\n" + "=" * 60)
    print("2. Optimizing Rastrigin function (D=10)...")
    
    optimizer = DPASAOptimizer(
        objective_func=rastrigin,
        dim=10,
        bounds=(-5, 5),
        max_iterations=300,
        seed=42
    )
    
    result = optimizer.optimize(verbose=True)
    
    print(f"\nResults:")
    print(f"  Best fitness: {result['best_fitness']:.6e}")
    print(f"  Best solution: {result['best_solution'][:3]}... (first 3 dims)")
    print(f"  Function evaluations: {result['history']['evaluation_count']}")


if __name__ == "__main__":
    main()
