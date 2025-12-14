"""
Basic Example - RIM Algorithm
Demonstrates simple usage on standard test functions
"""

from rim_enhanced import EnhancedRIM
import numpy as np
import time

# Define test functions
def sphere(x):
    """Unimodal test function"""
    return np.sum(x ** 2)

def rastrigin(x):
    """Multimodal test function"""
    return 10 * len(x) + np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x))

def main():
    print("="*60)
    print("RIM Algorithm - Basic Example")
    print("="*60)
    
    # Test on Sphere function
    print("\n1. Sphere Function (Unimodal)")
    print("-" * 60)
    
    start = time.time()
    optimizer = EnhancedRIM(
        objective_func=sphere,
        dim=10,
        bounds=(-5, 5),
        n_prophets=50,      # Moderate population
        n_followers=400,
        max_iterations=100,  # Quick demo
        seed=42
    )
    
    result = optimizer.optimize(verbose=False)
    elapsed = time.time() - start
    
    print(f"Best fitness: {result['best_fitness']:.6e}")
    print(f"Time: {elapsed:.2f}s")
    print(f"Evaluations: {result['history']['evaluation_count']}")
    
    # Test on Rastrigin function
    print("\n2. Rastrigin Function (Multimodal)")
    print("-" * 60)
    
    start = time.time()
    optimizer = EnhancedRIM(
        objective_func=rastrigin,
        dim=10,
        bounds=(-5, 5),
        n_prophets=50,
        n_followers=400,
        max_iterations=100,
        seed=42
    )
    
    result = optimizer.optimize(verbose=False)
    elapsed = time.time() - start
    
    print(f"Best fitness: {result['best_fitness']:.6e}")
    print(f"Time: {elapsed:.2f}s")
    print(f"Evaluations: {result['history']['evaluation_count']}")
    
    print("\n" + "="*60)
    print("Done! Try modifying parameters or functions above.")
    print("="*60)

if __name__ == "__main__":
    main()
