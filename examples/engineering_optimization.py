"""
Engineering Optimization Example
=================================

Demonstrates DPASA on constrained engineering design problems.
"""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

from dpasa import DPASAOptimizer
from benchmarks import (
    WeldedBeamDesign, 
    SpringDesign,
    ENGINEERING_PROBLEMS
)


def optimize_engineering_problem(problem_class, n_trials: int = 5, max_iter: int = 300):
    """
    Run DPASA on an engineering optimization problem.
    
    Args:
        problem_class: Engineering problem class
        n_trials: Number of independent runs
        max_iter: Maximum iterations per run
    """
    print(f"\n{'='*60}")
    print(f"{problem_class.name}")
    print(f"Variables: {problem_class.n_vars}, Constraints: {problem_class.n_constraints}")
    print(f"Best known: {problem_class.best_known}")
    print(f"{'='*60}")
    
    # Convert bounds to tuple format
    bounds_array = np.array(problem_class.bounds)
    bounds = (bounds_array[:, 0].min(), bounds_array[:, 1].max())
    
    results = []
    best_solutions = []
    feasible_count = 0
    
    for trial in range(n_trials):
        # Create optimizer with custom bounds handling
        optimizer = DPASAOptimizer(
            objective_func=problem_class.evaluate,
            dim=problem_class.n_vars,
            bounds=bounds,
            n_strategies=50,
            n_candidates=400,
            max_iterations=max_iter,
            seed=trial
        )
        
        # Custom initialization within proper bounds
        for i, (lb, ub) in enumerate(problem_class.bounds):
            optimizer.candidates[:, i] = np.random.uniform(lb, ub, len(optimizer.candidates))
        
        result = optimizer.optimize(verbose=False)
        
        # Check feasibility
        best_x = result['best_solution']
        constraints = problem_class.constraints(best_x)
        is_feasible = np.all(constraints <= 1e-6)
        
        if is_feasible:
            feasible_count += 1
            obj_value = problem_class.objective(best_x)
        else:
            obj_value = result['best_fitness']  # Penalized value
        
        results.append(obj_value)
        best_solutions.append(best_x)
        
        print(f"  Trial {trial+1}: f = {obj_value:.6f}, Feasible: {is_feasible}")
    
    results = np.array(results)
    
    print(f"\nStatistics ({n_trials} runs):")
    print(f"  Mean:   {np.mean(results):.6f}")
    print(f"  Std:    {np.std(results):.6f}")
    print(f"  Best:   {np.min(results):.6f}")
    print(f"  Feasibility rate: {feasible_count}/{n_trials} ({100*feasible_count/n_trials:.1f}%)")
    
    # Best solution details
    best_idx = np.argmin(results)
    best_x = best_solutions[best_idx]
    print(f"\nBest solution found:")
    print(f"  Variables: {best_x}")
    print(f"  Objective: {problem_class.objective(best_x):.6f}")
    print(f"  Gap from best known: {abs(np.min(results) - problem_class.best_known):.6f}")
    
    return results


def main():
    print("="*60)
    print("DPASA on Engineering Optimization Problems")
    print("="*60)
    
    # Test on key problems
    problems = [SpringDesign, WeldedBeamDesign]
    
    all_results = {}
    
    for problem in problems:
        results = optimize_engineering_problem(problem, n_trials=5, max_iter=300)
        all_results[problem.name] = results
    
    # Summary table
    print("\n" + "="*60)
    print("SUMMARY TABLE")
    print("="*60)
    print(f"{'Problem':<20} {'Best Known':>12} {'DPASA Best':>12} {'DPASA Mean':>12}")
    print("-"*60)
    
    for problem in problems:
        results = all_results[problem.name]
        print(f"{problem.name:<20} {problem.best_known:>12.4f} {np.min(results):>12.4f} {np.mean(results):>12.4f}")


if __name__ == "__main__":
    main()
