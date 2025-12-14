"""
Constrained Engineering Optimization Example
=============================================

Uses ConstrainedDPASAOptimizer with Deb's feasibility rules.
"""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dpasa import ConstrainedDPASAOptimizer
from benchmarks import (
    WeldedBeamDesign, 
    SpringDesign,
)


def optimize_problem(problem_class, n_trials: int = 5, max_iter: int = 500):
    """Run constrained SDM on an engineering problem."""
    
    print(f"\n{'='*60}")
    print(f"{problem_class.name}")
    print(f"Variables: {problem_class.n_vars}, Constraints: {problem_class.n_constraints}")
    print(f"Best known: {problem_class.best_known}")
    print(f"{'='*60}")
    
    results = []
    best_solutions = []
    feasible_count = 0
    
    for trial in range(n_trials):
        optimizer = ConstrainedDPASAOptimizer(
            objective_func=problem_class.objective,
            constraint_func=problem_class.constraints,
            bounds=problem_class.bounds,
            n_leaders=50,
            n_followers=300,
            max_iterations=max_iter,
            seed=trial
        )
        
        result = optimizer.optimize(verbose=False)
        
        if result['is_feasible']:
            feasible_count += 1
            results.append(result['best_fitness'])
        else:
            results.append(float('inf'))
        
        best_solutions.append(result['best_solution'])
        
        status = "✓" if result['is_feasible'] else "✗"
        print(f"  Trial {trial+1}: f = {result['best_fitness']:.6f} [{status}]")
    
    # Statistics for feasible solutions only
    feasible_results = [r for r in results if r != float('inf')]
    
    print(f"\nStatistics ({n_trials} runs):")
    if feasible_results:
        print(f"  Mean (feasible): {np.mean(feasible_results):.6f}")
        print(f"  Best:            {np.min(feasible_results):.6f}")
        print(f"  Best known:      {problem_class.best_known:.6f}")
        print(f"  Gap:             {100*(np.min(feasible_results)/problem_class.best_known - 1):.2f}%")
    print(f"  Feasibility:     {feasible_count}/{n_trials} ({100*feasible_count/n_trials:.1f}%)")
    
    return results


def main():
    print("="*60)
    print("SDM Constrained Engineering Optimization")
    print("Using Deb's Feasibility Rules")
    print("="*60)
    
    # Run on engineering problems
    problems = [SpringDesign, WeldedBeamDesign]
    
    all_results = {}
    for problem in problems:
        results = optimize_problem(problem, n_trials=5, max_iter=500)
        all_results[problem.name] = results
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"{'Problem':<18} {'Best Known':>12} {'SDM Best':>12} {'Gap':>10}")
    print("-"*55)
    
    for problem in problems:
        results = [r for r in all_results[problem.name] if r != float('inf')]
        if results:
            best = min(results)
            gap = 100 * (best / problem.best_known - 1)
            print(f"{problem.name:<18} {problem.best_known:>12.4f} {best:>12.4f} {gap:>9.1f}%")
        else:
            print(f"{problem.name:<18} {problem.best_known:>12.4f} {'N/A':>12} {'N/A':>10}")


if __name__ == "__main__":
    main()
