"""
Comprehensive SDM Testing on Engineering Problems
==================================================

Tests SDM on all 3 engineering design problems with paper-level rigor:
- Spring Design (3 vars, 4 constraints) - Best known: 0.012665
- Welded Beam (4 vars, 7 constraints) - Best known: 1.724852
- Speed Reducer (7 vars, 11 constraints) - Best known: 2994.471

This script runs 10 independent trials per problem and reports statistics.
"""

import numpy as np
import sys
import os

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dpasa import DPASAOptimizer
from benchmarks import SpringDesign, WeldedBeamDesign, SpeedReducerDesign


def test_problem(problem_class, n_trials: int = 10, max_iter: int = 1000):
    """
    Test SDM on an engineering problem with proper constrained handling.
    
    Uses the DPASAOptimizer's native constraint handling capabilities.
    """
    print(f"\n{'='*70}")
    print(f"PROBLEM: {problem_class.name}")
    print(f"Dimensions: {problem_class.n_vars}, Constraints: {problem_class.n_constraints}")
    print(f"Best Known Solution: {problem_class.best_known:.6f}")
    print(f"{'='*70}")
    
    results = []
    violations = []
    solutions = []
    feasible_count = 0
    
    for trial in range(n_trials):
        np.random.seed(trial * 42)  # Reproducible seeds
        
        # Use proper per-variable bounds
        bounds_list = problem_class.bounds
        
        # Create optimizer with constraint function
        optimizer = DPASAOptimizer(
            objective_func=problem_class.objective,
            constraint_func=problem_class.constraints,
            dim=problem_class.n_vars,
            bounds=bounds_list,  # Pass list of (lb, ub) tuples
            n_leaders=100,       # Paper values
            n_followers=800,
            n_cultures=5,
            max_iterations=max_iter,
            elite_size=20,
            seed=trial * 42
        )
        
        result = optimizer.optimize(verbose=False)
        
        # Final evaluation
        best_x = result['best_solution']
        obj_value = problem_class.objective(best_x)
        constraints = problem_class.constraints(best_x)
        max_violation = np.max(np.maximum(0, constraints))
        is_feasible = max_violation <= 1e-6
        
        results.append(obj_value)
        violations.append(max_violation)
        solutions.append(best_x)
        
        if is_feasible:
            feasible_count += 1
            status = "✓ Feasible"
        else:
            status = f"✗ Infeasible (viol={max_violation:.2e})"
        
        gap_pct = 100 * abs(obj_value - problem_class.best_known) / problem_class.best_known
        print(f"  Trial {trial+1:2d}: f = {obj_value:12.6f}  Gap: {gap_pct:6.2f}%  {status}")
    
    results = np.array(results)
    
    # Statistics
    print(f"\n--- Statistics ({n_trials} runs, {max_iter} iterations each) ---")
    print(f"  Best:       {np.min(results):12.6f}")
    print(f"  Mean:       {np.mean(results):12.6f}")
    print(f"  Std:        {np.std(results):12.6f}")
    print(f"  Worst:      {np.max(results):12.6f}")
    print(f"  Feasibility: {feasible_count}/{n_trials} ({100*feasible_count/n_trials:.0f}%)")
    
    # Gap analysis
    gap_best = 100 * abs(np.min(results) - problem_class.best_known) / problem_class.best_known
    gap_mean = 100 * abs(np.mean(results) - problem_class.best_known) / problem_class.best_known
    print(f"  Gap (Best): {gap_best:.4f}%")
    print(f"  Gap (Mean): {gap_mean:.4f}%")
    
    # Best solution details
    best_idx = np.argmin(results)
    best_x = solutions[best_idx]
    print(f"\n  Best solution vector:")
    for i, val in enumerate(best_x):
        lb, ub = problem_class.bounds[i]
        print(f"    x[{i}] = {val:.6f}  (bounds: [{lb:.2f}, {ub:.2f}])")
    
    return {
        'name': problem_class.name,
        'best_known': problem_class.best_known,
        'sdm_best': np.min(results),
        'sdm_mean': np.mean(results),
        'sdm_std': np.std(results),
        'gap_best_pct': gap_best,
        'gap_mean_pct': gap_mean,
        'feasibility_rate': feasible_count / n_trials,
        'best_solution': solutions[best_idx]
    }


def main():
    print("=" * 70)
    print("SDM ALGORITHM - ENGINEERING OPTIMIZATION BENCHMARKS")
    print("Testing on 3 Constrained Engineering Design Problems")
    print("=" * 70)
    
    # Test all three problems
    problems = [SpringDesign, WeldedBeamDesign, SpeedReducerDesign]
    
    all_results = []
    for problem in problems:
        result = test_problem(problem, n_trials=10, max_iter=1000)
        all_results.append(result)
    
    # Final summary table
    print("\n" + "=" * 70)
    print("FINAL SUMMARY TABLE")
    print("=" * 70)
    print(f"{'Problem':<18} {'Best Known':>12} {'SDM Best':>12} {'SDM Mean':>12} {'Gap%':>8}")
    print("-" * 70)
    
    for r in all_results:
        print(f"{r['name']:<18} {r['best_known']:>12.4f} {r['sdm_best']:>12.4f} {r['sdm_mean']:>12.4f} {r['gap_best_pct']:>7.3f}%")
    
    print("-" * 70)
    print("\nPaper Target Gaps: Spring ~0.03%, Welded ~0.47%, Speed ~0.45%")
    print("=" * 70)


if __name__ == "__main__":
    main()
