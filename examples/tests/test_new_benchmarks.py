"""
Test SDM on New Engineering Benchmarks
======================================

Tests SDM (Enhanced config: 150 leaders, 1600 followers) on:
- Three-Bar Truss (best known: 263.8958)
- Cantilever Beam (best known: 1.3399)
- Gear Train (best known: 2.7e-12)

Compares with literature baseline results from jSO, L-SHADE, DE, PSO, GA.
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dpasa import DPASAOptimizer
from benchmarks import ThreeBarTruss, CantileverBeam, GearTrainDesign


# Literature baseline results (from web search)
BASELINES = {
    "Three-Bar Truss": {
        "best_known": 263.8958,
        "GA": 263.9037,
        "PSO": 263.8959,
        "DE": 263.8958,
        "L-SHADE": 263.8958,  # Approximate
        "jSO": 263.8958,  # Approximate
    },
    "Cantilever Beam": {
        "best_known": 1.3399,
        "GA": 1.3400,
        "PSO": 1.3400,
        "DE": 1.3399,
        "L-SHADE": 1.3399,  # Approximate
        "jSO": 1.3399,  # Approximate
    },
    "Gear Train": {
        "best_known": 2.7e-12,
        "GA": 2.7e-12,
        "PSO": 2.7e-12,
        "DE": 2.7e-12,
        "L-SHADE": 2.7e-12,  # Approximate
        "jSO": 2.7e-12,  # Approximate
    },
}


def test_problem(problem_class, n_trials: int = 5, max_iter: int = 1000):
    """Test SDM with Enhanced config on a benchmark problem."""
    print(f"\n{'='*70}")
    print(f"PROBLEM: {problem_class.name}")
    print(f"Variables: {problem_class.n_vars}, Constraints: {problem_class.n_constraints}")
    print(f"Best Known: {problem_class.best_known:.6e}")
    print(f"{'='*70}")
    
    results = []
    solutions = []
    feasible_count = 0
    
    for trial in range(n_trials):
        np.random.seed(trial * 42)
        
        bounds_list = problem_class.bounds
        
        # Enhanced config: 150 leaders, 1600 followers
        optimizer = DPASAOptimizer(
            objective_func=problem_class.objective,
            constraint_func=problem_class.constraints if problem_class.n_constraints > 0 else None,
            dim=problem_class.n_vars,
            bounds=bounds_list,
            n_leaders=150,
            n_followers=1600,
            n_cultures=5,
            max_iterations=max_iter,
            elite_size=25,
            seed=trial * 42
        )
        
        result = optimizer.optimize(verbose=False)
        
        best_x = result['best_solution']
        obj_value = problem_class.objective(best_x)
        
        if problem_class.n_constraints > 0:
            constraints = problem_class.constraints(best_x)
            max_violation = np.max(np.maximum(0, constraints)) if len(constraints) > 0 else 0
            is_feasible = max_violation <= 1e-6
        else:
            is_feasible = True
            max_violation = 0
        
        if is_feasible:
            feasible_count += 1
        
        results.append(obj_value)
        solutions.append(best_x)
        
        gap_pct = 100 * abs(obj_value - problem_class.best_known) / max(abs(problem_class.best_known), 1e-12)
        status = "✓" if is_feasible else f"✗ (viol={max_violation:.2e})"
        print(f"  Trial {trial+1}: f = {obj_value:12.6e}  Gap: {gap_pct:8.4f}%  {status}")
    
    results = np.array(results)
    
    # Statistics
    print(f"\n--- SDM (150/1600) Statistics ---")
    print(f"  Best:        {np.min(results):12.6e}")
    print(f"  Mean:        {np.mean(results):12.6e}")
    print(f"  Std:         {np.std(results):12.6e}")
    print(f"  Feasibility: {feasible_count}/{n_trials} ({100*feasible_count/n_trials:.0f}%)")
    
    gap_best = 100 * abs(np.min(results) - problem_class.best_known) / max(abs(problem_class.best_known), 1e-12)
    print(f"  Best Gap:    {gap_best:.6f}%")
    
    return {
        'best': np.min(results),
        'mean': np.mean(results),
        'std': np.std(results),
        'gap_best_pct': gap_best,
        'feasibility': feasible_count / n_trials,
        'best_solution': solutions[np.argmin(results)]
    }


def main():
    print("=" * 70)
    print("SDM TESTING ON NEW ENGINEERING BENCHMARKS")
    print("Configuration: 150 leaders, 1600 followers, 1000 iterations")
    print("=" * 70)
    
    problems = [
        ("Three-Bar Truss", ThreeBarTruss),
        ("Cantilever Beam", CantileverBeam),
        ("Gear Train", GearTrainDesign),
    ]
    
    all_results = {}
    
    for name, problem_class in problems:
        result = test_problem(problem_class, n_trials=5, max_iter=1000)
        all_results[name] = result
    
    # Comparison table
    print("\n" + "=" * 70)
    print("COMPARISON WITH LITERATURE BASELINES")
    print("=" * 70)
    
    for name, problem_class in problems:
        print(f"\n{name} (Best Known: {problem_class.best_known:.6e})")
        print("-" * 50)
        print(f"{'Algorithm':<15} {'Best Result':>15} {'Gap %':>10}")
        print("-" * 50)
        
        baselines = BASELINES.get(name, {})
        for algo in ["GA", "PSO", "DE", "L-SHADE", "jSO"]:
            if algo in baselines:
                val = baselines[algo]
                gap = 100 * abs(val - problem_class.best_known) / max(abs(problem_class.best_known), 1e-12)
                print(f"{algo:<15} {val:>15.6e} {gap:>9.4f}%")
        
        sdm_best = all_results[name]['best']
        sdm_gap = all_results[name]['gap_best_pct']
        print(f"{'SDM (Ours)':<15} {sdm_best:>15.6e} {sdm_gap:>9.4f}%")
    
    print("\n" + "=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
