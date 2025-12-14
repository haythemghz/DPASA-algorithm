"""
Enhanced SDM Testing with Larger Populations
=============================================

Tests SDM with increased population sizes to see if results improve.
Focusing on Speed Reducer which had 0.57% gap (target: 0.45%).

Configurations tested:
- Standard: n_leaders=100, n_followers=800 (paper values)
- Enhanced: n_leaders=150, n_followers=1600 (2x followers)
- Maximum: n_leaders=200, n_followers=2000 (larger populations)
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dpasa import DPASAOptimizer
from benchmarks import SpringDesign, WeldedBeamDesign, SpeedReducerDesign


def test_with_config(problem_class, n_leaders: int, n_followers: int, 
                     n_trials: int = 5, max_iter: int = 1500):
    """
    Test SDM on an engineering problem with specific configuration.
    """
    print(f"\n  Config: n_leaders={n_leaders}, n_followers={n_followers}, max_iter={max_iter}")
    
    results = []
    best_solution = None
    best_fitness = float('inf')
    feasible_count = 0
    
    for trial in range(n_trials):
        np.random.seed(trial * 42)
        
        bounds_list = problem_class.bounds
        
        optimizer = DPASAOptimizer(
            objective_func=problem_class.objective,
            constraint_func=problem_class.constraints,
            dim=problem_class.n_vars,
            bounds=bounds_list,
            n_leaders=n_leaders,
            n_followers=n_followers,
            n_cultures=5,
            max_iterations=max_iter,
            elite_size=25,  # Larger elite archive
            seed=trial * 42
        )
        
        result = optimizer.optimize(verbose=False)
        
        best_x = result['best_solution']
        obj_value = problem_class.objective(best_x)
        constraints = problem_class.constraints(best_x)
        max_violation = np.max(np.maximum(0, constraints))
        is_feasible = max_violation <= 1e-6
        
        if is_feasible:
            feasible_count += 1
            if obj_value < best_fitness:
                best_fitness = obj_value
                best_solution = best_x
        
        results.append(obj_value)
        
        gap_pct = 100 * abs(obj_value - problem_class.best_known) / problem_class.best_known
        status = "✓" if is_feasible else "✗"
        print(f"    Trial {trial+1}: f = {obj_value:12.6f}  Gap: {gap_pct:5.2f}%  {status}")
    
    results = np.array(results)
    gap_best = 100 * abs(np.min(results) - problem_class.best_known) / problem_class.best_known
    
    return {
        'best': np.min(results),
        'mean': np.mean(results),
        'gap_best_pct': gap_best,
        'feasibility': feasible_count / n_trials,
        'best_solution': best_solution
    }


def main():
    print("=" * 70)
    print("ENHANCED SDM TESTING - LARGER POPULATIONS")
    print("=" * 70)
    
    # Configuration variants to test
    configs = [
        ("Standard (Paper)", 100, 800, 1000),
        ("Enhanced (2x)", 150, 1600, 1000),
        ("Maximum", 200, 2000, 1500),
    ]
    
    problems = [
        ("Spring Design", SpringDesign),
        ("Welded Beam", WeldedBeamDesign),
        ("Speed Reducer", SpeedReducerDesign),
    ]
    
    all_results = {}
    
    for prob_name, problem_class in problems:
        print(f"\n{'='*70}")
        print(f"PROBLEM: {prob_name}")
        print(f"Best Known: {problem_class.best_known}")
        print(f"{'='*70}")
        
        all_results[prob_name] = {}
        
        for config_name, n_leaders, n_followers, max_iter in configs:
            print(f"\n{config_name}:")
            result = test_with_config(
                problem_class, n_leaders, n_followers, 
                n_trials=5, max_iter=max_iter
            )
            all_results[prob_name][config_name] = result
            print(f"    --> Best: {result['best']:.6f} ({result['gap_best_pct']:.3f}% gap)")
    
    # Final summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY - BEST GAPS BY CONFIGURATION")
    print("=" * 70)
    print(f"{'Problem':<18} {'Paper Target':>12}", end="")
    for config_name, _, _, _ in configs:
        print(f" {config_name:>18}", end="")
    print()
    print("-" * 70)
    
    targets = {"Spring Design": 0.03, "Welded Beam": 0.47, "Speed Reducer": 0.45}
    
    for prob_name, _ in problems:
        print(f"{prob_name:<18} {targets[prob_name]:>11.2f}%", end="")
        for config_name, _, _, _ in configs:
            gap = all_results[prob_name][config_name]['gap_best_pct']
            print(f" {gap:>17.3f}%", end="")
        print()
    
    print("=" * 70)


if __name__ == "__main__":
    main()
