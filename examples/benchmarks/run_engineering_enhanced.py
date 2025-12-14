"""
Enhanced DPASA Run on Engineering Problems
========================================

Runs DPASA on 4 Constrained Engineering Design Problems with enhanced population settings:
- Strategies (Ns): 150
- Candidates (Nc): 1600
- Iterations: 1000
- Enhanced constraint handling (via DPASAOptimizer)

Problems:
1. Spring Design
2. Welded Beam Design
3. Speed Reducer Design
4. Three-Bar Truss Design
"""

import numpy as np
import sys
import os

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dpasa import DPASAOptimizer
from benchmarks import SpringDesign, WeldedBeamDesign, SpeedReducerDesign, ThreeBarTruss

def test_problem(problem_class, n_trials=10, max_iter=1000):
    """
    Test DPASA on an engineering problem with enhanced settings.
    """
    print(f"\n{'='*70}")
    print(f"PROBLEM: {problem_class.name}")
    print(f"Dimensions: {problem_class.n_vars}, Constraints: {problem_class.n_constraints}")
    print(f"Best Known Solution: {problem_class.best_known:.6f}")
    print(f"{'='*70}")
    
    results = []
    
    
    # Wrappers for batch evaluation support
    original_obj = problem_class.objective
    original_con = problem_class.constraints
    
    def batch_objective(x):
        if x.ndim == 2:
            return np.array([original_obj(xi) for xi in x])
        return original_obj(x)
        
    def batch_constraints(x):
        if x.ndim == 2:
            return np.array([original_con(xi) for xi in x])
        return original_con(x)
    
    for trial in range(n_trials):
        np.random.seed(trial * 42 + 100) # Slightly different seeds
        
        optimizer = DPASAOptimizer(
            objective_func=batch_objective,
            constraint_func=batch_constraints,
            dim=problem_class.n_vars,
            bounds=problem_class.bounds,
            n_strategies=150,    # Enhanced: 100 -> 150
            n_candidates=1600,   # Enhanced: 800 -> 1600
            n_partitions=5,
            max_iterations=max_iter,
            elite_size=20,
            seed=trial * 42 + 100
        )
        
        result = optimizer.optimize(verbose=False)
        
        # Verify result quality
        best_x = result['best_solution']
        obj_value = problem_class.objective(best_x)
        constraints = problem_class.constraints(best_x)
        max_violation = np.max(np.maximum(0, constraints))
        is_feasible = max_violation <= 1e-5 # Tolerance
        
        if is_feasible:
            results.append(obj_value)
            status = "✓"
        else:
            # If infeasible, we can penalize or just note it. 
            # Ideally DPASA returns feasible solutions if possible.
            status = f"✗ (v={max_violation:.1e})"
            results.append(obj_value + 1e6) # Penalty for stats

        gap_pct = 100 * abs(obj_value - problem_class.best_known) / abs(problem_class.best_known)
        print(f"  Run {trial+1:2d}: f = {obj_value:12.6f}  Gap: {gap_pct:6.4f}%  {status}")

    results = np.array(results)
    best_val = np.min(results)
    mean_val = np.mean(results)
    std_val = np.std(results)
    
    best_gap = 100 * abs(best_val - problem_class.best_known) / abs(problem_class.best_known)
    mean_gap = 100 * abs(mean_val - problem_class.best_known) / abs(problem_class.best_known)
    
    print(f"\n--- Statistics ({n_trials} runs) ---")
    print(f"  Best: {best_val:.6f} (Gap: {best_gap:.4f}%)")
    print(f"  Mean: {mean_val:.6f} (Gap: {mean_gap:.4f}%)")
    print(f"  Std:  {std_val:.6f}")
    
    return {
        'name': problem_class.name,
        'best': best_val,
        'mean': mean_val,
        'std': std_val,
        'gap_best': best_gap,
        'gap_mean': mean_gap
    }

def main():
    print("ENHANCED ENGINEERING BENCHMARKS (Ns=150, Nc=1600, Iter=1000)")
    
    problems = [SpringDesign, WeldedBeamDesign, SpeedReducerDesign, ThreeBarTruss]
    summary = []
    
    for p in problems:
        res = test_problem(p, n_trials=10, max_iter=1000)
        summary.append(res)
        
    print("\n" + "="*80)
    print(f"{'Problem':<20} {'Best Known':<12} {'DPASA Best':<12} {'DPASA Mean':<12} {'Gap(%)':<8}")
    print("-" * 80)
    for i, s in enumerate(summary):
        bk = problems[i].best_known
        print(f"{s['name']:<20} {bk:<12.6f} {s['best']:<12.6f} {s['mean']:<12.6f} {s['gap_best']:<8.4f}")
    print("="*80)

if __name__ == "__main__":
    main()

========================================

Runs SDM on 4 Constrained Engineering Design Problems with enhanced population settings:
- Leaders (Np): 150
- Followers (Nf): 1600
- Iterations: 1000
- Enhanced constraint handling (via DPASAOptimizer)

Problems:
1. Spring Design
2. Welded Beam Design
3. Speed Reducer Design
4. Three-Bar Truss Design
"""

import numpy as np
import sys
import os

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dpasa import DPASAOptimizer
from benchmarks import SpringDesign, WeldedBeamDesign, SpeedReducerDesign, ThreeBarTruss

def test_problem(problem_class, n_trials=10, max_iter=1000):
    """
    Test SDM on an engineering problem with enhanced settings.
    """
    print(f"\n{'='*70}")
    print(f"PROBLEM: {problem_class.name}")
    print(f"Dimensions: {problem_class.n_vars}, Constraints: {problem_class.n_constraints}")
    print(f"Best Known Solution: {problem_class.best_known:.6f}")
    print(f"{'='*70}")
    
    results = []
    
    
    # Wrappers for batch evaluation support
    original_obj = problem_class.objective
    original_con = problem_class.constraints
    
    def batch_objective(x):
        if x.ndim == 2:
            return np.array([original_obj(xi) for xi in x])
        return original_obj(x)
        
    def batch_constraints(x):
        if x.ndim == 2:
            return np.array([original_con(xi) for xi in x])
        return original_con(x)
    
    for trial in range(n_trials):
        np.random.seed(trial * 42 + 100) # Slightly different seeds
        
        optimizer = DPASAOptimizer(
            objective_func=batch_objective,
            constraint_func=batch_constraints,
            dim=problem_class.n_vars,
            bounds=problem_class.bounds,
            n_leaders=150,       # Enhanced: 100 -> 150
            n_followers=1600,    # Enhanced: 800 -> 1600
            n_cultures=5,
            max_iterations=max_iter,
            elite_size=20,
            seed=trial * 42 + 100
        )
        
        result = optimizer.optimize(verbose=False)
        
        # Verify result quality
        best_x = result['best_solution']
        obj_value = problem_class.objective(best_x)
        constraints = problem_class.constraints(best_x)
        max_violation = np.max(np.maximum(0, constraints))
        is_feasible = max_violation <= 1e-5 # Tolerance
        
        if is_feasible:
            results.append(obj_value)
            status = "✓"
        else:
            # If infeasible, we can penalize or just note it. 
            # Ideally SDM returns feasible solutions if possible.
            # If it returns infeasible despite constraints, we count it as a failure or penalize.
            # SDM logic usually prioritizes feasibility.
            status = f"✗ (v={max_violation:.1e})"
            results.append(obj_value + 1e6) # Penalty for stats

        gap_pct = 100 * abs(obj_value - problem_class.best_known) / abs(problem_class.best_known)
        print(f"  Run {trial+1:2d}: f = {obj_value:12.6f}  Gap: {gap_pct:6.4f}%  {status}")

    results = np.array(results)
    best_val = np.min(results)
    mean_val = np.mean(results)
    std_val = np.std(results)
    
    best_gap = 100 * abs(best_val - problem_class.best_known) / abs(problem_class.best_known)
    mean_gap = 100 * abs(mean_val - problem_class.best_known) / abs(problem_class.best_known)
    
    print(f"\n--- Statistics ({n_trials} runs) ---")
    print(f"  Best: {best_val:.6f} (Gap: {best_gap:.4f}%)")
    print(f"  Mean: {mean_val:.6f} (Gap: {mean_gap:.4f}%)")
    print(f"  Std:  {std_val:.6f}")
    
    return {
        'name': problem_class.name,
        'best': best_val,
        'mean': mean_val,
        'std': std_val,
        'gap_best': best_gap,
        'gap_mean': mean_gap
    }

def main():
    print("ENHANCED ENGINEERING BENCHMARKS (Np=150, Nf=1600, Iter=1000)")
    
    problems = [SpringDesign, WeldedBeamDesign, SpeedReducerDesign, ThreeBarTruss]
    summary = []
    
    for p in problems:
        res = test_problem(p, n_trials=10, max_iter=1000)
        summary.append(res)
        
    print("\n" + "="*80)
    print(f"{'Problem':<20} {'Best Known':<12} {'SDM Best':<12} {'SDM Mean':<12} {'Gap(%)':<8}")
    print("-" * 80)
    for i, s in enumerate(summary):
        bk = problems[i].best_known
        print(f"{s['name']:<20} {bk:<12.6f} {s['best']:<12.6f} {s['mean']:<12.6f} {s['gap_best']:<8.4f}")
    print("="*80)

if __name__ == "__main__":
    main()
