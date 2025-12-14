"""
SDM Hyperparameter Tuning for Welded Beam
==========================================

Tests different SDM configurations to find optimal settings.
"""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dpasa import ConstrainedDPASAOptimizer
from benchmarks import WeldedBeamDesign
import itertools


def run_config(config: dict, n_trials: int = 3) -> dict:
    """Run SDM with given config and return statistics."""
    results = []
    
    for trial in range(n_trials):
        optimizer = ConstrainedDPASAOptimizer(
            objective_func=WeldedBeamDesign.objective,
            constraint_func=WeldedBeamDesign.constraints,
            bounds=WeldedBeamDesign.bounds,
            n_leaders=config['n_leaders'],
            n_followers=config['n_followers'],
            n_cultures=config['n_cultures'],
            max_iterations=config['max_iter'],
            seed=trial
        )
        
        result = optimizer.optimize(verbose=False)
        
        if result['is_feasible']:
            results.append(result['best_fitness'])
        else:
            results.append(float('inf'))
    
    feasible_results = [r for r in results if r != float('inf')]
    
    return {
        'config': config,
        'best': min(feasible_results) if feasible_results else float('inf'),
        'mean': np.mean(feasible_results) if feasible_results else float('inf'),
        'feasibility': len(feasible_results) / n_trials,
        'n_feasible': len(feasible_results)
    }


def main():
    print("="*70)
    print("SDM Hyperparameter Tuning for Welded Beam")
    print(f"Best known: {WeldedBeamDesign.best_known}")
    print("="*70)
    
    # Hyperparameter grid
    param_grid = {
        'n_leaders': [30, 50, 80],
        'n_followers': [200, 400, 600],
        'n_cultures': [3, 5, 8],
        'max_iter': [500, 1000],
    }
    
    # Generate all combinations
    keys = list(param_grid.keys())
    combinations = list(itertools.product(*[param_grid[k] for k in keys]))
    
    print(f"\nTesting {len(combinations)} configurations...")
    print("-"*70)
    
    all_results = []
    
    for i, combo in enumerate(combinations):
        config = dict(zip(keys, combo))
        
        print(f"\n[{i+1}/{len(combinations)}] Testing: L={config['n_leaders']}, "
              f"F={config['n_followers']}, C={config['n_cultures']}, "
              f"I={config['max_iter']}")
        
        result = run_config(config, n_trials=3)
        all_results.append(result)
        
        if result['best'] != float('inf'):
            gap = 100 * (result['best'] / WeldedBeamDesign.best_known - 1)
            print(f"  -> Best: {result['best']:.4f} (Gap: {gap:.1f}%), "
                  f"Feas: {result['n_feasible']}/3")
        else:
            print(f"  -> No feasible solutions found")
    
    # Sort by best result
    valid_results = [r for r in all_results if r['best'] != float('inf')]
    valid_results.sort(key=lambda x: x['best'])
    
    print("\n" + "="*70)
    print("TOP 5 CONFIGURATIONS")
    print("="*70)
    
    for i, r in enumerate(valid_results[:5]):
        gap = 100 * (r['best'] / WeldedBeamDesign.best_known - 1)
        print(f"\n{i+1}. Best: {r['best']:.4f} (Gap: {gap:.1f}%)")
        print(f"   Config: L={r['config']['n_leaders']}, "
              f"F={r['config']['n_followers']}, "
              f"C={r['config']['n_cultures']}, "
              f"Iter={r['config']['max_iter']}")
        print(f"   Mean: {r['mean']:.4f}, Feasibility: {100*r['feasibility']:.0f}%")
    
    if valid_results:
        best = valid_results[0]
        print("\n" + "="*70)
        print("RECOMMENDED CONFIGURATION")
        print("="*70)
        print(f"n_leaders = {best['config']['n_leaders']}")
        print(f"n_followers = {best['config']['n_followers']}")
        print(f"n_cultures = {best['config']['n_cultures']}")
        print(f"max_iterations = {best['config']['max_iter']}")
        print(f"\nExpected performance: {best['best']:.4f} "
              f"(Gap: {100*(best['best']/WeldedBeamDesign.best_known-1):.1f}%)")


if __name__ == "__main__":
    main()
