"""Quick Welded Beam Tuning"""
import numpy as np
import sys
sys.path.insert(0, '.')
from dpasa import ConstrainedDPASAOptimizer
from benchmarks import WeldedBeamDesign

print('Quick Tuning: Welded Beam')
print('Best known:', WeldedBeamDesign.best_known)
print('='*50)

configs = [
    {'n_leaders': 30, 'n_followers': 200, 'n_cultures': 3, 'max_iter': 1500},
    {'n_leaders': 50, 'n_followers': 300, 'n_cultures': 3, 'max_iter': 1500},
    {'n_leaders': 80, 'n_followers': 400, 'n_cultures': 3, 'max_iter': 1500},
]

for config in configs:
    results = []
    for trial in range(3):
        opt = ConstrainedDPASAOptimizer(
            objective_func=WeldedBeamDesign.objective,
            constraint_func=WeldedBeamDesign.constraints,
            bounds=WeldedBeamDesign.bounds,
            n_leaders=config['n_leaders'],
            n_followers=config['n_followers'],
            n_cultures=config['n_cultures'],
            max_iterations=config['max_iter'],
            seed=trial
        )
        r = opt.optimize(verbose=False)
        if r['is_feasible']:
            results.append(r['best_fitness'])
    
    if results:
        best = min(results)
        gap = 100*(best/WeldedBeamDesign.best_known - 1)
        print(f"L={config['n_leaders']}, F={config['n_followers']}: Best={best:.4f} (Gap: {gap:.1f}%)")
    else:
        print(f"L={config['n_leaders']}, F={config['n_followers']}: No feasible")
