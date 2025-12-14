"""Quick Speed Reducer test with SDM+"""
import numpy as np
import sys
sys.path.insert(0, '.')
from dpasa import AdvancedSDM
from benchmarks import SpeedReducerDesign

print('Quick Test: Speed Reducer with SDM+')
print('Best known:', SpeedReducerDesign.best_known)
print('Variables:', SpeedReducerDesign.n_vars)
print('Constraints:', SpeedReducerDesign.n_constraints)
print('='*50)

results = []
for trial in range(3):
    opt = AdvancedSDM(
        objective_func=SpeedReducerDesign.objective,
        constraint_func=SpeedReducerDesign.constraints,
        bounds=SpeedReducerDesign.bounds,
        n_leaders=100,
        n_followers=500,
        max_iterations=500,
        seed=trial
    )
    r = opt.optimize(verbose=False)
    if r['is_feasible']:
        gap = 100*(r['best_fitness']/SpeedReducerDesign.best_known - 1)
        print(f"Trial {trial+1}: {r['best_fitness']:.2f} (Gap: {gap:.1f}%)")
        results.append(r['best_fitness'])
    else:
        print(f"Trial {trial+1}: Infeasible")

if results:
    print(f"\nBest: {min(results):.2f} (Gap: {100*(min(results)/SpeedReducerDesign.best_known-1):.1f}%)")
