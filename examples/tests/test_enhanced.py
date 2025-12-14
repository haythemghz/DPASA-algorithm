"""Test Enhanced SDM on Spring Design"""
import numpy as np
import sys
sys.path.insert(0, '.')
from dpasa import EnhancedConstrainedSDM
from benchmarks import SpringDesign

print('='*60)
print('Enhanced SDM Test: Spring Design')
print(f'Best known: {SpringDesign.best_known}')
print('='*60)

results = []
for trial in range(5):
    print(f"\nTrial {trial+1}/5:")
    opt = EnhancedConstrainedSDM(
        objective_func=SpringDesign.objective,
        constraint_func=SpringDesign.constraints,
        bounds=SpringDesign.bounds,
        n_leaders=80,
        n_followers=400,
        n_cultures=3,
        max_iterations=300,
        elite_size=10,
        stagnation_limit=30,
        local_search_freq=25,
        seed=trial
    )
    r = opt.optimize(verbose=True)
    
    if r['is_feasible']:
        results.append(r['best_fitness'])
        gap = 100*(r['best_fitness']/SpringDesign.best_known - 1)
        print(f"  Final: {r['best_fitness']:.6f} (Gap: {gap:.1f}%)")
    else:
        print("  Final: Infeasible")

print('\n' + '='*60)
print('RESULTS SUMMARY')
print('='*60)
if results:
    print(f"Best:   {min(results):.6f}")
    print(f"Mean:   {np.mean(results):.6f}")
    print(f"Std:    {np.std(results):.6f}")
    print(f"Gap:    {100*(min(results)/SpringDesign.best_known - 1):.1f}%")
    print(f"CV:     {100*np.std(results)/np.mean(results):.1f}% (variance coefficient)")
