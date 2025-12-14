"""Quick Spring Design Tuning with Verbose Output"""
import numpy as np
import sys
sys.path.insert(0, '.')
from dpasa import ConstrainedDPASAOptimizer
from benchmarks import SpringDesign

print('='*60)
print('SDM Tuning: Spring Design (Best known: 0.012665)')
print('='*60)

# Configurations to test
configs = [
    {'n_leaders': 30, 'n_followers': 200, 'n_cultures': 3, 'max_iter': 300},
    {'n_leaders': 30, 'n_followers': 200, 'n_cultures': 5, 'max_iter': 300},
    {'n_leaders': 50, 'n_followers': 200, 'n_cultures': 3, 'max_iter': 300},
    {'n_leaders': 50, 'n_followers': 300, 'n_cultures': 3, 'max_iter': 300},
    {'n_leaders': 50, 'n_followers': 300, 'n_cultures': 5, 'max_iter': 300},
    {'n_leaders': 80, 'n_followers': 400, 'n_cultures': 3, 'max_iter': 300},
    {'n_leaders': 50, 'n_followers': 300, 'n_cultures': 3, 'max_iter': 500},
    {'n_leaders': 50, 'n_followers': 300, 'n_cultures': 3, 'max_iter': 800},
]

best_overall = float('inf')
best_config = None

for i, config in enumerate(configs):
    print(f"\n[{i+1}/{len(configs)}] L={config['n_leaders']}, F={config['n_followers']}, C={config['n_cultures']}, I={config['max_iter']}")
    
    results = []
    for trial in range(3):
        opt = ConstrainedDPASAOptimizer(
            objective_func=SpringDesign.objective,
            constraint_func=SpringDesign.constraints,
            bounds=SpringDesign.bounds,
            n_leaders=config['n_leaders'],
            n_followers=config['n_followers'],
            n_cultures=config['n_cultures'],
            max_iterations=config['max_iter'],
            seed=trial
        )
        r = opt.optimize(verbose=False)
        if r['is_feasible']:
            results.append(r['best_fitness'])
            print(f"   Trial {trial+1}: {r['best_fitness']:.6f}")
        else:
            print(f"   Trial {trial+1}: Infeasible")
    
    if results:
        best = min(results)
        mean = np.mean(results)
        gap = 100*(best/SpringDesign.best_known - 1)
        print(f"   -> Best: {best:.6f}, Mean: {mean:.6f}, Gap: {gap:.1f}%")
        
        if best < best_overall:
            best_overall = best
            best_config = config.copy()
            best_config['best'] = best
            best_config['gap'] = gap

print('\n' + '='*60)
print('BEST CONFIGURATION FOUND')
print('='*60)
if best_config:
    print(f"n_leaders = {best_config['n_leaders']}")
    print(f"n_followers = {best_config['n_followers']}")
    print(f"n_cultures = {best_config['n_cultures']}")
    print(f"max_iterations = {best_config['max_iter']}")
    print(f"\nBest: {best_config['best']:.6f} (Gap: {best_config['gap']:.1f}%)")
    print(f"Known: {SpringDesign.best_known}")
