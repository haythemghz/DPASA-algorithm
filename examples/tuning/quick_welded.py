"""Quick Welded Beam test with SDM+"""
import numpy as np
import sys
sys.path.insert(0, '.')
from dpasa import AdvancedSDM
from benchmarks import WeldedBeamDesign

print('Quick Test: Welded Beam with SDM+')
print('Best known:', WeldedBeamDesign.best_known)
print('='*50)

results = []
for trial in range(3):
    opt = AdvancedSDM(
        objective_func=WeldedBeamDesign.objective,
        constraint_func=WeldedBeamDesign.constraints,
        bounds=WeldedBeamDesign.bounds,
        n_leaders=80,
        n_followers=400,
        max_iterations=500,
        seed=trial
    )
    r = opt.optimize(verbose=False)
    if r['is_feasible']:
        gap = 100*(r['best_fitness']/WeldedBeamDesign.best_known - 1)
        print(f"Trial {trial+1}: {r['best_fitness']:.4f} (Gap: {gap:.1f}%)")
        results.append(r['best_fitness'])
    else:
        print(f"Trial {trial+1}: Infeasible")

if results:
    print(f"\nBest: {min(results):.4f} (Gap: {100*(min(results)/WeldedBeamDesign.best_known-1):.1f}%)")
