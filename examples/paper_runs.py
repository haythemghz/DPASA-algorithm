"""Extended runs for paper-quality results"""
import numpy as np
import sys
sys.path.insert(0, '.')
from dpasa import AdvancedSDM
from benchmarks import SpringDesign, WeldedBeamDesign, SpeedReducerDesign

problems = [
    ("Spring Design", SpringDesign),
    ("Welded Beam", WeldedBeamDesign),
    ("Speed Reducer", SpeedReducerDesign),
]

print("="*70)
print("SDM+ Extended Runs (10 trials, 1000 iterations)")
print("="*70)

all_results = {}

for name, problem in problems:
    print(f"\n{name} (Best known: {problem.best_known})")
    print("-"*50)
    
    results = []
    for trial in range(10):
        opt = AdvancedSDM(
            objective_func=problem.objective,
            constraint_func=problem.constraints,
            bounds=problem.bounds,
            n_leaders=100,
            n_followers=500,
            max_iterations=1000,
            elite_size=20,
            seed=trial
        )
        r = opt.optimize(verbose=False)
        
        if r['is_feasible']:
            results.append(r['best_fitness'])
            gap = 100*(r['best_fitness']/problem.best_known - 1)
            status = "OK" if gap < 1 else ""
            print(f"  Trial {trial+1:2d}: {r['best_fitness']:.6f} (Gap: {gap:.2f}%) {status}")
        else:
            print(f"  Trial {trial+1:2d}: Infeasible")
    
    if results:
        best = min(results)
        mean = np.mean(results)
        std = np.std(results)
        gap = 100*(best/problem.best_known - 1)
        all_results[name] = {
            'best': best,
            'mean': mean,
            'std': std,
            'gap': gap,
            'known': problem.best_known,
            'n_feasible': len(results)
        }
        print(f"\n  Summary: Best={best:.6f} Mean={mean:.6f} Std={std:.6f} Gap={gap:.2f}%")

print("\n" + "="*70)
print("FINAL RESULTS FOR PAPER")
print("="*70)
print(f"{'Problem':<18} {'Best Known':>12} {'SDM+ Best':>12} {'SDM+ Mean':>12} {'Gap':>8}")
print("-"*62)
for name, r in all_results.items():
    print(f"{name:<18} {r['known']:>12.4f} {r['best']:>12.4f} {r['mean']:>12.4f} {r['gap']:>7.2f}%")
