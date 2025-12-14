
import numpy as np
import time
import sys
import os
sys.path.append(os.path.join(os.getcwd(), 'DPASA-algorithm'))
from dpasa import DPASAOptimizer
from benchmarks.engineering import PressureVesselDesign

class NoNegationDPASA(DPASAOptimizer):
    """
    Subclass that DISABLES the Negation operator to test if 'jitter' is the cause of failure.
    """
    def optimize(self, verbose: bool = True):
        # We need to copy the optimize method but REMOVE the Negation block.
        # To avoid giant code duplication in this script, we can hack the 'strategy_stagnation' update.
        # But 'optimize' is a monolithic function in the library.
        # So we MUST copy-paste it or use a monkey-patch.
        # Copy-paste is safer for a script.
        
        # ... (Identical initialization) ...
        step_scale = 0.4
        strategy_stagnation = np.zeros(self.n_strategies, dtype=int)
        self.current_fitness = np.zeros(self.n_candidates)
        self.current_violation = np.zeros(self.n_candidates)
        
        objs, viols, feass = self._evaluate_batch(self.candidates)
        self.current_fitness[:] = objs
        self.current_violation[:] = viols
        
        for i in range(self.n_candidates):
            self._update_elite(self.candidates[i], objs[i], viols[i])
            if self._compare_epsilon(objs[i], viols[i], self.best_fitness, self.best_violation):
                self.best_solution = self.candidates[i].copy()
                self.best_fitness = objs[i]
                self.best_violation = viols[i]
                self.best_is_feasible = feass[i]
        
        for iteration in range(self.max_iterations):
            tc = int(0.85 * self.max_iterations)
            if self.is_constrained and iteration < tc:
                self.epsilon = 1.0 * (1.0 - iteration / tc)
            else:
                self.epsilon = 0.0
            
            # ... Strategy Eval ...
            strategy_scores = []
            for strategy in self.strategies:
                sample_idx = np.random.choice(self.n_candidates, min(30, self.n_candidates), replace=False)
                sample = self.candidates[sample_idx]
                guided = strategy.guide_batch(sample)
                objs_g, viols_g, _ = self._evaluate_batch(guided)
                scores = objs_g + 1e6 * np.maximum(0, viols_g - self.epsilon)
                strategy_scores.append(np.mean(scores))
            strategy_scores = np.array(strategy_scores)
            elite_indices = np.argsort(strategy_scores)[:max(1, self.n_strategies // 8)]
            best_strategy = self.strategies[elite_indices[0]]
            
            # !!! MODIFICATION: DO NOT INCREMENT STAGNATION !!!
            # strategy_stagnation += 1  <-- DISABLED
            
            # ... Strategy Learning ...
            for i, strategy in enumerate(self.strategies):
                if i not in elite_indices:
                    lr, cr = self._get_adaptive_params()
                    strategy.a += step_scale * lr * (best_strategy.a - strategy.a) * np.random.rand(self.dim)
                    strategy.d += step_scale * lr * (best_strategy.d - strategy.d) * np.random.rand(self.dim)
                    if len(elite_indices) > 1 and np.random.rand() < cr:
                        random_elite = self.strategies[np.random.choice(elite_indices[1:])]
                        strategy.b += step_scale * 0.3 * (random_elite.b - strategy.b) * np.random.rand(self.dim)
                        strategy.c += step_scale * 0.3 * (random_elite.c - strategy.c) * np.random.rand(self.dim)
            
            # ... Candidate Guidance ...
            candidate_positions = self.candidates.copy()
            candidate_lrs = np.zeros(self.n_candidates)
            candidate_crs = np.zeros(self.n_candidates)
            for pid in range(self.n_partitions):
                c_indices = np.where(self.candidate_partition == pid)[0]
                if len(c_indices) == 0: continue
                partition_strategies = [s for s in self.strategies if s.partition_id == pid] or self.strategies
                active_strategy = np.random.choice(partition_strategies)
                current_pos = self.candidates[c_indices]
                guided = active_strategy.guide_batch(current_pos)
                lr, cr = self._get_adaptive_params()
                if self.elite_archive:
                    elite_mask = np.random.rand(len(c_indices)) < 0.15
                    if np.any(elite_mask):
                        n_updates = np.sum(elite_mask)
                        e_indices_rand = np.random.randint(0, len(self.elite_archive), n_updates)
                        elite_sols = np.array([self.elite_archive[ei][1] for ei in e_indices_rand])
                        guided[elite_mask] = (1 - lr) * guided[elite_mask] + lr * elite_sols
                perturb_mask = np.random.rand(len(c_indices)) < cr
                if np.any(perturb_mask):
                    local_std = step_scale * 0.08 * (self.ub - self.lb)
                    noise = np.random.normal(0, 1, (np.sum(perturb_mask), self.dim)) * local_std
                    guided[perturb_mask] += noise
                guided = np.clip(guided, self.lb, self.ub)
                candidate_positions[c_indices] = guided
                candidate_lrs[c_indices] = lr
                candidate_crs[c_indices] = cr
            
            # ... Evaluation ...
            cand_objs, cand_viols, cand_feass = self._evaluate_batch(candidate_positions)
            # ... (Selection Logic Identical) ...
            feas_cand = cand_viols <= self.epsilon
            feas_curr = self.current_violation <= self.epsilon
            both_feas = feas_cand & feas_curr
            cand_better_obj = cand_objs < self.current_fitness
            cand_better_viol = cand_viols < self.current_violation
            cand_feas_only = feas_cand & (~feas_curr)
            improved_mask = (cand_feas_only) | (both_feas & cand_better_obj) | ((~feas_cand) & (~feas_curr) & cand_better_viol)
            
            if np.any(improved_mask):
                self.candidates[improved_mask] = candidate_positions[improved_mask]
                self.current_fitness[improved_mask] = cand_objs[improved_mask]
                self.current_violation[improved_mask] = cand_viols[improved_mask]
                improved_indices = np.where(improved_mask)[0]
                for idx in improved_indices:
                    self._update_elite(candidate_positions[idx], cand_objs[idx], cand_viols[idx])
                    if self._compare_epsilon(cand_objs[idx], cand_viols[idx], self.best_fitness, self.best_violation):
                        self.best_solution = candidate_positions[idx].copy()
                        self.best_fitness = cand_objs[idx]
                        self.best_violation = cand_viols[idx]
                        self.best_is_feasible = cand_feass[idx]
                self._update_success_history(candidate_lrs[improved_mask].tolist(), candidate_crs[improved_mask].tolist())
            
            # ... Exchange ...
            if np.random.rand() < 0.05:
                # ... (Exchange logic) ...
                idx1, idx2 = np.random.choice(self.n_strategies, 2, replace=False)
                s1, s2 = self.strategies[idx1], self.strategies[idx2]
                s1.a, s2.a = s2.a.copy(), s1.a.copy() # etc
            
            # ... Elite Crossover ...
            if iteration % 10 == 0 and len(self.elite_archive) >= 2:
                for _ in range(5):
                    child = self._elite_crossover()
                    if child is not None:
                        obj, viol, feas = self._evaluate(child)
                        self._update_elite(child, obj, viol)
                        if self._compare_epsilon(obj, viol, self.best_fitness, self.best_violation):
                            self.best_solution = child.copy()
                            self.best_fitness = obj
                            self.best_violation = viol
                            self.best_is_feasible = feas
                            
            # ... Nelder-Mead (retain standard 50) ...
            if iteration % 50 == 0 and iteration > 0 and self.best_solution is not None:
                 improved = self._nelder_mead_search(self.best_solution, max_evals=50)
                 # ... update ...
                 obj, viol, feas = self._evaluate(improved)
                 self._update_elite(improved, obj, viol)
                 if self._compare_epsilon(obj, viol, self.best_fitness, self.best_violation):
                     self.best_solution = improved.copy()
                     self.best_fitness = obj
                     self.best_violation = viol
                     self.best_is_feasible = feas

            # !!! MODIFICATION: NEGATION BLOCK REMOVED !!!
            
            self.history['convergence'].append(self.best_fitness if self.best_is_feasible else float('inf'))
            step_scale *= 0.995
            
        return {
            'best_solution': self.best_solution,
            'best_fitness': self.best_fitness,
            'best_violation': self.best_violation
        }

def run_experiment():
    problem = PressureVesselDesign
    print(f"Goal: {problem.best_known}")
    print("\nMODE 3: No Negation (Testing Jitter Hypothesis)")
    
    # Run 1 run only for speed
    opt = NoNegationDPASA(
        objective_func=problem.objective,
        dim=problem.n_vars,
        bounds=problem.bounds,
        constraint_func=problem.constraints,
        n_strategies=150, n_candidates=1600, n_partitions=5, max_iterations=1000
    )
    r = opt.optimize(verbose=False)
    print(f"Result: {r['best_fitness']:.4f} (Viol: {r['best_violation']:.2e})")
    gap = 100*(r['best_fitness']/problem.best_known-1)
    print(f"Gap: {gap:.2f}%")

if __name__ == "__main__":
    run_experiment()
