"""
Dual-Population Adversarial Strategy Adaptation (DPASA)
=======================================================

State-of-the-art DPASA optimizer with advanced features:
1. Success-history based parameter adaptation (like L-SHADE)
2. Nelder-Mead local search for precise convergence
3. Epsilon-constraint relaxation for constrained problems
4. Elite crossover
5. Ranked-based selection
"""

import numpy as np
import random
from typing import Callable, Tuple, List, Dict, Optional, Union
from scipy.optimize import minimize
from .strategy import StrategyVector


class DPASAOptimizer:
    """
    Dual-Population Adversarial Strategy Adaptation (DPASA) Optimizer.
    
    A novel population-based metaheuristic for global optimization.
    Supports both unconstrained and constrained optimization problems.
    
    Features:
    - Success-history adaptation: learns optimal parameters during search
    - Nelder-Mead local search: precise local convergence
    - Epsilon-constraint handling for constrained problems
    - Elite archive: preserves best solutions
    - Multi-partition parallel search
    
    Example (unconstrained):
        >>> def sphere(x): return np.sum(x ** 2)
        >>> optimizer = DPASAOptimizer(sphere, dim=10, bounds=(-5, 5))
        >>> result = optimizer.optimize()
    
    Example (constrained):
        >>> optimizer = DPASAOptimizer(
        ...     objective_func=problem.objective,
        ...     constraint_func=problem.constraints,
        ...     bounds=problem.bounds
        ... )
        >>> result = optimizer.optimize()
    """
    
    def __init__(
        self,
        objective_func: Callable,
        dim: int = None,
        bounds: Union[Tuple[float, float], List[Tuple[float, float]]] = (-5, 5),
        constraint_func: Callable = None,
        n_strategies: int = 100,
        n_candidates: int = 800,
        n_partitions: int = 5,
        max_iterations: int = 300,
        elite_size: int = 15,
        seed: int = None
    ):
        """
        Initialize DPASA optimizer.
        
        Args:
            objective_func: Function to minimize (required)
            dim: Problem dimension (inferred from bounds if list)
            bounds: Search bounds - either (lb, ub) tuple or list of (lb, ub)
            constraint_func: Constraint function returning g(x) where g(x)<=0 is feasible
            n_strategies: Number of strategies (guidance functions) (default: 100)
            n_candidates: Number of candidates (solutions) (default: 800)
            n_partitions: Number of parallel partitions (default: 5)
            max_iterations: Maximum iterations (default: 300)
            elite_size: Size of elite archive (default: 15)
            seed: Random seed for reproducibility
        """
        self.objective = objective_func
        self.constraints = constraint_func
        self.is_constrained = constraint_func is not None
        
        # Handle bounds
        if isinstance(bounds, list):
            self.bounds = bounds
            self.dim = dim if dim else len(bounds)
        else:
            # Tuple bounds, need dim
            self.dim = dim if dim else 10
            self.bounds = [(bounds[0], bounds[1]) for _ in range(self.dim)]
        
        self.n_strategies = n_strategies
        self.n_candidates = n_candidates
        self.n_partitions = n_partitions
        self.max_iterations = max_iterations
        self.elite_size = elite_size
        
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        
        self.lb = np.array([b[0] for b in self.bounds])
        self.ub = np.array([b[1] for b in self.bounds])
        
        # Success-history for adaptation
        self.success_history_lr = [0.4] * 20  # Learning rate history
        self.success_history_cr = [0.5] * 20  # Crossover rate history
        
        # Initialize with Latin Hypercube Sampling
        self.candidates = self._latin_hypercube_init(n_candidates)
        self.candidate_partition = np.random.choice(n_partitions, n_candidates)
        
        global_bounds = (self.lb.min(), self.ub.max())
        self.strategies = [
            StrategyVector(self.dim, global_bounds, i % n_partitions)
            for i in range(n_strategies)
        ]
        
        # Elite archive
        self.elite_archive = []
        
        # Best tracking
        self.best_solution = None
        self.best_fitness = float('inf')
        self.best_violation = 0.0
        self.best_is_feasible = True
        
        # Epsilon constraint - starts relaxed, tightens over time
        self.epsilon = 1.0 if self.is_constrained else 0.0
        
        self.history = {
            'convergence': [],
            'evaluation_count': 0,
            'local_search_improvements': 0
        }
    
    def _latin_hypercube_init(self, n: int) -> np.ndarray:
        """Latin Hypercube Sampling for better coverage."""
        pop = np.zeros((n, self.dim))
        for d in range(self.dim):
            perm = np.random.permutation(n)
            for i in range(n):
                pop[i, d] = self.lb[d] + (self.ub[d] - self.lb[d]) * (perm[i] + np.random.rand()) / n
        return pop
    
    def _repair(self, x: np.ndarray) -> np.ndarray:
        return np.clip(x, self.lb, self.ub)
    
    def _evaluate(self, x: np.ndarray) -> Tuple[float, float, bool]:
        """Evaluate solution. For unconstrained, violation=0, feasible=True."""
        self.history['evaluation_count'] += 1
        obj = self.objective(x)
        
        if self.is_constrained:
            g = self.constraints(x)
            violation = np.sum(np.maximum(0, g) ** 2)
            is_feasible = violation < 1e-10
        else:
            violation = 0.0
            is_feasible = True
        
        return obj, violation, is_feasible

    def _evaluate_batch(self, x_batch: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Evaluate batch of solutions."""
        self.history['evaluation_count'] += len(x_batch)
        
        # Optimistic batch evaluation
        try:
            objs = self.objective(x_batch)
        except Exception:
            objs = None
            
        # Check if result is valid (array of correct length)
        if objs is None or np.isscalar(objs) or len(np.atleast_1d(objs)) != len(x_batch):
            # Fallback to sequential evaluation
            objs = np.array([self.objective(x) for x in x_batch])
        
        if self.is_constrained:
            try:
                g = self.constraints(x_batch)
                g = np.atleast_2d(g) if g.ndim == 1 else g
                # Check for dimensionality mismatch similar to objective
                if g.shape[0] != len(x_batch):
                     raise ValueError
                violation = np.sum(np.maximum(0, g) ** 2, axis=1)
            except Exception:
                 # Fallback for constraints
                 violation = np.zeros(len(x_batch))
                 for i, x in enumerate(x_batch):
                     g = self.constraints(x)
                     violation[i] = np.sum(np.maximum(0, g) ** 2)
            
            is_feasible = violation < 1e-10
        else:
            violation = np.zeros(len(x_batch))
            is_feasible = np.ones(len(x_batch), dtype=bool)
        
        return objs, violation, is_feasible
    
    def _compare_epsilon(self, obj1, viol1, obj2, viol2) -> bool:
        """Epsilon-constraint comparison."""
        feas1 = viol1 <= self.epsilon
        feas2 = viol2 <= self.epsilon
        
        if feas1 and not feas2:
            return True
        if feas2 and not feas1:
            return False
        if feas1 and feas2:
            return obj1 < obj2
        return viol1 < viol2
    
    def _update_elite(self, solution: np.ndarray, fitness: float, violation: float):
        """Update elite archive."""
        if violation > 1e-6:
            return
        
        self.elite_archive.append((fitness, solution.copy()))
        self.elite_archive.sort(key=lambda x: x[0])
        self.elite_archive = self.elite_archive[:self.elite_size]
    
    def _elite_crossover(self) -> np.ndarray:
        """Create new solution by crossing elite solutions."""
        if len(self.elite_archive) < 2:
            return None
        
        # Select two parents from elite
        idx1, idx2 = random.sample(range(len(self.elite_archive)), 2)
        p1 = self.elite_archive[idx1][1]
        p2 = self.elite_archive[idx2][1]
        
        # SBX crossover
        child = np.zeros(self.dim)
        for d in range(self.dim):
            if np.random.rand() < 0.5:
                child[d] = p1[d]
            else:
                child[d] = p2[d]
        
        # Small mutation (Sigma_local = 0.015)
        child += np.random.normal(0, 0.015 * (self.ub - self.lb))
        return self._repair(child)
    
    def _nelder_mead_search(self, x: np.ndarray, max_evals: int = 30) -> np.ndarray:
        """Precise local search using Nelder-Mead."""
        def penalized_objective(x_inner):
            x_clipped = self._repair(x_inner)
            obj, viol, _ = self._evaluate(x_clipped)
            return obj + 1e6 * viol
        
        result = minimize(
            penalized_objective,
            x,
            method='Nelder-Mead',
            options={'maxfev': max_evals, 'xatol': 1e-6, 'fatol': 1e-6}
        )
        
        return self._repair(result.x)
    
    def _get_adaptive_params(self) -> Tuple[float, float]:
        """Get adapted learning rate and crossover rate."""
        lr = np.random.choice(self.success_history_lr)
        lr = np.clip(np.random.normal(lr, 0.1), 0.1, 0.9)
        
        cr = np.random.choice(self.success_history_cr)
        cr = np.clip(np.random.normal(cr, 0.1), 0.1, 0.9)
        
        return lr, cr
    
    def _update_success_history(self, successful_lr: List[float], successful_cr: List[float]):
        """Update success history with parameters that led to improvements."""
        if successful_lr:
            self.success_history_lr.extend(successful_lr)
            self.success_history_lr = self.success_history_lr[-50:]
        if successful_cr:
            self.success_history_cr.extend(successful_cr)
            self.success_history_cr = self.success_history_cr[-50:]
    
    def optimize(self, verbose: bool = True) -> Dict:
        """Run advanced optimization (Fully Vectorized)."""
        step_scale = 0.4
        decay_rate = 0.993
        
        # Stagnation counters for strategies
        strategy_stagnation = np.zeros(self.n_strategies, dtype=int)
        
        # State arrays to avoid re-evaluation
        self.current_fitness = np.zeros(self.n_candidates)
        self.current_violation = np.zeros(self.n_candidates)
        
        # Initial Batch Evaluation
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
            # Extended Epsilon Relaxation: Decays linearly over 85% of iterations (Enhanced)
            # This allows longer exploration of infeasible regions compared to original 50%
            tc = int(0.85 * self.max_iterations)
            if self.is_constrained and iteration < tc:
                self.epsilon = 1.0 * (1.0 - iteration / tc)
            else:
                self.epsilon = 0.0
            
            successful_lr = []
            successful_cr = []
            
            # ========== 1. Strategy Evaluation & Ranking ==========
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
            
            strategy_stagnation += 1
            strategy_stagnation[elite_indices] = 0
            
            # ========== 2. Strategy Learning ==========
            for i, strategy in enumerate(self.strategies):
                if i not in elite_indices:
                    lr, cr = self._get_adaptive_params()
                    strategy.a += step_scale * lr * (best_strategy.a - strategy.a) * np.random.rand(self.dim)
                    strategy.d += step_scale * lr * (best_strategy.d - strategy.d) * np.random.rand(self.dim)
                    
                    if len(elite_indices) > 1 and np.random.rand() < cr:
                        random_elite = self.strategies[np.random.choice(elite_indices[1:])]
                        strategy.b += step_scale * 0.3 * (random_elite.b - strategy.b) * np.random.rand(self.dim)
                        strategy.c += step_scale * 0.3 * (random_elite.c - strategy.c) * np.random.rand(self.dim)
            
            # ========== 3. Vectorized Candidate Guidance ==========
            candidate_positions = self.candidates.copy()
            candidate_lrs = np.zeros(self.n_candidates)
            candidate_crs = np.zeros(self.n_candidates)
            
            # Process by partition to generate candidates (Guidance only)
            for pid in range(self.n_partitions):
                c_indices = np.where(self.candidate_partition == pid)[0]
                if len(c_indices) == 0: continue
                
                partition_strategies = [s for s in self.strategies if s.partition_id == pid] or self.strategies
                active_strategy = random.choice(partition_strategies)
                
                current_pos = self.candidates[c_indices]
                guided = active_strategy.guide_batch(current_pos)
                
                lr, cr = self._get_adaptive_params()
                
                # Elite Influence
                if self.elite_archive:
                    elite_mask = np.random.rand(len(c_indices)) < 0.15
                    if np.any(elite_mask):
                        n_updates = np.sum(elite_mask)
                        e_indices_rand = np.random.randint(0, len(self.elite_archive), n_updates)
                        elite_sols = np.array([self.elite_archive[ei][1] for ei in e_indices_rand])
                        guided[elite_mask] = (1 - lr) * guided[elite_mask] + lr * elite_sols
                
                # Perturbation
                perturb_mask = np.random.rand(len(c_indices)) < cr
                if np.any(perturb_mask):
                    local_std = step_scale * 0.08 * (self.ub - self.lb)
                    noise = np.random.normal(0, 1, (np.sum(perturb_mask), self.dim)) * local_std
                    guided[perturb_mask] += noise
                
                guided = np.clip(guided, self.lb, self.ub)
                
                candidate_positions[c_indices] = guided
                candidate_lrs[c_indices] = lr
                candidate_crs[c_indices] = cr

            # ========== 4. Batch Evaluation & Selection ==========
            cand_objs, cand_viols, cand_feass = self._evaluate_batch(candidate_positions)
            
            # Vectorized Comparison logic
            # Explicitly implementing _compare_epsilon logic vectorized
            feas_cand = cand_viols <= self.epsilon
            feas_curr = self.current_violation <= self.epsilon
            
            both_feas = feas_cand & feas_curr
            cand_better_obj = cand_objs < self.current_fitness
            cand_better_viol = cand_viols < self.current_violation
            cand_feas_only = feas_cand & (~feas_curr)
            
            # Improvement condition:
            # (Cand Feas AND NOT Curr Feas) OR (Both Feas AND Cand Obj < Curr Obj) OR (Neither Feas AND Cand Viol < Curr Viol)
            improved_mask = (cand_feas_only) | (both_feas & cand_better_obj) | ((~feas_cand) & (~feas_curr) & cand_better_viol)
            
            # Update Population
            if np.any(improved_mask):
                self.candidates[improved_mask] = candidate_positions[improved_mask]
                self.current_fitness[improved_mask] = cand_objs[improved_mask]
                self.current_violation[improved_mask] = cand_viols[improved_mask]
                
                # Update Elite (Sequential for improved only, to restart sort)
                improved_indices = np.where(improved_mask)[0]
                for idx in improved_indices:
                    self._update_elite(candidate_positions[idx], cand_objs[idx], cand_viols[idx])
                    
                    # Update Global Best
                    if self._compare_epsilon(cand_objs[idx], cand_viols[idx], self.best_fitness, self.best_violation):
                        self.best_solution = candidate_positions[idx].copy()
                        self.best_fitness = cand_objs[idx]
                        self.best_violation = cand_viols[idx]
                        self.best_is_feasible = cand_feass[idx]
                
                # History update
                self._update_success_history(candidate_lrs[improved_mask].tolist(), candidate_crs[improved_mask].tolist())

            # ========== 5. Partition Exchange (was Cultural Exchange) ==========
            if np.random.rand() < 0.05:
                idx1, idx2 = np.random.choice(self.n_strategies, 2, replace=False)
                s1, s2 = self.strategies[idx1], self.strategies[idx2]
                s1.a, s2.a = s2.a.copy(), s1.a.copy()
                s1.b, s2.b = s2.b.copy(), s1.b.copy()
                s1.c, s2.c = s2.c.copy(), s1.c.copy()
                s1.d, s2.d = s2.d.copy(), s1.d.copy()
            
            # ========== 6. Elite Crossover ==========
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
            
            # ========== 6. Nelder-Mead Local Search ==========
            if iteration % 50 == 0 and iteration > 0 and self.best_solution is not None:
                improved = self._nelder_mead_search(self.best_solution, max_evals=50)
                obj, viol, feas = self._evaluate(improved)
                self._update_elite(improved, obj, viol)
                if self._compare_epsilon(obj, viol, self.best_fitness, self.best_violation):
                    self.best_solution = improved.copy()
                    self.best_fitness = obj
                    self.best_violation = viol
                    self.best_is_feasible = feas
                    self.history['local_search_improvements'] += 1
            
            # ========== 7. Stagnation-Aware Negation (was Refutation) ==========
            if iteration % 5 == 0:
                stagnant_indices = np.where(strategy_stagnation > 10)[0]
                for idx in stagnant_indices[:min(len(stagnant_indices), 5)]:
                    strategy = self.strategies[idx]
                    counter = strategy.negate()
                    
                    sample_idx = np.random.choice(self.n_candidates, 10, replace=False)
                    sample = self.candidates[sample_idx]
                    
                    guided = strategy.guide_batch(sample)
                    objs_g, viols_g, _ = self._evaluate_batch(guided)
                    scores = objs_g + 1e6 * np.maximum(0, viols_g - self.epsilon)
                    
                    c_guided = counter.guide_batch(sample)
                    objs_c, viols_c, _ = self._evaluate_batch(c_guided)
                    c_scores = objs_c + 1e6 * np.maximum(0, viols_c - self.epsilon)
                    
                    if np.mean(c_scores) < np.mean(scores):
                        self.strategies[idx] = counter
                        strategy_stagnation[idx] = 0
            
            self.history['convergence'].append(self.best_fitness if self.best_is_feasible else float('inf'))
            step_scale *= 0.995
            
            if verbose and (iteration % 50 == 0 or iteration == self.max_iterations - 1):
                status = "✓" if self.best_is_feasible else "✗"
                print(f"Iter {iteration:3d}: Best = {self.best_fitness:.6e} [{status}], "
                      f"ε = {self.epsilon:.2f}, Elite = {len(self.elite_archive)}")
        
        return {
            'best_solution': self.best_solution,
            'best_fitness': self.best_fitness,
            'best_violation': self.best_violation,
            'is_feasible': self.best_is_feasible,
            'elite_archive': [(f, s.copy()) for f, s in self.elite_archive],
            'history': self.history
        }
