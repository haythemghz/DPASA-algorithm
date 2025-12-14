"""
Enhanced Stable SDM Optimizer
==============================

Improved version with mechanisms to reduce variance and improve consistency:
1. Elite archive - preserves best solutions found
2. Stagnation detection - restarts when stuck
3. Local search - Nelder-Mead style intensification
4. Opposition-based learning - better initial coverage
"""

import numpy as np
import random
from typing import Callable, Tuple, List, Dict, Optional
from .leaders import LeaderGuide


class EnhancedConstrainedSDM:
    """
    Enhanced SDM with stability improvements for constrained optimization.
    
    Key improvements over base ConstrainedDPASAOptimizer:
    - Elite archive preserves top solutions across all iterations
    - Stagnation detection triggers diversity injection
    - Local search intensifies around best solutions
    - Opposition-based initialization for better coverage
    
    Example:
        >>> optimizer = EnhancedConstrainedSDM(
        ...     objective_func=problem.objective,
        ...     constraint_func=problem.constraints,
        ...     bounds=problem.bounds
        ... )
        >>> result = optimizer.optimize()
    """
    
    def __init__(
        self,
        objective_func: Callable,
        constraint_func: Callable,
        bounds: List[Tuple[float, float]],
        dim: int = None,
        n_leaders: int = 80,
        n_followers: int = 400,
        n_cultures: int = 3,
        max_iterations: int = 500,
        elite_size: int = 10,
        stagnation_limit: int = 30,
        local_search_freq: int = 25,
        seed: int = None
    ):
        """
        Initialize enhanced DPASA optimizer.
        
        Args:
            objective_func: Function to minimize
            constraint_func: Returns g(x) where g(x) <= 0 is feasible
            bounds: List of (lower, upper) tuples
            dim: Problem dimension
            n_leaders: Number of leaders (default: 80)
            n_followers: Number of followers (default: 400)
            n_cultures: Number of cultures (default: 3)
            max_iterations: Max iterations (default: 500)
            elite_size: Size of elite archive (default: 10)
            stagnation_limit: Iterations before restart (default: 30)
            local_search_freq: Local search frequency (default: 25)
            seed: Random seed
        """
        self.objective = objective_func
        self.constraints = constraint_func
        self.bounds = bounds
        self.dim = dim if dim else len(bounds)
        self.n_leaders = n_leaders
        self.n_followers = n_followers
        self.n_cultures = n_cultures
        self.max_iterations = max_iterations
        self.elite_size = elite_size
        self.stagnation_limit = stagnation_limit
        self.local_search_freq = local_search_freq
        
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        
        # Bounds arrays
        self.lb = np.array([b[0] for b in bounds])
        self.ub = np.array([b[1] for b in bounds])
        
        # Initialize with opposition-based learning
        self.followers = self._init_opposition_population(n_followers)
        self.follower_culture = np.random.choice(n_cultures, n_followers)
        
        # Leaders
        global_bounds = (self.lb.min(), self.ub.max())
        self.leaders = [
            LeaderGuide(self.dim, global_bounds, i % n_cultures)
            for i in range(n_leaders)
        ]
        
        # Elite archive: stores (fitness, violation, solution) tuples
        self.elite_archive = []
        
        # Best solution tracking
        self.best_solution = None
        self.best_fitness = float('inf')
        self.best_violation = float('inf')
        self.best_is_feasible = False
        
        # Stagnation tracking
        self.stagnation_counter = 0
        self.last_improvement_iter = 0
        
        # History
        self.history = {
            'convergence': [],
            'violation': [],
            'feasibility_rate': [],
            'restarts': [],
            'local_searches': [],
            'evaluation_count': 0
        }
    
    def _init_opposition_population(self, n: int) -> np.ndarray:
        """Initialize population with opposition-based learning."""
        half = n // 2
        
        # Regular random initialization
        pop1 = np.zeros((half, self.dim))
        for i in range(self.dim):
            pop1[:, i] = np.random.uniform(self.lb[i], self.ub[i], half)
        
        # Opposition-based initialization
        pop2 = self.lb + self.ub - pop1
        
        # Combine
        return np.vstack([pop1, pop2[:n - half]])
    
    def _repair(self, x: np.ndarray) -> np.ndarray:
        """Repair solution to bounds."""
        return np.clip(x, self.lb, self.ub)
    
    def _evaluate(self, x: np.ndarray) -> Tuple[float, float, bool]:
        """Evaluate solution."""
        self.history['evaluation_count'] += 1
        obj = self.objective(x)
        g = self.constraints(x)
        violation = np.sum(np.maximum(0, g) ** 2)
        is_feasible = violation < 1e-10
        return obj, violation, is_feasible
    
    def _compare(self, obj1, viol1, feas1, obj2, viol2, feas2) -> bool:
        """Deb's feasibility rules."""
        if feas1 and not feas2:
            return True
        if feas2 and not feas1:
            return False
        if feas1 and feas2:
            return obj1 < obj2
        return viol1 < viol2
    
    def _update_elite(self, solution: np.ndarray, fitness: float, violation: float, is_feasible: bool):
        """Update elite archive with new solution."""
        if not is_feasible:
            return
        
        # Add to archive
        self.elite_archive.append((fitness, violation, solution.copy()))
        
        # Keep only best solutions
        self.elite_archive.sort(key=lambda x: x[0])
        self.elite_archive = self.elite_archive[:self.elite_size]
    
    def _local_search(self, x: np.ndarray, n_iter: int = 10) -> np.ndarray:
        """
        Nelder-Mead style local search around a solution.
        Simplified version focusing on coordinate-wise improvement.
        """
        best_x = x.copy()
        best_obj, best_viol, best_feas = self._evaluate(best_x)
        
        step = 0.1 * (self.ub - self.lb)
        
        for _ in range(n_iter):
            improved = False
            
            for d in range(self.dim):
                # Try positive step
                trial = best_x.copy()
                trial[d] = self._repair(np.array([trial[d] + step[d]]))[0]
                trial = self._repair(trial)
                obj, viol, feas = self._evaluate(trial)
                
                if self._compare(obj, viol, feas, best_obj, best_viol, best_feas):
                    best_x = trial
                    best_obj, best_viol, best_feas = obj, viol, feas
                    improved = True
                    continue
                
                # Try negative step
                trial = best_x.copy()
                trial[d] = self._repair(np.array([trial[d] - step[d]]))[0]
                trial = self._repair(trial)
                obj, viol, feas = self._evaluate(trial)
                
                if self._compare(obj, viol, feas, best_obj, best_viol, best_feas):
                    best_x = trial
                    best_obj, best_viol, best_feas = obj, viol, feas
                    improved = True
            
            if not improved:
                step *= 0.5  # Reduce step size
        
        return best_x
    
    def _detect_stagnation(self, iteration: int) -> bool:
        """Detect if optimization is stagnating."""
        return (iteration - self.last_improvement_iter) > self.stagnation_limit
    
    def _restart_diversity(self):
        """Inject diversity when stagnating."""
        # Reinitialize worst 30% of followers
        n_reinit = int(0.3 * self.n_followers)
        
        # Evaluate all followers
        scores = []
        for i, x in enumerate(self.followers):
            obj, viol, feas = self._evaluate(x)
            # Penalized score for ranking
            score = obj if feas else obj + 1e10 * viol
            scores.append((score, i))
        
        # Get worst indices
        scores.sort(reverse=True)
        worst_indices = [idx for _, idx in scores[:n_reinit]]
        
        # Reinitialize with opposition learning
        new_pop = self._init_opposition_population(n_reinit)
        for i, idx in enumerate(worst_indices):
            self.followers[idx] = new_pop[i]
        
        # Reinitialize some leaders too
        n_leader_reinit = max(1, self.n_leaders // 5)
        for i in range(n_leader_reinit):
            idx = np.random.randint(self.n_leaders)
            self.leaders[idx] = LeaderGuide(
                self.dim, 
                (self.lb.min(), self.ub.max()), 
                np.random.randint(self.n_cultures)
            )
    
    def optimize(self, verbose: bool = True) -> Dict:
        """Run enhanced optimization."""
        step_scale = 0.4
        decay_rate = 0.995
        
        # Initial evaluation
        for i, x in enumerate(self.followers):
            obj, viol, feas = self._evaluate(x)
            self._update_elite(x, obj, viol, feas)
            if self._compare(obj, viol, feas, self.best_fitness, self.best_violation, self.best_is_feasible):
                self.best_solution = x.copy()
                self.best_fitness = obj
                self.best_violation = viol
                self.best_is_feasible = feas
        
        for iteration in range(self.max_iterations):
            # ========== Stagnation Check ==========
            if self._detect_stagnation(iteration):
                self._restart_diversity()
                self.history['restarts'].append(iteration)
                self.stagnation_counter = 0
                if verbose:
                    print(f"  [Restart at iter {iteration}]")
            
            # ========== Local Search (periodic) ==========
            if iteration > 0 and iteration % self.local_search_freq == 0 and self.best_solution is not None:
                improved_x = self._local_search(self.best_solution, n_iter=5)
                obj, viol, feas = self._evaluate(improved_x)
                self._update_elite(improved_x, obj, viol, feas)
                
                if self._compare(obj, viol, feas, self.best_fitness, self.best_violation, self.best_is_feasible):
                    self.best_solution = improved_x.copy()
                    self.best_fitness = obj
                    self.best_violation = viol
                    self.best_is_feasible = feas
                    self.last_improvement_iter = iteration
                    self.history['local_searches'].append((iteration, 'improved'))
                else:
                    self.history['local_searches'].append((iteration, 'no_improve'))
            
            # ========== PHASE 1: Evaluate Leaders ==========
            leader_scores = []
            for leader in self.leaders:
                sample_idx = np.random.choice(self.n_followers, min(40, self.n_followers), replace=False)
                sample = self.followers[sample_idx]
                guided = np.array([self._repair(leader.guide(s)) for s in sample])
                
                objs, viols, feases = [], [], []
                for g in guided:
                    o, v, f = self._evaluate(g)
                    objs.append(o)
                    viols.append(v)
                    feases.append(f)
                
                leader_scores.append((np.mean(objs), np.mean(viols), np.mean(feases) > 0.5))
            
            # Rank leaders
            leader_ranks = []
            for i in range(self.n_leaders):
                rank = sum(1 for j in range(self.n_leaders) if i != j and 
                          self._compare(leader_scores[j][0], leader_scores[j][1], leader_scores[j][2],
                                       leader_scores[i][0], leader_scores[i][1], leader_scores[i][2]))
                leader_ranks.append(rank)
            
            elite_indices = np.argsort(leader_ranks)[:max(1, self.n_leaders // 10)]
            best_leader = self.leaders[elite_indices[0]]
            
            # ========== PHASE 2: Learning ==========
            for i, leader in enumerate(self.leaders):
                if i not in elite_indices:
                    learn_rate = 0.4
                    leader.a += step_scale * learn_rate * (best_leader.a - leader.a) * np.random.rand(self.dim)
                    leader.d += step_scale * learn_rate * (best_leader.d - leader.d) * np.random.rand(self.dim)
                    
                    if len(elite_indices) > 1:
                        random_elite = self.leaders[np.random.choice(elite_indices[1:])]
                        leader.b += step_scale * 0.3 * (random_elite.b - leader.b) * np.random.rand(self.dim)
                        leader.c += step_scale * 0.3 * (random_elite.c - leader.c) * np.random.rand(self.dim)
            
            # ========== PHASE 3: Follower Guidance ==========
            feasible_count = 0
            
            for i in range(self.n_followers):
                culture_leaders = [l for l in self.leaders if l.culture_id == self.follower_culture[i]]
                if not culture_leaders:
                    culture_leaders = self.leaders
                leader = random.choice(culture_leaders)
                
                guided = self._repair(leader.guide(self.followers[i]))
                
                # Local perturbation
                if np.random.rand() < 0.4:
                    local_std = step_scale * 0.1 * (self.ub - self.lb)
                    guided = self._repair(guided + np.random.normal(0, local_std))
                
                # Elite-guided search: occasionally move toward elite archive
                if self.elite_archive and np.random.rand() < 0.1:
                    elite_sol = random.choice(self.elite_archive)[2]
                    guided = self._repair(0.7 * guided + 0.3 * elite_sol)
                
                new_obj, new_viol, new_feas = self._evaluate(guided)
                old_obj, old_viol, old_feas = self._evaluate(self.followers[i])
                
                if self._compare(new_obj, new_viol, new_feas, old_obj, old_viol, old_feas):
                    self.followers[i] = guided
                    self._update_elite(guided, new_obj, new_viol, new_feas)
                    
                    if self._compare(new_obj, new_viol, new_feas, 
                                    self.best_fitness, self.best_violation, self.best_is_feasible):
                        self.best_solution = guided.copy()
                        self.best_fitness = new_obj
                        self.best_violation = new_viol
                        self.best_is_feasible = new_feas
                        self.last_improvement_iter = iteration
                
                if new_feas:
                    feasible_count += 1
            
            # ========== PHASE 4: Refutation ==========
            for i, leader in enumerate(self.leaders):
                if np.random.rand() < 0.3:  # Only refute 30% of leaders per iteration
                    counter = leader.refute()
                    sample_idx = np.random.choice(self.n_followers, 15, replace=False)
                    sample = self.followers[sample_idx]
                    
                    counter_guided = np.array([self._repair(counter.guide(s)) for s in sample])
                    
                    c_objs, c_viols, c_feases = [], [], []
                    for g in counter_guided:
                        o, v, f = self._evaluate(g)
                        c_objs.append(o)
                        c_viols.append(v)
                        c_feases.append(f)
                    
                    counter_score = (np.mean(c_objs), np.mean(c_viols), np.mean(c_feases) > 0.5)
                    
                    if self._compare(counter_score[0], counter_score[1], counter_score[2],
                                   leader_scores[i][0], leader_scores[i][1], leader_scores[i][2]):
                        self.leaders[i] = counter
            
            # Record history
            self.history['convergence'].append(self.best_fitness if self.best_is_feasible else float('inf'))
            self.history['violation'].append(self.best_violation)
            self.history['feasibility_rate'].append(feasible_count / self.n_followers)
            
            step_scale *= decay_rate
            
            if verbose and (iteration % 50 == 0 or iteration == self.max_iterations - 1):
                status = "✓" if self.best_is_feasible else "✗"
                print(f"Iter {iteration:3d}: Best = {self.best_fitness:.6e} [{status}], "
                      f"Feas = {100*feasible_count/self.n_followers:.1f}%, "
                      f"Elite = {len(self.elite_archive)}")
        
        return {
            'best_solution': self.best_solution,
            'best_fitness': self.best_fitness,
            'best_violation': self.best_violation,
            'is_feasible': self.best_is_feasible,
            'elite_archive': [(f, s.copy()) for f, _, s in self.elite_archive],
            'history': self.history
        }
