"""
Constrained SDM Optimizer
==========================

SDM variant with improved constraint handling for engineering optimization.

Features:
- Deb's feasibility rules (constraint-first selection)
- Bounds-aware initialization
- Adaptive penalty coefficient
- Repair mechanism for infeasible solutions
"""

import numpy as np
import random
from typing import Callable, Tuple, List, Dict, Optional

from .leaders import LeaderGuide


class ConstrainedDPASAOptimizer:
    """
    SDM Optimizer with advanced constraint handling.
    
    Implements Deb's feasibility rules:
    1. Feasible solution always beats infeasible
    2. Between two feasible solutions, better objective wins
    3. Between two infeasible solutions, lower violation wins
    
    Example:
        >>> from benchmarks import WeldedBeamDesign
        >>> optimizer = ConstrainedDPASAOptimizer(
        ...     objective_func=WeldedBeamDesign.objective,
        ...     constraint_func=WeldedBeamDesign.constraints,
        ...     bounds=WeldedBeamDesign.bounds,
        ...     dim=4
        ... )
        >>> result = optimizer.optimize()
    """
    
    def __init__(
        self,
        objective_func: Callable,
        constraint_func: Callable,
        bounds: List[Tuple[float, float]],
        dim: int = None,
        n_leaders: int = 50,
        n_followers: int = 400,
        n_cultures: int = 5,
        max_iterations: int = 500,
        seed: int = None
    ):
        """
        Initialize constrained DPASA optimizer.
        
        Args:
            objective_func: Function to minimize, returns scalar
            constraint_func: Returns array of g(x), where g(x) <= 0 is feasible
            bounds: List of (lower, upper) tuples for each variable
            dim: Problem dimension (inferred from bounds if not provided)
            n_leaders: Number of leaders
            n_followers: Number of followers
            n_cultures: Number of cultural groups
            max_iterations: Maximum iterations
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
        
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        
        # Bounds arrays for vectorized operations
        self.lb = np.array([b[0] for b in bounds])
        self.ub = np.array([b[1] for b in bounds])
        
        # Initialize populations within bounds
        self.followers = self._init_population(n_followers)
        self.follower_culture = np.random.choice(n_cultures, n_followers)
        
        # Initialize leaders with bounds-aware parameters
        global_bounds = (self.lb.min(), self.ub.max())
        self.leaders = [
            LeaderGuide(self.dim, global_bounds, i % n_cultures)
            for i in range(n_leaders)
        ]
        
        # Best solution tracking
        self.best_solution = None
        self.best_fitness = float('inf')
        self.best_violation = float('inf')
        self.best_is_feasible = False
        
        # History
        self.history = {
            'convergence': [],
            'violation': [],
            'feasibility_rate': [],
            'evaluation_count': 0
        }
    
    def _init_population(self, n: int) -> np.ndarray:
        """Initialize population within variable bounds."""
        pop = np.zeros((n, self.dim))
        for i in range(self.dim):
            pop[:, i] = np.random.uniform(self.lb[i], self.ub[i], n)
        return pop
    
    def _repair(self, x: np.ndarray) -> np.ndarray:
        """Repair solution to be within bounds."""
        return np.clip(x, self.lb, self.ub)
    
    def _evaluate(self, x: np.ndarray) -> Tuple[float, float, bool]:
        """
        Evaluate solution returning (objective, violation, is_feasible).
        """
        self.history['evaluation_count'] += 1
        obj = self.objective(x)
        g = self.constraints(x)
        violation = np.sum(np.maximum(0, g) ** 2)
        is_feasible = violation < 1e-10
        return obj, violation, is_feasible
    
    def _compare(self, obj1, viol1, feas1, obj2, viol2, feas2) -> bool:
        """
        Deb's feasibility rules: returns True if solution 1 is better.
        """
        # Rule 1: Feasible beats infeasible
        if feas1 and not feas2:
            return True
        if feas2 and not feas1:
            return False
        
        # Rule 2: Both feasible -> lower objective wins
        if feas1 and feas2:
            return obj1 < obj2
        
        # Rule 3: Both infeasible -> lower violation wins
        return viol1 < viol2
    
    def optimize(self, verbose: bool = True) -> Dict:
        """
        Run constrained optimization.
        
        Args:
            verbose: Print progress
            
        Returns:
            Dictionary with best solution, fitness, and history
        """
        # Adaptive parameters
        step_scale = 0.4
        decay_rate = 0.995
        
        # Evaluate initial population
        for i, x in enumerate(self.followers):
            obj, viol, feas = self._evaluate(x)
            if self._compare(obj, viol, feas, self.best_fitness, self.best_violation, self.best_is_feasible):
                self.best_solution = x.copy()
                self.best_fitness = obj
                self.best_violation = viol
                self.best_is_feasible = feas
        
        for iteration in range(self.max_iterations):
            # ========== PHASE 1: Evaluate Leaders ==========
            leader_scores = []  # (objective, violation, is_feasible) tuples
            
            for leader in self.leaders:
                # Sample followers for this leader
                sample_idx = np.random.choice(self.n_followers, min(50, self.n_followers), replace=False)
                sample = self.followers[sample_idx]
                guided = leader.guide_batch(sample)
                guided = np.array([self._repair(g) for g in guided])
                
                # Average performance
                objs, viols, feases = [], [], []
                for g in guided:
                    o, v, f = self._evaluate(g)
                    objs.append(o)
                    viols.append(v)
                    feases.append(f)
                
                leader_scores.append((np.mean(objs), np.mean(viols), np.mean(feases) > 0.5))
            
            # Rank leaders using feasibility rules
            leader_ranks = []
            for i in range(self.n_leaders):
                rank = sum(1 for j in range(self.n_leaders) if i != j and 
                          self._compare(leader_scores[j][0], leader_scores[j][1], leader_scores[j][2],
                                       leader_scores[i][0], leader_scores[i][1], leader_scores[i][2]))
                leader_ranks.append(rank)
            
            elite_indices = np.argsort(leader_ranks)[:max(1, self.n_leaders // 10)]
            best_idx = elite_indices[0]
            best_leader = self.leaders[best_idx]
            
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
                # Select leader from culture
                culture_leaders = [l for l in self.leaders if l.culture_id == self.follower_culture[i]]
                if not culture_leaders:
                    culture_leaders = self.leaders
                leader = random.choice(culture_leaders)
                
                # Guide and repair
                guided = self._repair(leader.guide(self.followers[i]))
                
                # Local search
                if np.random.rand() < 0.4:
                    local_std = step_scale * 0.1 * (self.ub - self.lb)
                    guided = self._repair(guided + np.random.normal(0, local_std))
                
                # Evaluate new position
                new_obj, new_viol, new_feas = self._evaluate(guided)
                old_obj, old_viol, old_feas = self._evaluate(self.followers[i])
                
                # Accept if better (feasibility rules)
                if self._compare(new_obj, new_viol, new_feas, old_obj, old_viol, old_feas):
                    self.followers[i] = guided
                    
                    # Update global best
                    if self._compare(new_obj, new_viol, new_feas, 
                                    self.best_fitness, self.best_violation, self.best_is_feasible):
                        self.best_solution = guided.copy()
                        self.best_fitness = new_obj
                        self.best_violation = new_viol
                        self.best_is_feasible = new_feas
                
                if new_feas:
                    feasible_count += 1
            
            # ========== PHASE 4: Refutation ==========
            for i, leader in enumerate(self.leaders):
                counter = leader.refute()
                sample_idx = np.random.choice(self.n_followers, 20, replace=False)
                sample = self.followers[sample_idx]
                
                counter_guided = counter.guide_batch(sample)
                counter_guided = np.array([self._repair(g) for g in counter_guided])
                
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
            
            # ========== PHASE 5: Diversity Injection ==========
            if iteration % 30 == 0 and iteration > 0:
                n_reinject = max(1, self.n_followers // 10)
                reinject_idx = np.random.choice(self.n_followers, n_reinject, replace=False)
                self.followers[reinject_idx] = self._init_population(n_reinject)
            
            # Record history
            self.history['convergence'].append(self.best_fitness if self.best_is_feasible else float('inf'))
            self.history['violation'].append(self.best_violation)
            self.history['feasibility_rate'].append(feasible_count / self.n_followers)
            
            step_scale *= decay_rate
            
            if verbose and (iteration % 50 == 0 or iteration == self.max_iterations - 1):
                status = "✓" if self.best_is_feasible else "✗"
                print(f"Iter {iteration:3d}: Best = {self.best_fitness:.6e} [{status}], "
                      f"Viol = {self.best_violation:.2e}, "
                      f"Feas = {100*feasible_count/self.n_followers:.1f}%")
        
        return {
            'best_solution': self.best_solution,
            'best_fitness': self.best_fitness,
            'best_violation': self.best_violation,
            'is_feasible': self.best_is_feasible,
            'history': self.history
        }
