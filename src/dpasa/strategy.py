"""
Strategy Vector Classes for DPASA Algorithm
===========================================

Contains the StrategyVector and StrategyMetrics classes that implement
the guidance mechanism for the DPASA algorithm.
"""

import numpy as np
from typing import List, Tuple
from dataclasses import dataclass


@dataclass
class StrategyMetrics:
    """Track strategy performance metrics throughout optimization."""
    fitness_history: List[float]
    diversity_contributions: List[float]
    influence_score: float = 0.0
    age: int = 0


class StrategyVector:
    """
    Strategy Guidance Function for DPASA.
    
    Each strategy defines a sinusoidal transformation that guides candidates
    toward promising regions of the search space.
    
    The guidance function is: g(x) = a * sin(b*x + c) + d*x
    
    Attributes:
        dim: Problem dimensionality
        bounds: Search space bounds (lower, upper)
        partition_id: Partition affiliation index
        a, b, c, d: Sinusoidal function parameters (D-dimensional vectors)
    """
    
    def __init__(self, dim: int, bounds: Tuple[float, float], partition_id: int):
        self.dim = dim
        self.bounds = bounds
        self.partition_id = partition_id
        
        # Guidance function parameters: g(x) = a*sin(b*x + c) + d*x
        self.a = np.random.uniform(-1, 1, dim)
        self.b = np.random.uniform(0.5, 2 * np.pi, dim)
        self.c = np.random.uniform(-np.pi, np.pi, dim)
        self.d = np.random.uniform(-1, 1, dim)
        
        self.metrics = StrategyMetrics(fitness_history=[], diversity_contributions=[])
    
    def guide(self, x: np.ndarray) -> np.ndarray:
        """
        Apply strategy's guidance transformation to a single position.
        
        Args:
            x: Current position (D-dimensional)
            
        Returns:
            Guided position (D-dimensional)
        """
        guided = self.a * np.sin(self.b * x + self.c) + self.d * x
        return np.clip(guided, self.bounds[0], self.bounds[1])
    
    def guide_batch(self, positions: np.ndarray) -> np.ndarray:
        """
        Vectorized batch guidance - applies transformation to multiple positions.
        
        Args:
            positions: Array of positions (N x D)
            
        Returns:
            Guided positions (N x D)
        """
        guided = (
            self.a[np.newaxis, :] * np.sin(self.b[np.newaxis, :] * positions + self.c[np.newaxis, :]) +
            self.d[np.newaxis, :] * positions
        )
        return np.clip(guided, self.bounds[0], self.bounds[1])
    
    def adaptive_mutate(self, convergence_rate: float, base_rate: float = 0.05):
        """
        Apply adaptive mutation to strategy parameters based on convergence state.
        
        Args:
            convergence_rate: Current convergence rate (0 = diverging, 1 = fully converged)
            base_rate: Base mutation rate
        """
        convergence_rate = np.clip(convergence_rate, 0.0, 1.0)
        rate = max(abs(base_rate * (1 + convergence_rate)), 1e-8)
        
        self.a += np.random.normal(0, rate, self.dim)
        self.b += np.random.normal(0, rate * 0.8, self.dim)
        self.c += np.random.normal(0, rate * 0.8, self.dim)
        self.d += np.random.normal(0, rate, self.dim)
    
    def negate(self) -> 'StrategyVector':
        """
        Create opposing counter-strategy through systematic negation.
        
        The counter-strategy explores the "opposite" direction in strategy space,
        implementing the adversarial challenge mechanism.
        
        Returns:
            New StrategyVector with negated/shifted parameters
        """
        counter = StrategyVector(self.dim, self.bounds, self.partition_id)
        counter.a = -self.a + np.random.normal(0, 0.08, self.dim)
        counter.b = self.b + np.random.normal(0, 0.08, self.dim)
        counter.c = self.c + np.pi / 2 * np.sign(np.random.randn(self.dim))
        counter.d = -self.d + np.random.normal(0, 0.08, self.dim)
        return counter
    
    def copy(self) -> 'StrategyVector':
        """Create a deep copy of this strategy."""
        new_strategy = StrategyVector(self.dim, self.bounds, self.partition_id)
        new_strategy.a = self.a.copy()
        new_strategy.b = self.b.copy()
        new_strategy.c = self.c.copy()
        new_strategy.d = self.d.copy()
        return new_strategy
