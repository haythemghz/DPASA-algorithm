"""
Dual-Population Adversarial Strategy Adaptation (DPASA)
=======================================================

A novel population-based metaheuristic for global optimization based on
dual-population adversarial search dynamics.

Main Classes:
    DPASAOptimizer: The main optimization algorithm
    StrategyVector: Strategy guidance function class

Usage:
    from dpasa import DPASAOptimizer
    
    # Unconstrained optimization
    optimizer = DPASAOptimizer(sphere, dim=10, bounds=(-5, 5))
    result = optimizer.optimize()
    
    # Constrained engineering optimization
    optimizer = DPASAOptimizer(
        objective_func=problem.objective,
        constraint_func=problem.constraints,
        bounds=problem.bounds,
        max_iterations=1000
    )
    result = optimizer.optimize()
"""

# Import the main DPASAOptimizer
from .optimizer import DPASAOptimizer
from .strategy import StrategyVector, StrategyMetrics

# Backward compatibility aliases (optional)
DPASAOptimizer = DPASAOptimizer
LeaderGuide = StrategyVector

__version__ = "1.0.0"
__author__ = "Haythem Ghazouani"

__all__ = [
    "DPASAOptimizer",
    "StrategyVector", 
    "StrategyMetrics",
    "DPASAOptimizer",  # Exported for backward compatibility
]
