"""
Benchmark Functions for Optimization Testing
=============================================

Provides classical and engineering benchmark functions.
"""

from .classical import (
    BenchmarkFunctions,
    BENCHMARK_SUITE,
    get_function_characteristics,
)

from .engineering import (
    WeldedBeamDesign,
    SpringDesign,
    SpeedReducerDesign,
    ThreeBarTruss,
    CantileverBeam,
    GearTrainDesign,
    ENGINEERING_PROBLEMS,
    get_problem_info,
)

# Convenience function aliases
sphere = BenchmarkFunctions.sphere
rosenbrock = BenchmarkFunctions.rosenbrock
rastrigin = BenchmarkFunctions.rastrigin
griewank = BenchmarkFunctions.griewank
ackley = BenchmarkFunctions.ackley
schwefel = BenchmarkFunctions.schwefel

__all__ = [
    # Classical
    "BenchmarkFunctions",
    "BENCHMARK_SUITE", 
    "get_function_characteristics",
    "sphere", "rosenbrock", "rastrigin", "griewank", "ackley", "schwefel",
    # Engineering
    "WeldedBeamDesign",
    "SpringDesign",
    "SpeedReducerDesign",
    "ThreeBarTruss",
    "CantileverBeam",
    "GearTrainDesign",
    "ENGINEERING_PROBLEMS",
    "get_problem_info",
]
