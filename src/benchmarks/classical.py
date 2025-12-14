"""
Comprehensive Benchmark Functions for Optimization Testing
===========================================================

Includes 15 standard benchmark functions with varying difficulty:
- Unimodal functions (convex, smooth)
- Multimodal functions (many local minima)
- Non-separable functions
- Ill-conditioned functions
"""

import numpy as np
from typing import Tuple, Dict, Callable


class BenchmarkFunctions:
    """Collection of standard optimization benchmark functions"""
    
    @staticmethod
    def sphere(x: np.ndarray) -> float:
        """
        Sphere function - Unimodal, separable, convex
        Global minimum: f(0,...,0) = 0
        Search domain: [-5, 5]^D
        """
        return np.sum(x ** 2)
    
    @staticmethod
    def ellipsoid(x: np.ndarray) -> float:
        """
        Ellipsoid function - Unimodal, separable, ill-conditioned
        Global minimum: f(0,...,0) = 0
        Search domain: [-5, 5]^D
        """
        return np.sum(np.arange(1, len(x) + 1) * x ** 2)
    
    @staticmethod
    def rosenbrock(x: np.ndarray) -> float:
        """
        Rosenbrock function - Unimodal, non-separable, narrow valley
        Global minimum: f(1,...,1) = 0
        Search domain: [-2, 2]^D
        """
        return np.sum(100 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2)
    
    @staticmethod
    def quartic(x: np.ndarray) -> float:
        """
        Quartic function - Unimodal with noise potential
        Global minimum: f(0,...,0) = 0
        Search domain: [-1, 1]^D
        """
        return np.sum(np.arange(1, len(x) + 1) * x ** 4)
    
    @staticmethod
    def step(x: np.ndarray) -> float:
        """
        Step function - Discontinuous, flat regions
        Global minimum: f(±0.5,...,±0.5) = 0
        Search domain: [-5, 5]^D
        """
        return np.sum(np.floor(x + 0.5) ** 2)
    
    @staticmethod
    def zakharov(x: np.ndarray) -> float:
        """
        Zakharov function - Unimodal, non-separable
        Global minimum: f(0,...,0) = 0
        Search domain: [-5, 5]^D
        """
        i = np.arange(1, len(x) + 1)
        sum_ix = np.sum(0.5 * i * x)
        return np.sum(x ** 2) + sum_ix ** 2 + sum_ix ** 4
    
    @staticmethod
    def ackley(x: np.ndarray) -> float:
        """
        Ackley function - Highly multimodal, many local minima
        Global minimum: f(0,...,0) = 0
        Search domain: [-32, 32]^D
        """
        a, b, c = 20, 0.2, 2 * np.pi
        n = len(x)
        sum_sq = np.sum(x ** 2) / n
        sum_cos = np.sum(np.cos(c * x)) / n
        return -a * np.exp(-b * np.sqrt(sum_sq)) - np.exp(sum_cos) + a + np.e
    
    @staticmethod
    def griewank(x: np.ndarray) -> float:
        """
        Griewank function - Multimodal with product term
        Global minimum: f(0,...,0) = 0
        Search domain: [-600, 600]^D
        """
        i = np.arange(1, len(x) + 1)
        sum_sq = np.sum(x ** 2) / 4000
        prod_cos = np.prod(np.cos(x / np.sqrt(i)))
        return sum_sq - prod_cos + 1
    
    @staticmethod
    def rastrigin(x: np.ndarray) -> float:
        """
        Rastrigin function - Highly multimodal, regular structure
        Global minimum: f(0,...,0) = 0
        Search domain: [-5, 5]^D
        """
        return 10 * len(x) + np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x))
    
    @staticmethod
    def schwefel(x: np.ndarray) -> float:
        """
        Schwefel function - Multimodal, deceptive (optimum far from origin)
        Global minimum: f(420.9687,...,420.9687) ≈ 0
        Search domain: [-500, 500]^D
        """
        return 418.9829 * len(x) - np.sum(x * np.sin(np.sqrt(np.abs(x))))
    
    @staticmethod
    def levy(x: np.ndarray) -> float:
        """
        Levy function - Multimodal
        Global minimum: f(1,...,1) = 0
        Search domain: [-10, 10]^D
        """
        w = 1 + (x - 1) / 4
        term1 = np.sin(np.pi * w[0]) ** 2
        term2 = np.sum((w[:-1] - 1) ** 2 * (1 + 10 * np.sin(np.pi * w[:-1] + 1) ** 2))
        term3 = (w[-1] - 1) ** 2 * (1 + np.sin(2 * np.pi * w[-1]) ** 2)
        return term1 + term2 + term3
    
    @staticmethod
    def michalewicz(x: np.ndarray) -> float:
        """
        Michalewicz function - Multimodal, steep ridges
        Global minimum: f(x*) ≈ -9.66 (D=10)
        Search domain: [0, π]^D
        """
        i = np.arange(1, len(x) + 1)
        return -np.sum(np.sin(x) * np.sin(i * x ** 2 / np.pi) ** 20)
    
    @staticmethod
    def perm(x: np.ndarray) -> float:
        """
        Perm function - Multimodal
        Global minimum: f(1, 1/2, 1/3, ..., 1/D) = 0
        Search domain: [-D, D]^D
        """
        n = len(x)
        result = 0
        for k in range(1, n + 1):
            inner_sum = 0
            for j in range(n):
                inner_sum += (j + 1 + k) * (x[j] ** k - 1 / (j + 1) ** k)
            result += inner_sum ** 2
        return result
    
    @staticmethod
    def katsuura(x: np.ndarray) -> float:
        """
        Katsuura function - Highly rugged, continuous multimodal
        Global minimum: f(0,...,0) = 0
        Search domain: [-100, 100]^D
        """
        d = len(x)
        prod = 1
        for i in range(d):
            sum_term = np.sum(
                np.abs(2 ** np.arange(1, 33) * x[i] - 
                       np.round(2 ** np.arange(1, 33) * x[i]))
            ) / (2 ** 32)
            prod *= (1 + (i + 1) * sum_term) ** (10 / d ** 1.2)
        return (10 / d ** 2) * prod - (10 / d ** 2)
    
    @staticmethod
    def happy_cat(x: np.ndarray) -> float:
        """
        Happy Cat function - Multimodal with plateaus
        Global minimum: f(-1,...,-1) ≈ 0
        Search domain: [-2, 2]^D
        """
        n = len(x)
        sum_sq = np.sum(x ** 2)
        sum_x = np.sum(x)
        return ((sum_sq - n) ** 2) ** 0.25 + (0.5 * sum_sq + sum_x) / n + 0.5


# Benchmark suite configuration
BENCHMARK_SUITE: Dict[str, Tuple[Callable, Tuple[float, float], float]] = {
    "Sphere": (BenchmarkFunctions.sphere, (-5, 5), 0.0),
    "Ellipsoid": (BenchmarkFunctions.ellipsoid, (-5, 5), 0.0),
    "Rosenbrock": (BenchmarkFunctions.rosenbrock, (-2, 2), 0.0),
    "Quartic": (BenchmarkFunctions.quartic, (-1, 1), 0.0),
    "Step": (BenchmarkFunctions.step, (-5, 5), 0.0),
    "Zakharov": (BenchmarkFunctions.zakharov, (-5, 5), 0.0),
    "Ackley": (BenchmarkFunctions.ackley, (-32, 32), 0.0),
    "Griewank": (BenchmarkFunctions.griewank, (-600, 600), 0.0),
    "Rastrigin": (BenchmarkFunctions.rastrigin, (-5, 5), 0.0),
    "Schwefel": (BenchmarkFunctions.schwefel, (-500, 500), 0.0),
    "Levy": (BenchmarkFunctions.levy, (-10, 10), 0.0),
    "Michalewicz": (BenchmarkFunctions.michalewicz, (0, np.pi), -9.66),
    "Perm": (BenchmarkFunctions.perm, (-10, 10), 0.0),
    "Katsuura": (BenchmarkFunctions.katsuura, (-100, 100), 0.0),
    "Happy Cat": (BenchmarkFunctions.happy_cat, (-2, 2), 0.0),
}


def get_function_characteristics(func_name: str) -> str:
    """Return characteristics of the benchmark function"""
    characteristics = {
        "Sphere": "Unimodal, Separable, Convex",
        "Ellipsoid": "Unimodal, Separable, Ill-conditioned",
        "Rosenbrock": "Unimodal, Non-separable, Narrow valley",
        "Quartic": "Unimodal, Noisy",
        "Step": "Discontinuous, Flat regions",
        "Zakharov": "Unimodal, Non-separable",
        "Ackley": "Multimodal, Many local minima",
        "Griewank": "Multimodal, Product-separable",
        "Rastrigin": "Highly multimodal, Regular",
        "Schwefel": "Multimodal, Deceptive",
        "Levy": "Multimodal",
        "Michalewicz": "Multimodal, Steep ridges",
        "Perm": "Multimodal",
        "Katsuura": "Highly rugged, Continuous multimodal",
        "Happy Cat": "Multimodal, Plateaus"
    }
    return characteristics.get(func_name, "Unknown")
