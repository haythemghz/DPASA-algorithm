"""
Engineering Optimization Benchmark Problems
=============================================

Classical constrained engineering design problems for algorithm validation.

Problems included:
- Welded Beam Design (4 vars, 7 constraints)
- Tension/Compression Spring Design (3 vars, 4 constraints)
- Speed Reducer Design (7 vars, 11 constraints)

References:
- Coello (2000): Constraint handling in evolutionary algorithms
- Deb (1991): Optimal design using GA
- Mezura-Montes & Coello (2011): Constraint-handling review
"""

import numpy as np
from typing import Tuple, Dict, Callable


class WeldedBeamDesign:
    """
    Welded Beam Design Problem
    
    Minimize the cost of a welded beam subject to constraints on
    shear stress, bending stress, buckling load, and end deflection.
    
    Variables:
        x[0] = h (weld thickness)
        x[1] = l (weld length)  
        x[2] = t (beam thickness)
        x[3] = b (beam width)
    
    Best known: f* = 1.724852
    Reference: Coello (2000), Deb (1991)
    """
    
    name = "Welded Beam"
    n_vars = 4
    n_constraints = 7
    bounds = [(0.1, 2.0), (0.1, 10.0), (0.1, 10.0), (0.1, 2.0)]
    best_known = 1.724852
    
    # Problem constants
    P = 6000  # Applied load (lb)
    L = 14    # Beam length (in)
    E = 30e6  # Young's modulus (psi)
    G = 12e6  # Shear modulus (psi)
    tau_max = 13600   # Max shear stress (psi)
    sigma_max = 30000 # Max bending stress (psi)
    delta_max = 0.25  # Max deflection (in)
    
    @classmethod
    def objective(cls, x: np.ndarray) -> float:
        """Cost function: material + welding costs"""
        h, l, t, b = x
        return 1.10471 * h**2 * l + 0.04811 * t * b * (14.0 + l)
    
    @classmethod
    def constraints(cls, x: np.ndarray) -> np.ndarray:
        """Return constraint violations (g <= 0 is feasible)"""
        h, l, t, b = x
        
        # Intermediate calculations
        M = cls.P * (cls.L + l/2)
        R = np.sqrt(l**2/4 + ((h + t)/2)**2)
        J = 2 * (np.sqrt(2) * h * l * (l**2/12 + ((h + t)/2)**2))
        
        tau_prime = cls.P / (np.sqrt(2) * h * l)
        tau_double_prime = M * R / J
        tau = np.sqrt(tau_prime**2 + 2*tau_prime*tau_double_prime*(l/(2*R)) + tau_double_prime**2)
        
        sigma = 6 * cls.P * cls.L / (b * t**2)
        
        P_c = (4.013 * cls.E * np.sqrt(t**2 * b**6 / 36) / cls.L**2) * (1 - t/(2*cls.L) * np.sqrt(cls.E/(4*cls.G)))
        
        delta = 4 * cls.P * cls.L**3 / (cls.E * t**3 * b)
        
        g = np.zeros(7)
        g[0] = tau - cls.tau_max          # Shear stress
        g[1] = sigma - cls.sigma_max      # Bending stress
        g[2] = h - b                       # Side constraint
        g[3] = 0.10471*h**2 + 0.04811*t*b*(14+l) - 5  # Cost limit
        g[4] = 0.125 - h                   # Min weld thickness
        g[5] = delta - cls.delta_max       # Deflection
        g[6] = cls.P - P_c                 # Buckling
        
        return g
    
    @classmethod
    def evaluate(cls, x: np.ndarray, penalty: float = 1e6) -> float:
        """Evaluate with penalty for constraint violations"""
        obj = cls.objective(x)
        g = cls.constraints(x)
        violation = np.sum(np.maximum(0, g)**2)
        return obj + penalty * violation


    



class SpringDesign:
    """
    Tension/Compression Spring Design Problem
    
    Minimize the weight of a tension/compression spring.
    
    Variables:
        x[0] = d (wire diameter)
        x[1] = D (mean coil diameter)
        x[2] = N (number of active coils)
    
    Best known: f* = 0.012665
    Reference: Belegundu (1982), Arora (1989)
    """
    
    name = "Spring Design"
    n_vars = 3
    n_constraints = 4
    bounds = [(0.05, 2.0), (0.25, 1.3), (2.0, 15.0)]
    best_known = 0.012665
    
    @classmethod
    def objective(cls, x: np.ndarray) -> float:
        """Weight of spring"""
        d, D, N = x
        return (N + 2) * D * d**2
    
    @classmethod
    def constraints(cls, x: np.ndarray) -> np.ndarray:
        """Return constraint violations (g <= 0 is feasible)"""
        d, D, N = x
        
        g = np.zeros(4)
        g[0] = 1 - D**3 * N / (71785 * d**4)  # Shear stress
        g[1] = (4*D**2 - d*D) / (12566*(D*d**3 - d**4)) + 1/(5108*d**2) - 1  # Surge frequency
        g[2] = 1 - 140.45 * d / (D**2 * N)     # Deflection
        g[3] = (D + d) / 1.5 - 1               # Diameter constraint
        
        return g
    
    @classmethod
    def evaluate(cls, x: np.ndarray, penalty: float = 1e6) -> float:
        """Evaluate with penalty for constraint violations"""
        obj = cls.objective(x)
        g = cls.constraints(x)
        violation = np.sum(np.maximum(0, g)**2)
        return obj + penalty * violation


class SpeedReducerDesign:
    """
    Speed Reducer (Gear Box) Design Problem
    
    Minimize the weight of a speed reducer.
    
    Variables:
        x[0] = b (face width)
        x[1] = m (module of teeth)
        x[2] = z (number of teeth on pinion) - integer
        x[3] = l1 (length of first shaft)
        x[4] = l2 (length of second shaft)
        x[5] = d1 (diameter of first shaft)
        x[6] = d2 (diameter of second shaft)
    
    Best known: f* = 2994.471
    Reference: Golinski (1970)
    """
    
    name = "Speed Reducer"
    n_vars = 7
    n_constraints = 11
    bounds = [(2.6, 3.6), (0.7, 0.8), (17, 28), (7.3, 8.3), 
              (7.8, 8.3), (2.9, 3.9), (5.0, 5.5)]
    best_known = 2994.471
    
    @classmethod
    def objective(cls, x: np.ndarray) -> float:
        """Weight of speed reducer"""
        b, m, z, l1, l2, d1, d2 = x
        
        return (0.7854 * b * m**2 * (3.3333*z**2 + 14.9334*z - 43.0934) -
                1.508 * b * (d1**2 + d2**2) +
                7.4777 * (d1**3 + d2**3) +
                0.7854 * (l1*d1**2 + l2*d2**2))
    
    @classmethod
    def constraints(cls, x: np.ndarray) -> np.ndarray:
        """Return constraint violations (g <= 0 is feasible)"""
        b, m, z, l1, l2, d1, d2 = x
        
        g = np.zeros(11)
        g[0] = 27 / (b * m**2 * z) - 1
        g[1] = 397.5 / (b * m**2 * z**2) - 1
        g[2] = 1.93 * l1**3 / (m * z * d1**4) - 1
        g[3] = 1.93 * l2**3 / (m * z * d2**4) - 1
        g[4] = np.sqrt((745*l1/(m*z))**2 + 16.9e6) / (110 * d1**3) - 1
        g[5] = np.sqrt((745*l2/(m*z))**2 + 157.5e6) / (85 * d2**3) - 1
        g[6] = m * z / 40 - 1
        g[7] = 5 * m / b - 1
        g[8] = b / (12 * m) - 1
        g[9] = (1.5*d1 + 1.9) / l1 - 1
        g[10] = (1.1*d2 + 1.9) / l2 - 1
        
        return g
    
    @classmethod
    def evaluate(cls, x: np.ndarray, penalty: float = 1e6) -> float:
        """Evaluate with penalty for constraint violations"""
        obj = cls.objective(x)
        g = cls.constraints(x)
        violation = np.sum(np.maximum(0, g)**2)
        return obj + penalty * violation


class ThreeBarTruss:
    """
    Three-Bar Truss Design Problem
    
    Minimize the volume (weight) of a three-bar planar truss subject to
    stress constraints.
    
    Variables:
        x[0] = A1 (cross-sectional area of bars 1 and 3)
        x[1] = A2 (cross-sectional area of bar 2)
    
    Best known: f* = 263.8958
    Reference: Coello (2000), Ray & Liew (2003)
    """
    
    name = "Three-Bar Truss"
    n_vars = 2
    n_constraints = 3
    bounds = [(0.0, 1.0), (0.0, 1.0)]
    best_known = 263.8958
    
    # Problem constants
    L = 100  # Length (cm)
    P = 2    # Load (kN/cm^2)
    sigma_max = 2  # Max stress (kN/cm^2)
    
    @classmethod
    def objective(cls, x: np.ndarray) -> float:
        """Volume (weight) of truss"""
        A1, A2 = x
        return (2 * np.sqrt(2) * A1 + A2) * cls.L
    
    @classmethod
    def constraints(cls, x: np.ndarray) -> np.ndarray:
        """Return constraint violations (g <= 0 is feasible)"""
        A1, A2 = x
        
        g = np.zeros(3)
        denom1 = 2 * A1 + np.sqrt(2) * A2
        denom2 = 2 * A1 + np.sqrt(2) * A2
        
        # Add epsilon to prevent division by zero
        eps = 1e-10
        
        # Stress constraints (normalized)
        denom_1 = np.sqrt(2) * A1**2 + 2*A1*A2 + eps
        denom_2 = A1 + np.sqrt(2) * A2 + eps
        
        g[0] = (np.sqrt(2) * A1 + A2) / denom_1 * cls.P - cls.sigma_max
        g[1] = A2 / denom_1 * cls.P - cls.sigma_max
        g[2] = 1 / denom_2 * cls.P - cls.sigma_max
        
        return g
    
    @classmethod
    def evaluate(cls, x: np.ndarray, penalty: float = 1e6) -> float:
        """Evaluate with penalty for constraint violations"""
        obj = cls.objective(x)
        g = cls.constraints(x)
        violation = np.sum(np.maximum(0, g)**2)
        return obj + penalty * violation


class CantileverBeam:
    """
    Cantilever Beam Design Problem
    
    Minimize the weight of a cantilever beam consisting of 5 hollow
    square cross-sections.
    
    Variables:
        x[0] to x[4] = widths of 5 sections
    
    Best known: f* = 1.3399
    Reference: Chickermane & Gea (1996)
    """
    
    name = "Cantilever Beam"
    n_vars = 5
    n_constraints = 1
    bounds = [(0.01, 100.0)] * 5
    best_known = 1.3399
    
    # Problem constants
    L = 100  # Total length
    P = 50000  # Applied load (N)
    E = 2e7   # Young's modulus
    delta_max = 2.7  # Max tip deflection
    
    @classmethod
    def objective(cls, x: np.ndarray) -> float:
        """Weight of beam (sum of section weights)"""
        return 0.0624 * np.sum(x)
    
    @classmethod
    def constraints(cls, x: np.ndarray) -> np.ndarray:
        """Return constraint violations (g <= 0 is feasible)"""
        # Deflection constraint
        deflection = 61 / x[0]**3 + 37 / x[1]**3 + 19 / x[2]**3 + 7 / x[3]**3 + 1 / x[4]**3
        g = np.array([deflection - 1])  # Normalized to 1
        return g
    
    @classmethod
    def evaluate(cls, x: np.ndarray, penalty: float = 1e6) -> float:
        """Evaluate with penalty for constraint violations"""
        obj = cls.objective(x)
        g = cls.constraints(x)
        violation = np.sum(np.maximum(0, g)**2)
        return obj + penalty * violation


class GearTrainDesign:
    """
    Gear Train Design Problem
    
    Minimize the error between a target gear ratio and the actual ratio
    produced by a compound gear train.
    
    Variables:
        x[0] to x[3] = number of teeth on 4 gears (integers 12-60)
    
    Best known: f* = 2.7e-12 (nearly zero)
    Reference: Sandgren (1990)
    """
    
    name = "Gear Train"
    n_vars = 4
    n_constraints = 0  # Only bound constraints
    bounds = [(12, 60)] * 4  # Integer bounds
    best_known = 2.7e-12
    
    # Target gear ratio
    target_ratio = 1.0 / 6.931
    
    @classmethod
    def objective(cls, x: np.ndarray) -> float:
        """Squared error between actual and target gear ratio"""
        # Round to integers since teeth counts are discrete
        n = np.round(x).astype(int)
        n = np.clip(n, 12, 60)
        
        # Gear ratio: (n[3] * n[1]) / (n[2] * n[0])
        actual_ratio = (n[3] * n[1]) / (n[2] * n[0])
        error = (cls.target_ratio - actual_ratio)**2
        return error
    
    @classmethod
    def constraints(cls, x: np.ndarray) -> np.ndarray:
        """No constraints (only bounds)"""
        return np.array([])
    
    @classmethod
    def evaluate(cls, x: np.ndarray, penalty: float = 1e6) -> float:
        """Evaluate (no constraints to penalize)"""
        return cls.objective(x)



# Engineering problems suite
ENGINEERING_PROBLEMS = {
    "Welded Beam": WeldedBeamDesign,
    "Spring Design": SpringDesign,
    "Speed Reducer": SpeedReducerDesign,
    "Three-Bar Truss": ThreeBarTruss,
    "Cantilever Beam": CantileverBeam,
    "Gear Train": GearTrainDesign,
}


def get_problem_info(name: str) -> Dict:
    """Get information about an engineering problem"""
    if name not in ENGINEERING_PROBLEMS:
        raise ValueError(f"Unknown problem: {name}")
    
    prob = ENGINEERING_PROBLEMS[name]
    return {
        "name": prob.name,
        "n_vars": prob.n_vars,
        "n_constraints": prob.n_constraints,
        "bounds": prob.bounds,
        "best_known": prob.best_known,
    }
