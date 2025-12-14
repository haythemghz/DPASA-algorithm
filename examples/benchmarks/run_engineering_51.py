
import numpy as np
import sys
import os
import time

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dpasa import DPASAOptimizer

# --- Vectorized Problem Definitions ---

class WeldedBeamDesign:
    name = "Welded Beam"
    n_vars = 4
    n_constraints = 7
    bounds = [(0.1, 2.0), (0.1, 10.0), (0.1, 10.0), (0.1, 2.0)]
    best_known = 1.724852
    
    P = 6000; L = 14; E = 30e6; G = 12e6
    tau_max = 13600; sigma_max = 30000; delta_max = 0.25

    @classmethod
    def objective(cls, x):
        # Handle (N, 4) or (4,)
        if x.ndim == 2: h, l, t, b = x.T
        else: h, l, t, b = x
        return 1.10471 * h**2 * l + 0.04811 * t * b * (14.0 + l)

    @classmethod
    def constraints(cls, x):
        if x.ndim == 2: h, l, t, b = x.T
        else: h, l, t, b = x
        
        M = cls.P * (cls.L + l/2)
        R = np.sqrt(l**2/4 + ((h + t)/2)**2)
        J = 2 * (np.sqrt(2) * h * l * (l**2/12 + ((h + t)/2)**2))
        
        tau_prime = cls.P / (np.sqrt(2) * h * l)
        tau_double_prime = M * R / J
        tau = np.sqrt(tau_prime**2 + 2*tau_prime*tau_double_prime*(l/(2*R)) + tau_double_prime**2)
        sigma = 6 * cls.P * cls.L / (b * t**2)
        P_c = (4.013 * cls.E * np.sqrt(t**2 * b**6 / 36) / cls.L**2) * (1 - t/(2*cls.L) * np.sqrt(cls.E/(4*cls.G)))
        delta = 4 * cls.P * cls.L**3 / (cls.E * t**3 * b)
        
        # Helper for scalar/vector compatibility
        g0 = tau - cls.tau_max
        g1 = sigma - cls.sigma_max
        g2 = h - b
        g3 = 0.10471*h**2 + 0.04811*t*b*(14+l) - 5
        g4 = 0.125 - h
        g5 = delta - cls.delta_max
        g6 = cls.P - P_c

        if x.ndim == 2:
            return np.column_stack([g0, g1, g2, g3, g4, g5, g6])
        return np.array([g0, g1, g2, g3, g4, g5, g6])

class SpringDesign:
    name = "Spring Design"
    n_vars = 3
    n_constraints = 4
    bounds = [(0.05, 2.0), (0.25, 1.3), (2.0, 15.0)]
    best_known = 0.012665

    @classmethod
    def objective(cls, x):
        if x.ndim == 2: d, D, N = x.T
        else: d, D, N = x
        return (N + 2) * D * d**2

    @classmethod
    def constraints(cls, x):
        if x.ndim == 2: d, D, N = x.T
        else: d, D, N = x
        
        g0 = 1 - D**3 * N / (71785 * d**4)
        g1 = (4*D**2 - d*D) / (12566*(D*d**3 - d**4)) + 1/(5108*d**2) - 1
        g2 = 1 - 140.45 * d / (D**2 * N)
        g3 = (D + d) / 1.5 - 1
        
        if x.ndim == 2:
            return np.column_stack([g0, g1, g2, g3])
        return np.array([g0, g1, g2, g3])

class SpeedReducerDesign:
    name = "Speed Reducer"
    n_vars = 7
    n_constraints = 11
    bounds = [(2.6, 3.6), (0.7, 0.8), (17, 28), (7.3, 8.3), 
              (7.8, 8.3), (2.9, 3.9), (5.0, 5.5)]
    best_known = 2994.471

    @classmethod
    def objective(cls, x):
        if x.ndim == 2: b, m, z, l1, l2, d1, d2 = x.T
        else: b, m, z, l1, l2, d1, d2 = x
        return (0.7854 * b * m**2 * (3.3333*z**2 + 14.9334*z - 43.0934) -
                1.508 * b * (d1**2 + d2**2) +
                7.4777 * (d1**3 + d2**3) +
                0.7854 * (l1*d1**2 + l2*d2**2))

    @classmethod
    def constraints(cls, x):
        if x.ndim == 2: b, m, z, l1, l2, d1, d2 = x.T
        else: b, m, z, l1, l2, d1, d2 = x
        
        g0 = 27 / (b * m**2 * z) - 1
        g1 = 397.5 / (b * m**2 * z**2) - 1
        g2 = 1.93 * l1**3 / (m * z * d1**4) - 1
        g3 = 1.93 * l2**3 / (m * z * d2**4) - 1
        g4 = np.sqrt((745*l1/(m*z))**2 + 16.9e6) / (110 * d1**3) - 1
        g5 = np.sqrt((745*l2/(m*z))**2 + 157.5e6) / (85 * d2**3) - 1
        g6 = m * z / 40 - 1
        g7 = 5 * m / b - 1
        g8 = b / (12 * m) - 1
        g9 = (1.5*d1 + 1.9) / l1 - 1
        g10 = (1.1*d2 + 1.9) / l2 - 1
        
        if x.ndim == 2:
            return np.column_stack([g0, g1, g2, g3, g4, g5, g6, g7, g8, g9, g10])
        return np.array([g0, g1, g2, g3, g4, g5, g6, g7, g8, g9, g10])

class ThreeBarTruss:
    name = "Three-Bar Truss"
    n_vars = 2
    n_constraints = 3
    bounds = [(0.0, 1.0), (0.0, 1.0)]
    best_known = 263.8958
    L = 100; P = 2; sigma_max = 2

    @classmethod
    def objective(cls, x):
        if x.ndim == 2: A1, A2 = x.T
        else: A1, A2 = x
        return (2 * np.sqrt(2) * A1 + A2) * cls.L

    @classmethod
    def constraints(cls, x):
        if x.ndim == 2: A1, A2 = x.T
        else: A1, A2 = x
        
        eps = 1e-10
        denom_1 = np.sqrt(2) * A1**2 + 2*A1*A2 + eps
        denom_2 = A1 + np.sqrt(2) * A2 + eps
        
        g0 = (np.sqrt(2) * A1 + A2) / denom_1 * cls.P - cls.sigma_max
        g1 = A2 / denom_1 * cls.P - cls.sigma_max
        g2 = 1 / denom_2 * cls.P - cls.sigma_max
        
        if x.ndim == 2:
            return np.column_stack([g0, g1, g2])
        return np.array([g0, g1, g2])

class CantileverBeam:
    name = "Cantilever Beam"
    n_vars = 5
    n_constraints = 1
    bounds = [(0.01, 100.0)] * 5
    best_known = 1.3399

    @classmethod
    def objective(cls, x):
        if x.ndim == 2: return 0.0624 * np.sum(x, axis=1)
        return 0.0624 * np.sum(x)

    @classmethod
    def constraints(cls, x):
        if x.ndim == 2:
            deflection = 61/x[:,0]**3 + 37/x[:,1]**3 + 19/x[:,2]**3 + 7/x[:,3]**3 + 1/x[:,4]**3
            return (deflection - 1).reshape(-1, 1)
        else:
            deflection = 61/x[0]**3 + 37/x[1]**3 + 19/x[2]**3 + 7/x[3]**3 + 1/x[4]**3
            return np.array([deflection - 1])

# --- Run Configuration ---

def run_engineering_51():
    problems = [WeldedBeamDesign, SpringDesign, SpeedReducerDesign, ThreeBarTruss, CantileverBeam]
    
    n_runs = 51
    n_leaders = 150
    n_followers = 1600
    max_iter = 1000
    
    output_file = "engineering_51runs.txt"
    
    print(f"Starting Engineering Benchmark (51 runs, {n_leaders}/{n_followers}, {max_iter} iter)")
    
    with open(output_file, "w") as f:
        header = f"{'Problem':<20} | {'Mean':<12} | {'Best':<12} | {'Std':<12} | {'Gap(Mean)%':<12} | {'Feasible':<10}\n"
        f.write(header)
        f.write("-" * 90 + "\n")
        f.flush()
        print(header, end='', flush=True)
        print("-" * 90, flush=True)
        
        for prob in problems:
            results = []
            feasibles = 0
            
            for i in range(n_runs):
                np.random.seed(i) # Reproducible
                
                optimizer = DPASAOptimizer(
                    objective_func=prob.objective,
                    constraint_func=prob.constraints,
                    dim=prob.n_vars,
                    bounds=prob.bounds,
                    n_leaders=n_leaders,
                    n_followers=n_followers,
                    max_iterations=max_iter,
                    n_cultures=5
                )
                
                res = optimizer.optimize(verbose=False)
                
                # Check feasibility
                x_best = res['best_solution']
                g = prob.constraints(x_best)
                if isinstance(g, np.ndarray) and g.ndim == 2: g = g.flatten()
                
                viol = np.sum(np.maximum(0, g)**2)
                is_feas = viol < 1e-6
                
                if is_feas:
                    results.append(res['best_fitness'])
                    feasibles += 1
                else:
                    results.append(res['best_fitness'])
                
                if (i+1) % 5 == 0:
                    print(f"[{prob.name}] Run {i+1}/{n_runs} complete...", flush=True)
            
            results = np.array(results)
            mean_val = np.mean(results)
            best_val = np.min(results)
            std_val = np.std(results)
            gap_mean = 100 * (mean_val - prob.best_known) / prob.best_known
            
            line = f"{prob.name:<20} | {mean_val:<12.5e} | {best_val:<12.5e} | {std_val:<12.5e} | {gap_mean:<12.2f} | {feasibles}/{n_runs}\n"
            print(line, end='', flush=True)
            f.write(line)
            f.flush()

if __name__ == "__main__":
    run_engineering_51()
