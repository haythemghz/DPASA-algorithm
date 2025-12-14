
import sys
import os
import time
import numpy as np
import traceback
import multiprocessing
from functools import partial

# Ensure root is in path
sys.path.append(os.getcwd())

from cec2022_impl import cec2022
from dpasa import DPASAOptimizer

# Configuration
RUNS = 51
DIM = 10
POP_SIZE = 100
ITERATIONS = 1000
BOUNDS = (-100, 100)

OUTPUT_FILE = "cec2022_results.txt"

# Function Definitions
# Must be picklable
FUNCTIONS = [
    ("F1", cec2022.F12022),
    ("F2", cec2022.F22022),
    ("F3", cec2022.F32022),
    ("F4", cec2022.F42022),
    ("F5", cec2022.F52022),
    ("F6", cec2022.F62022),
    ("F7", cec2022.F72022),
    ("F8", cec2022.F82022),
    ("F9", cec2022.F92022),
    ("F10", cec2022.F102022),
    ("F11", cec2022.F112022),
    ("F12", cec2022.F122022),
]

def run_single_trial(run_id, func_cls, dim, bounds, pop_size, iters):
    """
    Executes a single run of the benchmark.
    """
    try:
        # Re-import locally to ensure no pickling issues on Windows
        # (Though func_cls should be passed safely)
        
        # Initialize Benchmark
        bench = func_cls(ndim=dim)
        global_opt = bench.f_bias
        
        # Wrapper
        def objective_func(x):
            if hasattr(bench, 'evaluate'):
                if x.ndim == 2:
                    return np.array([bench.evaluate(xi) for xi in x])
                else:
                    return bench.evaluate(x)
            return 0.0

        # Initialize DPASA
        n_strategies = int(pop_size * 0.2)
        n_candidates = pop_size - n_strategies
        
        optimizer = DPASAOptimizer(
            objective_func=objective_func,
            dim=dim,
            bounds=bounds,
            n_strategies=n_strategies,
            n_candidates=n_candidates,
            max_iterations=iters,
            seed=int(time.time() * 1000) % 100000 + run_id
        )
        
        start_time = time.time()
        result_dict = optimizer.optimize(verbose=False)
        best_fitness = result_dict['best_fitness']
        elapsed = time.time() - start_time
        
        err = abs(best_fitness - global_opt)
        if err < 1e-8:
            err = 0.0
            
        return {
            "run": run_id,
            "best_fitness": best_fitness,
            "error": err,
            "time": elapsed,
            "status": "success"
        }
    except Exception as e:
        return {
            "run": run_id,
            "error": str(e),
            "status": "failed"
        }

def run_parallel_benchmark():
    print(f"Starting PARALLEL CEC 2022 Benchmark (D={DIM}, Runs={RUNS}, Pop={POP_SIZE}, Iters={ITERATIONS})")
    
    # Initialize file with header
    if not os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, "w") as f:
            f.write("Function,Run,BestFitness,Error,Time\n")

    # Use number of CPU cores minus 1 (or decent amount)
    cpu_count = max(1, multiprocessing.cpu_count() - 2) # Leave some room
    print(f"Using {cpu_count} Processes")

    with open(OUTPUT_FILE, "a") as f_out: # Append mode
        
        for func_name, func_cls in FUNCTIONS:
            print(f"\nProcessing {func_name} in parallel...")
            
            # Prepare args
            run_ids = list(range(1, RUNS + 1))
            
            # Use Pool
            pool = multiprocessing.Pool(processes=cpu_count)
            func_partial = partial(run_single_trial, func_cls=func_cls, dim=DIM, bounds=BOUNDS, pop_size=POP_SIZE, iters=ITERATIONS)
            
            results = []
            try:
                # Map
                for result in pool.imap_unordered(func_partial, run_ids):
                    if result["status"] == "success":
                        f_out.write(f"{func_name},{result['run']},{result['best_fitness']},{result['error']},{result['time']:.4f}\n")
                        f_out.flush()
                        results.append(result['error'])
                    else:
                        print(f"  Run {result['run']} FAILED: {result['error']}")
                
                pool.close()
                pool.join()
                
                # Stats
                if results:
                    mean = np.mean(results)
                    std = np.std(results)
                    print(f"  {func_name} Summary: Mean Error={mean:.2e}, Std={std:.2e}")
                    f_out.write(f"# SUMMARY {func_name}: Mean={mean}, Std={std}\n")
                    
            except KeyboardInterrupt:
                pool.terminate()
                print("Interrupted!")
                break
            except Exception as e:
                pool.terminate()
                print(f"Pool failed: {e}")
                traceback.print_exc()

if __name__ == "__main__":
    multiprocessing.freeze_support() # For Windows
    run_parallel_benchmark()

import traceback
import multiprocessing
from functools import partial

# Ensure root is in path
sys.path.append(os.getcwd())

from cec2022_impl import cec2022
from dpasa.algorithm import DPASAOptimizer

# Configuration
RUNS = 51
DIM = 10
POP_SIZE = 100
ITERATIONS = 1000
BOUNDS = (-100, 100)

OUTPUT_FILE = "cec2022_results.txt"

# Function Definitions
# Must be picklable
FUNCTIONS = [
    ("F1", cec2022.F12022),
    ("F2", cec2022.F22022),
    ("F3", cec2022.F32022),
    ("F4", cec2022.F42022),
    ("F5", cec2022.F52022),
    ("F6", cec2022.F62022),
    ("F7", cec2022.F72022),
    ("F8", cec2022.F82022),
    ("F9", cec2022.F92022),
    ("F10", cec2022.F102022),
    ("F11", cec2022.F112022),
    ("F12", cec2022.F122022),
]

def run_single_trial(run_id, func_cls, dim, bounds, pop_size, iters):
    """
    Executes a single run of the benchmark.
    """
    try:
        # Re-import locally to ensure no pickling issues on Windows
        # (Though func_cls should be passed safely)
        
        # Initialize Benchmark
        bench = func_cls(ndim=dim)
        global_opt = bench.f_bias
        
        # Wrapper
        def objective_func(x):
            if hasattr(bench, 'evaluate'):
                if x.ndim == 2:
                    return np.array([bench.evaluate(xi) for xi in x])
                else:
                    return bench.evaluate(x)
            return 0.0

        # Initialize SDM
        n_leaders = int(pop_size * 0.2)
        n_followers = pop_size - n_leaders
        
        optimizer = DPASAOptimizer(
            objective_func=objective_func,
            dim=dim,
            bounds=bounds,
            n_leaders=n_leaders,
            n_followers=n_followers,
            max_iterations=iters,
            seed=int(time.time() * 1000) % 100000 + run_id
        )
        
        start_time = time.time()
        result_dict = optimizer.optimize(verbose=False)
        best_fitness = result_dict['best_fitness']
        elapsed = time.time() - start_time
        
        err = abs(best_fitness - global_opt)
        if err < 1e-8:
            err = 0.0
            
        return {
            "run": run_id,
            "best_fitness": best_fitness,
            "error": err,
            "time": elapsed,
            "status": "success"
        }
    except Exception as e:
        return {
            "run": run_id,
            "error": str(e),
            "status": "failed"
        }

def run_parallel_benchmark():
    print(f"Starting PARALLEL CEC 2022 Benchmark (D={DIM}, Runs={RUNS}, Pop={POP_SIZE}, Iters={ITERATIONS})")
    
    # Initialize file with header
    if not os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, "w") as f:
            f.write("Function,Run,BestFitness,Error,Time\n")

    # Use number of CPU cores minus 1 (or decent amount)
    cpu_count = max(1, multiprocessing.cpu_count() - 2) # Leave some room
    print(f"Using {cpu_count} Processes")

    with open(OUTPUT_FILE, "a") as f_out: # Append mode
        
        for func_name, func_cls in FUNCTIONS:
            print(f"\nProcessing {func_name} in parallel...")
            
            # Prepare args
            run_ids = list(range(1, RUNS + 1))
            
            # Use Pool
            pool = multiprocessing.Pool(processes=cpu_count)
            func_partial = partial(run_single_trial, func_cls=func_cls, dim=DIM, bounds=BOUNDS, pop_size=POP_SIZE, iters=ITERATIONS)
            
            results = []
            try:
                # Map
                for result in pool.imap_unordered(func_partial, run_ids):
                    if result["status"] == "success":
                        f_out.write(f"{func_name},{result['run']},{result['best_fitness']},{result['error']},{result['time']:.4f}\n")
                        f_out.flush()
                        results.append(result['error'])
                    else:
                        print(f"  Run {result['run']} FAILED: {result['error']}")
                
                pool.close()
                pool.join()
                
                # Stats
                if results:
                    mean = np.mean(results)
                    std = np.std(results)
                    print(f"  {func_name} Summary: Mean Error={mean:.2e}, Std={std:.2e}")
                    f_out.write(f"# SUMMARY {func_name}: Mean={mean}, Std={std}\n")
                    
            except KeyboardInterrupt:
                pool.terminate()
                print("Interrupted!")
                break
            except Exception as e:
                pool.terminate()
                print(f"Pool failed: {e}")
                traceback.print_exc()

if __name__ == "__main__":
    multiprocessing.freeze_support() # For Windows
    run_parallel_benchmark()
