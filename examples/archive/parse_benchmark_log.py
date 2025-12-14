import re
import numpy as np

def parse_log(filename):
    results = {}
    current_func = None
    
    try:
        with open(filename, 'r', encoding='utf-16') as f:
            lines = f.readlines()
    except:
        with open(filename, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()

    for line in lines:
        if "Submitting tasks for" in line:
            # Extract function name, e.g., "F1 (Bent Cigar)"
            match = re.search(r"Submitting tasks for (.*)\.\.\.", line)
            if match:
                current_func = match.group(1)
                results[current_func] = []
        
        elif "Trial completed:" in line and current_func:
            # Extract score, e.g., "1.22e+01"
            # Line format: "  > Trial completed: 1.22e+01 (2/10)"
            match = re.search(r"Trial completed:\s+([0-9\.e\+\-]+)", line)
            if match:
                try:
                    val = float(match.group(1))
                    results[current_func].append(val)
                except ValueError:
                    pass

    with open('parsed_results.txt', 'w') as out:
        out.write(f"{'Function':<20} | {'Mean':<12} | {'Best':<12} | {'Std':<12} | {'Count':<5}\n")
        out.write("-" * 75 + "\n")
        
        for func, vals in results.items():
            if vals:
                v = np.array(vals)
                out.write(f"{func:<20} | {np.mean(v):.2e}     | {np.min(v):.2e}     | {np.std(v):.2e}     | {len(v)}\n")
            else:
                out.write(f"{func:<20} | N/A\n")

if __name__ == "__main__":
    parse_log('final_benchmark_51runs.txt')
