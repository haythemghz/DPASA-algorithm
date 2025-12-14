
import os
import glob
import numpy as np

def parse_file(filepath):
    print(f"--- Peeking {filepath} ---")
    try:
        with open(filepath, 'r', encoding='utf-16') as f:
            print(f.readline().strip())
            print(f.readline().strip())
            encoding = 'utf-16'
    except:
        with open(filepath, 'r') as f:
            print(f.readline().strip())
            print(f.readline().strip())
            encoding = 'utf-8'

    data = {}
    with open(filepath, 'r', encoding=encoding) as f:
        for line in f:
            # Skip separators
            if '=' in line or 'Function' in line: continue
            
            # Format likely: Function, Run, Best, Error...
            parts = line.strip().split(',')
            if len(parts) < 3: 
                # Try tab/space verify
                parts = line.strip().split()
            
            if len(parts) < 3: continue
            
            try:
                # Identify function name (F1, F2...)
                func = parts[0]
                if not func.startswith('F'): continue
                
                # Try to find error value. 
                # If comma separated: Function, Run, Best, Error, Time
                if ',' in line:
                    error = float(parts[3])
                else:
                    # Space separated: F1 1 1.23 0.00 ...
                    error = float(parts[3]) # Index 3 typically error if Run is included
                
                if func not in data: data[func] = []
                data[func].append(error)
            except:
                pass
    
    for func in sorted(data.keys()):
        vals = data[func]
        print(f"{func}: Mean={np.mean(vals):.4e}, Min={np.min(vals):.4e}, Max={np.max(vals):.4e} (N={len(vals)})")

files = [
    'DPASA-algorithm/cec2022_results.txt',
    'DPASA-algorithm/final_benchmark_51runs.txt'
]

for f in files:
    if os.path.exists(f):
        parse_file(f)
    else:
        print(f"File not found: {f}")
