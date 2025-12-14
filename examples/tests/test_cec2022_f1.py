
import sys
import os
sys.path.append(os.getcwd()) # Ensure root is in path

try:
    from cec2022_impl.cec2022 import F12022
    import numpy as np

    print("Successfully imported F12022")
    
    # Initialize Function
    f1 = F12022(ndim=10)
    print("Initialized F12022(ndim=10)")
    
    # Test Evaluation
    x = np.zeros(10) # Should be shifted? F1 optimum is at shift vector.
    # Check shift vector
    print(f"Shift vector[0]: {f1.f_shift[0]}")
    
    # Evaluate at some point
    score = f1.evaluate(x)
    print(f"Evaluated f(0) = {score}")
    
    # Evaluate at optimum (approx)
    opt = f1.f_shift
    score_opt = f1.evaluate(opt)
    print(f"Evaluated f(opt) = {score_opt} (Expected: {f1.f_bias})")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
