
try:
    import mealpy
    print(f"Mealpy version: {mealpy.__version__}")
    # Inspect for CEC2024
    # Common locations for benchmarks in mealpy
    import inspect
    from mealpy.utils import problem
    
    print("Checking mealpy.utils.problem for CEC2024...")
    # This is a guess, I need to see what's available
    # Actually mealpy >= 3.0 has a different structure often
    
except ImportError as e:
    print(f"ImportError: {e}")

try:
    import opfunu
    print(f"Opfunu version: {opfunu.__version__}")
    # Check if cec2024 is in opfunu
    if hasattr(opfunu, 'cec_based'):
        print("opfunu.cec_based contents:", dir(opfunu.cec_based))
    
    # Try to find 2024
    import opfunu.cec_based.cec2022 as cec2022
    print("Found CEC2022")
    
    try:
        import opfunu.cec_based.cec2024 as cec2024
        print("Found CEC2024 in opfunu!")
        print(dir(cec2024))
    except ImportError:
        print("CEC2024 not found in opfunu.cec_based")

except ImportError as e:
    print(f"Opfunu ImportError: {e}")
