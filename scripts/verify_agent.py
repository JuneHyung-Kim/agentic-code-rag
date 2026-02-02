import sys
import os

# Add src to python path
sys.path.append(os.path.join(os.getcwd(), "src"))

try:
    from agent.core import CodeAgent
    print("Import successful")
    
    # Try initialization (might fail if keys aren't set, but we just want to check imports)
    try:
        agent = CodeAgent()
        print("Initialization successful (or mocked)")
    except Exception as e:
        print(f"Initialization failed (expected if keys missing): {e}")

except Exception as e:
    print(f"Import failed: {e}")
    sys.exit(1)
