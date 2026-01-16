
import numpy as np
from scipy.integrate import solve_ivp
import time

def test_sine_integration():
    T = 10000.0
    dt = 0.01
    
    print(f"Testing Sine Integration for T={T}s, dt={dt}")
    
    # dy/dt = sin(t). y(0)=0.
    # y(t) = 1 - cos(t).
    
    def model_func(t, y):
        return [np.sin(t)]
    
    y0 = [0.0]
    t_span = (0.0, T)
    t_eval = np.arange(0, T + dt, dt)
    
    start = time.time()
    # Using tighter tolerances
    sol = solve_ivp(model_func, t_span, y0, t_eval=t_eval, method='RK45', rtol=1e-9, atol=1e-12)
    dur = time.time() - start
    
    print(f"Solver finished in {dur:.4f}s. Success: {sol.success}")
    print(f"Message: {sol.message}")
    
    if not sol.success:
        return
        
    y_out = sol.y[0]
    t_out = sol.t
    
    print(f"Output shape: {y_out.shape}")
    
    # Error analysis
    y_true = 1.0 - np.cos(t_out)
    error = np.abs(y_out - y_true)
    
    print(f"Max Error: {np.max(error):.4e}")
    print(f"Mean Error: {np.mean(error):.4e}")
    
    # Check end of simulation stats
    print(f"Values at end (last 10): {y_out[-10:]}")
    print(f"True Values at end: {y_true[-10:]}")
    
    if np.max(error) < 1e-3:
        print("Test PASSED.")
    else:
        print("Test FAILED (Significant drift/error).")

if __name__ == "__main__":
    test_sine_integration()
