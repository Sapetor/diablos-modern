
import numpy as np
from scipy import signal

def test_discrete_tf():
    # Example: y[k] = 0.5 * y[k-1] + u[k]
    # H(z) = 1 / (1 - 0.5 z^-1) = z / (z - 0.5)
    # Num = [1, 0] (for z) or [1] (for 1)?
    # Let's use positive powers of z.
    # H(z) = z / (z - 0.5)
    # Num = [1, 0]
    # Den = [1, -0.5]
    
    num = [1, 0]
    den = [1, -0.5]
    
    print(f"Num: {num}")
    print(f"Den: {den}")
    
    A, B, C, D = signal.tf2ss(num, den)
    
    print("State Space Matrices:")
    print(f"A: {A}")
    print(f"B: {B}")
    print(f"C: {C}")
    print(f"D: {D}")
    
    # Simulate step response manually
    x = np.zeros((A.shape[0], 1))
    u = 1.0
    
    print("\nSimulation:")
    for k in range(5):
        y = C @ x + D * u
        x_new = A @ x + B * u
        print(f"k={k}, y={y.item()}, x={x.flatten()}")
        x = x_new

    # Expected:
    # k=0: y[0] = 1 (Direct feedthrough if proper?)
    # H(z) = z/(z-0.5) -> y[k] - 0.5y[k-1] = u[k] -> y[k] = 0.5y[k-1] + u[k]
    # y[0] = 0 + 1 = 1
    # y[1] = 0.5(1) + 1 = 1.5
    # y[2] = 0.5(1.5) + 1 = 1.75
    
if __name__ == "__main__":
    test_discrete_tf()
