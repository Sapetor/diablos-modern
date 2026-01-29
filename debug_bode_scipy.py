
import numpy as np
from scipy import signal

# Define system: z / (z - 0.5), dt = 1.0
num = [1.0, 0.0]
den = [1.0, -0.5]
dt = 1.0

print(f"System: Num={num}, Den={den}, dt={dt}")
sys = signal.TransferFunction(num, den, dt=dt)

# Calculate Bode at specific freq points
w_targets = [0.001, np.pi/2, np.pi] # approx DC, Half-band, Nyquist
w_out, mag_out, phase_out = sys.bode(w=w_targets)

print("\n--- Scipy Bode Results ---")
for i in range(len(w_targets)):
    print(f"w = {w_targets[i]:.4f} rad/s")
    print(f"  Mag = {mag_out[i]:.4f} dB")
    print(f"  Phase = {phase_out[i]:.4f} deg")

# Manual Calculation
# H(z) = z / (z - 0.5)
# z = exp(j w dt)
# z_nyq = exp(j * pi * 1) = -1
# H(-1) = -1 / (-1 - 0.5) = -1 / -1.5 = 2/3 = 0.666
# Mag = 20 log10(0.666) approx -3.52 dB

print("\n--- Manual Check (Nyquist) ---")
mag_nyq_calc = 20 * np.log10(2/3)
print(f"Manual Mag at Nyquist: {mag_nyq_calc:.4f} dB")

print("\n--- Manual Check (w=0.001 approx DC) ---")
# z -> 1
# H(1) = 1 / (1 - 0.5) = 2
# Mag = 20 log10(2) approx 6.02 dB
mag_dc_calc = 20 * np.log10(2)
print(f"Manual Mag at DC: {mag_dc_calc:.4f} dB")
