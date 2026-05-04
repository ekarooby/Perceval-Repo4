# This test code confirms that Samuel was correct that ignoring the input phase shifters 
# of our Perceval code, does not affect the Z= P0 - P1
# Therefore, we don't need input phase shifters on mode0 and mode1 on the Belenos chip


import numpy as np
import perceval as pcvl
import perceval.components as comp
from math import pi

# Test if ignoring input PSs changes Z significantly
FUNC_NAME    = "STEP"
ANGLE_METHOD = "pq"
L            = 1
x_val        = 0.5

theta_arr = np.load(f"theta_{FUNC_NAME.lower()}_{ANGLE_METHOD}_L{L}.npy")
phi_arr   = np.load(f"phi_{FUNC_NAME.lower()}_{ANGLE_METHOD}_L{L}.npy")

# ── Original full circuit ──────────────────────────────────
full = pcvl.Circuit(2)
full.add(0,      comp.PS(float(-phi_arr[0] / 2)))
full.add(1,      comp.PS(float( phi_arr[0] / 2)))
full.add((0, 1), comp.BS.Ry(theta=float(theta_arr[0])))
full.add(0,      comp.PS(float(-x_val / 2)))
full.add(1,      comp.PS(float( x_val / 2)))
full.add(0,      comp.PS(float(-phi_arr[1] / 2)))
full.add(1,      comp.PS(float( phi_arr[1] / 2)))
full.add((0, 1), comp.BS.Ry(theta=float(theta_arr[1])))

U = np.array(full.compute_unitary())
psi = U @ np.array([1.0, 0.0])
Z_full = abs(psi[0])**2 - abs(psi[1])**2
print(f"Z full circuit: {Z_full:.4f}")

# ── Circuit WITHOUT input PSs ──────────────────────────────
no_input = pcvl.Circuit(2)
no_input.add((0, 1), comp.BS.Ry(theta=float(theta_arr[0])))
no_input.add(0,      comp.PS(float(-x_val / 2)))
no_input.add(1,      comp.PS(float( x_val / 2)))
no_input.add(0,      comp.PS(float(-phi_arr[1] / 2)))
no_input.add(1,      comp.PS(float( phi_arr[1] / 2)))
no_input.add((0, 1), comp.BS.Ry(theta=float(theta_arr[1])))

U2 = np.array(no_input.compute_unitary())
psi2 = U2 @ np.array([1.0, 0.0])
Z_no_input = abs(psi2[0])**2 - abs(psi2[1])**2
print(f"Z without input PSs: {Z_no_input:.4f}")

print(f"\nDifference: {abs(Z_full - Z_no_input):.4f}")
print(f"Are they equal? {np.isclose(Z_full, Z_no_input, atol=1e-3)}")