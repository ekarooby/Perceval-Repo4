import numpy as np
import perceval as pcvl
import perceval.components as comp
from math import pi

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
print(f"Z full circuit:           {Z_full:.4f}")

# ── Samuel's combined approach: only mode 0 PSs ───────────
# Replace each pair of PSs with single mode0 PS = phi0 - phi1
# Use BS.Rx (50:50) as Belenos uses
combined = pcvl.Circuit(2)
combined.add((0, 1), comp.BS.Rx())                    # 1st BS
combined.add(0,      comp.PS(0 - 1.570105))           # after 1st BS
combined.add((0, 1), comp.BS.Rx())                    # 2nd BS
combined.add(0,      comp.PS(-0.090646 - 5.121287))   # after 2nd BS
combined.add((0, 1), comp.BS.Rx())                    # 3rd BS
combined.add(0,      comp.PS(0 - 1.252545))           # after 3rd BS
combined.add((0, 1), comp.BS.Rx())                    # 4th BS
combined.add(0,      comp.PS(0 - 3.141593))           # after 4th BS

U2 = np.array(combined.compute_unitary())
psi2 = U2 @ np.array([1.0, 0.0])
Z_combined = abs(psi2[0])**2 - abs(psi2[1])**2
print(f"Z Samuel combined (phi0-phi1): {Z_combined:.4f}")

# ── Try phi1 - phi0 ───────────────────────────────────────
combined2 = pcvl.Circuit(2)
combined2.add((0, 1), comp.BS.Rx())
combined2.add(0,      comp.PS(1.570105 - 0))
combined2.add((0, 1), comp.BS.Rx())
combined2.add(0,      comp.PS(5.121287 - (-0.090646)))
combined2.add((0, 1), comp.BS.Rx())
combined2.add(0,      comp.PS(1.252545 - 0))
combined2.add((0, 1), comp.BS.Rx())
combined2.add(0,      comp.PS(3.141593 - 0))

U3 = np.array(combined2.compute_unitary())
psi3 = U3 @ np.array([1.0, 0.0])
Z_combined2 = abs(psi3[0])**2 - abs(psi3[1])**2
print(f"Z Samuel combined (phi1-phi0): {Z_combined2:.4f}")

print(f"\nExpected Z: {Z_full:.4f}")
print(f"Which matches? phi0-phi1={np.isclose(Z_full, Z_combined, atol=1e-3)}  phi1-phi0={np.isclose(Z_full, Z_combined2, atol=1e-3)}")