import numpy as np
import perceval as pcvl
import perceval.components as comp
from perceval.components import BS, PS
import re

# ============================================================
# QSP CIRCUIT LAYER-BY-LAYER DECOMPOSITION INTO BELENOS MZIs
# ============================================================
#
# GOAL:
#   Implement the full QSP circuit for L=1, x=0.5 directly
#   on the Belenos chip, gate by gate, without letting the
#   compiler collapse everything into a single 2x2 unitary.
#
# APPROACH:
#   1. Decompose each Ry(theta_j) gate into a physical MZI:
#      BS() -- PS(phi0) -- BS() -- PS(phi1)
#   2. Combine adjacent PSs on the same mode by summing them
#   3. Verify the rebuilt circuit matches the original unitary
#   4. Use Samuel's code to set PS values directly on Belenos
#
# WHY pcvl.random_seed(0):
#   Circuit.decomposition() uses numerical search from random
#   starting point. Seed 0 was found to give correct phi_out=pi
#   for both gates. Other seeds may give wrong solutions.
# ============================================================

# ── Reproducibility ───────────────────────────────────────
pcvl.random_seed(0)

# ── Settings ──────────────────────────────────────────────
FUNC_NAME    = "STEP"
ANGLE_METHOD = "pq"
L            = 1
x_val        = 0.5

# ── Load angles ───────────────────────────────────────────
theta_arr = np.load(f"theta_{FUNC_NAME.lower()}_{ANGLE_METHOD}_L{L}.npy")
phi_arr   = np.load(f"phi_{FUNC_NAME.lower()}_{ANGLE_METHOD}_L{L}.npy")

print(f"Loaded angles: L={L}, x={x_val}")
print(f"theta_arr: {np.round(theta_arr, 6)}")
print(f"phi_arr:   {np.round(phi_arr, 6)}")

# ── Helper: extract phi from PS description string ────────
def extract_phi(desc):
    m = re.search(r"phi\s*=\s*([\-\+]?[\d\.eE\+\-]+|pi)", desc)
    if m:
        val_str = m.group(1)
        return np.pi if val_str == "pi" else float(val_str)
    return None

# ── MZI building block (Belenos style: BS() = BS.Rx 50:50) ─
mzi = pcvl.Circuit(2) // BS() // (1, PS(pcvl.P("phi0"))) // BS() // (1, PS(pcvl.P("phi1")))

# ── Decompose a single Ry gate into MZI PS values ─────────
def decompose_ry(theta_j):
    """
    Decompose BS.Ry(theta_j) into:
      BS() -- PS(phi0) -- BS() -- PS(phi1)
    Returns: phi_in_m0, phi_in_m1, phi_mid_m1, phi_out_m1
    """
    single_gate = comp.BS.Ry(theta=float(theta_j))
    U_gate = pcvl.Matrix(np.array(single_gate.compute_unitary()).tolist())
    decomposed = pcvl.Circuit.decomposition(U_gate, mzi, phase_shifter_fn=PS)
    all_components = list(decomposed)

    bs_count = 0
    mode0_ps = []
    mode1_ps = []
    for modes, component in all_components:
        desc = component.describe()
        if desc.startswith("BS"):
            bs_count += 1
        if desc.startswith("PS(") and modes == (0,):
            phi = extract_phi(desc)
            if phi is not None:
                mode0_ps.append(phi)
        if desc.startswith("PS(") and modes == (1,):
            phi = extract_phi(desc)
            if phi is not None:
                mode1_ps.append((bs_count, phi))

    phi_in_m0  = mode0_ps[0] if mode0_ps else 0.0
    phi_in_m1  = next((phi for bc, phi in mode1_ps if bc == 0), 0.0)
    phi_mid_m1 = next((phi for bc, phi in mode1_ps if bc == 1), 0.0)
    phi_out_m1 = next((phi for bc, phi in mode1_ps if bc == 2), 0.0)
    return phi_in_m0, phi_in_m1, phi_mid_m1, phi_out_m1

# ── Step 1: Decompose Ry(theta_0) ─────────────────────────
in0_m0, in0_m1, mid0_m1, out0_m1 = decompose_ry(theta_arr[0])
print(f"\n=== Ry(theta_0={theta_arr[0]:.4f}) MZI values ===")
print(f"  phi_in  mode0 = {in0_m0:.6f} rad ({np.degrees(in0_m0):.4f} deg)")
print(f"  phi_in  mode1 = {in0_m1:.6f} rad ({np.degrees(in0_m1):.4f} deg)")
print(f"  phi_mid mode1 = {mid0_m1:.6f} rad ({np.degrees(mid0_m1):.4f} deg)")
print(f"  phi_out mode1 = {out0_m1:.6f} rad ({np.degrees(out0_m1):.4f} deg)")

# ── Step 2: Decompose Ry(theta_1) ─────────────────────────
in1_m0, in1_m1, mid1_m1, out1_m1 = decompose_ry(theta_arr[1])
print(f"\n=== Ry(theta_1={theta_arr[1]:.4f}) MZI values ===")
print(f"  phi_in  mode0 = {in1_m0:.6f} rad ({np.degrees(in1_m0):.4f} deg)")
print(f"  phi_in  mode1 = {in1_m1:.6f} rad ({np.degrees(in1_m1):.4f} deg)")
print(f"  phi_mid mode1 = {mid1_m1:.6f} rad ({np.degrees(mid1_m1):.4f} deg)")
print(f"  phi_out mode1 = {out1_m1:.6f} rad ({np.degrees(out1_m1):.4f} deg)")

# ── Step 3: QSP PS values ─────────────────────────────────
rz_phi0_m0 = -phi_arr[0] / 2
rz_phi0_m1 = +phi_arr[0] / 2
rz_x_m0    = -x_val / 2
rz_x_m1    = +x_val / 2
rz_phi1_m0 = -phi_arr[1] / 2
rz_phi1_m1 = +phi_arr[1] / 2

print(f"\n=== QSP PS values ===")
print(f"  Rz(phi_0) mode0 = {rz_phi0_m0:.6f} rad ({np.degrees(rz_phi0_m0):.4f} deg)")
print(f"  Rz(phi_0) mode1 = {rz_phi0_m1:.6f} rad ({np.degrees(rz_phi0_m1):.4f} deg)")
print(f"  Rz(x=0.5) mode0 = {rz_x_m0:.6f} rad ({np.degrees(rz_x_m0):.4f} deg)")
print(f"  Rz(x=0.5) mode1 = {rz_x_m1:.6f} rad ({np.degrees(rz_x_m1):.4f} deg)")
print(f"  Rz(phi_1) mode0 = {rz_phi1_m0:.6f} rad ({np.degrees(rz_phi1_m0):.4f} deg)")
print(f"  Rz(phi_1) mode1 = {rz_phi1_m1:.6f} rad ({np.degrees(rz_phi1_m1):.4f} deg)")

# ── Step 4: Combine adjacent PSs ──────────────────────────
# MZI_0 input: Rz(phi_0) + Ry0_in
mzi0_input_m0 = rz_phi0_m0 + in0_m0
mzi0_input_m1 = rz_phi0_m1 + in0_m1

# Between MZI_0 and MZI_1: Ry0_out + Rz(x) + Rz(phi_1) + Ry1_in
# Note: mode 0 has no output PS from MZI
between_m0 = rz_x_m0 + rz_phi1_m0 + in1_m0
between_m1 = out0_m1 + rz_x_m1 + rz_phi1_m1 + in1_m1

print(f"\n=== Combined Adjacent PSs ===")
print(f"  MZI_0 input  mode0 = {rz_phi0_m0:.4f} + {in0_m0:.4f} = {mzi0_input_m0:.6f} rad ({np.degrees(mzi0_input_m0):.4f} deg)")
print(f"  MZI_0 input  mode1 = {rz_phi0_m1:.4f} + {in0_m1:.4f} = {mzi0_input_m1:.6f} rad ({np.degrees(mzi0_input_m1):.4f} deg)")
print(f"  MZI_0 middle mode1 = {mid0_m1:.6f} rad ({np.degrees(mid0_m1):.4f} deg)  [unchanged]")
print(f"  Between MZIs mode0 = {rz_x_m0:.4f} + {rz_phi1_m0:.4f} + {in1_m0:.4f} = {between_m0:.6f} rad ({np.degrees(between_m0):.4f} deg)")
print(f"  Between MZIs mode1 = {out0_m1:.4f} + {rz_x_m1:.4f} + {rz_phi1_m1:.4f} + {in1_m1:.4f} = {between_m1:.6f} rad ({np.degrees(between_m1):.4f} deg)")
print(f"  MZI_1 middle mode1 = {mid1_m1:.6f} rad ({np.degrees(mid1_m1):.4f} deg)  [unchanged]")
print(f"  MZI_1 output mode1 = {out1_m1:.6f} rad ({np.degrees(out1_m1):.4f} deg)  [unchanged]")

# ── Step 5: Rebuild circuit and verify ────────────────────
original_circuit = pcvl.Circuit(2)
original_circuit.add(0,      comp.PS(float(-phi_arr[0] / 2)))
original_circuit.add(1,      comp.PS(float( phi_arr[0] / 2)))
original_circuit.add((0, 1), comp.BS.Ry(theta=float(theta_arr[0])))
original_circuit.add(0,      comp.PS(float(-x_val / 2)))
original_circuit.add(1,      comp.PS(float( x_val / 2)))
original_circuit.add(0,      comp.PS(float(-phi_arr[1] / 2)))
original_circuit.add(1,      comp.PS(float( phi_arr[1] / 2)))
original_circuit.add((0, 1), comp.BS.Ry(theta=float(theta_arr[1])))

U_original = np.array(original_circuit.compute_unitary())
print(f"\n=== Original QSP unitary ===")
print(np.round(U_original, 4))

# Rebuilt circuit using decomposed PS values
rebuilt = pcvl.Circuit(2)
rebuilt.add(0,      comp.PS(float(mzi0_input_m0)))
rebuilt.add(1,      comp.PS(float(mzi0_input_m1)))
rebuilt.add((0, 1), comp.BS.Rx())
rebuilt.add(1,      comp.PS(float(mid0_m1)))
rebuilt.add((0, 1), comp.BS.Rx())
rebuilt.add(0,      comp.PS(float(between_m0)))
rebuilt.add(1,      comp.PS(float(between_m1)))
rebuilt.add((0, 1), comp.BS.Rx())
rebuilt.add(1,      comp.PS(float(mid1_m1)))
rebuilt.add((0, 1), comp.BS.Rx())
rebuilt.add(1,      comp.PS(float(out1_m1)))

U_rebuilt = np.array(rebuilt.compute_unitary())
print(f"\n=== Rebuilt circuit unitary ===")
print(np.round(U_rebuilt, 4))

error = np.linalg.norm(U_original - U_rebuilt)
print(f"\n=== Verification ===")
print(f"Error: {error:.2e}")
print(f"Match: {np.allclose(U_original, U_rebuilt, atol=1e-3)}")

# ── Step 6: Final PS summary ──────────────────────────────
print(f"\n{'='*65}")
print(f"FINAL PS VALUES FOR BELENOS (L={L}, x={x_val})")
print(f"{'='*65}")
print(f"{'Position':<28} {'Mode':<6} {'phi (rad)':>10} {'phi (deg)':>12}")
print(f"-"*60)
print(f"{'MZI_0 input':<28} {'0':<6} {mzi0_input_m0:>10.6f} {np.degrees(mzi0_input_m0):>12.4f}")
print(f"{'MZI_0 input':<28} {'1':<6} {mzi0_input_m1:>10.6f} {np.degrees(mzi0_input_m1):>12.4f}")
print(f"{'MZI_0 [BS 50:50]':<28} {'(0,1)':<6} {'---':>10}")
print(f"{'MZI_0 middle':<28} {'1':<6} {mid0_m1:>10.6f} {np.degrees(mid0_m1):>12.4f}")
print(f"{'MZI_0 [BS 50:50]':<28} {'(0,1)':<6} {'---':>10}")
print(f"{'Between MZIs':<28} {'0':<6} {between_m0:>10.6f} {np.degrees(between_m0):>12.4f}")
print(f"{'Between MZIs':<28} {'1':<6} {between_m1:>10.6f} {np.degrees(between_m1):>12.4f}")
print(f"{'MZI_1 [BS 50:50]':<28} {'(0,1)':<6} {'---':>10}")
print(f"{'MZI_1 middle':<28} {'1':<6} {mid1_m1:>10.6f} {np.degrees(mid1_m1):>12.4f}")
print(f"{'MZI_1 [BS 50:50]':<28} {'(0,1)':<6} {'---':>10}")
print(f"{'MZI_1 output':<28} {'1':<6} {out1_m1:>10.6f} {np.degrees(out1_m1):>12.4f}")