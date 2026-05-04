import numpy as np
import perceval as pcvl
import perceval.components as comp
from perceval.components import BS, PS
import re

# ── Reproducibility ───────────────────────────────────────
pcvl.random_seed(42)

# ── Load your QSP angles ──────────────────────────────────
FUNC_NAME    = "STEP"
ANGLE_METHOD = "pq"
L            = 1

theta_arr = np.load(f"theta_{FUNC_NAME.lower()}_{ANGLE_METHOD}_L{L}.npy")
phi_arr   = np.load(f"phi_{FUNC_NAME.lower()}_{ANGLE_METHOD}_L{L}.npy")

print(f"Loaded angles: L={L}")
print(f"Number of Ry gates: {L+1}  (indices 0 to {L})")

# ── MZI building block (Belenos style) ───────────────────
mzi = pcvl.Circuit(2) // BS() // (1, PS(pcvl.P("phi0"))) // BS() // (1, PS(pcvl.P("phi1")))

# ── Helper: extract phi from PS description string ────────
def extract_phi(desc):
    m = re.search(r"phi\s*=\s*([\-\+]?[\d\.eE\+\-]+|pi)", desc)
    if m:
        val_str = m.group(1)
        return np.pi if val_str == "pi" else float(val_str)
    return None

# ── Decompose each Ry gate individually ───────────────────
print(f"\n{'Gate':<6} {'theta_j':>10} {'phi_mid (rad)':>14} {'phi_out (rad)':>14} {'phi_mid (deg)':>14} {'phi_out (deg)':>14} {'error':>10}")
print("-" * 90)

gate_ps_values = []

for j in range(L + 1):
    theta_j = theta_arr[j]

    # Build the single Ry gate unitary
    single_gate = comp.BS.Ry(theta=float(theta_j))
    U_gate = pcvl.Matrix(np.array(single_gate.compute_unitary()).tolist())

    # Decompose into MZI
    decomposed = pcvl.Circuit.decomposition(U_gate, mzi, phase_shifter_fn=PS)

    # ── Print all components for inspection ───────────────
    print(f"\n  j={j} all components:")
    all_components = list(decomposed)
    for modes, component in all_components:
        print(f"    modes={modes}  desc={component.describe()}")

    # ── Extract PS values on mode 1 only ──────────────────
    # Mode 1 PSs are: input PS, middle PS (between BSs), output PS
    # We want: phi_mid (between the two BSs) and phi_out (after second BS)
    mode1_ps = []
    bs_count  = 0
    for modes, component in all_components:
        desc = component.describe()
        if desc.startswith("BS"):
            bs_count += 1
        if desc.startswith("PS(") and modes == (1,):
            phi = extract_phi(desc)
            if phi is not None:
                mode1_ps.append((bs_count, phi))  # (after how many BSs, value)

    print(f"  Mode 1 PS values (bs_count, phi): {mode1_ps}")
    mode0_ps = []
    for modes, component in all_components:
        desc = component.describe()
        if desc.startswith("PS(") and modes == (0,):
            phi = extract_phi(desc)
            if phi is not None:
                mode0_ps.append(phi)

    print(f"  Mode 0 PS values: {mode0_ps}")

    # phi_mid = PS after first BS (bs_count=1)
    # phi_out = PS after second BS (bs_count=2)
    phi_mid = next((phi for bc, phi in mode1_ps if bc == 1), None)
    phi_out = next((phi for bc, phi in mode1_ps if bc == 2), None)

    # Verify decomposition
    error = np.linalg.norm(
        np.array(U_gate) - np.array(decomposed.compute_unitary())
    )

    if phi_mid is not None and phi_out is not None:
        gate_ps_values.append((phi_mid, phi_out))
        print(f"\n  j={j:<3} theta={theta_j:.4f}  "
              f"phi_mid={phi_mid:.6f} rad ({np.degrees(phi_mid):.4f} deg)  "
              f"phi_out={phi_out:.6f} rad ({np.degrees(phi_out):.4f} deg)  "
              f"error={error:.2e}")
    else:
        gate_ps_values.append((None, None))
        print(f"\n  j={j:<3} could not extract phi_mid/phi_out")

# ── Final summary table ───────────────────────────────────
print(f"\n{'='*70}")
print(f"FINAL SUMMARY: MZI PS values for each Ry gate (L={L})")
print(f"{'='*70}")
print(f"{'Gate':<6} {'theta_j':>10} {'phi_mid (rad)':>14} {'phi_out (rad)':>14} {'phi_mid (deg)':>14} {'phi_out (deg)':>14}")
print("-" * 70)
for j, (phi_mid, phi_out) in enumerate(gate_ps_values):
    if phi_mid is not None:
        print(f"  j={j:<3} {theta_arr[j]:>10.4f} {phi_mid:>14.6f} {phi_out:>14.6f} "
              f"{np.degrees(phi_mid):>14.4f} {np.degrees(phi_out):>14.4f}")
        
