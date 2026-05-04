import numpy as np
import perceval as pcvl
from perceval import RemoteProcessor
from perceval.components import BS, PS
from math import pi
import time
import perceval.components as comp
import re

# ============================================================
# QSP CIRCUIT LAYER-BY-LAYER ON BELENOS -- ANY L, x=0.5
# ============================================================
#
# GOAL:
#   Implement QSP circuit for any L directly on Belenos chip
#   gate by gate, using Samuel's approach of setting PS values
#   directly on the chip architecture.
#
# KEY INSIGHTS FROM SAMUEL:
#   1. For PSs between BSs on both modes, set only mode 0 to
#      phi_mode0 - phi_mode1, and mode 1 to identity (0)
#   2. Input PSs at the start of circuit are ignored because
#      they don't affect probability measurements
#   3. Remaining MZIs after our circuit set to identity:
#      mode 0 = pi, mode 1 = 0 (phase difference = pi)
#
# NOTE ON SEED:
#   pcvl.Circuit.decomposition() uses numerical search from
#   random starting point. For some gate angles, seed 0 fails.
#   This code automatically tries seeds 0-99 until it finds
#   a working seed for each gate.
#
# NOTE ON BS THETAS:
#   Uses actual Belenos chip BS thetas (not ideal 50:50).
#   This explains difference between analytic and local/QPU Z.
#
# HOW TO USE:
#   Change L at the top. Make sure angle files exist.
#   Code works for any L automatically.
# ============================================================

# ── Settings ──────────────────────────────────────────────
FUNC_NAME    = "STEP"
ANGLE_METHOD = "pq"
L            = 5      # <-- change this to any L
x_val        = 0.5

TOKEN = "_T_eyJhbGciOiJIUzUxMiIsInR5cCI6IkpXVCJ9.eyJpZCI6MjU5MiwiZXhwIjoxODA2NTkwNjc1LjE4NjQ3NDN9.eCJCJNcF7BqyUtKiUSaOPK1RM0Nd42B-2UXwH0IIRiYy0TOqi9eQWcUM3049KIkcjgyKkHT-xyIxzoqzmuxyaQ"

# ── Load angles ───────────────────────────────────────────
theta_arr = np.load(f"theta_{FUNC_NAME.lower()}_{ANGLE_METHOD}_L{L}.npy")
phi_arr   = np.load(f"phi_{FUNC_NAME.lower()}_{ANGLE_METHOD}_L{L}.npy")

print(f"Loaded angles: L={L}, x={x_val}")
print(f"theta_arr: {np.round(theta_arr, 6)}")
print(f"phi_arr:   {np.round(phi_arr, 6)}")

# ── Connect to Belenos and get actual BS thetas ───────────
remote_processor = RemoteProcessor("qpu:belenos", token=TOKEN)
arch = remote_processor.specs["architecture"]
belenos_circuit = arch.unitary_circuit()

# Extract actual BS thetas from chip
bs_thetas = []
for r, c in belenos_circuit:
    if r == (0, 1):
        desc = c.describe()
        m = re.search(r"theta=([\-\+]?[\d\.eE\+\-]+)", desc)
        if m:
            bs_thetas.append(float(m.group(1)))

# ── Check we have enough BSs ──────────────────────────────
n_mzis      = L + 1
n_bs_needed = n_mzis * 2

print(f"\nn_mzis      = {n_mzis}")
print(f"n_bs_needed = {n_bs_needed}")
print(f"BSs on chip = {len(bs_thetas)}")
print(f"theta_arr length = {len(theta_arr)}")

assert len(bs_thetas) >= n_bs_needed, \
    f"Not enough BSs on chip! Need {n_bs_needed}, have {len(bs_thetas)}"
assert len(theta_arr) == n_mzis, \
    f"theta_arr length {len(theta_arr)} != n_mzis {n_mzis}"

print(f"\nActual Belenos BS thetas (first {n_bs_needed}):")
for i, t in enumerate(bs_thetas[:n_bs_needed]):
    print(f"  BS #{i+1}: theta={t:.6f} rad ({np.degrees(t):.4f} deg)")

# ── Helper: extract phi from PS description string ────────
def extract_phi(desc):
    m = re.search(r"phi\s*=\s*([\-\+]?[\d\.eE\+\-]+|pi)", desc)
    if m:
        val_str = m.group(1)
        return np.pi if val_str == "pi" else float(val_str)
    return None

# ── Decompose ALL Ry gates with automatic seed finding ────
print(f"\n=== Decomposing {n_mzis} Ry gates ===")
mzi_values = []

for j in range(n_mzis):
    bs1 = bs_thetas[j * 2]
    bs2 = bs_thetas[j * 2 + 1]

    print(f"\n  Trying to decompose j={j}: theta={theta_arr[j]:.4f} "
          f"with BS thetas {bs1:.4f}, {bs2:.4f}")

    success = False
    for seed in range(200):   # try up to 200 seeds
        pcvl.random_seed(seed)
        single_gate = comp.BS.Ry(theta=float(theta_arr[j]))
        U_gate = pcvl.Matrix(np.array(single_gate.compute_unitary()).tolist())
        mzi = (pcvl.Circuit(2)
               // BS.Rx(theta=bs1)
               // (1, PS(pcvl.P("phi0")))
               // BS.Rx(theta=bs2)
               // (1, PS(pcvl.P("phi1"))))
        decomposed = pcvl.Circuit.decomposition(U_gate, mzi, phase_shifter_fn=PS, max_try=10)

        if decomposed is not None:
            error = np.linalg.norm(
                np.array(U_gate) - np.array(decomposed.compute_unitary()))
            print(f"    seed={seed:3d}  error={error:.2e}")
            if error < 1e-6:
                print(f"  ✅ j={j}: found working seed={seed}  error={error:.2e}")
                success = True
                break

    if not success:
        raise ValueError(
            f"Could not decompose Ry gate j={j} "
            f"(theta={theta_arr[j]:.4f}) with any seed 0-199!")

    # Extract PS values from decomposed circuit
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
    mzi_values.append((phi_in_m0, phi_in_m1, phi_mid_m1, phi_out_m1))

    print(f"       phi_in_m0={phi_in_m0:.4f}  phi_in_m1={phi_in_m1:.4f}  "
          f"phi_mid_m1={phi_mid_m1:.4f}  phi_out_m1={phi_out_m1:.4f}")

# ── Build custom_phases for any L ─────────────────────────
custom_phases = []

for j in range(n_mzis):
    in_m0_j, in_m1_j, mid_m1_j, out_m1_j = mzi_values[j]

    # After first BS of MZI_j: mode0=0, mode1=mid_m1_j
    custom_phases.append(0 - mid_m1_j)

    # After second BS of MZI_j
    if j < n_mzis - 1:
        # Between MZI_j and MZI_{j+1}
        in_m0_next, in_m1_next, _, _ = mzi_values[j + 1]
        between_m0 = (-x_val/2) + (-phi_arr[j+1]/2) + in_m0_next
        between_m1 = out_m1_j + (x_val/2) + (phi_arr[j+1]/2) + in_m1_next
        custom_phases.append(between_m0 - between_m1)
    else:
        # Last MZI: after final BS, mode0=0, mode1=out_m1_j
        custom_phases.append(0 - out_m1_j)

print(f"\n=== custom_phases for Belenos (L={L}, x={x_val}) ===")
print(f"Total custom phases: {len(custom_phases)}  (should be {2*n_mzis})")
for i, p in enumerate(custom_phases):
    print(f"  PS #{i+1}: {p:.6f} rad ({np.degrees(p):.4f} deg)")

# ── Set PS values on chip ─────────────────────────────────
i = 0
for r, c in belenos_circuit:
    if isinstance(c, PS):
        params = c.get_parameters()
        if not params:
            continue
        if r[0] == 0:
            if i < len(custom_phases):
                params[0].set_value(custom_phases[i])
                i += 1
            else:
                params[0].set_value(pi)   # identity for remaining MZIs
        elif r[0] % 2 == 1:
            params[0].set_value(0)        # odd modes: identity = 0
        else:
            params[0].set_value(pi)       # even modes: identity = pi

print(f"\nTotal mode 0 PSs set with custom values: {i}")

# ── Local verification ────────────────────────────────────
U_belenos = np.array(belenos_circuit.compute_unitary())
psi = U_belenos[:, 0]
Z_local = abs(psi[0])**2 - abs(psi[1])**2

# Expected Z from analytic circuit
analytic_circuit = pcvl.Circuit(2)
analytic_circuit.add(0,      comp.PS(float(-phi_arr[0] / 2)))
analytic_circuit.add(1,      comp.PS(float( phi_arr[0] / 2)))
analytic_circuit.add((0, 1), comp.BS.Ry(theta=float(theta_arr[0])))
for j in range(1, L + 1):
    analytic_circuit.add(0,      comp.PS(float(-x_val / 2)))
    analytic_circuit.add(1,      comp.PS(float( x_val / 2)))
    analytic_circuit.add(0,      comp.PS(float(-phi_arr[j] / 2)))
    analytic_circuit.add(1,      comp.PS(float( phi_arr[j] / 2)))
    analytic_circuit.add((0, 1), comp.BS.Ry(theta=float(theta_arr[j])))

U = np.array(analytic_circuit.compute_unitary())
psi_expected = U @ np.array([1.0, 0.0])
Z_expected = abs(psi_expected[0])**2 - abs(psi_expected[1])**2

print(f"\n=== Local verification ===")
print(f"  Z local (real BS thetas) = {Z_local:.4f}")
print(f"  Z expected analytic      = {Z_expected:.4f}")
print(f"  Match: {np.isclose(Z_local, Z_expected, atol=0.05)}")

# ── Submit job only if local verification passes ──────────
if not np.isclose(Z_local, Z_expected, atol=0.05):
    print("\n⚠️  Local verification failed! NOT submitting job.")
    print("    Check the decomposition values above.")
else:
    remote_processor.set_circuit(belenos_circuit)
    remote_processor.with_input(pcvl.BasicState([1, 0] + [0] * 22))
    remote_processor.min_detected_photons_filter(1)

    from perceval.algorithm import Sampler
    sampler = Sampler(remote_processor, max_shots_per_call=5000)
    job = sampler.sample_count.execute_async(5000)

    print(f"\n=== Job submitted ===")
    print(f"Job ID: {job.id}")

    # ── Wait for results ───────────────────────────────────
    print("\nWaiting for job to complete...")
    while not job.is_complete:
        print(f"  Status: {job.status}...")
        time.sleep(5)

    results = job.get_results()
    counts = dict(results['results'])
    total = sum(counts.values())

    print("\n=== Results ===")
    for state, count in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"  {str(state):<50} {count:>8}  ({count/total:.4f})")

    p0 = counts.get(pcvl.BasicState([1, 0] + [0] * 22), 0) / total
    p1 = counts.get(pcvl.BasicState([0, 1] + [0] * 22), 0) / total
    Z  = p0 - p1

    print(f"\n  p0 = {p0:.4f}")
    print(f"  p1 = {p1:.4f}")
    print(f"  Z  = p0 - p1 = {Z:.4f}")
    print(f"\n=== Comparison ===")
    print(f"  Z from Belenos QPU            : {Z:.4f}")
    print(f"  Z local (real BS thetas)      : {Z_local:.4f}")
    print(f"  Z expected analytic           : {Z_expected:.4f}")
    print(f"  Difference QPU vs analytic    : {abs(Z - Z_expected):.4f}")
    print(f"  Difference QPU vs local       : {abs(Z - Z_local):.4f}")