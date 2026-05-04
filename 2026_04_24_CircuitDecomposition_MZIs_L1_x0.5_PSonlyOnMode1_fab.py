import numpy as np
import perceval as pcvl
from perceval import RemoteProcessor
from perceval.components import BS, PS
from math import pi
import time
import perceval.components as comp
import re

# ── Reproducibility ───────────────────────────────────────
pcvl.random_seed(0)

# ── Settings ──────────────────────────────────────────────
FUNC_NAME    = "STEP"
ANGLE_METHOD = "pq"
L            = 1
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

print(f"\nActual Belenos BS thetas (first 4):")
for i, t in enumerate(bs_thetas[:4]):
    print(f"  BS #{i+1}: theta={t:.6f} rad ({np.degrees(t):.4f} deg)")

# ── Helper: extract phi from PS description string ────────
def extract_phi(desc):
    m = re.search(r"phi\s*=\s*([\-\+]?[\d\.eE\+\-]+|pi)", desc)
    if m:
        val_str = m.group(1)
        return np.pi if val_str == "pi" else float(val_str)
    return None

# ── Decompose Ry gate using actual Belenos BS thetas ──────
def decompose_ry_belenos(theta_j, bs_theta_1, bs_theta_2):
    """
    Decompose BS.Ry(theta_j) into MZI using actual Belenos BS thetas:
      BS.Rx(bs_theta_1) -- PS(phi0) -- BS.Rx(bs_theta_2) -- PS(phi1)
    """
    single_gate = comp.BS.Ry(theta=float(theta_j))
    U_gate = pcvl.Matrix(np.array(single_gate.compute_unitary()).tolist())

    mzi = pcvl.Circuit(2) // BS.Rx(theta=bs_theta_1) // (1, PS(pcvl.P("phi0"))) // BS.Rx(theta=bs_theta_2) // (1, PS(pcvl.P("phi1")))
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

# ── Decompose each Ry gate with actual BS thetas ──────────
print(f"\n=== Decomposing Ry gates with actual Belenos BS thetas ===")

# MZI_0 uses BS#1 and BS#2
in0_m0, in0_m1, mid0_m1, out0_m1 = decompose_ry_belenos(
    theta_arr[0], bs_thetas[0], bs_thetas[1])
print(f"\nRy(theta_0={theta_arr[0]:.4f}) with BS thetas {bs_thetas[0]:.4f}, {bs_thetas[1]:.4f}:")
print(f"  phi_in  mode0 = {in0_m0:.6f} rad")
print(f"  phi_in  mode1 = {in0_m1:.6f} rad")
print(f"  phi_mid mode1 = {mid0_m1:.6f} rad")
print(f"  phi_out mode1 = {out0_m1:.6f} rad")

# MZI_1 uses BS#3 and BS#4
in1_m0, in1_m1, mid1_m1, out1_m1 = decompose_ry_belenos(
    theta_arr[1], bs_thetas[2], bs_thetas[3])
print(f"\nRy(theta_1={theta_arr[1]:.4f}) with BS thetas {bs_thetas[2]:.4f}, {bs_thetas[3]:.4f}:")
print(f"  phi_in  mode0 = {in1_m0:.6f} rad")
print(f"  phi_in  mode1 = {in1_m1:.6f} rad")
print(f"  phi_mid mode1 = {mid1_m1:.6f} rad")
print(f"  phi_out mode1 = {out1_m1:.6f} rad")

# ── QSP PS values ─────────────────────────────────────────
rz_phi0_m0 = -phi_arr[0] / 2
rz_phi0_m1 = +phi_arr[0] / 2
rz_x_m0    = -x_val / 2
rz_x_m1    = +x_val / 2
rz_phi1_m0 = -phi_arr[1] / 2
rz_phi1_m1 = +phi_arr[1] / 2

# ── Combine adjacent PSs ──────────────────────────────────
mzi0_input_m0 = rz_phi0_m0 + in0_m0
mzi0_input_m1 = rz_phi0_m1 + in0_m1
between_m0    = rz_x_m0 + rz_phi1_m0 + in1_m0
between_m1    = out0_m1 + rz_x_m1 + rz_phi1_m1 + in1_m1

print(f"\n=== Combined PS values ===")
print(f"  MZI_0 input  mode0 = {mzi0_input_m0:.6f} rad (ignored)")
print(f"  MZI_0 input  mode1 = {mzi0_input_m1:.6f} rad (ignored)")
print(f"  MZI_0 middle mode1 = {mid0_m1:.6f} rad")
print(f"  Between MZIs mode0 = {between_m0:.6f} rad")
print(f"  Between MZIs mode1 = {between_m1:.6f} rad")
print(f"  MZI_1 middle mode1 = {mid1_m1:.6f} rad")
print(f"  MZI_1 output mode1 = {out1_m1:.6f} rad")

# ── Apply Samuel's insight: combined on mode 0 only ───────
# phi_combined = phi_mode0 - phi_mode1
custom_phases = [
    0          - mid0_m1,    # after BS#1: mode0=0, mode1=mid0_m1
    between_m0 - between_m1, # after BS#2: both modes
    0          - mid1_m1,    # after BS#3: mode0=0, mode1=mid1_m1
    0          - out1_m1,    # after BS#4: mode0=0, mode1=out1_m1
]

print(f"\n=== custom_phases for Belenos (mode 0 only) ===")
for i, p in enumerate(custom_phases):
    print(f"  PS #{i+1}: {p:.6f} rad ({np.degrees(p):.4f} deg)")

# ── Set PS values on chip ─────────────────────────────────
i = 0
print("\n=== Setting PS values on Belenos ===")
for r, c in belenos_circuit:
    if isinstance(c, PS):
        params = c.get_parameters()
        if not params:
            continue
        if r[0] == 0:
            if i < len(custom_phases):
                params[0].set_value(custom_phases[i])
                print(f"  Mode 0 PS #{i}: set to {custom_phases[i]:.6f} rad")
                i += 1
            else:
                params[0].set_value(pi)
        elif r[0] % 2 == 1:
            params[0].set_value(0)
        else:
            params[0].set_value(pi)

print(f"\nTotal mode 0 PSs set: {i}")

# ── Local verification ────────────────────────────────────
U_belenos = np.array(belenos_circuit.compute_unitary())
psi = U_belenos[:, 0]
p0_local = abs(psi[0])**2
p1_local = abs(psi[1])**2
Z_local = p0_local - p1_local

# Expected Z
analytic_circuit = pcvl.Circuit(2)
analytic_circuit.add(0,      comp.PS(float(-phi_arr[0] / 2)))
analytic_circuit.add(1,      comp.PS(float( phi_arr[0] / 2)))
analytic_circuit.add((0, 1), comp.BS.Ry(theta=float(theta_arr[0])))
analytic_circuit.add(0,      comp.PS(float(-x_val / 2)))
analytic_circuit.add(1,      comp.PS(float( x_val / 2)))
analytic_circuit.add(0,      comp.PS(float(-phi_arr[1] / 2)))
analytic_circuit.add(1,      comp.PS(float( phi_arr[1] / 2)))
analytic_circuit.add((0, 1), comp.BS.Ry(theta=float(theta_arr[1])))

U = np.array(analytic_circuit.compute_unitary())
psi_expected = U @ np.array([1.0, 0.0])
Z_expected = abs(psi_expected[0])**2 - abs(psi_expected[1])**2

print(f"\n=== Local verification ===")
print(f"  Z local (with real BS thetas) = {Z_local:.4f}")
print(f"  Z expected analytic           = {Z_expected:.4f}")
print(f"  Match: {np.isclose(Z_local, Z_expected, atol=0.05)}")

# ── Submit job ────────────────────────────────────────────
remote_processor.set_circuit(belenos_circuit)
remote_processor.with_input(pcvl.BasicState([1, 0] + [0] * 22))
remote_processor.min_detected_photons_filter(1)

from perceval.algorithm import Sampler
sampler = Sampler(remote_processor, max_shots_per_call=5000)
job = sampler.sample_count.execute_async(5000)

print(f"\n=== Job submitted ===")
print(f"Job ID: {job.id}")

# ── Wait for results ──────────────────────────────────────
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