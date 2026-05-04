import numpy as np
import perceval as pcvl
from perceval import RemoteProcessor
from perceval.components import PS
from math import pi
import time
import perceval.components as comp

# ============================================================
# QSP CIRCUIT LAYER-BY-LAYER ON BELENOS -- L=1, x=0.5
# ============================================================
#
# GOAL:
#   Implement QSP circuit for L=1, x=0.5 directly on Belenos
#   chip gate by gate, using Samuel's approach of setting PS
#   values directly on the chip architecture.
#
# KEY INSIGHTS FROM SAMUEL:
#   1. For PSs between BSs on both modes, set only mode 0 to
#      phi_mode0 - phi_mode1, and mode 1 to identity (0)
#   2. Input PSs at the start of circuit are ignored because
#      they don't affect probability measurements
#
# custom_phases: mode 0 PS values in circuit order
#   Computed as phi_mode0 - phi_mode1 for each PS position
#   Trying NEGATIVE sign based on verification analysis
#
# VERIFIED PS VALUES (Match=True, Error=5.13e-07):
#   MZI_0 middle mode1 = 1.570105
#   Between MZIs mode0 = -0.090646, mode1 = 5.121287
#   MZI_1 middle mode1 = 1.252545
#   MZI_1 output mode1 = 3.141593
# ============================================================

TOKEN = "_T_eyJhbGciOiJIUzUxMiIsInR5cCI6IkpXVCJ9.eyJpZCI6MjU5MiwiZXhwIjoxODA2NTkwNjc1LjE4NjQ3NDN9.eCJCJNcF7BqyUtKiUSaOPK1RM0Nd42B-2UXwH0IIRiYy0TOqi9eQWcUM3049KIkcjgyKkHT-xyIxzoqzmuxyaQ"

# ── Connect to Belenos ────────────────────────────────────
remote_processor = RemoteProcessor("qpu:belenos", token=TOKEN)
arch = remote_processor.specs["architecture"]
belenos_circuit = arch.unitary_circuit()

print(f"Belenos circuit: {belenos_circuit.m} modes")

# ── custom_phases: mode 0 PS values in circuit order ──────
# Formula: phi_combined = phi_mode0 - phi_mode1
# MZI_0 middle: 0       - 1.570105 = -1.570105
# Between MZIs: -0.090646 - 5.121287 = -5.211933
# MZI_1 middle: 0       - 1.252545 = -1.252545
# MZI_1 output: 0       - 3.141593 = -3.141593
custom_phases = [
    -1.570105,   # MZI_0 middle
    -5.211933,   # Between MZIs
    -1.252545,   # MZI_1 middle
    -3.141593,   # MZI_1 output
]

# ── Set PS values directly on chip ────────────────────────
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
                params[0].set_value(pi)  # identity for remaining PSs

        elif r[0] % 2 == 1:
            params[0].set_value(0)   # odd modes: identity = 0
        else:
            params[0].set_value(pi)  # even modes: identity = pi

print(f"\nTotal mode 0 PSs set with custom values: {i}")


# ── Local verification BEFORE submitting job ──────────────
U_belenos = np.array(belenos_circuit.compute_unitary())
psi = U_belenos[:, 0]
p0_local = abs(psi[0])**2
p1_local = abs(psi[1])**2
Z_local = p0_local - p1_local
print(f"\n=== Local verification (no credits used) ===")
print(f"  Z local    = {Z_local:.4f}")
print(f"  Z expected = 0.4552")
print(f"  Match: {np.isclose(Z_local, 0.4552, atol=0.05)}")

# ── Submit job ────────────────────────────────────────────
remote_processor.set_circuit(belenos_circuit)
remote_processor.with_input(pcvl.BasicState([1, 0] + [0] * 22))
remote_processor.min_detected_photons_filter(1)

from perceval.algorithm import Sampler
sampler = Sampler(remote_processor, max_shots_per_call=1000)
job = sampler.sample_count.execute_async(1000)

print(f"\n=== Job submitted ===")
print(f"Job ID: {job.id}")

# ── Wait for results ──────────────────────────────────────
print("\nWaiting for job to complete...")
while not job.is_complete:
    print(f"  Status: {job.status}...")
    time.sleep(5)

results = job.get_results()

# ── Process results ───────────────────────────────────────
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

# ── Expected Z from analytic computation ──────────────────
FUNC_NAME    = "STEP"
ANGLE_METHOD = "pq"
L            = 1
x_val        = 0.5

theta_arr = np.load(f"theta_{FUNC_NAME.lower()}_{ANGLE_METHOD}_L{L}.npy")
phi_arr   = np.load(f"phi_{FUNC_NAME.lower()}_{ANGLE_METHOD}_L{L}.npy")

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
psi = U @ np.array([1.0, 0.0])
Z_expected = abs(psi[0])**2 - abs(psi[1])**2

print(f"\n=== Comparison ===")
print(f"  Z from Belenos QPU : {Z:.4f}")
print(f"  Z expected analytic: {Z_expected:.4f}")
print(f"  Difference         : {abs(Z - Z_expected):.4f}")