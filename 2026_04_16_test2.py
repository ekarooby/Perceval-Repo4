import numpy as np
import perceval as pcvl
from perceval import RemoteProcessor
from perceval.components import PS
from math import pi
import time

# ── Our verified PS values (Match=True, Error=5.13e-07) ───
# Full QSP circuit L=1, x=0.5 decomposed layer by layer

TOKEN = "_T_eyJhbGciOiJIUzUxMiIsInR5cCI6IkpXVCJ9.eyJpZCI6MjU5MiwiZXhwIjoxODA2NTkwNjc1LjE4NjQ3NDN9.eCJCJNcF7BqyUtKiUSaOPK1RM0Nd42B-2UXwH0IIRiYy0TOqi9eQWcUM3049KIkcjgyKkHT-xyIxzoqzmuxyaQ"
remote_processor = RemoteProcessor("qpu:belenos", token=TOKEN)
arch = remote_processor.specs["architecture"]
belenos_circuit = arch.unitary_circuit()

print(f"Belenos circuit: {belenos_circuit.m} modes")

# ── Set PS values directly ────────────────────────────────
# Pattern on modes 0,1 per MZI:
#   PS(mode1) → BS → PS(mode1) + PS(mode0) → BS → PS(mode1) + PS(mode0) → ...
#
# Our QSP uses 2 MZIs. Remaining 10 MZIs set to identity.
#
# MZI_0:
#   input  mode1 = mzi0_input_m1 = -0.785052
#   middle mode1 = mid0_m1       =  1.570105
#   middle mode0 = mzi0_input_m0 =  2.356540
# Between MZI_0 and MZI_1:
#   mode1  = between_m1          =  5.121287
#   mode0  = between_m0          = -0.090646
# MZI_1:
#   middle mode1 = mid1_m1       =  1.252545
#   middle mode0 = identity      =  pi
# After MZI_1:
#   mode1  = out1_m1             =  pi
#   mode0  = identity            =  0

# PS sequence for mode 1 (in order they appear in circuit)
ps_m1_values = [
    -0.785052,   # MZI_0 input mode 1
     1.570105,   # MZI_0 middle mode 1
     5.121287,   # between MZIs mode 1
     1.252545,   # MZI_1 middle mode 1
     3.141593,   # MZI_1 output mode 1 (=pi)
]

# PS sequence for mode 0 (in order they appear in circuit)
ps_m0_values = [
     2.356540,   # MZI_0 middle mode 0
    -0.090646,   # between MZIs mode 0 (phi_top)
     pi,         # MZI_1 middle mode 0 (identity)
]

i_m0 = 0
i_m1 = 0

print("\n=== Setting PS values on Belenos ===")
for r, c in belenos_circuit:
    if isinstance(c, PS):
        params = c.get_parameters()
        if not params:
            continue

        if r == (1,):
            if i_m1 < len(ps_m1_values):
                params[0].set_value(ps_m1_values[i_m1])
                print(f"  Mode 1 PS #{i_m1}: set to {ps_m1_values[i_m1]:.6f} rad")
                i_m1 += 1
            else:
                params[0].set_value(0)  # identity for remaining mode 1 PSs

        elif r == (0,):
            if i_m0 < len(ps_m0_values):
                params[0].set_value(ps_m0_values[i_m0])
                print(f"  Mode 0 PS #{i_m0}: set to {ps_m0_values[i_m0]:.6f} rad")
                i_m0 += 1
            else:
                params[0].set_value(0)  # identity for remaining mode 0 PSs

        elif r[0] % 2 == 1:
            params[0].set_value(0)   # odd modes: identity = 0
        else:
            params[0].set_value(pi)  # even modes: identity = pi

print(f"\nMode 0 PSs set: {i_m0} (expected {len(ps_m0_values)})")
print(f"Mode 1 PSs set: {i_m1} (expected {len(ps_m1_values)})")

# ── Submit job ────────────────────────────────────────────
remote_processor.set_circuit(belenos_circuit)
remote_processor.with_input(pcvl.BasicState([1, 0]))
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
print("\n=== Results ===")
raw = results['results']
total = sum(raw.values())
for state, count in sorted(raw.items(), key=lambda kv: -kv[1]):
    print(f"  {str(state):<20} {count:>8}  ({count/total:.4f})")

p0 = raw.get(pcvl.BasicState([1, 0]), 0) / total
p1 = raw.get(pcvl.BasicState([0, 1]), 0) / total
Z  = p0 - p1
print(f"\n  p0 = {p0:.4f}")
print(f"  p1 = {p1:.4f}")
print(f"  Z  = p0 - p1 = {Z:.4f}")

# ── Expected Z from analytic computation ──────────────────
import perceval.components as comp
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