import numpy as np
import perceval as pcvl
from perceval import RemoteProcessor
from perceval.components import PS
from math import pi
import time
import perceval.components as comp

TOKEN = "_T_eyJhbGciOiJIUzUxMiIsInR5cCI6IkpXVCJ9.eyJpZCI6MjU5MiwiZXhwIjoxODA2NTkwNjc1LjE4NjQ3NDN9.eCJCJNcF7BqyUtKiUSaOPK1RM0Nd42B-2UXwH0IIRiYy0TOqi9eQWcUM3049KIkcjgyKkHT-xyIxzoqzmuxyaQ"

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
print(f"Z expected analytic: {Z_expected:.4f}")

# ── Base PS values ─────────────────────────────────────────
# Combined = phi_mode0 - phi_mode1 for each position
v1 =  0       - 1.570105   # after 1st BS: -1.570105
v2 = -0.090646 - 5.121287  # after 2nd BS: -5.211933
v3 =  0       - 1.252545   # after 3rd BS: -1.252545
v4 =  0       - 3.141593   # after 4th BS: -3.141593

# ── 4 sign combinations to test ───────────────────────────
sign_combinations = {
    "all negative (phi0-phi1)": [ v1,  v2,  v3,  v4],
    "all positive (phi1-phi0)": [-v1, -v2, -v3, -v4],
    "mixed +--+":               [-v1,  v2,  v3, -v4],
    "mixed -++-":               [ v1, -v2, -v3,  v4],
}

for label, custom_phases in sign_combinations.items():
    print(f"\n{'='*60}")
    print(f"Testing: {label}")
    print(f"custom_phases: {[round(p,4) for p in custom_phases]}")

    # ── Connect and set PS values ──────────────────────────
    remote_processor = RemoteProcessor("qpu:belenos", token=TOKEN)
    arch = remote_processor.specs["architecture"]
    belenos_circuit = arch.unitary_circuit()

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
                    params[0].set_value(0)
            elif r[0] % 2 == 1:
                params[0].set_value(0)
            else:
                params[0].set_value(pi)

    # ── Submit job ─────────────────────────────────────────
    remote_processor.set_circuit(belenos_circuit)
    remote_processor.with_input(pcvl.BasicState([1, 0] + [0] * 22))
    remote_processor.min_detected_photons_filter(1)

    from perceval.algorithm import Sampler
    sampler = Sampler(remote_processor, max_shots_per_call=1000)
    job = sampler.sample_count.execute_async(1000)
    print(f"Job ID: {job.id}")

    # ── Wait for results ───────────────────────────────────
    while not job.is_complete:
        time.sleep(5)

    results = job.get_results()
    counts = dict(results['results'])
    total = sum(counts.values())

    p0 = counts.get(pcvl.BasicState([1, 0] + [0] * 22), 0) / total
    p1 = counts.get(pcvl.BasicState([0, 1] + [0] * 22), 0) / total
    Z  = p0 - p1

    print(f"p0={p0:.4f}  p1={p1:.4f}  Z={Z:.4f}  expected={Z_expected:.4f}  diff={abs(Z-Z_expected):.4f}")