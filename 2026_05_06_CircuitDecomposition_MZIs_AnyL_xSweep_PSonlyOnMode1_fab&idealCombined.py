# ============================================================
# This code corrects the problem of Code "2026_05_04_CircuitDecomposition_MZIs_AnyL_xSweep_PSonlyOnMode1_fab" by using both non-ideal chip BSs and ideal 50:50 BSs for the Ry(theta) gates with theta close to pi or zero
# This code Handles extreme theta values (near 0 or pi) by falling back to ideal 50:50 BSs
# in the decomposition (which has a closed-form solution). Phi values are then
# applied to chip's real BSs, introducing a small per-gate error that is
# verified via local check before QPU submission.

# ============================================================

import numpy as np
import matplotlib.pyplot as plt
import perceval as pcvl
from perceval import RemoteProcessor
from perceval.components import BS, PS
from perceval.algorithm import Sampler
from math import pi
import time
import perceval.components as comp
import re

# ============================================================
# QSP FULL PIPELINE -- LAYER BY LAYER ON BELENOS
# ============================================================
#
# GOAL:
#   Implement QSP circuit for any L directly on Belenos chip
#   gate by gate, sweeping over multiple x values to get the
#   full QSP approximation curve Z(x).
#   Plots and compares 6 curves:
#     1. True STEP function (black solid)
#     2. Surrogate function (black dashed)
#     3. Pure math analytic -- ideal BS.Ry (red)
#     4. SLOS -- real chip BS thetas, layer by layer (green)
#     5. QPU  -- real chip BS thetas, real photons (blue dots)
#
# KEY DIFFERENCE FROM OLD QPU CODE:
#   Old: builds 2x2 unitary, sends to auto-compiler.
#   This: decomposes each Ry(theta_j) individually into a
#         physical MZI using real chip BS thetas, combines
#         adjacent PSs, sets PS values directly on chip.
#
# WHAT CHANGES WITH x:
#   Gate decompositions (mid_m1, out_m1) are FIXED for all x.
#   Only between-MZI PSs change with x:
#     between_m0 = (-x/2) + (-phi_{j+1}/2) + in_m0_next
#     between_m1 = out_m1_j + (x/2) + (phi_{j+1}/2) + in_m1_next
#   So decomposition runs ONCE, x sweep only updates between PSs.
#
# ERROR HANDLING:
#   If decomposition fails for any gate (common for theta ≈ pi),
#   code raises a clear ValueError and stops immediately.
#   This is intentional -- do not use allow_error=True.
#   Fix: choose a different L or regenerate angles.
#
# SLOS SIMULATION:
#   Uses real chip BS thetas (not ideal 50:50).
#   Builds full decomposed circuit with BS.Rx components.
#   SLOS fires photons layer by layer -- no matrix multiplication.
#   Difference between SLOS and QPU = hardware noise only.
#
# MSEs REPORTED:
#   All MSEs computed against SURROGATE function (not true STEP)
#   because QSP angles were optimized against surrogate.
#
# KEY INSIGHTS FROM SAMUEL (Quandela):
#   1. Combined PS on mode 0 = phi_mode0 - phi_mode1
#   2. Input PSs ignored (don't affect probabilities)
#   3. Remaining MZIs set to identity: mode0=pi, mode1=0
#
# HOW TO USE:
#   Change L, N_SHOTS, N_X at the top.
#   Make sure angle files exist for chosen L.
#   Results saved as .npy files and plot as .png.
# ============================================================

# ── Settings ──────────────────────────────────────────────
FUNC_NAME    = "STEP"
ANGLE_METHOD = "pq"
L            = 11       # <-- change to any L
N_SHOTS_SLOS = 5000    # <-- shots per x value for SLOS
N_SHOTS_QPU  = 5000    # <-- shots per x value for QPU
N_X          = 30      # <-- number of x values
N_approx     = 100     # <-- surrogate sharpness

TOKEN = "_T_eyJhbGciOiJIUzUxMiIsInR5cCI6IkpXVCJ9.eyJpZCI6MjU5MiwiZXhwIjoxODA2NTkwNjc1LjE4NjQ3NDN9.eCJCJNcF7BqyUtKiUSaOPK1RM0Nd42B-2UXwH0IIRiYy0TOqi9eQWcUM3049KIkcjgyKkHT-xyIxzoqzmuxyaQ"

# ── Derived settings ───────────────────────────────────────
x_values  = np.linspace(-np.pi, np.pi, N_X)
x_fine    = np.linspace(-np.pi, np.pi, 300)
FILE_TAG  = f"{FUNC_NAME}_L{L}_NSLOS{N_SHOTS_SLOS}_NQPU{N_SHOTS_QPU}_x{N_X}"
FUNC_LOWER = FUNC_NAME.lower()

print(f"File tag: {FILE_TAG}")

# ── Target functions ───────────────────────────────────────
surrogate_func = lambda x: (2.0 / np.pi) * np.arctan(N_approx * x)
true_func      = lambda x: np.where(x >= 0, 1.0, -1.0)

# ── Load angles ───────────────────────────────────────────
theta_arr = np.load(f"theta_{FUNC_LOWER}_{ANGLE_METHOD}_L{L}.npy")
phi_arr   = np.load(f"phi_{FUNC_LOWER}_{ANGLE_METHOD}_L{L}.npy")

print(f"\nLoaded angles: L={L}")
print(f"theta_arr: {np.round(theta_arr, 6)}")
print(f"phi_arr:   {np.round(phi_arr, 6)}")

# ── Connect to Belenos and get actual BS thetas ───────────
print("\nConnecting to Belenos...")

remote_processor = RemoteProcessor("qpu:belenos", token=TOKEN)
print("Available keys in specs:")
for key in remote_processor.specs.keys():
    print(f"  {key}")
belenos_circuit_template = remote_processor.specs["specific_circuit"]

# Extract actual BS thetas from chip
bs_thetas = []
for r, c in belenos_circuit_template:
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

assert len(bs_thetas) >= n_bs_needed, \
    f"Not enough BSs! Need {n_bs_needed}, have {len(bs_thetas)}"
assert len(theta_arr) == n_mzis, \
    f"theta_arr length {len(theta_arr)} != n_mzis {n_mzis}"

print(f"\nActual Belenos BS thetas (first {n_bs_needed}):")
for i, t in enumerate(bs_thetas[:n_bs_needed]):
    print(f"  BS #{i+1}: theta={t:.6f} rad ({np.degrees(t):.4f} deg)")

# ── Helper: extract phi from PS description ───────────────
def extract_phi(desc):
    m = re.search(r"phi\s*=\s*([\-\+]?[\d\.eE\+\-]+|pi)", desc)
    if m:
        val_str = m.group(1)
        return np.pi if val_str == "pi" else float(val_str)
    return None

# ── Decompose ALL Ry gates ONCE ───────────────────────────
# For extreme theta (near 0 or pi), use IDEAL 50:50 BSs in decomposition
# (closed-form solvable). Phi values are then applied to chip's real BSs.
# Small per-gate error introduced — verified via local check before QPU.
print(f"\n{'='*60}")
print(f"Decomposing {n_mzis} Ry gates (runs ONCE for all x)")
print(f"{'='*60}")
mzi_values = []
EXTREME_THETA_THRESHOLD = 0.1  # within this distance of 0 or pi → ideal BSs

for j in range(n_mzis):
    theta_j  = float(theta_arr[j])
    bs1_real = bs_thetas[j * 2]
    bs2_real = bs_thetas[j * 2 + 1]

    # Detect extreme theta and use ideal BSs in decomposition if so
    is_extreme = (theta_j < EXTREME_THETA_THRESHOLD or
                  abs(theta_j - pi) < EXTREME_THETA_THRESHOLD)

    if is_extreme:
        bs1_dec = bs2_dec = pi/2  # ideal 50:50 for decomposition only
        print(f"\n  j={j}: theta={theta_j:.4f}  ⚠️  extreme — using IDEAL BSs")
        print(f"         (real chip BSs are {bs1_real:.4f}, {bs2_real:.4f})")
    else:
        bs1_dec, bs2_dec = bs1_real, bs2_real
        print(f"\n  j={j}: theta={theta_j:.4f}  BS thetas {bs1_dec:.4f}, {bs2_dec:.4f}")

    success = False
    for seed in range(200):
        pcvl.random_seed(seed)
        single_gate = comp.BS.Ry(theta=theta_j)
        U_gate = pcvl.Matrix(np.array(single_gate.compute_unitary()).tolist())
        mzi = (pcvl.Circuit(2)
               // BS.Rx(theta=bs1_dec)
               // (1, PS(pcvl.P("phi0")))
               // BS.Rx(theta=bs2_dec)
               // (1, PS(pcvl.P("phi1"))))
        decomposed = pcvl.Circuit.decomposition(
            U_gate, mzi, phase_shifter_fn=PS, max_try=10)

        if decomposed is not None:
            error = np.linalg.norm(
                np.array(U_gate) - np.array(decomposed.compute_unitary()))
            if error < 1e-6:
                print(f"  ✅ seed={seed}  error={error:.2e}")
                success = True
                break

    if not success:
        raise ValueError(
            f"\n{'='*60}\n"
            f"DECOMPOSITION FAILED for gate j={j}\n"
            f"  theta={theta_j:.6f} rad ({np.degrees(theta_j):.4f} deg)\n"
            f"  BS thetas used in decomposition: {bs1_dec:.6f}, {bs2_dec:.6f}\n"
            f"  Real chip BSs:                   {bs1_real:.6f}, {bs2_real:.6f}\n"
            f"  is_extreme = {is_extreme}\n"
            f"  Tried seeds 0-199 with max_try=10 -- all failed\n"
            f"{'='*60}")

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

    print(f"     phi_in_m0={phi_in_m0:.4f}  phi_in_m1={phi_in_m1:.4f}  "
          f"phi_mid_m1={phi_mid_m1:.4f}  phi_out_m1={phi_out_m1:.4f}")

print(f"\n✅ All {n_mzis} gates decomposed successfully!")

# ── Helper: build custom_phases for a given x ─────────────
def build_custom_phases(x_val):
    phases = []
    for j in range(n_mzis):
        in_m0_j, in_m1_j, mid_m1_j, out_m1_j = mzi_values[j]
        phases.append(0 - mid_m1_j)
        if j < n_mzis - 1:
            in_m0_next, in_m1_next, _, _ = mzi_values[j + 1]
            between_m0 = (-x_val/2) + (-phi_arr[j+1]/2) + in_m0_next
            between_m1 = out_m1_j + (x_val/2) + (phi_arr[j+1]/2) + in_m1_next
            phases.append(between_m0 - between_m1)
        else:
            phases.append(0 - out_m1_j)
    return phases

# ── Helper: build decomposed Perceval circuit ─────────────
# Used for SLOS simulation with real chip BS thetas
def build_decomposed_circuit(x_val):
    """
    Build the full decomposed QSP circuit using real chip BS thetas.
    Uses BS.Rx components -- same as physical chip.
    Input PSs (before first BS) are included but don't affect Z.
    """
    custom_phases = build_custom_phases(x_val)
    circuit = pcvl.Circuit(2, name=f"QSP_decomposed_L{L}")

    ps_idx = 0
    for j in range(n_mzis):
        bs1 = bs_thetas[j * 2]
        bs2 = bs_thetas[j * 2 + 1]
        in_m0_j, in_m1_j, mid_m1_j, out_m1_j = mzi_values[j]

        # Input PSs (ignored as per Samuel -- but included for completeness)
        # For j=0: input = Rz(phi_0) combined with MZI input
        # We skip input PSs as they don't affect probabilities

        # First BS
        circuit.add((0, 1), comp.BS.Rx(theta=bs1))

        # After first BS: combined PS on mode 0 only
        circuit.add(0, comp.PS(custom_phases[ps_idx]))
        ps_idx += 1

        # Second BS
        circuit.add((0, 1), comp.BS.Rx(theta=bs2))

        # After second BS: combined PS on mode 0 only
        circuit.add(0, comp.PS(custom_phases[ps_idx]))
        ps_idx += 1

    return circuit

# ── Helper: set PS values on Belenos chip circuit ─────────
def set_ps_values_on_chip(circuit, custom_phases):
    i = 0
    for r, c in circuit:
        if isinstance(c, PS):
            params = c.get_parameters()
            if not params:
                continue
            if r[0] == 0:
                if i < len(custom_phases):
                    params[0].set_value(custom_phases[i])
                    i += 1
                else:
                    params[0].set_value(pi)
            elif r[0] % 2 == 1:
                params[0].set_value(0)
            else:
                params[0].set_value(pi)
    return i

# ── Helper: compute pure math analytic Z ──────────────────
def compute_analytic_Z(x_val):
    """Pure math with ideal BS.Ry gates -- no chip imperfections."""
    circuit = pcvl.Circuit(2)
    circuit.add(0,      comp.PS(float(-phi_arr[0] / 2)))
    circuit.add(1,      comp.PS(float( phi_arr[0] / 2)))
    circuit.add((0, 1), comp.BS.Ry(theta=float(theta_arr[0])))
    for j in range(1, L + 1):
        circuit.add(0,      comp.PS(float(-x_val / 2)))
        circuit.add(1,      comp.PS(float( x_val / 2)))
        circuit.add(0,      comp.PS(float(-phi_arr[j] / 2)))
        circuit.add(1,      comp.PS(float( phi_arr[j] / 2)))
        circuit.add((0, 1), comp.BS.Ry(theta=float(theta_arr[j])))
    U   = np.array(circuit.compute_unitary())
    psi = U @ np.array([1.0, 0.0])
    return abs(psi[0])**2 - abs(psi[1])**2

# ── Helper: compute local Z with real BS thetas ───────────
def compute_local_Z(x_val):
    """Local simulation with real chip BS thetas -- no sampling."""
    custom_phases   = build_custom_phases(x_val)
    belenos_circuit = remote_processor.specs["specific_circuit"]
    set_ps_values_on_chip(belenos_circuit, custom_phases)
    U_belenos = np.array(belenos_circuit.compute_unitary())
    psi = U_belenos[:, 0]
    return abs(psi[0])**2 - abs(psi[1])**2

# ============================================================
# Step 1: Pure math analytic sweep (instant)
# ============================================================
print(f"\n{'='*60}")
print(f"Step 1: Pure math analytic sweep ({N_X} x values)")
print(f"{'='*60}")
z_analytic = np.array([compute_analytic_Z(x) for x in x_values])
print(f"Done. Z range: [{z_analytic.min():.4f}, {z_analytic.max():.4f}]")

# ============================================================
# Step 2: Local simulation with real BS thetas (instant)
# ============================================================
print(f"\n{'='*60}")
print(f"Step 2: Local simulation with real BS thetas ({N_X} x values)")
print(f"{'='*60}")
z_local = np.array([compute_local_Z(x) for x in x_values])
print(f"Done. Z range: [{z_local.min():.4f}, {z_local.max():.4f}]")

# ============================================================
# Step 3: SLOS simulation -- real chip BS thetas, layer by layer
# ============================================================
print(f"\n{'='*60}")
print(f"Step 3: SLOS simulation ({N_X} x values, {N_SHOTS_SLOS} shots each)")
print(f"Uses real chip BS thetas -- layer by layer, no matrix multiply")
print(f"{'='*60}")

z_slos  = np.zeros(N_X)
p0_slos = np.zeros(N_X)
p1_slos = np.zeros(N_X)

for idx, x_val in enumerate(x_values):
    circuit    = build_decomposed_circuit(x_val)
    local_proc = pcvl.Processor("SLOS", circuit)
    local_proc.with_input(pcvl.BasicState([1, 0]))
    local_proc.min_detected_photons_filter(1)

    sampler = Sampler(local_proc)
    results = sampler.sample_count(N_SHOTS_SLOS)
    counts  = dict(results['results'])

    count_mode0 = counts.get(pcvl.BasicState([1, 0]), 0)
    count_mode1 = counts.get(pcvl.BasicState([0, 1]), 0)
    total       = count_mode0 + count_mode1

    if total > 0:
        p0 = count_mode0 / total
        p1 = count_mode1 / total
        z  = p0 - p1
    else:
        p0, p1, z = 0.0, 0.0, 0.0
        print(f"  WARNING: no counts at x={x_val:.3f}")

    z_slos[idx]  = z
    p0_slos[idx] = p0
    p1_slos[idx] = p1

    print(f"  [{idx+1:3d}/{N_X}] x={x_val:+.3f}  "
          f"p0={p0:.3f}  p1={p1:.3f}  Z={z:+.4f}")

print(f"SLOS done.")

# ============================================================
# Step 4: QPU sweep -- real chip, real photons
# (Throttled to respect Academic Offer 5-job concurrent limit)
# ============================================================
print(f"\n{'='*60}")
print(f"Step 4: QPU sweep ({N_X} jobs, {N_SHOTS_QPU} shots each)")
print(f"Throttling at MAX_CONCURRENT=4 to respect Academic Offer cap")
print(f"{'='*60}")

MAX_CONCURRENT = 4   # ← stay under the 5-job academic cap with 1 slot buffer
POLL_INTERVAL  = 15  # ← seconds between queue checks

job_ids       = []
submitted_jobs = []  # track {id: job_obj} for status polling
z_local_check = np.zeros(N_X)

def count_active_jobs(jobs):
    """Return number of jobs still waiting/running."""
    n = 0
    for j in jobs:
        try:
            if not j.is_complete:
                n += 1
        except Exception:
            n += 1  # if we can't tell, assume active
    return n

for idx, x_val in enumerate(x_values):
    print(f"\n[{idx+1:3d}/{N_X}] x={x_val:+.4f}")

    custom_phases   = build_custom_phases(x_val)
    belenos_circuit = remote_processor.specs["specific_circuit"]
    set_ps_values_on_chip(belenos_circuit, custom_phases)

    # Local verification
    U_belenos = np.array(belenos_circuit.compute_unitary())
    psi = U_belenos[:, 0]
    Z_loc = abs(psi[0])**2 - abs(psi[1])**2
    z_local_check[idx] = Z_loc

    match = np.isclose(Z_loc, z_analytic[idx], atol=0.05)
    print(f"  Z local={Z_loc:.4f}  Z analytic={z_analytic[idx]:.4f}  Match={match}")

   
    if not match:
        print(f"  ⚠️  Local verification failed — submitting anyway (user override)")
        # job_ids.append("SKIPPED")   # ← was here, now disabled
        # continue                     # ← was here, now disabled

    # ── Wait if we already have MAX_CONCURRENT jobs pending ──
    while count_active_jobs(submitted_jobs) >= MAX_CONCURRENT:
        n_active = count_active_jobs(submitted_jobs)
        print(f"  ⏳ {n_active} jobs active (cap={MAX_CONCURRENT}). Waiting {POLL_INTERVAL}s...")
        time.sleep(POLL_INTERVAL)

    # Submit job
    rp = RemoteProcessor("qpu:belenos", token=TOKEN)
    rp.set_circuit(belenos_circuit)
    rp.with_input(pcvl.BasicState([1, 0] + [0] * 22))
    rp.min_detected_photons_filter(1)

    sampler = Sampler(rp, max_shots_per_call=N_SHOTS_QPU)
    try:
        job = sampler.sample_count.execute_async(N_SHOTS_QPU)
        job_ids.append(job.id)
        submitted_jobs.append(job)
        print(f"  ✅ Job submitted: {job.id}  (active: {count_active_jobs(submitted_jobs)}/{MAX_CONCURRENT})")
        time.sleep(5)
    except Exception as e:
        print(f"  ⚠️  Job submission failed: {e}")
        job_ids.append("FAILED")
        continue

# Save job IDs
job_id_file = f"job_ids_{FILE_TAG}.txt"
with open(job_id_file, "w") as f:
    for idx, (x_val, job_id) in enumerate(zip(x_values, job_ids)):
        f.write(f"x={x_val:.6f}  job_id={job_id}  "
                f"z_local={z_local_check[idx]:.6f}  "
                f"z_analytic={z_analytic[idx]:.6f}\n")
print(f"\n✅ All jobs submitted! IDs saved to: {job_id_file}")

# ── Retrieve QPU results ───────────────────────────────────
print(f"\nRetrieving QPU results...")
z_qpu  = np.zeros(N_X)
p0_qpu = np.zeros(N_X)
p1_qpu = np.zeros(N_X)

for idx, (x_val, job_id) in enumerate(zip(x_values, job_ids)):
    if job_id in ["SKIPPED", "FAILED"]:
        z_qpu[idx] = np.nan
        print(f"  [{idx+1:3d}/{N_X}] x={x_val:+.4f}  {job_id}")
        continue

    rp_ret = RemoteProcessor("qpu:belenos", token=TOKEN)
    job    = rp_ret.resume_job(job_id)

    while not job.is_complete:
        time.sleep(10)

    results = job.get_results()
    counts  = dict(results['results'])
    total   = sum(counts.values())
     
    # ── Guard against zero detected counts ──
    if total == 0:
        z_qpu[idx] = np.nan
        print(f"  [{idx+1:3d}/{N_X}] x={x_val:+.4f}  ⚠️  no counts")
        continue

    p0 = counts.get(pcvl.BasicState([1, 0] + [0] * 22), 0) / total
    p1 = counts.get(pcvl.BasicState([0, 1] + [0] * 22), 0) / total
    Z  = p0 - p1


    p0_qpu[idx] = p0
    p1_qpu[idx] = p1
    z_qpu[idx]  = Z

    print(f"  [{idx+1:3d}/{N_X}] x={x_val:+.4f}  "
          f"p0={p0:.3f}  p1={p1:.3f}  Z={Z:+.4f}")
# ============================================================
# Step 5: Save all results
# ============================================================
np.save(f"x_values_{FILE_TAG}.npy",   x_values)
np.save(f"z_analytic_{FILE_TAG}.npy", z_analytic)
np.save(f"z_local_{FILE_TAG}.npy",    z_local)
np.save(f"z_slos_{FILE_TAG}.npy",     z_slos)
np.save(f"z_qpu_{FILE_TAG}.npy",      z_qpu)
np.save(f"p0_qpu_{FILE_TAG}.npy",     p0_qpu)
np.save(f"p1_qpu_{FILE_TAG}.npy",     p1_qpu)
print(f"\n✅ All results saved with tag: {FILE_TAG}")

# ============================================================
# Step 6: MSE report (all vs surrogate + new QPU comparisons)
# ============================================================
valid = ~np.isnan(z_qpu)
surr  = surrogate_func(x_values)

mse_analytic_vs_surr = np.mean((z_analytic - surr)**2)
mse_local_vs_surr    = np.mean((z_local    - surr)**2)
mse_slos_vs_surr     = np.mean((z_slos     - surr)**2)
mse_qpu_vs_surr      = np.mean((z_qpu[valid] - surr[valid])**2)
mse_qpu_vs_local     = np.mean((z_qpu[valid] - z_local[valid])**2)
mse_slos_vs_local    = np.mean((z_slos     - z_local)**2)

# ── New MSEs: Analytic (ideal BSs) vs QPU and Analytic (real BSs) vs QPU ──
mse_analytic_vs_qpu  = np.mean((z_analytic[valid] - z_qpu[valid])**2)
mse_local_vs_qpu     = np.mean((z_local[valid]    - z_qpu[valid])**2)

print(f"\n========== MSE Report [{FILE_TAG}] ==========")
print(f"  MSE analytic vs surrogate         : {mse_analytic_vs_surr:.4f}")
print(f"  MSE local    vs surrogate         : {mse_local_vs_surr:.4f}")
print(f"  MSE SLOS     vs surrogate         : {mse_slos_vs_surr:.4f}")
print(f"  MSE QPU      vs surrogate         : {mse_qpu_vs_surr:.4f}")
print(f"  MSE QPU      vs local             : {mse_qpu_vs_local:.4f}")
print(f"  MSE SLOS     vs local             : {mse_slos_vs_local:.4f}")
print(f"  MSE Analytic (ideal BSs) vs QPU   : {mse_analytic_vs_qpu:.4f}")
print(f"  MSE Analytic (real BSs)  vs QPU   : {mse_local_vs_qpu:.4f}")
print(f"==============================================")

# ============================================================
# Step 7: Plot all curves
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle(
    f"QSP Layer-by-Layer on Belenos  {FUNC_NAME}  L={L}  "
    f"N_SLOS={N_SHOTS_SLOS}  N_QPU={N_SHOTS_QPU}  N_x={N_X}",
    fontsize=12, fontweight='bold'
)

xt = [-np.pi, 0, np.pi]
xl = [r"$-\pi$", r"$0$", r"$\pi$"]

# ── Left panel: all Z curves ──────────────────────────────
ax = axes[0]
ax.plot(x_fine, true_func(x_fine),     'k-',  lw=2.5,
        label=f"True {FUNC_NAME}")
ax.plot(x_fine, surrogate_func(x_fine),'k--', lw=1.5,
        label="Surrogate")
ax.plot(x_values, z_analytic,          'r-',  lw=1.5,
        label=f"Analytic (ideal BSs) vs Surr  "
              f"MSE={mse_analytic_vs_surr:.4f}")
ax.plot(x_values, z_local,             'g-',  lw=1.5,
        label=f"Analytic (real BSs) vs Surr "
              f"MSE={mse_local_vs_surr:.4f}")
ax.plot(x_values, z_slos,              'm.',  ms=8,
        label=f"SLOS (real BSs) vs Surr "
              f"MSE={mse_slos_vs_surr:.4f}")
ax.plot(x_values[valid], z_qpu[valid], 'b.',  ms=10,
        label=f"QPU (real BSs) vs Surr "
              f"MSE={mse_qpu_vs_surr:.4f}")

# ── Invisible entries to add the new MSEs to the legend ──
ax.plot([], [], ' ',
        label=f"Analytic (ideal BSs) vs QPU  "
              f"MSE={mse_analytic_vs_qpu:.4f}")
ax.plot([], [], ' ',
        label=f"Analytic (real BSs)  vs QPU  "
              f"MSE={mse_local_vs_qpu:.4f}")

ax.set_xlim([-np.pi, np.pi])
ax.set_ylim([-1.3, 1.3])
ax.set_xticks(xt); ax.set_xticklabels(xl, fontsize=12)
ax.set_xlabel(r"$x$", fontsize=12)
ax.set_ylabel("Z = p0 - p1", fontsize=12)
ax.set_title(f"QSP on Belenos  {FUNC_NAME}  L={L}", fontsize=11)
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# ── Right panel: QPU vs local residual ───────────────────
ax2 = axes[1]
diff_qpu_local  = z_qpu[valid]  - z_local[valid]
diff_slos_local = z_slos        - z_local

ax2.plot(x_values[valid], diff_qpu_local,  'b.',
         ms=8, label=f"QPU minus local  "
                     f"MSE={mse_qpu_vs_local:.4f}")
ax2.plot(x_values, diff_slos_local, 'm-',
         lw=1.5, label=f"SLOS minus local  "
                       f"MSE={mse_slos_vs_local:.4f}")
ax2.axhline(0, color='k', lw=0.8, linestyle='--')
ax2.fill_between(x_values[valid], diff_qpu_local,
                 alpha=0.15, color='blue')
ax2.fill_between(x_values, diff_slos_local,
                 alpha=0.15, color='magenta')
ax2.set_xlim([-np.pi, np.pi])
ax2.set_xticks(xt); ax2.set_xticklabels(xl, fontsize=12)
ax2.set_xlabel(r"$x$", fontsize=12)
ax2.set_ylabel("Residual", fontsize=12)
ax2.set_title("QPU and SLOS vs local residual", fontsize=11)
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plot_filename = f"qsp_qpu_{FILE_TAG}.png"
plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
plt.show()
print(f"Plot saved: {plot_filename}")