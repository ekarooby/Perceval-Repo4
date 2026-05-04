# ============================================================
# QSVT EIGENVALUE FILTER ON BELENOS QPU -- d=6 (T_6 FILTER)
# ============================================================
#
# GOAL:
#   Submit the d=6 QSVT eigenvalue filter on the Belenos chip,
#   one job per Hamiltonian eigenstate input (4 jobs total),
#   with 2000 shots each, and compare measured filter success
#   probabilities to T_6(lambda_k / alpha)^2.
#
# PIPELINE (mirrors the 2-mode QSP layer-by-layer pattern from
# 2026_04_24_CircuitDecomposition_MZIs_AnyL_x0.5_PSonlyOnMode1_fab.py):
#
#   Step A: Build U_QSVT (8x8) with d=6 layers of U_H alternating
#           with 7 R(phi) phase rotations. Top-left 4x4 = T_6(H/alpha).
#
#   Step B: Connect to Belenos and extract chip BS thetas. The chip
#           BSs are NOT 50:50; each is fixed at a known theta. The
#           cloud's universalchipworker compiler inverts the
#           constrained mesh to find PSs; locally we rely on the
#           seed-search Reck pattern (proven robust for the QSP
#           workflow) and submit the resulting circuit so the cloud
#           reproduces U_24 (top-left 8x8 = U_QSVT, identity rest).
#
#   Step C: For each Hamiltonian eigenstate index k = 0..3, build a
#           4x4 state-prep unitary V_k whose first column is V_H[:,k]
#           so that |mode 0> -> |0_anc> kron |lambda_k>. Compose
#           U_full_k = U_24 @ block_diag(V_k, I_20).
#
#   Step D: Local verification (threshold 1e-3) -- compute_unitary
#           on the prepared circuit gives U_full_k (ideal); check
#           the photon density on the ancilla=|0> register matches
#           |T_6(lambda_k / alpha)|^2. Only submit if all 4 pass.
#
#   Step E: Submit 4 async jobs (2000 shots each, no repeats).
#           For each job, count the photons that fall in modes 0..3
#           (ancilla=|0> = filter success) vs modes 4..7
#           (ancilla=|1> = filter rejected) vs modes 8..23 (leakage).
#
#   Step F: Plot measured P(filter success | k) against
#           |T_6(lambda_k / alpha)|^2 and save to
#           results_qsvt_d6_belenos.png + .npz.
#
# CREDIT NOTE:
#   4 jobs * 2000 shots = 8000 shots total (down from 4 * 5000).
#   Set SUBMIT_TO_QPU = False to dry-run locally without spending
#   credits.
# ============================================================

import os
import re
import time
import numpy as np
import matplotlib.pyplot as plt

import perceval as pcvl
import perceval.components as comp
from perceval.components import BS, PS, Unitary
from perceval.algorithm import Sampler

from qsvt_angles import chebyshev_qsp_angles
from qsvt_tfim_matrix_reference import (
    build_tfim_hamiltonian,
    build_block_encoding,
    build_qsvt_unitary,
    DIM_SYS, DIM_ANC, DIM_BE,
)

# ── Settings ──────────────────────────────────────────────
H_FIELD          = 1.0
QSVT_DEGREE      = 6
N_MODES_CHIP     = 24
SHOTS_PER_JOB    = 2000     # down from 5000 to save credit
N_EIGENSTATES    = 4        # 4 jobs total, no repeats
LOCAL_TOL        = 1e-3
SUBMIT_TO_QPU    = False    # MUST be flipped to True manually before submitting -- jobs cost credits
SEED             = 42

TOKEN = "_T_eyJhbGciOiJIUzUxMiIsInR5cCI6IkpXVCJ9.eyJpZCI6MjU5MiwiZXhwIjoxODA2NTkwNjc1LjE4NjQ3NDN9.eCJCJNcF7BqyUtKiUSaOPK1RM0Nd42B-2UXwH0IIRiYy0TOqi9eQWcUM3049KIkcjgyKkHT-xyIxzoqzmuxyaQ"

OUT_PNG = os.path.join(os.path.dirname(__file__),
                       "results_qsvt_d6_belenos.png")
OUT_NPZ = os.path.join(os.path.dirname(__file__),
                       "results_qsvt_d6_belenos.npz")
DIAG_PNG = os.path.join(os.path.dirname(__file__), "diagnostics_d6.png")
DIAG_NPZ = os.path.join(os.path.dirname(__file__), "diagnostics_d6.npz")

# ── Diagnostics: fetch already-completed jobs by ID (NO new submission) ──
# Set to {} to skip the diagnostic Step G entirely.
RETRIEVE_AND_DIAGNOSE = True
JOB_IDS = {
    0: "28cf3a16-ce34-4353-ad91-d38dcc8dbb9c",   # k=0  lambda=-2.236
    1: "437a7a53-4b1b-4eb2-abc6-edb82e5566e3",   # k=1  lambda=-1.000
    2: "0604f60b-48db-4fae-bf88-e4827a76983e",   # k=2  lambda=+1.000
    3: "8df96ed5-0070-45fc-a604-99bc1ddea4f6",   # k=3  lambda=+2.236
}

pcvl.random_seed(SEED)
np.random.seed(SEED)


# ── Step A: Build d=6 U_QSVT ──────────────────────────────
print("=" * 66)
print("  Step A -- build U_QSVT (d=6, T_6 filter)")
print("=" * 66)

H = build_tfim_hamiltonian(H_FIELD)
eigs_H, V_H = np.linalg.eigh(H)
U_H, alpha, A = build_block_encoding(H)
angles = chebyshev_qsp_angles(QSVT_DEGREE)
U_qsvt = build_qsvt_unitary(U_H, angles)

T6_eigs = np.cos(QSVT_DEGREE *
                 np.arccos(np.clip(eigs_H / alpha, -1, 1)))

print(f"H eigenvalues       = {np.round(eigs_H, 6)}")
print(f"alpha = ||H||_2     = {alpha:.6f}")
print(f"angles (d={QSVT_DEGREE})       = {np.round(angles, 6)}")
print(f"#layers: {QSVT_DEGREE} U_H + {QSVT_DEGREE+1} R(phi)")
print(f"T_6(eigvals/alpha) = {np.round(T6_eigs, 6)}")
print(f"|T_6(eigvals/alpha)|^2 (target probabilities) = "
      f"{np.round(T6_eigs**2, 6)}")
print(f"||U_QSVT[:4,:4] - T_6(A)|| = "
      f"{np.linalg.norm(U_qsvt[:DIM_SYS, :DIM_SYS] - V_H @ np.diag(T6_eigs) @ V_H.conj().T):.2e}")

U_24 = np.eye(N_MODES_CHIP, dtype=complex)
U_24[:DIM_BE, :DIM_BE] = U_qsvt


# ── Step B: Connect to Belenos, extract chip BS thetas ────
print("\n" + "=" * 66)
print("  Step B -- connect to Belenos, extract chip BS thetas")
print("=" * 66)

bs_thetas = []
belenos_online = False
try:
    remote_processor = pcvl.RemoteProcessor("qpu:belenos", token=TOKEN)
    arch = remote_processor.specs["architecture"]
    belenos_circuit = arch.unitary_circuit()
    for r, c in belenos_circuit:
        if r == (0, 1):
            m = re.search(r"theta=([\-\+]?[\d\.eE\+\-]+)", c.describe())
            if m:
                bs_thetas.append(float(m.group(1)))
    belenos_online = True
    print(f"Belenos modes = {arch.m},  BSs on (0,1) = {len(bs_thetas)}")
    for i, t in enumerate(bs_thetas[:8]):
        print(f"  BS #{i+1:2d}: theta={t:.6f} rad ({np.degrees(t):.3f} deg)")
    print("...")
except Exception as e:
    print(f"Belenos unreachable ({e}) -- continuing in local-only mode.")
    print("(Step E will be skipped automatically.)")


# ── Step C: Build state-prep + full circuits per eigenstate
print("\n" + "=" * 66)
print(f"  Step C -- build {N_EIGENSTATES} state-prep + QSVT circuits")
print("=" * 66)

def state_prep_4x4(k):
    """4x4 unitary whose first column is V_H[:,k] (Hamiltonian eigvec)."""
    cols = [V_H[:, k]] + [V_H[:, j] for j in range(DIM_SYS) if j != k]
    return np.column_stack(cols).astype(complex)

U_full_per_k = []
expected_amps_top = []
for k in range(N_EIGENSTATES):
    Vk = state_prep_4x4(k)
    P24 = np.eye(N_MODES_CHIP, dtype=complex)
    P24[:DIM_SYS, :DIM_SYS] = Vk
    Uk = U_24 @ P24

    # Predicted single-photon amplitudes after circuit acting on |mode 0>:
    psi_in  = np.zeros(N_MODES_CHIP, dtype=complex); psi_in[0] = 1.0
    psi_out = Uk @ psi_in
    U_full_per_k.append(Uk)
    expected_amps_top.append(psi_out[:DIM_BE])

    p_filter = float(np.sum(np.abs(psi_out[:DIM_SYS])**2))
    print(f"  k={k}  lambda(H)={eigs_H[k]:+.4f}  "
          f"|T_6(lambda/alpha)|^2={T6_eigs[k]**2:.4f}  "
          f"predicted P(modes 0..3) = {p_filter:.4f}")


# ── Step D: Local verification (threshold 1e-3) ───────────
print("\n" + "=" * 66)
print(f"  Step D -- local verification (threshold {LOCAL_TOL})")
print("=" * 66)

local_ok = True
for k in range(N_EIGENSTATES):
    ckt = pcvl.Circuit(N_MODES_CHIP, name=f"QSVT_d{QSVT_DEGREE}_k{k}")
    ckt.add(0, Unitary(pcvl.Matrix(U_full_per_k[k])))
    U_check = np.array(ckt.compute_unitary())
    psi0 = np.zeros(N_MODES_CHIP, dtype=complex); psi0[0] = 1.0
    psi_chk = U_check @ psi0
    err = np.linalg.norm(psi_chk[:DIM_BE] - expected_amps_top[k])
    pf_ideal = float(np.sum(np.abs(psi_chk[:DIM_SYS])**2))
    pf_theory = T6_eigs[k]**2
    diff_p   = abs(pf_ideal - pf_theory)
    ok = (err < LOCAL_TOL) and (diff_p < LOCAL_TOL)
    local_ok = local_ok and ok
    print(f"  k={k}  ||amp - predicted||={err:.2e}  "
          f"|P_ideal - |T_6|^2|={diff_p:.2e}  "
          f"-> {'PASS' if ok else 'FAIL'}")

print(f"\nLocal verification: {'PASS' if local_ok else 'FAIL'}  "
      f"(submission {'allowed' if local_ok else 'BLOCKED'})")


# ── Step E: Submit 4 jobs (2000 shots each) ───────────────
qpu_p_filter = np.full(N_EIGENSTATES, np.nan)

if not local_ok:
    print("\nLocal verification failed -- not submitting any jobs.")
elif not SUBMIT_TO_QPU:
    print("\nSUBMIT_TO_QPU=False -- skipping QPU jobs (dry run).")
elif not belenos_online:
    print("\nBelenos offline (Step B failed) -- skipping QPU jobs.")
else:
    print("\n" + "=" * 66)
    print(f"  Step E -- submit {N_EIGENSTATES} jobs to Belenos "
          f"({SHOTS_PER_JOB} shots each)")
    print("=" * 66)

    jobs = []
    for k in range(N_EIGENSTATES):
        ckt = pcvl.Circuit(N_MODES_CHIP, name=f"QSVT_d{QSVT_DEGREE}_k{k}")
        ckt.add(0, Unitary(pcvl.Matrix(U_full_per_k[k])))

        rp = pcvl.RemoteProcessor("qpu:belenos", token=TOKEN)
        rp.set_circuit(ckt)
        rp.with_input(pcvl.BasicState([1] + [0] * (N_MODES_CHIP - 1)))
        rp.min_detected_photons_filter(1)

        sampler = Sampler(rp, max_shots_per_call=SHOTS_PER_JOB)
        job = sampler.sample_count.execute_async(SHOTS_PER_JOB)
        print(f"  k={k}  job_id={job.id}")
        jobs.append((k, job))

    for k, job in jobs:
        print(f"\n  waiting for k={k} (job_id={job.id})...")
        while not job.is_complete:
            print(f"    status={job.status}")
            time.sleep(5)
        results = job.get_results()
        counts  = dict(results["results"])
        total   = sum(counts.values())

        # Filter success = photon detected in modes 0..3.
        n_succ = 0
        for state, cnt in counts.items():
            occ = list(state)
            if any(occ[m] > 0 for m in range(DIM_SYS)):
                n_succ += cnt
        p_succ = n_succ / total if total > 0 else 0.0
        qpu_p_filter[k] = p_succ
        print(f"  k={k}  total={total}  "
              f"P(filter success, modes 0..3) = {p_succ:.4f}  "
              f"|T_6(lambda/alpha)|^2 = {T6_eigs[k]**2:.4f}")


# ── Step F: Plot + save ───────────────────────────────────
print("\n" + "=" * 66)
print("  Step F -- plot + save results")
print("=" * 66)

x = np.arange(N_EIGENSTATES)
target = T6_eigs**2
predicted = np.array([
    float(np.sum(np.abs(expected_amps_top[k][:DIM_SYS])**2))
    for k in range(N_EIGENSTATES)
])

fig, ax = plt.subplots(figsize=(8, 5))
w = 0.28
ax.bar(x - w, target,    w, label=r"target $|T_6(\lambda/\alpha)|^2$",
       color="#3060c0")
ax.bar(x,      predicted, w, label="ideal Perceval (matrix)",
       color="#c08020")
ax.bar(x + w, qpu_p_filter, w, label=f"Belenos QPU ({SHOTS_PER_JOB} shots)",
       color="#a02050")
ax.set_xticks(x)
ax.set_xticklabels([f"k={k}\n$\\lambda$={eigs_H[k]:+.3f}"
                    for k in range(N_EIGENSTATES)])
ax.set_ylabel("P(filter success | eigenstate k)")
ax.set_title(r"QSVT eigenvalue filter on Belenos -- $T_6(H/\alpha)$, "
             f"h={H_FIELD}")
ax.set_ylim(0, 1.1)
ax.grid(True, axis="y", alpha=0.3)
ax.legend()
fig.tight_layout()
fig.savefig(OUT_PNG, dpi=140)
print(f"Saved plot: {OUT_PNG}")

np.savez(
    OUT_NPZ,
    H=H, eigs_H=eigs_H, V_H=V_H, alpha=alpha,
    angles=angles, U_qsvt=U_qsvt, U_24=U_24,
    T6_eigvals_over_alpha=T6_eigs,
    target_probabilities=target,
    predicted_probabilities=predicted,
    qpu_p_filter=qpu_p_filter,
    bs_thetas_belenos=np.array(bs_thetas),
    shots_per_job=SHOTS_PER_JOB,
)
print(f"Saved data: {OUT_NPZ}")


# ── Step G: Diagnostics from already-completed jobs ───────
# READ-ONLY: never submits anything. Pulls results for the IDs in
# JOB_IDS and produces per-eigenstate mode-distribution diagnostics
# to understand why middle eigenstates (k=1,2) overshoot the T_6 target.
if not RETRIEVE_AND_DIAGNOSE or not JOB_IDS:
    print("\n[Step G skipped -- RETRIEVE_AND_DIAGNOSE=False or JOB_IDS empty]")
else:
    print("\n" + "=" * 66)
    print("  Step G -- diagnostics from completed jobs (READ-ONLY)")
    print("=" * 66)
    print("Fetching results for", len(JOB_IDS), "job IDs (no submission).")

    pcvl.RemoteConfig.set_token(TOKEN)
    pcvl.RemoteConfig().save()
    rp_fetch = pcvl.RemoteProcessor("qpu:belenos")

    # Per-mode photon distribution: counts[k, m] = #photons detected in mode m.
    n_modes_inspect = DIM_BE   # modes 0..7 (system + ancilla)
    measured_mode_counts = np.zeros((N_EIGENSTATES, n_modes_inspect))
    measured_mode_total  = np.zeros(N_EIGENSTATES)
    mode0_only_counts    = np.zeros(N_EIGENSTATES)
    detected_total       = np.zeros(N_EIGENSTATES, dtype=int)
    requested_shots      = np.zeros(N_EIGENSTATES, dtype=int)
    physical_perf        = np.full(N_EIGENSTATES, np.nan)
    n_components_chip    = np.zeros(N_EIGENSTATES, dtype=int)
    n_bs_chip            = np.zeros(N_EIGENSTATES, dtype=int)
    n_ps_chip            = np.zeros(N_EIGENSTATES, dtype=int)

    for k, jid in JOB_IDS.items():
        print(f"\n--- k={k}  lambda={eigs_H[k]:+.4f}  job_id={jid} ---")
        job = rp_fetch.resume_job(jid)
        results = job.get_results()

        # (5) Raw shots vs detected events
        keys_avail = list(results.keys())
        print(f"  result keys: {keys_avail}")
        raw  = results["results"]
        total = sum(raw.values())
        detected_total[k] = total
        for meta_key in ("physical_perf", "platform_perf",
                         "global_perf", "n_input"):
            if meta_key in results:
                print(f"  {meta_key} = {results[meta_key]}")
        if "physical_perf" in results:
            physical_perf[k] = float(results["physical_perf"])
        # Try multiple metadata names for "shots requested".
        for shot_key in ("nshot", "n_shots", "shots", "input_shots"):
            if shot_key in results:
                requested_shots[k] = int(results[shot_key])
                print(f"  requested {shot_key} = {results[shot_key]}")
                break

        # (1) Per-mode photon count distribution within modes 0..7.
        # Each detected event contributes its photon occupation per mode.
        # mode-0-only = events where exactly one photon and it sits in mode 0.
        m0_only = 0
        for state, cnt in raw.items():
            occ = list(state)
            for m in range(n_modes_inspect):
                measured_mode_counts[k, m] += occ[m] * cnt
            measured_mode_total[k] += sum(occ[:n_modes_inspect]) * cnt
            if sum(occ) == 1 and occ[0] == 1:
                m0_only += cnt
        mode0_only_counts[k] = m0_only

        # Theoretical distribution: |U_full_k @ |mode 0>|^2 over modes 0..7.
        psi0 = np.zeros(N_MODES_CHIP, dtype=complex); psi0[0] = 1.0
        psi_th = U_full_per_k[k] @ psi0
        th_dist = (np.abs(psi_th[:n_modes_inspect])**2)
        # Normalize measured to a probability over modes 0..7.
        meas_dist = (measured_mode_counts[k, :n_modes_inspect]
                     / max(measured_mode_total[k], 1))

        print(f"  {'mode':<6} {'theory P':>10} {'measured P':>12} "
              f"{'|diff|':>10}")
        for m in range(n_modes_inspect):
            tag = " <- ancilla=1 starts" if m == DIM_SYS else ""
            print(f"  {m:<6} {th_dist[m]:>10.4f} {meas_dist[m]:>12.4f} "
                  f"{abs(th_dist[m] - meas_dist[m]):>10.4f}{tag}")

        # (3) Mode-0-only concentration flag.
        p_mode0_only = m0_only / total if total else 0.0
        flag = "  <-- !! >80% in single mode 0 -- eigenvector amplitudes collapsed !!" \
               if p_mode0_only > 0.80 else ""
        print(f"  P(photon in mode 0 only) = {p_mode0_only:.4f}{flag}")

        # (2) Inspect the actual chip-side circuit if returned.
        cc = results.get("computed_circuit", None)
        if cc is not None:
            try:
                n_components_chip[k] = sum(1 for _ in cc)
                n_bs_chip[k] = sum(1 for _, c in cc if isinstance(c, BS))
                n_ps_chip[k] = sum(1 for _, c in cc if isinstance(c, PS))
                print(f"  computed_circuit: {n_components_chip[k]} components "
                      f"({n_bs_chip[k]} BS, {n_ps_chip[k]} PS), "
                      f"modes={cc.m}")
            except Exception as e:
                print(f"  computed_circuit inspection failed: {e}")
        else:
            print("  (no computed_circuit in results)")

    # ── Diagnostic plot: theory vs measured per-mode for each k ──
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True, sharey=True)
    x = np.arange(n_modes_inspect); w = 0.4
    for k, ax in zip(range(N_EIGENSTATES), axes.flatten()):
        psi0 = np.zeros(N_MODES_CHIP, dtype=complex); psi0[0] = 1.0
        psi_th = U_full_per_k[k] @ psi0
        th_dist = (np.abs(psi_th[:n_modes_inspect])**2)
        meas_dist = (measured_mode_counts[k, :n_modes_inspect]
                     / max(measured_mode_total[k], 1))
        ax.bar(x - w/2, th_dist,   w, label="theory",   color="#3060c0")
        ax.bar(x + w/2, meas_dist, w, label="measured", color="#a02050")
        ax.axvline(DIM_SYS - 0.5, color="gray", linestyle="--", alpha=0.6)
        ax.set_title(f"k={k}, $\\lambda$={eigs_H[k]:+.3f}, "
                     f"N={detected_total[k]}")
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend(fontsize=8)
        ax.set_xticks(x)
    for ax in axes[-1, :]:
        ax.set_xlabel("output mode (0..3 = sys/anc=|0>,  4..7 = sys/anc=|1>)")
    for ax in axes[:, 0]:
        ax.set_ylabel("P(photon in mode m)")
    fig.suptitle(r"d=6 QSVT diagnostics: per-mode distribution per "
                 r"eigenstate input $|\lambda_k\rangle$")
    fig.tight_layout()
    fig.savefig(DIAG_PNG, dpi=140)
    print(f"\nSaved diagnostic plot: {DIAG_PNG}")

    np.savez(
        DIAG_NPZ,
        eigs_H=eigs_H, alpha=alpha,
        T6_eigs=T6_eigs, target_probabilities=target,
        job_ids=np.array([JOB_IDS[k] for k in range(N_EIGENSTATES)]),
        measured_mode_counts=measured_mode_counts,
        measured_mode_total=measured_mode_total,
        mode0_only_counts=mode0_only_counts,
        detected_total=detected_total,
        requested_shots=requested_shots,
        physical_perf=physical_perf,
        n_components_chip=n_components_chip,
        n_bs_chip=n_bs_chip,
        n_ps_chip=n_ps_chip,
        U_full_per_k=np.array(U_full_per_k),
    )
    print(f"Saved diagnostic data: {DIAG_NPZ}")

    # ── Summary table ──
    print("\n" + "=" * 66)
    print("  Step G summary -- where the deviation comes from")
    print("=" * 66)
    print(f"{'k':>2} {'lambda':>8} {'target':>8} {'detected':>9} "
          f"{'P(filter)':>10} {'P(mode-0 only)':>14}")
    for k in range(N_EIGENSTATES):
        meas_dist_top = measured_mode_counts[k, :DIM_SYS].sum() \
                        / max(measured_mode_total[k], 1)
        print(f"{k:>2} {eigs_H[k]:>+8.4f} {target[k]:>8.4f} "
              f"{detected_total[k]:>9d} {meas_dist_top:>10.4f} "
              f"{mode0_only_counts[k]/max(detected_total[k],1):>14.4f}")
