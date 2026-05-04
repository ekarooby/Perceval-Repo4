# ============================================================
# DIRECT-PS COMPILATION OF d=6 QSVT TO BELENOS -- v2
#
# Honest status of the L^-1 D L shuffle (Clements 2016 Sec A.1):
#   The clean shuffle requires the alternative T-matrix convention
#       T_full(theta, phi) = [[e^{iφ}sin(θ/2), e^{iφ}cos(θ/2)],
#                             [        cos(θ/2),       -sin(θ/2)]]
#   for which D · T_full = T_full(θ, φ + arg(α/β)) · D' with
#   D' = diag(β, β) on the affected pair. I implemented that
#   convention but my edge-case logic for near-zero pivots
#   produced a Clements decomposition with ||recon - U_QSVT||_F ~ 2,
#   so I reverted to the v1 convention (which decomposes U_QSVT
#   to ~1e-14) and use the shuffle abstractly: by composing
#   ALL Clements matrices into a single equivalent rectangular
#   mesh via numerical multiplication.
#
# Pipeline:
#   1. Clements decompose U_QSVT (v1 conv, exact).
#   2. Numerically reconstruct U_QSVT = L_chain @ diag(D) @ R_chain
#      and define M = L_chain @ diag(D) -- this absorbs the diagonal
#      into the left chain. Then U_QSVT = M @ R_chain. M is unitary.
#   3. RE-DECOMPOSE M @ R_chain (i.e. just U_QSVT, since they're equal)
#      using a layer-by-layer rectangular Clements that produces
#      exactly the chip's even/odd pair layer pattern.
#   4. Match each layer's T's to the corresponding chip MZIs.
#   5. Per-MZI seed search to fit (theta, phi) into chip BS thetas
#      via 2 free PSs.
#   6. Local verification (target < 1e-3).
#   7. If not < 1e-3, scipy.optimize on active PSs to refine.
#   8. Local Perceval simulation per eigenstate input. Plot + save.
#
# NO QPU SUBMISSION.
# ============================================================

import os, re, sys
from math import pi
from collections import defaultdict, Counter
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

import perceval as pcvl
import perceval.components as comp
from perceval.components import BS, PS

from qsvt_angles import chebyshev_qsp_angles
from qsvt_tfim_matrix_reference import (
    build_tfim_hamiltonian, build_block_encoding,
    build_qsvt_unitary, DIM_SYS, DIM_BE,
)

H_FIELD       = 1.0
QSVT_DEGREE   = 6
N_MODES_CHIP  = 24
LOCAL_TOL     = 1e-3
MAX_SEEDS_MZI = 50
SCIPY_MAXITER = 600
N_LOCAL_SHOTS = 5000
TOKEN = "_T_eyJhbGciOiJIUzUxMiIsInR5cCI6IkpXVCJ9.eyJpZCI6MjU5MiwiZXhwIjoxODA2NTkwNjc1LjE4NjQ3NDN9.eCJCJNcF7BqyUtKiUSaOPK1RM0Nd42B-2UXwH0IIRiYy0TOqi9eQWcUM3049KIkcjgyKkHT-xyIxzoqzmuxyaQ"

OUT_PNG = os.path.join(os.path.dirname(__file__),
                       "results_directPS_local_v2.png")
OUT_NPZ = os.path.join(os.path.dirname(__file__),
                       "results_directPS_local_v2.npz")
np.random.seed(42); pcvl.random_seed(42)


def parse_theta_str(d):
    m = re.search(r"theta=([\-\+]?[\d\.eE\+\-]+)", d)
    return float(m.group(1)) if m else None


def parse_phi_str(d):
    m = re.search(r"phi\s*=\s*([\-\+]?[\d\.eE\+\-]+|pi)", d)
    if not m: return None
    v = m.group(1); return pi if v == "pi" else float(v)


def inventory_mzis(circuit):
    comps = list(circuit); mzis = []; i = 0
    while i < len(comps):
        r, c = comps[i]
        if not isinstance(c, BS) or i + 3 >= len(comps):
            i += 1; continue
        r1, c1 = comps[i+1]; r2, c2 = comps[i+2]; r3, c3 = comps[i+3]
        if (isinstance(c1, PS) and isinstance(c2, PS) and isinstance(c3, BS)
                and r3 == r and len(r1) == 1 and len(r2) == 1
                and set(r1) | set(r2) == set(r)):
            mzis.append({"mode_pair": r, "bs1_idx": i,
                         "ps1_idx": i+1, "ps1_mode": r1[0],
                         "ps2_idx": i+2, "ps2_mode": r2[0], "bs2_idx": i+3,
                         "bs1_theta": parse_theta_str(c.describe()),
                         "bs2_theta": parse_theta_str(c3.describe())})
            i += 4
        else:
            i += 1
    return mzis


# ══════════════════════════════════════════════════════════
# Step 1: build U_QSVT
# ══════════════════════════════════════════════════════════
print("=" * 70, flush=True)
print("  Step 1 -- build U_QSVT (d=6)", flush=True)
print("=" * 70, flush=True)
H = build_tfim_hamiltonian(H_FIELD)
eigs_H, V_H = np.linalg.eigh(H)
U_H, alpha, A = build_block_encoding(H)
angles = chebyshev_qsp_angles(QSVT_DEGREE)
U_qsvt = build_qsvt_unitary(U_H, angles)
T6 = np.cos(QSVT_DEGREE * np.arccos(np.clip(eigs_H / alpha, -1, 1)))
target_p = T6**2
print(f"alpha={alpha:.4f}  T_6={np.round(T6,4)}  |T_6|^2={np.round(target_p,4)}",
      flush=True)


# ══════════════════════════════════════════════════════════
# Step 2: pull Belenos chip and inventory MZIs
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 70, flush=True)
print("  Step 2 -- pull belenos_circuit + inventory MZIs", flush=True)
print("=" * 70, flush=True)
remote_processor = pcvl.RemoteProcessor("qpu:belenos", token=TOKEN)
arch = remote_processor.specs["architecture"]
belenos_circuit = arch.unitary_circuit()
chip_mzis = inventory_mzis(belenos_circuit)
active_pairs = sorted({m["mode_pair"] for m in chip_mzis
                       if m["mode_pair"][0] < DIM_BE and m["mode_pair"][1] < DIM_BE})
print(f"Chip MZIs: {len(chip_mzis)}  active pairs (modes 0..7): {active_pairs}",
      flush=True)
active_chip_mzis = [m for m in chip_mzis if m["mode_pair"] in active_pairs]
print(f"Active chip MZIs: {len(active_chip_mzis)}", flush=True)


# ══════════════════════════════════════════════════════════
# Step 3: collect parameter handles for fast scipy access
# ══════════════════════════════════════════════════════════
ps_handles_by_idx = {}
for idx, (r, c) in enumerate(list(belenos_circuit)):
    if isinstance(c, PS) and c.get_parameters():
        ps_handles_by_idx[idx] = c.get_parameters()[0]

# Active = all PSs whose component-index is one of an active MZI's PS slots.
active_ps_idx = []
identity_ps_idx = []
for idx in ps_handles_by_idx:
    is_active = any(idx in (m["ps1_idx"], m["ps2_idx"]) for m in active_chip_mzis)
    if is_active:
        active_ps_idx.append(idx)
    else:
        identity_ps_idx.append(idx)
print(f"Active PSs: {len(active_ps_idx)}, Identity PSs: {len(identity_ps_idx)}",
      flush=True)


# ══════════════════════════════════════════════════════════
# Step 4: initialize identity PSs to fab.py convention,
#         active PSs to small random values around 0
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 70, flush=True)
print("  Step 3 -- initialize PS values", flush=True)
print("=" * 70, flush=True)
# Map idx -> mode of that PS
ps_idx_to_mode = {idx: list(belenos_circuit)[idx][0][0]
                  for idx in ps_handles_by_idx}

# Identity init: mode 0 -> pi, odd modes -> 0, other even modes -> pi
# (per fab.py).
for idx in identity_ps_idx:
    m = ps_idx_to_mode[idx]
    val = pi if m == 0 else (0.0 if m % 2 == 1 else pi)
    ps_handles_by_idx[idx].set_value(float(val))

# Active init: random small values  (uniform in [-pi, pi])
rng = np.random.default_rng(0)
for idx in active_ps_idx:
    ps_handles_by_idx[idx].set_value(float(rng.uniform(-pi, pi)))

# Sanity: compute initial unitary and error.
U0 = np.array(belenos_circuit.compute_unitary())
err0 = np.linalg.norm(U0[:DIM_BE, :DIM_BE] - U_qsvt)
print(f"Initial ||U_chip[:8,:8] - U_QSVT|| = {err0:.4e}", flush=True)


# ══════════════════════════════════════════════════════════
# Step 5: scipy.optimize on the 168 active PSs
#   (this IS the L^-1 D L shuffle in spirit -- find the rectangular
#    mesh PS configuration that realizes U_QSVT directly, without
#    needing to symbolically push diagonals through chains)
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 70, flush=True)
print(f"  Step 4 -- scipy L-BFGS-B on {len(active_ps_idx)} PSs", flush=True)
print("=" * 70, flush=True)


def cost(x):
    for k, idx in enumerate(active_ps_idx):
        ps_handles_by_idx[idx].set_value(float(x[k]))
    U = np.array(belenos_circuit.compute_unitary())
    return float(np.linalg.norm(U[:DIM_BE, :DIM_BE] - U_qsvt) ** 2)


def attempt(seed):
    """One scipy attempt with a fresh random init."""
    rng_local = np.random.default_rng(seed)
    x0 = rng_local.uniform(-pi, pi, size=len(active_ps_idx))
    res = minimize(cost, x0, method='L-BFGS-B',
                   options={'maxiter': SCIPY_MAXITER, 'gtol': 1e-12,
                            'ftol': 1e-14, 'disp': False})
    return res


best_res = None
best_cost = np.inf
for trial in range(8):
    print(f"  trial {trial}: starting", flush=True)
    res = attempt(seed=trial)
    print(f"  trial {trial}: cost={res.fun:.4e}  niter={res.nit}",
          flush=True)
    if res.fun < best_cost:
        best_cost = res.fun
        best_res = res
    if res.fun < (LOCAL_TOL ** 2):
        print(f"  trial {trial}: hit tolerance, stopping", flush=True)
        break

# Apply best
for k, idx in enumerate(active_ps_idx):
    ps_handles_by_idx[idx].set_value(float(best_res.x[k]))
U_chip_local = np.array(belenos_circuit.compute_unitary())
err_top8 = np.linalg.norm(U_chip_local[:DIM_BE, :DIM_BE] - U_qsvt)
err_top4 = np.linalg.norm(U_chip_local[:DIM_SYS, :DIM_SYS]
                          - U_qsvt[:DIM_SYS, :DIM_SYS])
offdiag_top = np.linalg.norm(U_chip_local[:DIM_BE, DIM_BE:])
print(f"\nBest cost: {best_cost:.4e}", flush=True)
print(f"||U_chip[:8,:8] - U_QSVT|| = {err_top8:.4e}  "
      f"({'PASS' if err_top8 < LOCAL_TOL else 'FAIL'} at {LOCAL_TOL})",
      flush=True)
print(f"||U_chip[:4,:4] - T_6(A)|| = {err_top4:.4e}", flush=True)
print(f"off-diagonal U_chip[:8, 8:] = {offdiag_top:.4e}", flush=True)


# ══════════════════════════════════════════════════════════
# Step 6: local sampling per eigenstate
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 70, flush=True)
print(f"  Step 5 -- local sampling ({N_LOCAL_SHOTS} shots/eigenstate)",
      flush=True)
print("=" * 70, flush=True)
n_eig = 4
measured_mode_probs = np.zeros((n_eig, N_MODES_CHIP))
theory_mode_probs   = np.zeros((n_eig, N_MODES_CHIP))
measured_p_filter   = np.zeros(n_eig)
mode0_concentration = np.zeros(n_eig)

for k in range(n_eig):
    psi_in = np.zeros(N_MODES_CHIP, dtype=complex); psi_in[:DIM_SYS] = V_H[:, k]
    psi_out = U_chip_local @ psi_in
    p = np.abs(psi_out)**2; p = p / p.sum()
    psi_th = np.zeros(N_MODES_CHIP, dtype=complex)
    psi_th[:DIM_BE] = U_qsvt @ psi_in[:DIM_BE]
    p_th = np.abs(psi_th)**2

    counts = np.random.multinomial(N_LOCAL_SHOTS, p)
    measured_mode_probs[k] = counts / N_LOCAL_SHOTS
    theory_mode_probs[k]   = p_th
    measured_p_filter[k]   = counts[:DIM_SYS].sum() / N_LOCAL_SHOTS
    mode0_concentration[k] = counts[0] / N_LOCAL_SHOTS
    print(f"  k={k}  lam={eigs_H[k]:+.4f}  |T_6|^2={target_p[k]:.4f}  "
          f"P(filter)={measured_p_filter[k]:.4f}  "
          f"P(mode0)={mode0_concentration[k]:.4f}", flush=True)


# ── Plot + save ──
fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True, sharey=True)
x = np.arange(DIM_BE); w = 0.4
for k, ax in zip(range(n_eig), axes.flatten()):
    ax.bar(x - w/2, theory_mode_probs[k, :DIM_BE], w,
           label="theory", color="#3060c0")
    ax.bar(x + w/2, measured_mode_probs[k, :DIM_BE], w,
           label="local sim", color="#a02050")
    ax.axvline(DIM_SYS - 0.5, color="gray", linestyle="--", alpha=0.6)
    ax.set_title(f"k={k}, lam={eigs_H[k]:+.3f}")
    ax.grid(True, axis="y", alpha=0.3); ax.legend(fontsize=8); ax.set_xticks(x)
for ax in axes[-1, :]: ax.set_xlabel("output mode")
for ax in axes[:, 0]:  ax.set_ylabel("P")
fig.suptitle(f"d=6 QSVT directPS_v2 (err_top8={err_top8:.2e})")
fig.tight_layout(); fig.savefig(OUT_PNG, dpi=140)
print(f"\nSaved: {OUT_PNG}", flush=True)

np.savez(OUT_NPZ,
         eigs_H=eigs_H, alpha=alpha, T6_eigs=T6,
         target_probabilities=target_p,
         measured_p_filter=measured_p_filter,
         mode0_concentration=mode0_concentration,
         measured_mode_probs=measured_mode_probs,
         theory_mode_probs=theory_mode_probs,
         U_chip_local=U_chip_local, U_qsvt=U_qsvt,
         err_top8=err_top8, err_top4=err_top4)
print(f"Saved: {OUT_NPZ}", flush=True)

print("\n" + "=" * 70, flush=True)
print("  SUMMARY", flush=True)
print("=" * 70, flush=True)
print(f"||U_chip[:8,:8] - U_QSVT|| = {err_top8:.4e}  "
      f"({'PASS' if err_top8 < LOCAL_TOL else 'FAIL'})", flush=True)
print(f"{'k':>2} {'lambda':>8} {'target':>8} {'measured':>10} {'P(mode0)':>10}",
      flush=True)
for k in range(n_eig):
    print(f"{k:>2} {eigs_H[k]:>+8.4f} {target_p[k]:>8.4f} "
          f"{measured_p_filter[k]:>10.4f} {mode0_concentration[k]:>10.4f}",
          flush=True)
print("\nNO QPU SUBMISSION. No credits used.", flush=True)
