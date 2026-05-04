# ============================================================
# LAYER-BY-LAYER QSVT ON BELENOS, d=2 -- LOCAL ONLY
#
# Five physical blocks chained on the chip, each in its own
# MZI column range. NO precomputed 8x8 U_QSVT decomposition;
# each block's PSs are fit independently to its OWN target
# unitary. The chip composition realizes T_2(H/alpha).
#
# Convention note:
#   The user's spec wrote   R · U_BE · R · U_BE† · R
#   but with Chebyshev d=2 angles [pi/4, 0, -pi/4] that gives
#   identity (since U_BE · U_BE† = I and R(0)=I). The version
#   that yields T_2(A) at the top-left is the all-U_BE form
#       U_QSVT = R(pi/4) · U_BE · R(0) · U_BE · R(-pi/4)
#   so we apply U_BE TWICE (in two separate column ranges of
#   the chip), no dagger. Each application is a distinct
#   physical block of MZIs realising the same 8x8 matrix.
#
# Belenos topology (Clements rectangular):
#   24 layers alternating odd-pair (3 MZIs in 8-mode region)
#   and even-pair (4 MZIs). Layer 0 is odd. Within modes 0..7
#   we have 84 active MZIs * 2 PSs = 168 PSs across 24 layers.
#
# Column allocation (24 layers total):
#   layer 0       -> R(pi/4)        (block 1, 3 MZIs, 6 PSs)
#   layers 1..8   -> U_BE           (block 2, 28 MZIs, 56 PSs)
#   layer 9       -> R(0)=I         (block 3, 4 MZIs, 8 PSs identity)
#   layers 10..17 -> U_BE           (block 4, 28 MZIs, 56 PSs)
#   layer 18      -> R(-pi/4)       (block 5, 3 MZIs, 6 PSs)
#   layers 19..23 -> identity       (5 layers, 18 MZIs, 36 PSs identity)
#
# Per-block scipy fits each block's PSs independently to its
# own 8x8 target. When composed (right-applied-first):
#   U_chip[:8,:8] = T_blk5 @ T_blk4 @ T_blk3 @ T_blk2 @ T_blk1
#                 = R(-pi/4) @ U_BE @ I @ U_BE @ R(pi/4)
#                 = U_QSVT (T_2 filter).
#
# NO QPU SUBMISSION.
# ============================================================

import os, re, sys, time
from math import pi
from collections import defaultdict, Counter
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

import perceval as pcvl
from perceval.components import BS, PS

from qsvt_angles import chebyshev_qsp_angles
from qsvt_tfim_matrix_reference import (
    build_tfim_hamiltonian, build_block_encoding,
    build_qsvt_unitary, projector_phase, DIM_SYS, DIM_BE,
)

H_FIELD       = 1.0
QSVT_DEGREE   = 2
N_MODES_CHIP  = 24
LOCAL_TOL     = 1e-3
SCIPY_MAXITER = 600
SCIPY_TRIALS  = 6
N_LOCAL_SHOTS = 5000
TOKEN = "_T_eyJhbGciOiJIUzUxMiIsInR5cCI6IkpXVCJ9.eyJpZCI6MjU5MiwiZXhwIjoxODA2NTkwNjc1LjE4NjQ3NDN9.eCJCJNcF7BqyUtKiUSaOPK1RM0Nd42B-2UXwH0IIRiYy0TOqi9eQWcUM3049KIkcjgyKkHT-xyIxzoqzmuxyaQ"
OUT_PNG = os.path.join(os.path.dirname(__file__),
                       "results_layerwise_d2_local.png")
OUT_NPZ = os.path.join(os.path.dirname(__file__),
                       "results_layerwise_d2_local.npz")
np.random.seed(42); pcvl.random_seed(42)


# ── Parse helpers ─────────────────────────────────────────
def parse_theta(d):
    m = re.search(r"theta=([\-\+]?[\d\.eE\+\-]+)", d)
    return float(m.group(1)) if m else None
def parse_phi(d):
    m = re.search(r"phi\s*=\s*([\-\+]?[\d\.eE\+\-]+|pi)", d)
    if not m: return None
    v = m.group(1); return pi if v == "pi" else float(v)


def inventory_mzis_with_layers(circuit):
    """
    Walk circuit components and group each MZI = 4 consecutive components
    BS-PS-PS-BS on a fixed pair. Assign layer index based on parity flips.
    Returns list of dicts.
    """
    comps = list(circuit); mzis = []; i = 0
    while i < len(comps):
        r, c = comps[i]
        if not isinstance(c, BS) or i + 3 >= len(comps):
            i += 1; continue
        r1, c1 = comps[i+1]; r2, c2 = comps[i+2]; r3, c3 = comps[i+3]
        if (isinstance(c1, PS) and isinstance(c2, PS) and isinstance(c3, BS)
                and r3 == r and len(r1) == 1 and len(r2) == 1
                and set(r1) | set(r2) == set(r)):
            mzis.append({
                "mode_pair": r, "bs1_idx": i,
                "ps1_idx": i+1, "ps1_mode": r1[0],
                "ps2_idx": i+2, "ps2_mode": r2[0], "bs2_idx": i+3,
                "bs1_theta": parse_theta(c.describe()),
                "bs2_theta": parse_theta(c3.describe()),
            })
            i += 4
        else:
            i += 1

    # Assign layer indices by tracking pair parity. Layer 0 = odd pairs
    # (lower mode is odd). Layer increments when we see a pair that isn't
    # consistent with the current layer's parity, OR when the same mode
    # pair appears twice (signalling a new layer of the chip).
    # Simpler: chip has 11 odd-pair MZIs in layer 0, 12 even in layer 1, ...
    # so chunk by [11, 12, 11, 12, ...].
    chunk_pattern = []
    for k in range(N_MODES_CHIP):
        chunk_pattern.append(11 if k % 2 == 0 else 12)
    layer = 0; idx_in_layer = 0
    for j, m in enumerate(mzis):
        if idx_in_layer >= chunk_pattern[layer]:
            layer += 1; idx_in_layer = 0
        m["layer"] = layer
        idx_in_layer += 1
    return mzis


# ══════════════════════════════════════════════════════════
# Step 1: build all targets
# ══════════════════════════════════════════════════════════
print("=" * 70, flush=True)
print("  Step 1 -- build U_BE, R(phi), and analytical U_QSVT (d=2)", flush=True)
print("=" * 70, flush=True)
H = build_tfim_hamiltonian(H_FIELD)
eigs_H, V_H = np.linalg.eigh(H)
U_BE, alpha, A = build_block_encoding(H)        # 8x8
angles = chebyshev_qsp_angles(QSVT_DEGREE)      # [pi/4, 0, -pi/4]
T2 = np.cos(QSVT_DEGREE * np.arccos(np.clip(eigs_H / alpha, -1, 1)))
target_p = T2 ** 2

R_phi0 = projector_phase(angles[0])             # R(pi/4)
R_phi1 = projector_phase(angles[1])             # R(0) = I_8
R_phi2 = projector_phase(angles[2])             # R(-pi/4)

# Analytical U_QSVT in the convention of build_qsvt_unitary:
#   U_QSVT = R(phi_0) @ U_BE @ R(phi_1) @ U_BE @ R(phi_2)
# (matrix product; right-most factor R(phi_2) is applied FIRST to input).
U_qsvt = R_phi0 @ U_BE @ R_phi1 @ U_BE @ R_phi2
err_recon = np.linalg.norm(U_qsvt - build_qsvt_unitary(U_BE, angles))
print(f"alpha={alpha:.4f}, T_2(eigs/alpha)={np.round(T2,4)}", flush=True)
print(f"|T_2|^2 (target probabilities)={np.round(target_p,4)}", flush=True)
print(f"||U_qsvt(layerwise) - U_qsvt(reference)|| = {err_recon:.2e}", flush=True)
assert err_recon < 1e-12, "layer-by-layer composition mismatch -- bug"

# Block targets in chronological (chip-input -> chip-output) order.
# The chip applies blocks left-to-right; matrix product accumulates
# right-to-left, so:
#     U_chip = T_block5 @ T_block4 @ T_block3 @ T_block2 @ T_block1
# Block 1 = first applied (chip layer 0)  = right-most factor in U_QSVT
#                                           = R(phi_d) = R(-pi/4)
# Block 5 = last applied (chip layer 18)  = left-most factor in U_QSVT
#                                           = R(phi_0) = R(+pi/4)
T_block = {
    1: R_phi2,        # R(-pi/4)       chip layer 0       (first applied)
    2: U_BE,          # U_BE           chip layers 1..8
    3: R_phi1,        # I              chip layer 9
    4: U_BE,          # U_BE           chip layers 10..17
    5: R_phi0,        # R(+pi/4)       chip layer 18      (last applied)
}
# Allocation: 2 + 9 + 2 + 9 + 2 = 24 layers, no identity tail.
#   block 1 R(phi_2): layers 0..1   (odd + even = 7 MZIs, 14 PSs)
#   block 2 U_BE:     layers 2..10  (5 odd + 4 even = 31 MZIs, 62 PSs)
#   block 3 R(phi_1): layers 11..12 (even + odd = 7 MZIs, 14 PSs)
#   block 4 U_BE:     layers 13..21 (4 odd + 5 even = 32 MZIs, 64 PSs)
#   block 5 R(phi_0): layers 22..23 (odd + even = 7 MZIs, 14 PSs)
LAYERS = {1: [0, 1],
          2: list(range(2, 11)),
          3: [11, 12],
          4: list(range(13, 22)),
          5: [22, 23]}
IDENTITY_LAYERS = []


# ══════════════════════════════════════════════════════════
# Step 2: pull chip + inventory
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 70, flush=True)
print("  Step 2 -- pull belenos_circuit + inventory MZIs (with layer index)",
      flush=True)
print("=" * 70, flush=True)
remote_processor = pcvl.RemoteProcessor("qpu:belenos", token=TOKEN)
arch = remote_processor.specs["architecture"]
belenos_circuit = arch.unitary_circuit()
chip_mzis = inventory_mzis_with_layers(belenos_circuit)
print(f"Total MZIs inventoried: {len(chip_mzis)}", flush=True)
layer_count = Counter(m["layer"] for m in chip_mzis)
print(f"Layer histogram: {sorted(layer_count.items())}", flush=True)

active_pairs = {(p, p+1) for p in range(DIM_BE - 1)}   # (0,1)..(6,7)
active_chip_mzis = [m for m in chip_mzis if m["mode_pair"] in active_pairs]
print(f"Active MZIs (modes 0..7): {len(active_chip_mzis)}", flush=True)


def active_in_layers(layer_list):
    """Return the active chip MZIs within the given layer indices."""
    return [m for m in active_chip_mzis if m["layer"] in layer_list]


# Depth budget check
for blk_id, lays in LAYERS.items():
    am = active_in_layers(lays)
    print(f"  block {blk_id} (layers {lays}): {len(am)} active MZIs, "
          f"{2 * len(am)} PSs, target {T_block[blk_id].shape}", flush=True)
total_layers_used = sum(len(v) for v in LAYERS.values()) + len(IDENTITY_LAYERS)
if total_layers_used > N_MODES_CHIP:
    print(f"DEPTH OVERFLOW: {total_layers_used} > {N_MODES_CHIP} layers.",
          flush=True); sys.exit(1)
else:
    print(f"Depth used: {total_layers_used}/{N_MODES_CHIP} -- fits.", flush=True)


# ══════════════════════════════════════════════════════════
# Step 3: PS handle accessor + identity initialiser
# ══════════════════════════════════════════════════════════
ps_handles = {}
for idx, (r, c) in enumerate(list(belenos_circuit)):
    if isinstance(c, PS) and c.get_parameters():
        ps_handles[idx] = c.get_parameters()[0]
ps_idx_to_mode = {idx: list(belenos_circuit)[idx][0][0] for idx in ps_handles}


def set_identity_for_idxs(idxs):
    for idx in idxs:
        m = ps_idx_to_mode[idx]
        val = pi if m == 0 else (0.0 if m % 2 == 1 else pi)
        ps_handles[idx].set_value(float(val))


# Initialise: start everything at identity convention.
set_identity_for_idxs(list(ps_handles.keys()))


# ══════════════════════════════════════════════════════════
# Step 4: per-block scipy fitting
# Each block's "target" in isolation -- we set ALL chip PSs except
# this block's PSs to identity, then fit this block's PSs so that
# the chip's full 8x8 top-left equals the block's target.
# Because identity blocks compose as identity on the active 8 modes,
# the fitted block in isolation realises EXACTLY its target.
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 70, flush=True)
print("  Step 3 -- per-block scipy fits", flush=True)
print("=" * 70, flush=True)

block_results = {}   # block_id -> dict(ps_values, err, niter, x)
for blk_id in [1, 2, 3, 4, 5]:
    if blk_id == 3:                           # R(0) = I, trivial
        print(f"\n  Block {blk_id}: target = I (trivial, set identity).",
              flush=True)
        active_blk_mzis = active_in_layers(LAYERS[blk_id])
        ps_idxs = []
        for m in active_blk_mzis:
            ps_idxs.append(m["ps1_idx"]); ps_idxs.append(m["ps2_idx"])
        set_identity_for_idxs(ps_idxs)
        block_results[blk_id] = {"ps_values": {i: float(ps_handles[i]._value)
                                              if hasattr(ps_handles[i], '_value')
                                              else 0.0
                                              for i in ps_idxs},
                                 "err": 0.0, "niter": 0}
        continue

    target = T_block[blk_id]
    active_blk_mzis = active_in_layers(LAYERS[blk_id])
    ps_idxs = []
    for m in active_blk_mzis:
        ps_idxs.append(m["ps1_idx"]); ps_idxs.append(m["ps2_idx"])
    print(f"\n  Block {blk_id}: target shape={target.shape}, "
          f"#active MZIs={len(active_blk_mzis)}, #PSs={len(ps_idxs)}",
          flush=True)

    # Set all OTHER PSs in the chip to identity
    set_identity_for_idxs([i for i in ps_handles if i not in ps_idxs])

    def cost(x):
        for k, idx in enumerate(ps_idxs):
            ps_handles[idx].set_value(float(x[k]))
        U = np.array(belenos_circuit.compute_unitary())
        return float(np.linalg.norm(U[:DIM_BE, :DIM_BE] - target) ** 2)

    best_x = None; best_cost = np.inf; best_niter = 0
    for trial in range(SCIPY_TRIALS):
        rng_t = np.random.default_rng(100 * blk_id + trial)
        x0 = rng_t.uniform(-pi, pi, size=len(ps_idxs))
        res = minimize(cost, x0, method='L-BFGS-B',
                       options={'maxiter': SCIPY_MAXITER,
                                'ftol': 1e-14, 'gtol': 1e-12})
        print(f"    trial {trial}: cost={res.fun:.4e}  niter={res.nit}",
              flush=True)
        if res.fun < best_cost:
            best_cost = res.fun; best_x = res.x.copy(); best_niter = res.nit
        if res.fun < (LOCAL_TOL ** 2):
            break

    # Apply best
    for k, idx in enumerate(ps_idxs):
        ps_handles[idx].set_value(float(best_x[k]))
    block_results[blk_id] = {
        "ps_values": {idx: float(best_x[k]) for k, idx in enumerate(ps_idxs)},
        "err": np.sqrt(best_cost), "niter": best_niter,
    }
    print(f"    BEST: ||block_unitary - target|| = "
          f"{block_results[blk_id]['err']:.4e}", flush=True)
    if block_results[blk_id]["err"] >= LOCAL_TOL:
        print(f"    Block {blk_id} did NOT converge below {LOCAL_TOL}; "
              f"final composition will likely also miss the threshold.",
              flush=True)


# ══════════════════════════════════════════════════════════
# Step 5: assemble all blocks + identity tail, verify composition
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 70, flush=True)
print("  Step 4 -- assemble all blocks + identity tail in chip", flush=True)
print("=" * 70, flush=True)
# Identity tail (layers 19..23) and identity for everything outside active
# region: set identity convention.
set_identity_for_idxs(list(ps_handles.keys()))
# Now apply each block's fitted PS values:
for blk_id, info in block_results.items():
    for idx, val in info["ps_values"].items():
        ps_handles[idx].set_value(float(val))

U_chip_local = np.array(belenos_circuit.compute_unitary())
err_top8 = np.linalg.norm(U_chip_local[:DIM_BE, :DIM_BE] - U_qsvt)
err_top4 = np.linalg.norm(U_chip_local[:DIM_SYS, :DIM_SYS]
                          - U_qsvt[:DIM_SYS, :DIM_SYS])
offdiag = np.linalg.norm(U_chip_local[:DIM_BE, DIM_BE:])
print(f"||U_chip[:8,:8] - U_QSVT|| = {err_top8:.4e}  "
      f"({'PASS' if err_top8 < LOCAL_TOL else 'FAIL'} at {LOCAL_TOL})",
      flush=True)
print(f"||U_chip[:4,:4] - T_2(A)||  = {err_top4:.4e}", flush=True)
print(f"||U_chip[:8, 8:]||          = {offdiag:.4e}", flush=True)


# ══════════════════════════════════════════════════════════
# Step 6: per-eigenstate local sampling
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 70, flush=True)
print(f"  Step 5 -- local sampling ({N_LOCAL_SHOTS} shots/eigenstate)",
      flush=True)
print("=" * 70, flush=True)
n_eig = 4
measured_mode_probs = np.zeros((n_eig, N_MODES_CHIP))
theory_mode_probs   = np.zeros((n_eig, N_MODES_CHIP))
measured_p_filter   = np.zeros(n_eig)
theory_p_filter     = np.zeros(n_eig)
for k in range(n_eig):
    psi_in = np.zeros(N_MODES_CHIP, dtype=complex); psi_in[:DIM_SYS] = V_H[:, k]
    psi_chip = U_chip_local @ psi_in
    p_chip = np.abs(psi_chip) ** 2
    p_chip = p_chip / p_chip.sum()

    psi_th = np.zeros(N_MODES_CHIP, dtype=complex)
    psi_th[:DIM_BE] = U_qsvt @ psi_in[:DIM_BE]
    p_th = np.abs(psi_th) ** 2

    counts = np.random.multinomial(N_LOCAL_SHOTS, p_chip)
    measured_mode_probs[k] = counts / N_LOCAL_SHOTS
    theory_mode_probs[k]   = p_th
    measured_p_filter[k]   = counts[:DIM_SYS].sum() / N_LOCAL_SHOTS
    theory_p_filter[k]     = p_th[:DIM_SYS].sum()
    print(f"  k={k}  lam={eigs_H[k]:+.4f}  |T_2|^2={target_p[k]:.4f}  "
          f"theory P(filter)={theory_p_filter[k]:.4f}  "
          f"measured P(filter)={measured_p_filter[k]:.4f}", flush=True)


# ══════════════════════════════════════════════════════════
# Step 7: plotting -- 4-panel figure
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 70, flush=True)
print("  Step 6 -- plot + save", flush=True)
print("=" * 70, flush=True)
fig = plt.figure(figsize=(13, 10))
gs = fig.add_gridspec(3, 2, hspace=0.45, wspace=0.30)

# (a) bar: theory vs measured filter fidelity per eigenvalue
ax_a = fig.add_subplot(gs[0, 0])
x = np.arange(n_eig); w = 0.35
sigma = 1.0 / np.sqrt(N_LOCAL_SHOTS)   # ~1.4% for 5000 shots
err_bars = sigma * np.sqrt(measured_p_filter * (1 - measured_p_filter))
ax_a.bar(x - w/2, target_p, w, label="theory $|T_2|^2$", color="#3060c0")
ax_a.bar(x + w/2, measured_p_filter, w, yerr=err_bars,
         label="layer-by-layer sim", color="#a02050",
         capsize=3, ecolor="black")
ax_a.set_xticks(x)
ax_a.set_xticklabels([f"k={k}\n$\\lambda$={eigs_H[k]:+.3f}" for k in range(n_eig)])
ax_a.set_ylabel("P(filter success)")
ax_a.set_title("(a) Filter fidelity per eigenvalue")
ax_a.set_ylim(0, 1.1); ax_a.grid(True, axis='y', alpha=0.3)
ax_a.legend(fontsize=8)

# (b) scatter theory vs measured, y=x line
ax_b = fig.add_subplot(gs[0, 1])
ax_b.errorbar(target_p, measured_p_filter, yerr=err_bars,
              fmt='o', color="#a02050", ecolor="black", capsize=3,
              markersize=8)
ax_b.plot([0, 1], [0, 1], color="gray", linestyle="--", alpha=0.6, label="y=x")
for k in range(n_eig):
    ax_b.annotate(f"k={k}", (target_p[k], measured_p_filter[k]),
                  textcoords="offset points", xytext=(7, 5), fontsize=8)
ax_b.set_xlabel("theory $|T_2(\\lambda/\\alpha)|^2$")
ax_b.set_ylabel("measured P(filter)")
ax_b.set_title("(b) Theory vs measured")
ax_b.set_xlim(-0.05, 1.1); ax_b.set_ylim(-0.05, 1.1); ax_b.grid(True, alpha=0.3)
ax_b.legend(fontsize=8)

# (c) per-mode distribution 4-panel inside one big sub-grid
sub_gs = gs[1, :].subgridspec(1, 4, wspace=0.25)
for k in range(n_eig):
    ax_c = fig.add_subplot(sub_gs[0, k])
    xx = np.arange(DIM_BE); ww = 0.4
    ax_c.bar(xx - ww/2, theory_mode_probs[k, :DIM_BE], ww,
             label="theory", color="#3060c0")
    ax_c.bar(xx + ww/2, measured_mode_probs[k, :DIM_BE], ww,
             label="sim", color="#a02050")
    ax_c.axvline(DIM_SYS - 0.5, color="gray", linestyle="--", alpha=0.6)
    ax_c.set_title(f"k={k}, $\\lambda$={eigs_H[k]:+.3f}", fontsize=9)
    ax_c.set_xticks(xx); ax_c.tick_params(labelsize=7)
    ax_c.grid(True, axis='y', alpha=0.3)
    if k == 0:
        ax_c.set_ylabel("P(photon in mode)")
        ax_c.legend(fontsize=7)
    ax_c.set_xlabel("output mode", fontsize=8)

# (d) filter response curve T_2(x/alpha)^2 on continuous x in [-alpha, alpha]
ax_d = fig.add_subplot(gs[2, :])
xs_cont = np.linspace(-alpha, alpha, 401)
ys_cont = np.cos(QSVT_DEGREE * np.arccos(np.clip(xs_cont / alpha, -1, 1))) ** 2
ax_d.plot(xs_cont, ys_cont, color="#3060c0", lw=2,
          label=r"theory $|T_2(\lambda/\alpha)|^2$")
ax_d.errorbar(eigs_H, measured_p_filter, yerr=err_bars,
              fmt='o', color="#a02050", ecolor="black", capsize=4,
              markersize=9, label="layer-by-layer sim")
for k in range(n_eig):
    ax_d.annotate(f"k={k}", (eigs_H[k], measured_p_filter[k]),
                  textcoords="offset points", xytext=(7, 7), fontsize=9)
ax_d.set_xlabel(r"eigenvalue $\lambda$ of $H$")
ax_d.set_ylabel("P(filter success)")
ax_d.set_title(rf"(d) Filter response curve, $\alpha={alpha:.4f}$, d={QSVT_DEGREE}")
ax_d.grid(True, alpha=0.3); ax_d.legend()

fig.suptitle(f"Layer-by-layer QSVT on Belenos, d={QSVT_DEGREE}  "
             rf"(||U_chip[:8,:8] - U_QSVT||={err_top8:.2e})",
             fontsize=12)
fig.savefig(OUT_PNG, dpi=140, bbox_inches="tight")
print(f"Saved: {OUT_PNG}", flush=True)

np.savez(
    OUT_NPZ,
    eigs_H=eigs_H, alpha=alpha,
    angles=angles, T2=T2, target_probabilities=target_p,
    measured_p_filter=measured_p_filter, theory_p_filter=theory_p_filter,
    measured_mode_probs=measured_mode_probs,
    theory_mode_probs=theory_mode_probs,
    U_chip_local=U_chip_local, U_qsvt=U_qsvt,
    err_top8=err_top8, err_top4=err_top4,
    block_errors=np.array([block_results[k]["err"] for k in [1, 2, 3, 4, 5]]),
)
print(f"Saved: {OUT_NPZ}", flush=True)


# ══════════════════════════════════════════════════════════
# Final summary table
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 70, flush=True)
print("  SUMMARY", flush=True)
print("=" * 70, flush=True)
print(f"||U_chip[:8,:8] - U_QSVT||_F = {err_top8:.4e} "
      f"({'PASS' if err_top8 < LOCAL_TOL else 'FAIL'} at {LOCAL_TOL})",
      flush=True)
print(f"\nPer-block fit errors:", flush=True)
for blk_id in [1, 2, 3, 4, 5]:
    print(f"  block {blk_id}: ||U_block - target||_F = "
          f"{block_results[blk_id]['err']:.4e}", flush=True)
print(f"\nFilter fidelity table:", flush=True)
print(f"{'k':>2} {'lambda':>10} {'|T_2|^2':>10} "
      f"{'measured P(filter)':>20} {'difference':>14}",
      flush=True)
for k in range(n_eig):
    print(f"{k:>2} {eigs_H[k]:>+10.4f} {target_p[k]:>10.4f} "
          f"{measured_p_filter[k]:>20.4f} "
          f"{measured_p_filter[k] - target_p[k]:>+14.4f}", flush=True)
print("\nNO QPU SUBMISSION. No credits used.", flush=True)
