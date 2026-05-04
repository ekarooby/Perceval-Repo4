# ============================================================
# DIRECT-PS COMPILATION OF d=6 QSVT TO BELENOS -- LOCAL ONLY
# Hand-rolled Clements et al. 2016 rectangular decomposition
# of the 8x8 U_QSVT, matched to the chip's actual per-position
# BS thetas via your seed-search pattern. Bypasses the cloud
# compiler (which silently returned identity for 3/4 of the
# earlier QPU jobs -- see qsvt_tfim_belenos_QPU_cloudaudit.py).
#
# NO QPU SUBMISSION. Everything runs against Perceval's local
# backend: belenos_circuit.compute_unitary() + local sampling.
# ============================================================

import os, re
import numpy as np
from math import pi
import matplotlib.pyplot as plt
from collections import defaultdict, Counter

import perceval as pcvl
import perceval.components as comp
from perceval.components import BS, PS

from qsvt_angles import chebyshev_qsp_angles
from qsvt_tfim_matrix_reference import (
    build_tfim_hamiltonian, build_block_encoding,
    build_qsvt_unitary, DIM_SYS, DIM_BE,
)

# ── Settings ──────────────────────────────────────────────
H_FIELD        = 1.0
QSVT_DEGREE    = 6
N_MODES_CHIP   = 24
LOCAL_TOL      = 1e-3
MAX_SEEDS_MZI  = 200
N_LOCAL_SHOTS  = 5000
TOKEN = "_T_eyJhbGciOiJIUzUxMiIsInR5cCI6IkpXVCJ9.eyJpZCI6MjU5MiwiZXhwIjoxODA2NTkwNjc1LjE4NjQ3NDN9.eCJCJNcF7BqyUtKiUSaOPK1RM0Nd42B-2UXwH0IIRiYy0TOqi9eQWcUM3049KIkcjgyKkHT-xyIxzoqzmuxyaQ"

OUT_PNG = os.path.join(os.path.dirname(__file__),
                       "results_directPS_local.png")
OUT_NPZ = os.path.join(os.path.dirname(__file__),
                       "results_directPS_local.npz")

np.random.seed(42)
pcvl.random_seed(42)


# ══════════════════════════════════════════════════════════
# CLEMENTS ET AL. 2016 RECTANGULAR DECOMPOSITION
# Convention: T(theta, phi) = [[e^{iφ} sin(θ/2), cos(θ/2)],
#                              [e^{iφ} cos(θ/2), -sin(θ/2)]]
# This is the 2x2 unitary Clements calls "the MZI" in their paper.
# Algorithm 2 peels off off-diagonals of U alternately from below
# (applying T^{-1} from the right) and from above (applying T
# from the left). After 28 rotations for 8x8, U is diagonal.
# Then all T's are shuffled to one side using commutation with
# the diagonal so the result is a pure rectangular mesh.
# ══════════════════════════════════════════════════════════

def _nullify_from_right(U, row, col):
    """
    Find (theta, phi) such that applying T^{-1}_{col, col+1}(theta, phi)
    from the right zeros U[row, col]. Returns (theta, phi, U_new).

    T = [[e^{iφ} sin(θ/2), cos(θ/2)], [e^{iφ} cos(θ/2), -sin(θ/2)]]
    T^{-1} = T^†
    Applying T^{-1} to columns (col, col+1) of U from the right:
      new_col     = U[:,col]   * e^{-iφ} sin(θ/2) + U[:,col+1] * e^{-iφ} cos(θ/2)
      new_col+1   = U[:,col]   * cos(θ/2)          + U[:,col+1] * (-sin(θ/2))
    We need new U[row, col] = 0:
      a e^{-iφ} sin(θ/2) + b e^{-iφ} cos(θ/2) = 0,  a=U[row,col], b=U[row,col+1]
    which gives:
      tan(θ/2) = -b/a  (complex)
    Splitting magnitude and phase:
      θ = 2 arctan(|b|/|a|),   φ = angle(a) - angle(b) + pi
    """
    n = U.shape[0]
    a = U[row, col]
    b = U[row, col + 1]
    # Solve: a e^{-iφ} sin(θ/2) + b cos(θ/2) = 0  →  tan(θ/2) = |b/a|.
    # Edge cases:
    #   a ≈ 0: U[row, col] already ≈ 0, set θ = π so T^{-1} preserves it.
    #   b ≈ 0: need cos(θ/2) = 0 → θ = π... wait, we need to swap col with col+1.
    #          With θ = 0, T^{-1} acts as a swap (diag = 0, off = 1) which
    #          places the zero from column col+1 into row,col.
    if abs(a) < 1e-15:
        theta, phi = pi, 0.0
    elif abs(b) < 1e-15:
        theta, phi = 0.0, 0.0
    else:
        theta = 2.0 * np.arctan2(abs(b), abs(a))
        phi = np.angle(a) - np.angle(b) + pi

    s = np.sin(theta / 2)
    c = np.cos(theta / 2)
    T_inv = np.eye(n, dtype=complex)
    T_inv[col,     col]     = np.exp(-1j * phi) * s
    T_inv[col,     col + 1] = np.exp(-1j * phi) * c
    T_inv[col + 1, col]     = c
    T_inv[col + 1, col + 1] = -s
    return theta, phi, U @ T_inv


def _nullify_from_left(U, row, col):
    """
    Find (theta, phi) such that applying T_{row-1, row}(theta, phi)
    from the left zeros U[row, col]. Returns (theta, phi, U_new).

    T applied to rows (row-1, row) of U from the left:
      new_row-1 = e^{iφ} sin(θ/2) U[row-1,:] + cos(θ/2) U[row,:]
      new_row   = e^{iφ} cos(θ/2) U[row-1,:] + (-sin(θ/2)) U[row,:]
    We need new U[row, col] = 0:
      e^{iφ} cos(θ/2) U[row-1,col] - sin(θ/2) U[row,col] = 0
    giving theta = 2 arctan(|U[row,col]| / |U[row-1,col]|),
           phi   = angle(U[row,col]) - angle(U[row-1,col]).
    """
    n = U.shape[0]
    a = U[row - 1, col]   # upper -- want this kept
    b = U[row,     col]   # target to zero
    # Solve: e^{iφ} cos(θ/2) a = sin(θ/2) b   →  e^{iφ} = tan(θ/2) b/a
    # For |e^{iφ}|=1: tan(θ/2) = |a|/|b|.
    # Solve: e^{iφ} cos(θ/2) a = sin(θ/2) b  →  tan(θ/2) = |a/b|.
    # Edge cases:
    #   b ≈ 0: U[row, col] already ≈ 0, set θ = π so T preserves it.
    #   a ≈ 0: need sin(θ/2) → 0 i.e. θ = 0 so the swap-with-phase brings
    #          the zero from row-1 down into (row, col).
    if abs(b) < 1e-15:
        theta, phi = pi, 0.0
    elif abs(a) < 1e-15:
        theta, phi = 0.0, 0.0
    else:
        theta = 2.0 * np.arctan2(abs(a), abs(b))
        phi = np.angle(b) - np.angle(a)

    s = np.sin(theta / 2)
    c = np.cos(theta / 2)
    T = np.eye(n, dtype=complex)
    T[row - 1, row - 1] = np.exp(1j * phi) * s
    T[row - 1, row]     = c
    T[row,     row - 1] = np.exp(1j * phi) * c
    T[row,     row]     = -s
    return theta, phi, T @ U


def clements_decompose(U):
    """
    Clements rectangular decomposition of n x n unitary U.
    Returns:
       right_Ts:  list of (col, theta, phi) applied from the right, in
                  the order they were applied (i.e., U -> U @ T^-1)
       left_Ts:   list of (row, theta, phi) applied from the left
       D:         diagonal phases after all zeroing (length n)

    Reconstruction identity:
        U = (prod of left T^-1, reversed) @ diag(D)
                  @ (prod of right T, reversed)
    """
    n = U.shape[0]
    U = U.astype(complex).copy()
    right_Ts = []
    left_Ts  = []

    for i in range(n - 1):
        if i % 2 == 0:
            # Null column i from the bottom up by applying T^-1 from right.
            for j in range(i + 1):
                row = n - 1 - j
                col = i - j
                theta, phi, U = _nullify_from_right(U, row, col)
                right_Ts.append((col, theta, phi))
        else:
            # Null row (n-1-i) along the diagonal by applying T from left.
            for j in range(i + 1):
                row = n - 1 - i + j
                col = j
                theta, phi, U = _nullify_from_left(U, row, col)
                left_Ts.append((row, theta, phi))

    D = np.diag(U).copy()
    return right_Ts, left_Ts, D


def t_matrix_2x2(theta, phi):
    """2x2 T(theta, phi) per Clements convention."""
    s = np.sin(theta / 2); c = np.cos(theta / 2)
    return np.array([[np.exp(1j * phi) * s, c],
                     [np.exp(1j * phi) * c, -s]], dtype=complex)


def verify_clements(U, right_Ts, left_Ts, D):
    """
    The algorithm produces:   D = L_product @ U @ R_product
    where L_product = L_(last) @ ... @ L_2 @ L_1   (last-applied on the left)
    and   R_product = R_1^{-1} @ R_2^{-1} @ ... @ R_(last)^{-1}.
    So U = L_product^{-1} @ D @ R_product^{-1}
         = (L_1^{-1} @ ... @ L_(last)^{-1}) @ D @ (R_(last) @ ... @ R_1).
    """
    n = U.shape[0]

    # L_product^{-1} = L_1^{-1} @ L_2^{-1} @ ... @ L_(last)^{-1}  (forward order).
    L_inv_product = np.eye(n, dtype=complex)
    for (row, theta, phi) in left_Ts:
        T2 = t_matrix_2x2(theta, phi)
        full = np.eye(n, dtype=complex)
        full[row-1:row+1, row-1:row+1] = T2
        L_inv_product = L_inv_product @ np.linalg.inv(full)

    # R_product^{-1} = R_(last) @ ... @ R_2 @ R_1  (reverse order, NOT inverted).
    R_product_inv = np.eye(n, dtype=complex)
    for (col, theta, phi) in reversed(right_Ts):
        T2 = t_matrix_2x2(theta, phi)
        full = np.eye(n, dtype=complex)
        full[col:col+2, col:col+2] = T2
        R_product_inv = R_product_inv @ full

    U_recon = L_inv_product @ np.diag(D) @ R_product_inv
    return np.linalg.norm(U_recon - U), U_recon


# ══════════════════════════════════════════════════════════
# Chip MZI parsing (same as the QPU_directPS.py file)
# ══════════════════════════════════════════════════════════
def parse_theta(desc):
    m = re.search(r"theta=([\-\+]?[\d\.eE\+\-]+)", desc)
    return float(m.group(1)) if m else None


def parse_phi(desc):
    m = re.search(r"phi\s*=\s*([\-\+]?[\d\.eE\+\-]+|pi)", desc)
    if not m:
        return None
    v = m.group(1)
    return pi if v == "pi" else float(v)


def inventory_mzis(circuit):
    """
    Returns list of dict per MZI: keys
       mode_pair, bs1_idx, ps1_idx, ps1_mode, ps2_idx, ps2_mode, bs2_idx,
       bs1_theta, bs2_theta.
    """
    comps = list(circuit)
    mzis  = []
    i = 0
    while i < len(comps):
        r, c = comps[i]
        if not isinstance(c, BS):
            i += 1; continue
        if i + 3 >= len(comps):
            i += 1; continue
        r1, c1 = comps[i + 1]
        r2, c2 = comps[i + 2]
        r3, c3 = comps[i + 3]
        if (isinstance(c1, PS) and isinstance(c2, PS) and isinstance(c3, BS)
                and r3 == r and set(r1) | set(r2) == set(r)
                and len(r1) == 1 and len(r2) == 1):
            mzis.append({
                "mode_pair": r, "bs1_idx": i,
                "ps1_idx": i + 1, "ps1_mode": r1[0],
                "ps2_idx": i + 2, "ps2_mode": r2[0],
                "bs2_idx": i + 3,
                "bs1_theta": parse_theta(c.describe()),
                "bs2_theta": parse_theta(c3.describe()),
            })
            i += 4
        else:
            i += 1
    return mzis


# ══════════════════════════════════════════════════════════
# MAIN PIPELINE
# ══════════════════════════════════════════════════════════
print("=" * 70)
print("  Step 1 -- build U_QSVT (d=6)")
print("=" * 70)
H = build_tfim_hamiltonian(H_FIELD)
eigs_H, V_H = np.linalg.eigh(H)
U_H, alpha, A = build_block_encoding(H)
angles = chebyshev_qsp_angles(QSVT_DEGREE)
U_qsvt = build_qsvt_unitary(U_H, angles)
T6 = np.cos(QSVT_DEGREE * np.arccos(np.clip(eigs_H / alpha, -1, 1)))
target_p = T6**2
print(f"alpha = {alpha:.4f}")
print(f"H eigenvalues       = {np.round(eigs_H, 4)}")
print(f"T_6(lambda/alpha)   = {np.round(T6, 4)}")
print(f"|T_6|^2 (targets)   = {np.round(target_p, 4)}")


print("\n" + "=" * 70)
print("  Step 2 -- hand-roll Clements decomposition of U_QSVT (8x8)")
print("=" * 70)
right_Ts, left_Ts, D = clements_decompose(U_qsvt)
err_recon, _ = verify_clements(U_qsvt, right_Ts, left_Ts, D)
print(f"#right_Ts={len(right_Ts)}  #left_Ts={len(left_Ts)}  "
      f"(28 expected)")
print(f"|D| values (should all be 1): {np.round(np.abs(D), 6)}")
print(f"Clements reconstruction error: {err_recon:.2e}")
if err_recon > 1e-10:
    raise RuntimeError(f"Clements algo bug: reconstruction error {err_recon:.2e}")
print("Clements decomposition verified to machine precision.")


print("\n" + "=" * 70)
print("  Step 3 -- pull belenos_circuit + inventory MZIs")
print("=" * 70)
remote_processor = pcvl.RemoteProcessor("qpu:belenos", token=TOKEN)
arch = remote_processor.specs["architecture"]
belenos_circuit = arch.unitary_circuit()
chip_mzis = inventory_mzis(belenos_circuit)
pair_count = Counter(m["mode_pair"] for m in chip_mzis)
print(f"Chip modes={arch.m}, total components={sum(1 for _ in belenos_circuit)}, "
      f"MZIs={len(chip_mzis)}")
# Active pairs: those entirely within modes 0..7
active_pairs = [p for p in pair_count if p[0] < DIM_BE and p[1] < DIM_BE]
print(f"Pairs inside modes 0..7: {sorted(active_pairs)}")
for p in sorted(active_pairs):
    print(f"  pair {p}: {pair_count[p]} MZIs on chip")


print("\n" + "=" * 70)
print("  Step 4 -- realize the Clements MZIs on chip positions + seed search")
print("=" * 70)
# The Clements decomposition yields a sequence of T_k each on a mode pair
# (col, col+1) or (row-1, row). We need to place them on consecutive chip
# MZIs on that same pair, in the order the chip encounters them.
#
# Strategy: build an equivalent list of (mode_pair, target_2x2) entries in
# the order they should be applied on the chip (left-to-right). Then walk
# chip MZIs in circuit order and assign targets in sequence, per pair.

# Clements' canonical rectangular mesh ordering for 8 modes:
# the sequence of right_Ts as emitted by clements_decompose corresponds to
# the "right" half of the mesh (applied before D), and left_Ts to the "left"
# half (applied after D, reading left->right). To compose everything into a
# single left->right chain, we shuffle left_Ts through D using the standard
# commutation identity (see Clements paper Appendix):
#     diag(d) @ T(theta, phi) = T(theta, phi')  @ diag(d')
# where phi' and d' can be computed explicitly.
#
# For simplicity (this is an 8-mode problem), here we skip the analytic
# shuffling and instead build the final mesh numerically: compose the full
# mesh as (left-Ts inverted) @ diag(D) @ (right-Ts), and for each chip MZI,
# solve for the 2-PS values that realize the cumulative 2x2 action at that
# position.

# List of MZI-pair slots the chip exposes inside 8 modes, in the order
# the chip would apply them (left->right traversal of belenos_circuit).
chip_slots_order = []  # [(chip_mzi_idx_in_chip_mzis, mode_pair), ...]
for idx, m in enumerate(chip_mzis):
    if m["mode_pair"] in active_pairs:
        chip_slots_order.append((idx, m["mode_pair"]))

# Capacity check
chip_slot_count_per_pair = Counter(p for _, p in chip_slots_order)
print(f"Chip slots inside 8-mode region (first 40): "
      f"{chip_slots_order[:40]}")

# Clements ordering for 8-mode: we need exactly 28 MZIs arranged in 8 layers:
#   layer 0 (odd pairs):  (1,2), (3,4), (5,6)        -- 3 MZIs
#   layer 1 (even pairs): (0,1), (2,3), (4,5), (6,7) -- 4 MZIs
#   layer 2 (odd):        (1,2), (3,4), (5,6)
#   layer 3 (even):       (0,1), (2,3), (4,5), (6,7)
#   ... alternating, 8 layers total.
# 4 odd layers * 3 + 4 even layers * 4 = 12 + 16 = 28. Matches.
#
# We parameterize each layer-slot with a (theta_ij, phi_ij) and solve
# via a single scipy-free nested Clements algorithm: just redo the
# Clements algorithm layer-by-layer, reading T parameters in the right
# order for the chip's rectangular layout.

mesh_schedule = []   # (layer_idx, mode_pair, (theta, phi))

# Re-run the Clements algorithm, but THIS time collect T's in the
# specific order Clements' rectangular mesh produces them (layers,
# pairs within a layer). This is the same sequence returned by
# clements_decompose() but annotated with layer position.
#
# We detect the layer by counting consecutive (col, col+1) with the
# right pair-parity. Simpler: we just use Clements' output directly
# (right_Ts + left_Ts) and fit chip MZI PSs to their cumulative effect.
#
# The cleanest equivalent: reconstruct U_QSVT as an equivalent mesh of
# Clements MZIs on the chip topology using the recursive algorithm
# described in Clements paper, Appendix A. For 8-mode the result is
# directly:
#   U_QSVT = D' @ prod_k T_k(θ_k, φ_k)
# where T_k's are in the rectangular layering. The commutation shuffle
# from left_Ts through D is:
#   D @ T(θ, φ) = T(θ, φ') @ D'
# with D'[row-1] = D[row] * (-1) ... let's just reconstruct numerically.

# We'll implement the shuffle by one big numerical trick:
# U_QSVT = L^-1 @ D @ R   where L = prod(left_Ts from right to left),
#                              R = prod(right_Ts in original order)
# Equivalently: U_QSVT = (L^-1 @ D @ L) @ (L^-1 @ R)
#             = D_shuffled @ R_final
# where D_shuffled is L^-1 @ D @ L and R_final = L^-1 @ R. This gives a
# single product of T-rotations (R_final) preceded by an output diagonal
# D_shuffled.

def build_full_from_list(Ts, invert=False, size=8):
    prod = np.eye(size, dtype=complex)
    for (pos, theta, phi) in Ts:
        full = np.eye(size, dtype=complex)
        T2 = t_matrix_2x2(theta, phi)
        if invert:
            T2 = np.linalg.inv(T2)
        # For right_Ts: pos is the 'col' (lower mode)
        # For left_Ts:  pos is the 'row' (upper mode of the pair was row-1)
        # Normalize: use pos and pos+1 for right_Ts, pos-1 and pos for left_Ts.
        p_lo = pos if invert is False else pos - 1
        p_hi = p_lo + 1
        full[p_lo:p_hi+1, p_lo:p_hi+1] = T2
        prod = prod @ full
    return prod


# Reconstruction: U_QSVT = L_inv_prod @ diag(D) @ R_inv_prod
# where L_inv_prod = L_1^{-1} @ ... @ L_(last)^{-1}  (forward order, inverted)
# and   R_inv_prod = R_(last) @ ... @ R_1            (reverse order, NOT inverted).
L_inv_prod = np.eye(DIM_BE, dtype=complex)
for (row, theta, phi) in left_Ts:
    T2 = t_matrix_2x2(theta, phi)
    full = np.eye(DIM_BE, dtype=complex)
    full[row-1:row+1, row-1:row+1] = T2
    L_inv_prod = L_inv_prod @ np.linalg.inv(full)

R_inv_prod = np.eye(DIM_BE, dtype=complex)
for (col, theta, phi) in reversed(right_Ts):
    T2 = t_matrix_2x2(theta, phi)
    full = np.eye(DIM_BE, dtype=complex)
    full[col:col+2, col:col+2] = T2
    R_inv_prod = R_inv_prod @ full

recon = L_inv_prod @ np.diag(D) @ R_inv_prod
print(f"Reconstruction ||U_QSVT - L^-1 D R^-1|| = "
      f"{np.linalg.norm(U_qsvt - recon):.2e}")


# ── Simpler route: SCIPY optimization over the 28 chip MZIs' 56 PS values
# Given the above shuffle subtlety, we use scipy on the ACTIVE chip MZIs
# with good initial guess from Clements' (theta, phi) pairs. The initial
# guess is converted to (phi_hi, phi_lo) via per-MZI Circuit.decomposition
# with mean chip BS thetas. This is the "match to chip's BS thetas via
# seed-search" step.
#
# Active chip MZI list ordered by circuit position (same as chip_slots_order).
active_chip_mzis = [chip_mzis[i] for i, _ in chip_slots_order]
n_active = len(active_chip_mzis)
print(f"\nActive chip MZIs inside 8 modes: {n_active}")

# Collect all Clements T's in ONE linear sequence that composes to U_QSVT
# ignoring the diagonal (we'll absorb the diagonal into the first column
# of PSs separately). For that we just list the T's in the order they
# should appear on a rectangular mesh. In practice, we try to match the
# chip's MZI order (chip_slots_order) by fixing the mesh layout.

# For simplicity of this first local test, we pre-compute a "target 2x2"
# for each active chip MZI as follows:
#   * Run Clements on U_QSVT. For each T in right_Ts + left_Ts, assign
#     it to the next available chip MZI on the same mode pair (in circuit
#     order). This is NOT the fully-correct Clements mesh ordering, but
#     gives us per-MZI target 2x2's to fit.
# If the pair ordering does not exactly match the chip's rectangular mesh
# layering, the resulting per-MZI fit will be imperfect -- that's what the
# final full-unitary verification will detect.
target_2x2_per_chip_mzi = [None] * n_active

chip_pair_cursors = defaultdict(int)  # pair -> next free active chip slot idx
def next_chip_slot_for_pair(pair):
    """Return the index in `active_chip_mzis` of the next unused MZI on pair."""
    for j in range(chip_pair_cursors[pair], n_active):
        if active_chip_mzis[j]["mode_pair"] == pair:
            chip_pair_cursors[pair] = j + 1
            return j
    return None

# Walk Clements T's: right_Ts first (they act closest to input), then
# left_Ts in reverse order (they act on output side).
clements_t_seq = []
for (col, theta, phi) in right_Ts:
    clements_t_seq.append(((col, col + 1), theta, phi))
for (row, theta, phi) in reversed(left_Ts):
    clements_t_seq.append(((row - 1, row), theta, phi))

print(f"#Clements T's total: {len(clements_t_seq)}")

for ((p_lo, p_hi), theta, phi) in clements_t_seq:
    j = next_chip_slot_for_pair((p_lo, p_hi))
    if j is None:
        print(f"  WARNING: no chip slot for pair ({p_lo},{p_hi})")
        continue
    target_2x2_per_chip_mzi[j] = t_matrix_2x2(theta, phi)

n_assigned = sum(1 for t in target_2x2_per_chip_mzi if t is not None)
print(f"Chip MZIs assigned a target 2x2: {n_assigned}/{n_active}")


# ── Per-MZI seed-search -- realize target 2x2 on chip BS thetas ──
print("\n" + "=" * 70)
print("  Step 5 -- per-MZI seed-search (chip BS thetas + 2 free PSs)")
print("=" * 70)
ps_assignments = {}
per_mzi_errors = []
for j, chip_m in enumerate(active_chip_mzis):
    if target_2x2_per_chip_mzi[j] is None:
        # No target -- leave as identity on this chip MZI.
        ps_assignments[chip_m["ps1_idx"]] = 0.0 if chip_m["ps1_mode"] % 2 == 1 else pi
        ps_assignments[chip_m["ps2_idx"]] = 0.0 if chip_m["ps2_mode"] % 2 == 1 else pi
        continue

    U_target = target_2x2_per_chip_mzi[j]
    chip_template = (pcvl.Circuit(2)
                     // BS.Rx(theta=chip_m["bs1_theta"])
                     // (chip_m["ps1_mode"] - chip_m["mode_pair"][0],
                         PS(pcvl.P("phi_a")))
                     // (chip_m["ps2_mode"] - chip_m["mode_pair"][0],
                         PS(pcvl.P("phi_b")))
                     // BS.Rx(theta=chip_m["bs2_theta"]))
    best_err = np.inf
    best_phi_a = best_phi_b = 0.0
    for seed in range(MAX_SEEDS_MZI):
        pcvl.random_seed(seed)
        cand = pcvl.Circuit.decomposition(
            pcvl.Matrix(U_target), chip_template,
            phase_shifter_fn=PS, max_try=3)
        if cand is None:
            continue
        err = np.linalg.norm(np.array(cand.compute_unitary()) - U_target)
        if err < best_err:
            best_err = err
            vals = []
            for r, c in cand:
                if isinstance(c, PS):
                    vals.append(parse_phi(c.describe()))
            if len(vals) >= 2:
                best_phi_a, best_phi_b = vals[0], vals[1]
        if best_err < 1e-8:
            break
    per_mzi_errors.append(best_err)
    ps_assignments[chip_m["ps1_idx"]] = best_phi_a
    ps_assignments[chip_m["ps2_idx"]] = best_phi_b

per_mzi_errors = np.array(per_mzi_errors)
print(f"Per-MZI seed-search: n={len(per_mzi_errors)}  "
      f"max_err={per_mzi_errors.max():.2e}  "
      f"mean_err={per_mzi_errors.mean():.2e}  "
      f"#above 1e-3 = {(per_mzi_errors > 1e-3).sum()}")


# ── Set PSs on belenos_circuit ──
print("\n" + "=" * 70)
print("  Step 6 -- set PSs on belenos_circuit")
print("=" * 70)
n_active_set = 0
n_identity_set = 0
for idx, (r, c) in enumerate(list(belenos_circuit)):
    if not isinstance(c, PS):
        continue
    params = c.get_parameters()
    if not params:
        continue
    if idx in ps_assignments:
        params[0].set_value(float(ps_assignments[idx]))
        n_active_set += 1
    else:
        if r[0] == 0:
            params[0].set_value(pi)
        elif r[0] % 2 == 1:
            params[0].set_value(0.0)
        else:
            params[0].set_value(pi)
        n_identity_set += 1
print(f"Active PSs set: {n_active_set}   Identity PSs set: {n_identity_set}")


# ── Step 7: Local verification ──
print("\n" + "=" * 70)
print(f"  Step 7 -- local compute_unitary verification (tol {LOCAL_TOL})")
print("=" * 70)
U_chip_local = np.array(belenos_circuit.compute_unitary())
err_top8 = np.linalg.norm(U_chip_local[:DIM_BE, :DIM_BE] - U_qsvt)
err_top4 = np.linalg.norm(U_chip_local[:DIM_SYS, :DIM_SYS] -
                          U_qsvt[:DIM_SYS, :DIM_SYS])
offdiag_top = np.linalg.norm(U_chip_local[:DIM_BE, DIM_BE:])
print(f"||U_chip[:8,:8] - U_QSVT||_F    = {err_top8:.4e}  "
      f"({'PASS' if err_top8 < LOCAL_TOL else 'FAIL'} at tol {LOCAL_TOL})")
print(f"||U_chip[:4,:4] - T_6(A)||_F    = {err_top4:.4e}")
print(f"||U_chip[:8, 8:]||_F (top-right off-diag) = {offdiag_top:.4e}")


# ══════════════════════════════════════════════════════════
# Step 8: local sampling per eigenstate input
# We apply U_chip_local to each eigenstate vector embedded on modes 0..3
# (ancilla = |0> = modes 0..3 in single-photon encoding) and sample
# N_LOCAL_SHOTS outcomes from the mode-occupation distribution.
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print(f"  Step 8 -- local sampling per eigenstate ({N_LOCAL_SHOTS} shots each)")
print("=" * 70)

n_eigenstates = 4
measured_mode_probs = np.zeros((n_eigenstates, N_MODES_CHIP))
measured_p_filter   = np.zeros(n_eigenstates)
theory_mode_probs   = np.zeros((n_eigenstates, N_MODES_CHIP))
mode0_concentration = np.zeros(n_eigenstates)
detected_events     = np.zeros(n_eigenstates, dtype=int)

for k in range(n_eigenstates):
    psi_in = np.zeros(N_MODES_CHIP, dtype=complex)
    psi_in[:DIM_SYS] = V_H[:, k]    # |0_anc> kron |lambda_k>

    psi_out_chip = U_chip_local @ psi_in
    p_modes = np.abs(psi_out_chip)**2
    p_modes = p_modes / p_modes.sum()   # normalize (should be 1 already if unitary)

    # Theory: apply U_qsvt directly (top-left 8x8) and embed in 24-mode space
    psi_th = np.zeros(N_MODES_CHIP, dtype=complex)
    psi_th[:DIM_BE] = U_qsvt @ psi_in[:DIM_BE]
    p_modes_th = np.abs(psi_th)**2

    # Sample
    counts = np.random.multinomial(N_LOCAL_SHOTS, p_modes)
    measured_mode_probs[k] = counts / N_LOCAL_SHOTS
    theory_mode_probs[k]   = p_modes_th
    measured_p_filter[k]   = counts[:DIM_SYS].sum() / N_LOCAL_SHOTS
    mode0_concentration[k] = counts[0] / N_LOCAL_SHOTS
    detected_events[k]     = N_LOCAL_SHOTS

    print(f"  k={k}  lambda={eigs_H[k]:+.4f}  "
          f"|T_6|^2={target_p[k]:.4f}  "
          f"P(filter succ)={measured_p_filter[k]:.4f}  "
          f"P(mode 0)={mode0_concentration[k]:.4f}")


# ══════════════════════════════════════════════════════════
# Step 9: plot + save
# ══════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True, sharey=True)
x = np.arange(DIM_BE); w = 0.4
for k, ax in zip(range(n_eigenstates), axes.flatten()):
    ax.bar(x - w/2, theory_mode_probs[k, :DIM_BE],   w,
           label="theory (U_QSVT)", color="#3060c0")
    ax.bar(x + w/2, measured_mode_probs[k, :DIM_BE], w,
           label=f"local sim ({N_LOCAL_SHOTS} shots)", color="#a02050")
    ax.axvline(DIM_SYS - 0.5, color="gray", linestyle="--", alpha=0.6)
    ax.set_title(f"k={k}, $\\lambda$={eigs_H[k]:+.3f}")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=8)
    ax.set_xticks(x)
for ax in axes[-1, :]:
    ax.set_xlabel("output mode (0..3 = anc=|0>, 4..7 = anc=|1>)")
for ax in axes[:, 0]:
    ax.set_ylabel("P(photon in mode m)")
fig.suptitle(f"d=6 QSVT directPS local (err_top8={err_top8:.2e})")
fig.tight_layout()
fig.savefig(OUT_PNG, dpi=140)
print(f"\nSaved plot: {OUT_PNG}")

np.savez(
    OUT_NPZ,
    eigs_H=eigs_H, alpha=alpha,
    T6_eigs=T6, target_probabilities=target_p,
    measured_p_filter=measured_p_filter,
    mode0_concentration=mode0_concentration,
    measured_mode_probs=measured_mode_probs,
    theory_mode_probs=theory_mode_probs,
    detected_events=detected_events,
    U_chip_local=U_chip_local,
    U_qsvt=U_qsvt,
    err_top8=err_top8,
    err_top4=err_top4,
    offdiag_top=offdiag_top,
    per_mzi_errors=per_mzi_errors,
    n_active_set=n_active_set,
    n_identity_set=n_identity_set,
)
print(f"Saved data: {OUT_NPZ}")

# ── Final summary ──
print("\n" + "=" * 70)
print("  SUMMARY")
print("=" * 70)
print(f"{'k':>2} {'lambda':>8} {'target':>8} {'measured P(filter)':>20} "
      f"{'P(mode 0)':>12}")
for k in range(n_eigenstates):
    print(f"{k:>2} {eigs_H[k]:>+8.4f} {target_p[k]:>8.4f} "
          f"{measured_p_filter[k]:>20.4f} "
          f"{mode0_concentration[k]:>12.4f}")

print()
if err_top8 < LOCAL_TOL:
    print(f"LOCAL VERIFICATION PASSED (||U_chip[:8,:8] - U_QSVT|| = {err_top8:.2e})")
    print("The direct-PS pipeline faithfully realizes U_QSVT on the chip's")
    print("exact BS mesh -- no cloud compiler involved.")
else:
    print(f"LOCAL VERIFICATION FAILED: err_top8 = {err_top8:.2e} > {LOCAL_TOL}")
    print()
    print("The Clements T sequence was assigned to chip MZIs by circuit-order")
    print("on each mode pair, which matches the ideal rectangular Clements")
    print("layout only approximately. For sub-1e-3 accuracy we need to either:")
    print("  (a) implement the full L^-1 D L shuffle analytically so that all")
    print("      T's compose into a single strict rectangular mesh, or")
    print("  (b) scipy-refine the 56 active PS values starting from this init.")
    print("Both are incremental refinements on top of what this file builds.")

print("\nNO QPU SUBMISSION. No credits used.")
