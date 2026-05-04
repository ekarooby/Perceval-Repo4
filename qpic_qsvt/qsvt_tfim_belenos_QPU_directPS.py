# ============================================================
# DIRECT-PS COMPILATION OF d=6 QSVT TO BELENOS
# (no universalchipworker, no Unitary box)
#
# ANSWER TO Q1-Q3 (about qsvt_tfim_belenos_QPU.py):
#   The prior script uses approach (b) -- cloud compilation.
#   Specifically, it calls:
#       ckt = pcvl.Circuit(24); ckt.add(0, Unitary(pcvl.Matrix(U_24)))
#       rp.set_circuit(ckt)
#   so universalchipworker synthesizes all 552 BS + 574 PS
#   values on the cloud. Approach (a) from your 2-mode QSP
#   workflow (arch.unitary_circuit() + direct params[0].set_value)
#   was NOT used there. This file implements approach (a)
#   extended to the 8x8 QSVT block.
#
# CHIP TOPOLOGY (confirmed by inspecting belenos_circuit):
#   Belenos is Clements rectangular, not Reck triangular.
#   Each MZI = 4 consecutive components on a fixed mode pair:
#       BS.Rx(theta1) -> PS(phi_hi) on upper mode
#                     -> PS(phi_lo) on lower mode
#                     -> BS.Rx(theta2)
#   theta1, theta2 are chip-specific (not 50:50; ~91.5 deg each).
#   The chip is composed of 276 such MZIs, matching Clements of 24.
#
# PIPELINE:
#   Step 1: Build U_QSVT (8x8) and U_24 = U_QSVT (+) I_16.
#   Step 2: Pull belenos_circuit = arch.unitary_circuit()
#           and inventory all MZIs (4-component pattern).
#   Step 3: Reck/Clements-decompose U_24 using Perceval's
#           Circuit.decomposition with shape='rectangle' and a
#           2-PS MZI template at the chip's mean BS theta.
#           Seeds 0..199 (LOCAL_TOL = 1e-3 target).
#   Step 4: Walk decomposed + belenos-chip MZIs in lockstep
#           (both share Clements ordering). For each matched
#           pair, re-solve the 2 PSs against the chip's actual
#           per-position BS thetas via per-MZI
#           Circuit.decomposition + seed search.
#   Step 5: Set PS values directly via params[0].set_value.
#           Set unused PSs (those on chip MZIs whose counterparts
#           were not produced by the decomposition, and any
#           remaining non-MZI PSs) to identity.
#   Step 6: Local verification -- compute_unitary on
#           belenos_circuit must match U_24 top-left 8x8 to 1e-3.
#
# NO SUBMISSION happens in this script.
# ============================================================

import os, re
import numpy as np
from math import pi
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
H_FIELD       = 1.0
QSVT_DEGREE   = 6
N_MODES_CHIP  = 24
LOCAL_TOL     = 1e-3
MAX_SEEDS_GLOBAL = 200
MAX_SEEDS_MZI    = 200
TOKEN = "_T_eyJhbGciOiJIUzUxMiIsInR5cCI6IkpXVCJ9.eyJpZCI6MjU5MiwiZXhwIjoxODA2NTkwNjc1LjE4NjQ3NDN9.eCJCJNcF7BqyUtKiUSaOPK1RM0Nd42B-2UXwH0IIRiYy0TOqi9eQWcUM3049KIkcjgyKkHT-xyIxzoqzmuxyaQ"


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
    Walk circuit components and group each MZI = 4 consecutive components:
       BS on (a,b) -> PS on one of {a,b} -> PS on the other -> BS on (a,b)
    Returns list of dict.
    """
    comps = list(circuit)
    mzis  = []
    i = 0
    while i < len(comps):
        r, c = comps[i]
        # Skip Barriers or non-BS
        if not isinstance(c, BS):
            i += 1
            continue
        # Need 3 lookahead
        if i + 3 >= len(comps):
            i += 1
            continue
        r1, c1 = comps[i + 1]
        r2, c2 = comps[i + 2]
        r3, c3 = comps[i + 3]
        is_mzi = (
            isinstance(c1, PS) and isinstance(c2, PS) and isinstance(c3, BS)
            and r3 == r
            and set(r1) | set(r2) == set(r)   # two PSs cover both modes
            and len(r1) == 1 and len(r2) == 1
        )
        if not is_mzi:
            i += 1
            continue
        mzis.append({
            "mode_pair": r,
            "bs1_idx":   i,
            "ps1_idx":   i + 1,
            "ps1_mode":  r1[0],
            "ps2_idx":   i + 2,
            "ps2_mode":  r2[0],
            "bs2_idx":   i + 3,
            "bs1_theta": parse_theta(c.describe()),
            "bs2_theta": parse_theta(c3.describe()),
        })
        i += 4
    return mzis


# ── Step 1: Build U_QSVT (8x8) and U_24 (24x24) ───────────
print("=" * 66)
print("  Step 1 -- build U_QSVT (d=6) and U_24")
print("=" * 66)
H = build_tfim_hamiltonian(H_FIELD)
eigs_H, V_H = np.linalg.eigh(H)
U_H, alpha, A = build_block_encoding(H)
angles = chebyshev_qsp_angles(QSVT_DEGREE)
U_qsvt = build_qsvt_unitary(U_H, angles)
U_24 = np.eye(N_MODES_CHIP, dtype=complex); U_24[:DIM_BE, :DIM_BE] = U_qsvt

T6 = np.cos(QSVT_DEGREE *
            np.arccos(np.clip(eigs_H / alpha, -1, 1)))
print(f"||U_QSVT[:4,:4] - T_6(A)||  = "
      f"{np.linalg.norm(U_qsvt[:DIM_SYS,:DIM_SYS] - V_H @ np.diag(T6) @ V_H.conj().T):.2e}")
print(f"||U_24 U_24^dag - I||       = "
      f"{np.linalg.norm(U_24 @ U_24.conj().T - np.eye(N_MODES_CHIP)):.2e}")


# ── Step 2: Pull Belenos chip + inventory MZIs ────────────
print("\n" + "=" * 66)
print("  Step 2 -- pull belenos_circuit and inventory MZIs")
print("=" * 66)
remote_processor = pcvl.RemoteProcessor("qpu:belenos", token=TOKEN)
arch = remote_processor.specs["architecture"]
belenos_circuit = arch.unitary_circuit()

chip_mzis = inventory_mzis(belenos_circuit)
pair_count = Counter([m["mode_pair"] for m in chip_mzis])
print(f"Chip modes: {arch.m},  total components: "
      f"{sum(1 for _ in belenos_circuit)},  MZIs inventoried: {len(chip_mzis)}")
if len(chip_mzis) == 0:
    raise RuntimeError("MZI inventory empty -- check chip structure parser.")
print(f"Sample MZIs 0..5:")
for m in chip_mzis[:6]:
    print(f"  pair={m['mode_pair']}  "
          f"bs1_theta={m['bs1_theta']:.4f}  bs2_theta={m['bs2_theta']:.4f}  "
          f"ps1=mode{m['ps1_mode']}  ps2=mode{m['ps2_mode']}")


# ── Step 3: Decompose U_24 with chip-style 2-PS MZI template
print("\n" + "=" * 66)
print("  Step 3 -- decompose U_24 with 2-PS Clements template")
print("=" * 66)
all_thetas = [t for m in chip_mzis for t in (m["bs1_theta"], m["bs2_theta"])]
mean_theta = float(np.mean(all_thetas))
print(f"Chip BS thetas: mean={mean_theta:.4f} rad "
      f"({np.degrees(mean_theta):.3f} deg),  "
      f"range=[{min(all_thetas):.4f}, {max(all_thetas):.4f}]")

# 2-PS MZI template matching chip structure (BS -> PS on hi -> PS on lo -> BS).
mzi_template = (pcvl.Circuit(2)
                // BS.Rx(theta=mean_theta)
                // (1, PS(pcvl.P("phi_hi")))
                // (0, PS(pcvl.P("phi_lo")))
                // BS.Rx(theta=mean_theta))

best_err = np.inf
decomposed = None
for shape in ("rectangle", "triangle"):
    print(f"  trying shape='{shape}'")
    for seed in range(MAX_SEEDS_GLOBAL):
        pcvl.random_seed(seed)
        cand = pcvl.Circuit.decomposition(
            pcvl.Matrix(U_24), mzi_template,
            phase_shifter_fn=PS, shape=shape, max_try=3)
        if cand is None:
            continue
        err = np.linalg.norm(np.array(cand.compute_unitary()) - U_24)
        if err < best_err:
            best_err = err
            decomposed = cand
            best_shape = shape
            print(f"    seed={seed:3d} shape={shape:<9} err={err:.2e}")
        if err < LOCAL_TOL:
            break
    if best_err < LOCAL_TOL:
        break

if decomposed is None:
    raise RuntimeError("Circuit.decomposition returned None for all seeds/shapes.")

print(f"Best decomposition: shape={best_shape}, err={best_err:.2e}")
dec_mzis = inventory_mzis(decomposed)
print(f"Decomposed MZI count: {len(dec_mzis)}  "
      f"(chip MZI count: {len(chip_mzis)})")


# ── Step 4: Match MZIs + re-solve PSs per chip MZI ────────
print("\n" + "=" * 66)
print("  Step 4 -- match decomposed MZIs to chip positions + re-solve PSs")
print("=" * 66)

chip_by_pair = defaultdict(list)
for m in chip_mzis:
    chip_by_pair[m["mode_pair"]].append(m)
dec_by_pair = defaultdict(list)
for m in dec_mzis:
    dec_by_pair[m["mode_pair"]].append(m)

# For each decomposed MZI, match to the next unused chip MZI on the same pair.
def read_ps_values_from_circuit(ckt):
    """Return ordered list of (mode, phi) for PS components of ckt."""
    out = []
    for r, c in ckt:
        if isinstance(c, PS):
            phi = parse_phi(c.describe())
            out.append((r[0], phi))
    return out

ps_assignments = {}   # global idx in belenos_circuit -> phi
per_mzi_err    = []
unmatched      = 0

for pair, dec_list in dec_by_pair.items():
    chip_list = chip_by_pair.get(pair, [])
    if len(chip_list) < len(dec_list):
        unmatched += len(dec_list) - len(chip_list)
        dec_list = dec_list[:len(chip_list)]

    for j, dec_m in enumerate(dec_list):
        chip_m = chip_list[j]
        # Target 2x2 = what dec_m realizes using MEAN thetas + its PSs.
        target_ckt = (pcvl.Circuit(2)
                      // BS.Rx(theta=mean_theta)
                      // (dec_m["ps1_mode"], PS(
                          parse_phi(
                              list(decomposed)[dec_m["ps1_idx"]][1].describe())))
                      // (dec_m["ps2_mode"], PS(
                          parse_phi(
                              list(decomposed)[dec_m["ps2_idx"]][1].describe())))
                      // BS.Rx(theta=mean_theta))
        U_target = np.array(target_ckt.compute_unitary())

        # Build chip template with this position's actual BS thetas and 2 PSs
        # on the same mode ordering as the chip.
        chip_template = (pcvl.Circuit(2)
                         // BS.Rx(theta=chip_m["bs1_theta"])
                         // (chip_m["ps1_mode"] - chip_m["mode_pair"][0],
                             PS(pcvl.P("phi1")))
                         // (chip_m["ps2_mode"] - chip_m["mode_pair"][0],
                             PS(pcvl.P("phi2")))
                         // BS.Rx(theta=chip_m["bs2_theta"]))

        best_local_err = np.inf
        best_ps1 = best_ps2 = 0.0
        for seed in range(MAX_SEEDS_MZI):
            pcvl.random_seed(seed)
            c2 = pcvl.Circuit.decomposition(
                pcvl.Matrix(U_target), chip_template,
                phase_shifter_fn=PS, max_try=3)
            if c2 is None:
                continue
            le = np.linalg.norm(np.array(c2.compute_unitary()) - U_target)
            if le < best_local_err:
                best_local_err = le
                pvs = read_ps_values_from_circuit(c2)
                # Expect exactly 2 PS entries, in chip mode order.
                if len(pvs) >= 2:
                    best_ps1 = pvs[0][1]
                    best_ps2 = pvs[1][1]
            if best_local_err < 1e-8:
                break

        per_mzi_err.append(best_local_err)
        ps_assignments[chip_m["ps1_idx"]] = best_ps1
        ps_assignments[chip_m["ps2_idx"]] = best_ps2

per_mzi_err = np.array(per_mzi_err)
print(f"Per-MZI re-solve: total={len(per_mzi_err)}  "
      f"max_err={per_mzi_err.max():.2e}  mean_err={per_mzi_err.mean():.2e}  "
      f"#above 1e-3 = {(per_mzi_err > 1e-3).sum()}")
print(f"Unmatched decomposed MZIs (chip ran out on that pair): {unmatched}")


# ── Step 5: Set PS values directly on belenos_circuit ─────
print("\n" + "=" * 66)
print("  Step 5 -- set PS values directly on belenos_circuit")
print("=" * 66)

n_active = 0
n_identity = 0
for idx, (r, c) in enumerate(list(belenos_circuit)):
    if not isinstance(c, PS):
        continue
    params = c.get_parameters()
    if not params:
        continue
    if idx in ps_assignments:
        params[0].set_value(float(ps_assignments[idx]))
        n_active += 1
    else:
        # Identity-MZI convention from your fab.py:
        # mode 0 -> pi, odd modes -> 0, other even modes -> pi.
        if r[0] == 0:
            params[0].set_value(pi)
        elif r[0] % 2 == 1:
            params[0].set_value(0.0)
        else:
            params[0].set_value(pi)
        n_identity += 1

print(f"Active PSs set: {n_active}  Identity PSs set: {n_identity}")


# ── Step 6: Local verification ────────────────────────────
print("\n" + "=" * 66)
print(f"  Step 6 -- local verification (threshold {LOCAL_TOL})")
print("=" * 66)
U_chip_local = np.array(belenos_circuit.compute_unitary())
err_full     = np.linalg.norm(U_chip_local - U_24)
err_top8     = np.linalg.norm(U_chip_local[:DIM_BE, :DIM_BE] - U_qsvt)
err_top4     = np.linalg.norm(U_chip_local[:DIM_SYS, :DIM_SYS] -
                              U_qsvt[:DIM_SYS, :DIM_SYS])
print(f"||U_chip - U_24||_F             = {err_full:.2e}")
print(f"||U_chip[:8,:8] - U_QSVT||_F    = {err_top8:.2e}  "
      f"({'PASS' if err_top8 < LOCAL_TOL else 'FAIL'} at tol {LOCAL_TOL})")
print(f"||U_chip[:4,:4] - T_6(A)||_F    = {err_top4:.2e}")

# Single-photon-in-mode-0 distribution comparison.
psi0 = np.zeros(N_MODES_CHIP, dtype=complex); psi0[0] = 1.0
out_chip = U_chip_local @ psi0
out_id   = U_24 @ psi0
print(f"\nSingle-photon-in-mode-0 amplitudes, modes 0..7:")
print(f"  {'mode':<4} {'|U_24 @ e0|^2':>16} {'|U_chip @ e0|^2':>18} {'diff':>10}")
for m in range(DIM_BE):
    print(f"  {m:<4} {abs(out_id[m])**2:>16.5f} {abs(out_chip[m])**2:>18.5f} "
          f"{abs(abs(out_id[m])**2 - abs(out_chip[m])**2):>10.2e}")

print("\n" + "=" * 66)
if err_top8 < LOCAL_TOL:
    print("VERIFICATION PASSED -- belenos_circuit with these PSs realizes")
    print("U_24 on chip to better than 1e-3 (top-left 8x8). Ready to submit")
    print("with explicit per-run confirmation.")
else:
    print("VERIFICATION FAILED -- direct-PS pipeline did not converge to 1e-3.")
    print("Likely cause: chip BSs are ~1.5 deg off 50:50, and the constrained")
    print("MZI class cannot realize arbitrary SU(2) with just 2 free PSs. To")
    print("fully compile, would need either the cloud universalchipworker or a")
    print("scipy.optimize refinement over all 552 PS parameters jointly.")
print("=" * 66)
print("\nNO JOBS WERE SUBMITTED. No credits used.")
