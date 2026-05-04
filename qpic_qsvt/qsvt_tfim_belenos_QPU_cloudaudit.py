# ============================================================
# OPTION (D): AUDIT CLOUD COMPILATION OF d=6 QSVT
#
# Read-only. Uses the 4 completed job IDs from the earlier
# submission. For each, pulls results['computed_circuit'] --
# the exact 24x24 chip-native circuit that universalchipworker
# synthesized -- computes its unitary locally, and compares to
# the target we fed in.
#
# Tells us: is the cloud compiler accurate (so the QPU
# distortion we saw is loss + post-selection), or did it
# drift (so direct-PS compilation could actually help)?
# ============================================================

import numpy as np
import perceval as pcvl
from perceval import RemoteProcessor

from qsvt_angles import chebyshev_qsp_angles
from qsvt_tfim_matrix_reference import (
    build_tfim_hamiltonian, build_block_encoding,
    build_qsvt_unitary, DIM_SYS, DIM_BE,
)

N_MODES_CHIP = 24
TOKEN = "_T_eyJhbGciOiJIUzUxMiIsInR5cCI6IkpXVCJ9.eyJpZCI6MjU5MiwiZXhwIjoxODA2NTkwNjc1LjE4NjQ3NDN9.eCJCJNcF7BqyUtKiUSaOPK1RM0Nd42B-2UXwH0IIRiYy0TOqi9eQWcUM3049KIkcjgyKkHT-xyIxzoqzmuxyaQ"

JOB_IDS = {
    0: "28cf3a16-ce34-4353-ad91-d38dcc8dbb9c",   # lambda=-2.236
    1: "437a7a53-4b1b-4eb2-abc6-edb82e5566e3",   # lambda=-1.000
    2: "0604f60b-48db-4fae-bf88-e4827a76983e",   # lambda=+1.000
    3: "8df96ed5-0070-45fc-a604-99bc1ddea4f6",   # lambda=+2.236
}

# ── Rebuild what the chip was ASKED to realize ─
H = build_tfim_hamiltonian(1.0)
eigs_H, V_H = np.linalg.eigh(H)
U_H, alpha, A = build_block_encoding(H)
angles = chebyshev_qsp_angles(6)
U_qsvt = build_qsvt_unitary(U_H, angles)
U_24 = np.eye(N_MODES_CHIP, dtype=complex); U_24[:DIM_BE, :DIM_BE] = U_qsvt

def state_prep_4x4(k):
    cols = [V_H[:, k]] + [V_H[:, j] for j in range(DIM_SYS) if j != k]
    return np.column_stack(cols).astype(complex)

U_full_per_k = []
P24_per_k    = []
for k in range(4):
    Vk = state_prep_4x4(k)
    P24 = np.eye(N_MODES_CHIP, dtype=complex); P24[:DIM_SYS, :DIM_SYS] = Vk
    U_full_per_k.append(U_24 @ P24)
    P24_per_k.append(P24)


# ── Fetch + audit ─
pcvl.RemoteConfig.set_token(TOKEN)
pcvl.RemoteConfig().save()
rp = RemoteProcessor("qpu:belenos")

print("=" * 72)
print("  OPTION (D): cloud-compilation audit -- compare computed_circuit")
print("               to what we asked for, and to U_QSVT after un-prep")
print("=" * 72)

summary = []
for k, jid in JOB_IDS.items():
    print(f"\n--- k={k}  lambda(H)={eigs_H[k]:+.4f}  job_id={jid} ---")
    job = rp.resume_job(jid)
    results = job.get_results()
    cc = results["computed_circuit"]
    print(f"  computed_circuit: {sum(1 for _ in cc)} components, "
          f"modes={cc.m}")

    # Local unitary of the chip-native compiled circuit.
    U_comp = np.array(cc.compute_unitary())

    # (A) Compare to what the chip was ASKED to realize:
    U_target_full = U_full_per_k[k]
    err_full   = np.linalg.norm(U_comp - U_target_full)
    err_top8   = np.linalg.norm(U_comp[:DIM_BE, :DIM_BE]
                                - U_target_full[:DIM_BE, :DIM_BE])
    err_top4   = np.linalg.norm(U_comp[:DIM_SYS, :DIM_SYS]
                                - U_target_full[:DIM_SYS, :DIM_SYS])
    maxel_full = np.max(np.abs(U_comp - U_target_full))
    maxel_top8 = np.max(np.abs(U_comp[:DIM_BE, :DIM_BE]
                               - U_target_full[:DIM_BE, :DIM_BE]))

    # (B) Off-diagonal block coupling: top-right 8 x 16 and bottom-left 16 x 8.
    offdiag_top     = np.linalg.norm(U_comp[:DIM_BE, DIM_BE:])
    offdiag_bot     = np.linalg.norm(U_comp[DIM_BE:, :DIM_BE])
    # Expected off-diag for U_target_full: should be zero (since U_target_full
    # is block-diag with U_full on top-left 8x8 and I_16 on bottom-right).
    expected_off    = np.linalg.norm(U_target_full[:DIM_BE, DIM_BE:])

    # (C) Un-apply the state-prep and compare to U_QSVT:
    #      U_target_full = U_24 @ P24_k
    #      => U_24 = U_target_full @ P24_k^dag
    # So U_comp @ P24_k^dag, top-left 8x8, should match U_QSVT (if chip is
    # accurate).
    U_recovered = U_comp @ P24_per_k[k].conj().T
    err_vsQSVT_top8 = np.linalg.norm(U_recovered[:DIM_BE, :DIM_BE] - U_qsvt)
    err_vsQSVT_top4 = np.linalg.norm(U_recovered[:DIM_SYS, :DIM_SYS]
                                     - U_qsvt[:DIM_SYS, :DIM_SYS])
    maxel_vsQSVT    = np.max(np.abs(U_recovered[:DIM_BE, :DIM_BE] - U_qsvt))

    # Print block summary
    print(f"  (1) ||U_comp - U_target||_F (full 24x24)     = {err_full:.4e}")
    print(f"      ||U_comp[:8,:8] - U_target[:8,:8]||_F    = {err_top8:.4e}")
    print(f"      ||U_comp[:4,:4] - U_target[:4,:4]||_F    = {err_top4:.4e}")
    print(f"      max|U_comp - U_target|  (full)           = {maxel_full:.4e}")
    print(f"      max|U_comp - U_target|  (8x8)            = {maxel_top8:.4e}")
    print(f"  (2) ||U_comp[:8, 8:]||_F (top-right block)   = {offdiag_top:.4e}")
    print(f"      ||U_comp[8:, :8]||_F (bot-left block)    = {offdiag_bot:.4e}")
    print(f"      expected = {expected_off:.4e} (0 if block-diag preserved)")
    print(f"  (3) After un-prep (U_comp @ P24_k^dag):")
    print(f"      ||...[:8,:8] - U_QSVT||_F                = "
          f"{err_vsQSVT_top8:.4e}")
    print(f"      ||...[:4,:4] - T_6(A)||_F                = "
          f"{err_vsQSVT_top4:.4e}")
    print(f"      max|...[:8,:8] - U_QSVT|                 = "
          f"{maxel_vsQSVT:.4e}")

    # Side-by-side of a few key elements (top-left 4x4 only to keep it short).
    print(f"\n  side-by-side top-left 4x4 (U_comp  vs  U_target, after un-prep"
          f" -> compare to U_QSVT):")
    print(f"  {'i,j':<5} {'|U_recov[i,j]|':>14} {'|U_QSVT[i,j]|':>14} "
          f"{'|diff|':>10} {'arg diff (rad)':>16}")
    for i in range(4):
        for j in range(4):
            a = U_recovered[i, j]
            b = U_qsvt[i, j]
            d_mag = abs(abs(a) - abs(b))
            d_arg = np.angle(a) - np.angle(b) if abs(a) > 1e-6 and abs(b) > 1e-6 else 0.0
            print(f"  {i},{j:<3} {abs(a):>14.5f} {abs(b):>14.5f} "
                  f"{d_mag:>10.2e} {d_arg:>16.4f}")

    summary.append({
        "k": k, "lambda": eigs_H[k],
        "err_full": err_full, "err_top8": err_top8, "err_top4": err_top4,
        "maxel_top8": maxel_top8,
        "offdiag_top": offdiag_top, "offdiag_bot": offdiag_bot,
        "err_vsQSVT_top8": err_vsQSVT_top8,
        "err_vsQSVT_top4": err_vsQSVT_top4,
    })


# ── Final verdict ─
print("\n" + "=" * 72)
print("  SUMMARY")
print("=" * 72)
print(f"{'k':<2} {'lambda':>8} {'||U_comp-U_target||8x8':>24} "
      f"{'||off-diag top-right||':>22} {'||U_recov[:8,:8]-U_QSVT||':>26}")
for s in summary:
    print(f"{s['k']:<2} {s['lambda']:>+8.4f} {s['err_top8']:>24.4e} "
          f"{s['offdiag_top']:>22.4e} {s['err_vsQSVT_top8']:>26.4e}")

# Verdict: drift or no?
max_err_top8 = max(s["err_top8"] for s in summary)
max_offdiag  = max(s["offdiag_top"] for s in summary)
print()
if max_err_top8 < 1e-3:
    print("VERDICT: cloud compilation is accurate (top-8x8 err < 1e-3 for all k).")
    print("         The measured QPU distortion is therefore NOT compilation drift.")
    print("         It is dominated by mode-dependent photon loss + post-selection")
    print("         bias (physical_perf ~ 0.4-0.9% across the 552-BS mesh).")
    print("         -> Direct-PS compilation would not fix this.")
elif max_err_top8 < 1e-1:
    print(f"VERDICT: mild compilation drift (top-8x8 err up to {max_err_top8:.2e}).")
    print("         Worth trying direct-PS; but loss is still the bigger issue.")
else:
    print(f"VERDICT: SIGNIFICANT compilation drift (top-8x8 err up to "
          f"{max_err_top8:.2e}).")
    print("         Direct-PS compilation (Clements algorithm against the chip's")
    print("         actual BS thetas) is worth implementing.")
print(f"Max off-diagonal coupling U_comp[:8, 8:]: {max_offdiag:.2e}")
if max_offdiag > 1e-3:
    print("  -> The compiled circuit is NOT block-diagonal: mode 0..7 leaks into")
    print("     modes 8..23 at the unitary level. This is another pathway for")
    print("     the measured distortion -- before considering loss.")
else:
    print("  -> Active 8-mode block is well-isolated at the unitary level.")
