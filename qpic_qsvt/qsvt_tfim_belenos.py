# ============================================================
# QSVT EIGENVALUE FILTER ON 2-SITE TFIM -- PERCEVAL / BELENOS
# ============================================================
#
# GOAL:
#   Run the same QSVT eigenvalue filter as
#   qsvt_tfim_matrix_reference.py, but in Perceval, embedded
#   into Belenos' 24-mode photonic chip architecture, and
#   validate against the matrix reference.
#
# PIPELINE (mirrors the QSP/Belenos workflow we already use
# in 2026_04_24_CircuitDecomposition_MZIs_AnyL_x0.5_*.py):
#
#   Step 0: Build the matrix-reference QSVT unitary (8x8).
#           This is the source of truth.
#
#   Step 1: Embed the 8x8 QSVT unitary into a 24x24 unitary
#           U_24 = U_QSVT (top-left 8x8) (+) I_16. The system
#           qubits live on modes 0..3 and the ancilla qubit
#           on modes 4..7 in single-photon (qudit) encoding.
#
#   Step 2: Build a Perceval Circuit on 24 modes that
#           realises U_24 as a single Unitary component.
#           ---> "ideal" Perceval simulation: compute_unitary
#                must equal U_24 to numerical precision.
#
#   Step 3: Decompose U_24 into a BS/PS mesh (Reck triangle).
#           This is the closest *local* analogue to the chip's
#           BS/PS mesh -- universal but with arbitrary BS thetas.
#           Verify the decomposed circuit's unitary matches U_24.
#
#   Step 4 (optional, requires QPU token):
#           Try to load Belenos chip BS thetas from
#           RemoteProcessor("qpu:belenos").specs["architecture"].
#           Print them so we can confirm the chip would compile
#           U_24 on the actual fixed-BS mesh on the cloud
#           (universalchipworker handles the constrained inverse).
#           No job is submitted here -- ekarooby's existing scripts
#           are the place for QPU submission.
#
#   Step 5: Validation -- compare against qsvt_tfim_matrix_reference:
#             (a) Perceval ideal U_24 top-left 8x8  vs. U_QSVT
#             (b) U_QSVT top-left 4x4               vs. f(A)
#             (c) f(A) * alpha                      vs. f(H) (direct
#                 numpy diagonalisation)
#           All three Frobenius differences are printed; in the
#           ideal simulation they should all be ~ 1e-12 or better.
#
#   Step 6: Filter demo -- act U_QSVT on |0>_anc x |lambda_k>_sys
#           for every Hamiltonian eigenstate, and report the
#           projection probability onto |0>_anc (the success
#           probability of the BE) and the resulting filtered
#           amplitude. Compare against the polynomial value.
#
# NOTE ON THE BELENOS CONSTRAINED-BS MESH:
#   Belenos has 552 PSs and a fixed (non-50:50) BS mesh.
#   Inverting that mesh to realise an arbitrary 24x24 U is the
#   universalchipworker compiler's job and runs on the cloud.
#   Locally we can only confirm that *some* BS/PS decomposition
#   reproduces U_24 -- which is what Step 3 verifies via Reck.
#   To actually run on Belenos: copy this circuit's setup into
#   the 2026_04_24 layer-by-layer pipeline (same approach used
#   for the QSP scripts) and submit via RemoteProcessor.
# ============================================================

import numpy as np
import perceval as pcvl
import perceval.components as comp
from perceval.components import BS, PS, Unitary

from qsvt_angles import chebyshev_qsp_angles
from qsvt_tfim_matrix_reference import (
    build_tfim_hamiltonian,
    build_block_encoding,
    build_qsvt_unitary,
    exact_polynomial_of_A,
    DIM_SYS, DIM_ANC, DIM_BE,
)

# ── Settings ──────────────────────────────────────────────
H_FIELD       = 1.0
QSVT_DEGREE   = 6          # d=6 -- 6 layers of U_H + 7 phase rotations
N_MODES_CHIP  = 24
TRY_BELENOS   = False      # set True (and provide TOKEN) to fetch chip BS thetas
TOKEN         = None       # e.g. "_T_eyJ..." -- only used if TRY_BELENOS=True
RUN_RECK      = True       # set False to skip the heavy 24x24 decomposition
SEED          = 42

pcvl.random_seed(SEED)
np.random.seed(SEED)


# ── Step 0: Build matrix-reference QSVT unitary ───────────
print("=" * 66)
print("  Step 0 -- matrix reference: build U_QSVT (8x8)")
print("=" * 66)

H = build_tfim_hamiltonian(H_FIELD)
eigs_H, V_H = np.linalg.eigh(H)
U_H, alpha, A = build_block_encoding(H)
angles = chebyshev_qsp_angles(QSVT_DEGREE)
U_qsvt = build_qsvt_unitary(U_H, angles)
f_A    = exact_polynomial_of_A(A, angles)

print(f"H eigenvalues       = {np.round(eigs_H, 6)}")
print(f"alpha = ||H||_2     = {alpha:.6f}")
print(f"angles (d={QSVT_DEGREE})       = {np.round(angles, 6)}")
print(f"#layers: {QSVT_DEGREE} U_H blocks alternating with "
      f"{QSVT_DEGREE+1} R(phi) phase rotations")
print(f"||U_H U_H^dag - I||      = "
      f"{np.linalg.norm(U_H @ U_H.conj().T - np.eye(DIM_BE)):.2e}")
print(f"||U_H[:4,:4] - H/alpha|| = "
      f"{np.linalg.norm(U_H[:DIM_SYS, :DIM_SYS] - A):.2e}")
print(f"||U_QSVT[:4,:4] - T_{QSVT_DEGREE}(A)|| = "
      f"{np.linalg.norm(U_qsvt[:DIM_SYS, :DIM_SYS] - f_A):.2e}")


# ── Step 1: Embed 8x8 -> 24x24 ────────────────────────────
print("\n" + "=" * 66)
print("  Step 1 -- embed U_QSVT (8x8) into U_24 (24x24)")
print("=" * 66)

U_24 = np.eye(N_MODES_CHIP, dtype=complex)
U_24[:DIM_BE, :DIM_BE] = U_qsvt

embed_unitary_err = np.linalg.norm(U_24 @ U_24.conj().T - np.eye(N_MODES_CHIP))
print(f"||U_24 U_24^dag - I||_F = {embed_unitary_err:.2e}")
print(f"top-left 8x8 == U_QSVT  : "
      f"{np.allclose(U_24[:DIM_BE, :DIM_BE], U_qsvt, atol=1e-12)}")
print(f"identity on modes 8..23 : "
      f"{np.allclose(U_24[DIM_BE:, DIM_BE:], np.eye(N_MODES_CHIP - DIM_BE))}")


# ── Step 2: Perceval circuit (ideal) ──────────────────────
print("\n" + "=" * 66)
print("  Step 2 -- Perceval ideal simulation (single Unitary box)")
print("=" * 66)

ideal_circuit = pcvl.Circuit(N_MODES_CHIP, name=f"QSVT_TFIM_d{QSVT_DEGREE}")
ideal_circuit.add(0, Unitary(pcvl.Matrix(U_24)))

U_pcvl_ideal = np.array(ideal_circuit.compute_unitary())
diff_ideal_24 = np.linalg.norm(U_pcvl_ideal - U_24)
diff_ideal_top_left = np.linalg.norm(
    U_pcvl_ideal[:DIM_BE, :DIM_BE] - U_qsvt)

print(f"Perceval circuit modes      = {ideal_circuit.m}")
print(f"||U_pcvl - U_24||_F         = {diff_ideal_24:.2e}")
print(f"||U_pcvl[:8,:8] - U_QSVT||_F = {diff_ideal_top_left:.2e}")


# ── Step 3: Reck BS/PS decomposition (realistic local) ────
if RUN_RECK:
    print("\n" + "=" * 66)
    print("  Step 3 -- Reck decomposition into BS + PS mesh (24-mode)")
    print("=" * 66)
    print("This is the local analogue of the chip's BS/PS mesh; the BS")
    print("thetas are free, unlike the constrained Belenos mesh.")

    decomposed = pcvl.Circuit.decomposition(
        pcvl.Matrix(U_24),
        BS(theta=pcvl.P("theta"), phi_tr=pcvl.P("phi")),
        phase_shifter_fn=PS,
        max_try=20,
    )

    if decomposed is None:
        print("Reck decomposition FAILED -- skipping decomposed verification.")
    else:
        U_decomp = np.array(decomposed.compute_unitary())
        diff_reck_24  = np.linalg.norm(U_decomp - U_24)
        diff_reck_top = np.linalg.norm(U_decomp[:DIM_BE, :DIM_BE] - U_qsvt)
        n_bs = sum(1 for _, c in decomposed if isinstance(c, BS))
        n_ps = sum(1 for _, c in decomposed if isinstance(c, PS))

        # Depth = number of non-overlapping BS columns (greedy column packing).
        bs_modes = [r for r, c in decomposed if isinstance(c, BS)]
        cols = []
        for r in bs_modes:
            placed = False
            for col in cols:
                if not any(set(r) & set(rr) for rr in col):
                    col.append(r); placed = True; break
            if not placed:
                cols.append([r])
        bs_depth = len(cols)

        print(f"Reck circuit components: {n_bs} BS, {n_ps} PS  "
              f"(N(N-1)/2 = {N_MODES_CHIP*(N_MODES_CHIP-1)//2} BS upper bound; "
              f"Perceval prunes identity blocks)")
        depth_msg = ("fits" if bs_depth <= 24
                     else "EXCEEDS chip depth -- cloud compiler "
                          "repacks into a single 24x24 unitary")
        print(f"BS-column depth          = {bs_depth}  "
              f"(Belenos has ~24 MZI columns -> {depth_msg})")
        print(f"||U_decomp - U_24||_F         = {diff_reck_24:.2e}")
        print(f"||U_decomp[:8,:8] - U_QSVT||  = {diff_reck_top:.2e}")
        print(f"top-left 8x8 matches U_QSVT (atol=1e-5): "
              f"{np.allclose(U_decomp[:DIM_BE, :DIM_BE], U_qsvt, atol=1e-5)}")
else:
    print("\n[Step 3 skipped -- set RUN_RECK=True for Reck decomposition]")


# ── Step 4: Belenos chip BS thetas (optional) ─────────────
if TRY_BELENOS and TOKEN:
    print("\n" + "=" * 66)
    print("  Step 4 -- inspect Belenos chip BS thetas")
    print("=" * 66)
    try:
        import re
        rp   = pcvl.RemoteProcessor("qpu:belenos", token=TOKEN)
        arch = rp.specs["architecture"]
        belenos_circuit = arch.unitary_circuit()
        bs_thetas = []
        for r, c in belenos_circuit:
            if r == (0, 1):
                m = re.search(r"theta=([\-\+]?[\d\.eE\+\-]+)",
                              c.describe())
                if m:
                    bs_thetas.append(float(m.group(1)))
        print(f"Belenos chip modes = {arch.m}")
        print(f"#BS extracted on modes (0,1) = {len(bs_thetas)}")
        for i, t in enumerate(bs_thetas[:8]):
            print(f"  BS #{i+1:2d}: theta = {t:.6f} rad "
                  f"({np.degrees(t):.3f} deg)")
        print("...")
        print("To realise U_24 on Belenos, push it through the same")
        print("layer-by-layer PS-only assignment used in the QSP scripts.")
    except Exception as e:
        print(f"Belenos inspection failed: {e}")
else:
    print("\n[Step 4 skipped -- set TRY_BELENOS=True and provide TOKEN]")


# ── Step 5: Cross-validation ──────────────────────────────
print("\n" + "=" * 66)
print("  Step 5 -- cross-validation summary")
print("=" * 66)

# (a) Perceval (ideal) vs matrix-reference U_QSVT
err_a = diff_ideal_top_left

# (b) matrix-reference top-left vs polynomial of A
err_b = np.linalg.norm(U_qsvt[:DIM_SYS, :DIM_SYS] - f_A)

# (c) polynomial of A scaled to H vs direct polynomial of H eigvals
P_eigs = np.array([
    np.cos(QSVT_DEGREE * np.arccos(np.clip(eigs_H[k] / alpha, -1, 1)))
    for k in range(DIM_SYS)
])  # T_d evaluated on eigvals of A
f_H_direct = V_H @ np.diag(P_eigs) @ V_H.conj().T
err_c = np.linalg.norm(f_A - f_H_direct)

print(f"(a) ||U_pcvl[:8,:8] - U_QSVT||_F                   = {err_a:.2e}")
print(f"(b) ||U_QSVT[:4,:4] - f(A)||_F                      = {err_b:.2e}")
print(f"(c) ||f(A) - V * T_d(diag(eigs_A)) * V^dag||_F      = {err_c:.2e}")

print("\nAll three should be ~ 1e-12 in ideal simulation.")
all_ok = err_a < 1e-9 and err_b < 1e-9 and err_c < 1e-9
print(f"VALIDATION {'PASSED' if all_ok else 'FAILED'}")


# ── Step 6: Per-eigenstate filter demo ────────────────────
print("\n" + "=" * 66)
print("  Step 6 -- per-eigenstate filter fidelity")
print("=" * 66)
print(f"{'k':>2}  {'lambda(H)':>11}  {'lambda(A)':>11}  "
      f"{'|T_d(lambda)|^2':>16}  {'|<0_anc,lam|U_QSVT|0_anc,lam>|^2':>34}")
for k in range(DIM_SYS):
    v_k     = V_H[:, k]
    lam_A_k = eigs_H[k] / alpha
    Td_val  = np.cos(QSVT_DEGREE * np.arccos(np.clip(lam_A_k, -1, 1)))

    # Inject |0>_anc kron |lam_k> -> photon amplitudes in modes 0..3
    psi_in  = np.zeros(DIM_BE, dtype=complex)
    psi_in[:DIM_SYS] = v_k

    psi_out = U_qsvt @ psi_in
    # Project on |0>_anc (modes 0..3 amplitudes)
    psi_out_top = psi_out[:DIM_SYS]
    overlap_sq  = abs(v_k.conj() @ psi_out_top) ** 2

    print(f"{k:>2}  {eigs_H[k]:>11.6f}  {lam_A_k:>11.6f}  "
          f"{Td_val**2:>16.6f}  {overlap_sq:>34.6f}")


# ── Step 7: Single-photon-in-mode-0 sanity check ──────────
print("\n" + "=" * 66)
print("  Step 7 -- single photon in mode 0 (baseline test)")
print("=" * 66)
psi0 = np.zeros(N_MODES_CHIP, dtype=complex); psi0[0] = 1.0
psi_out_24      = U_24 @ psi0
psi_out_pcvl_24 = U_pcvl_ideal @ psi0
print("Single-photon amplitudes in modes 0..7 after U_24 (matrix):")
for m in range(DIM_BE):
    print(f"  mode {m}: amp = {psi_out_24[m]:+.6f}")
print(f"\n||U_24 @ e_0  -  U_pcvl @ e_0|| = "
      f"{np.linalg.norm(psi_out_24 - psi_out_pcvl_24):.2e}")
