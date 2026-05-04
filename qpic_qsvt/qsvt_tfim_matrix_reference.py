# ============================================================
# QSVT EIGENVALUE FILTER ON 2-SITE TFIM -- PURE NUMPY REFERENCE
# ============================================================
#
# GOAL:
#   Build, run, and validate a QSVT eigenvalue filter for the
#   transverse-field Ising Hamiltonian H = -ZZ - h*(XI + IX)
#   in pure numpy. This is the ground truth that
#   qsvt_tfim_belenos.py is checked against.
#
# WHAT IT DOES:
#   1. Build H (4x4) for h=1, print eigenvalues / spectral norm.
#   2. Build single-ancilla block encoding U_H (8x8):
#         U_H = [[A, i*sqrt(I-A^2)], [i*sqrt(I-A^2), A]]
#      with A = H/alpha,  alpha = ||H||_2.
#      Verify (a) U_H is unitary and (b) top-left 4x4 = A.
#   3. Build R(phi) = exp(i*phi*(2*Pi - I)) where Pi = |0><0|_anc x I_4.
#   4. Compose QSVT:
#         U_QSVT = R(phi_0) U_H R(phi_1) U_H R(phi_2) ... U_H R(phi_d)
#   5. Compute the *exact* polynomial filter applied directly to H:
#         f(H) = V * diag(P_d(lambda_i / alpha)) * V^dagger
#      where P_d is the polynomial implied by the angles (T_d here).
#   6. Compare:
#         (a) top-left 4x4 of U_QSVT  vs.  f(H/alpha)
#         (b) every action U_QSVT @ (|0>_anc kron |psi>) projected on |0>_anc
#             vs. f(A) @ |psi>
#      Both should match to machine precision.
#
# EXPORTED FUNCTIONS (used by qsvt_tfim_belenos.py):
#   build_tfim_hamiltonian(h)
#   build_block_encoding(H)        -> (U_H, alpha, A)
#   build_qsvt_unitary(U_H, angles) -> 8x8
#   exact_polynomial_of_A(A, angles) -> 4x4 reference for top-left block
# ============================================================

import numpy as np
from qsvt_angles import (chebyshev_qsp_angles,
                         polynomial_from_angles)

# ── Constants ─────────────────────────────────────────────
N_SITES = 2
DIM_SYS = 2 ** N_SITES        # 4
DIM_ANC = 2                   # 1 ancilla qubit
DIM_BE  = DIM_SYS * DIM_ANC   # 8


# ── 1. Hamiltonian ────────────────────────────────────────
def build_tfim_hamiltonian(h=1.0):
    I2 = np.eye(2, dtype=complex)
    X  = np.array([[0, 1], [1, 0]], dtype=complex)
    Z  = np.array([[1, 0], [0, -1]], dtype=complex)
    ZZ = np.kron(Z, Z)
    XI = np.kron(X, I2)
    IX = np.kron(I2, X)
    return -ZZ - h * (XI + IX)


# ── 2. Block encoding ─────────────────────────────────────
def build_block_encoding(H):
    """
    Single-ancilla-qubit block encoding U_H of A = H/alpha:

        U_H = [[ A,        i*sqrt(I-A^2) ],
               [ i*sqrt(I-A^2),       A  ]]

    H must be Hermitian.  alpha is the spectral norm so |A| <= 1.
    Returns (U_H, alpha, A).
    """
    assert np.allclose(H, H.conj().T, atol=1e-12), "H not Hermitian"
    eigs, V = np.linalg.eigh(H)
    alpha = np.max(np.abs(eigs))
    A = H / alpha

    eigs_A = eigs / alpha
    sqrt_term = V @ np.diag(np.sqrt(np.clip(1 - eigs_A**2, 0, None))) @ V.conj().T

    n = H.shape[0]
    U_H = np.block([[A,             1j * sqrt_term],
                    [1j * sqrt_term, A           ]])
    return U_H, alpha, A


# ── 3. Projector-controlled phase rotation ────────────────
def projector_phase(phi, dim_sys=DIM_SYS):
    """
    R(phi) = exp(i*phi*(2*Pi - I)) with Pi = |0><0|_anc kron I_sys.
    Returns (2*dim_sys) x (2*dim_sys) diagonal unitary.
    """
    top = np.exp( 1j * phi) * np.eye(dim_sys, dtype=complex)
    bot = np.exp(-1j * phi) * np.eye(dim_sys, dtype=complex)
    return np.block([[top, np.zeros_like(top)],
                     [np.zeros_like(bot), bot]])


# ── 4. Compose the QSVT unitary ───────────────────────────
def build_qsvt_unitary(U_H, angles):
    """
    Compose:
        U_QSVT = R(phi_0) U_H R(phi_1) U_H R(phi_2) ... U_H R(phi_d)
    angles has length d+1.
    """
    d = len(angles) - 1
    dim_sys = U_H.shape[0] // 2
    U = projector_phase(angles[-1], dim_sys)              # rightmost R(phi_d)
    for k in range(d - 1, -1, -1):
        U = U_H @ U
        U = projector_phase(angles[k], dim_sys) @ U
    return U


# ── 5. Exact polynomial of A via diagonalization ──────────
def exact_polynomial_of_A(A, angles, sample_grid=None):
    """
    Apply the QSP polynomial implied by `angles` directly to A:
        f(A) = V * diag(P(lambda_i)) * V^dagger
    where P is computed by polynomial_from_angles on a sample grid
    and lambda_i are eigenvalues of A.
    """
    eigs, V = np.linalg.eigh(A)
    P_at_eigs = polynomial_from_angles(angles, eigs)
    return V @ np.diag(P_at_eigs) @ V.conj().T


# ── Reference / validation script ─────────────────────────
def main(h=1.0, deg=2, verbose=True):
    angles = chebyshev_qsp_angles(deg)

    # 1. Build H, eigenvalues
    H = build_tfim_hamiltonian(h)
    eigs_H, V_H = np.linalg.eigh(H)

    if verbose:
        print("=" * 60)
        print(f"  TFIM Hamiltonian  H = -ZZ - h*(XI+IX),  h={h}")
        print("=" * 60)
        print(f"H =\n{np.round(H.real, 4)}")
        print(f"eigenvalues(H) = {np.round(eigs_H, 6)}")
        print(f"||H||_2 = {np.max(np.abs(eigs_H)):.6f}")

    # 2. Block encoding
    U_H, alpha, A = build_block_encoding(H)

    be_unitary_err = np.linalg.norm(U_H @ U_H.conj().T - np.eye(DIM_BE))
    be_block_err   = np.linalg.norm(U_H[:DIM_SYS, :DIM_SYS] - A)
    if verbose:
        print("\n--- Block encoding ---")
        print(f"alpha = ||H||_2 = {alpha:.6f}")
        print(f"||U_H U_H^dagger - I||_F = {be_unitary_err:.2e}")
        print(f"||U_H[:4,:4] - H/alpha||_F = {be_block_err:.2e}")

    # 3, 4. QSVT unitary
    U_qsvt = build_qsvt_unitary(U_H, angles)
    qsvt_unitary_err = np.linalg.norm(U_qsvt @ U_qsvt.conj().T - np.eye(DIM_BE))

    # 5. Exact polynomial
    f_A_exact = exact_polynomial_of_A(A, angles)
    top_left  = U_qsvt[:DIM_SYS, :DIM_SYS]
    diff_qsvt_vs_exact = np.linalg.norm(top_left - f_A_exact)

    if verbose:
        print("\n--- QSVT unitary ---")
        print(f"angles (d={deg}): {np.round(angles, 6)}")
        print(f"||U_QSVT U_QSVT^dagger - I||_F = {qsvt_unitary_err:.2e}")
        print(f"||top_left(U_QSVT) - f(H/alpha)||_F = {diff_qsvt_vs_exact:.2e}")

    # 6. Per-eigenstate filter check
    if verbose:
        print("\n--- Per-eigenstate filter fidelity ---")
        print(f"{'lambda(H)':>12}  {'lambda(A)':>12}  {'|P(lambda)|^2':>14}  "
              f"{'|<lam|TopLeft|lam>|^2':>22}")
        for k in range(DIM_SYS):
            v_k = V_H[:, k]
            lam_A = eigs_H[k] / alpha
            p_val = polynomial_from_angles(angles, np.array([lam_A]))[0].real
            amp_eigen = v_k.conj() @ top_left @ v_k
            print(f"{eigs_H[k]:>12.6f}  {lam_A:>12.6f}  "
                  f"{abs(p_val)**2:>14.6f}  {abs(amp_eigen)**2:>22.6f}")

    return {
        "H": H, "eigs_H": eigs_H, "V_H": V_H,
        "alpha": alpha, "A": A,
        "U_H": U_H, "U_qsvt": U_qsvt,
        "f_A_exact": f_A_exact,
        "angles": angles,
        "diff_qsvt_vs_exact": diff_qsvt_vs_exact,
    }


if __name__ == "__main__":
    main(h=1.0, deg=2)
    print()
    main(h=1.0, deg=3)
    print()
    out = main(h=1.0, deg=6)
    print()
    # Print T_6(eigvals(A)) so we know what the d=6 filter does.
    A = out["A"]; eigs_A, _ = np.linalg.eigh(A)
    print("--- T_6 evaluated on eigenvalues of A=H/alpha ---")
    for la in eigs_A:
        print(f"  lambda(A) = {la:+.6f}   T_6(lambda) = "
              f"{np.cos(6 * np.arccos(np.clip(la, -1, 1))):+.6f}")
