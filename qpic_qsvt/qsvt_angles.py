# ============================================================
# QSVT ANGLE GENERATION -- CHEBYSHEV FILTERS T_d(x)
# ============================================================
#
# GOAL:
#   Generate QSP/QSVT phase angles for a degree-d polynomial
#   filter applied to H/alpha. Used by qsvt_tfim_*.py.
#
# CONVENTION (Wx Re, matches our BE):
#   QSVT sequence (left -> right matrix product, right applied first):
#     U_QSVT = R(phi_0) U_H R(phi_1) U_H R(phi_2) ... U_H R(phi_d)
#   with R(phi) = exp(i*phi*(2*Pi - I))   [Pi = ancilla |0> projector]
#   and U_H = [[A, i*sqrt(I-A^2)], [i*sqrt(I-A^2), A]],  A = H/alpha.
#   Top-left 4x4 block of U_QSVT  ==  P_d(A).
#
#   For the Chebyshev polynomial T_d (P_d = T_d), the angles in
#   this convention are [pi/4, 0, 0, ..., 0, -pi/4] (d+1 entries).
#
#   Verified analytically for d=1,2,3 -- top-left equals T_d(A)
#   exactly (no boundary phase factor since e^{i pi/4} e^{-i pi/4} = 1).
#
# WHY T_d AS THE FILTER POLYNOMIAL:
#   T_d on [-1,1] oscillates between -1 and +1 with d roots.
#   For our TFIM at h=1, alpha ~ 2.414 and A=H/alpha has eigenvalues
#   roughly [-1, -0.414, 0.414, 1]. Then T_d(A) acts as an
#   eigenvalue filter:
#     T_2(A) maps eigvals to [1, -0.66, -0.66, 1]   -- high-pass
#     T_3(A) maps eigvals to [-1, 0.96, -0.96, 1]   -- sign-flip on small
#   Higher-degree minimax sign approximations give true step filters
#   (see pyqsp's poly_sign / poly_threshold) -- swap in via pyqsp_angles().
#
# REFERENCE:
#   Gilyen, Su, Low, Wiebe -- "Quantum singular value transformation"
#   Martyn, Rossi, Tan, Chuang -- "Grand unification of QSP/QSVT"
# ============================================================

import numpy as np


def chebyshev_qsp_angles(d):
    """
    QSP/QSVT phases that produce P_d(A) = T_d(A) in our convention.

    Returns array of length d+1 with [pi/4, 0, ..., 0, -pi/4].
    """
    if d < 1:
        raise ValueError("d must be >= 1")
    angles = np.zeros(d + 1)
    angles[0]  =  np.pi / 4
    angles[-1] = -np.pi / 4
    return angles


def pyqsp_angles(poly_name="sign", deg=3, **kwargs):
    """
    Try pyqsp for a real eigenvalue filter polynomial.
    Falls back to chebyshev_qsp_angles(deg) if pyqsp is missing.

    poly_name choices (when pyqsp is available):
       "sign"       -- minimax sign function approximation
       "threshold"  -- step at +- threshold
       "cheb"       -- pure T_d
    """
    try:
        import pyqsp
        from pyqsp import angle_sequence
        from pyqsp.poly import (PolySign, PolyThreshold, PolyTaylorSeries)

        if poly_name == "sign":
            poly = PolySign().generate(degree=deg, delta=kwargs.get("delta", 0.5))
        elif poly_name == "threshold":
            poly = PolyThreshold().generate(degree=deg,
                                            delta=kwargs.get("delta", 0.5))
        else:
            return chebyshev_qsp_angles(deg)

        phiset = angle_sequence.QuantumSignalProcessingPhases(
            poly, signal_operator="Wx", method="laurent")
        return np.array(phiset)

    except ImportError:
        print("[qsvt_angles] pyqsp not installed -- using Chebyshev T_d angles.")
        return chebyshev_qsp_angles(deg)
    except Exception as e:
        print(f"[qsvt_angles] pyqsp angle gen failed ({e}) -- using T_d angles.")
        return chebyshev_qsp_angles(deg)


def polynomial_from_angles(angles, x_grid):
    """
    Evaluate the scalar QSP polynomial implied by `angles` at points x_grid,
    in the *same* convention as build_qsvt_unitary in
    qsvt_tfim_matrix_reference.py:

        U_scalar = e^{i phi_0 Z} W_x e^{i phi_1 Z} W_x ... W_x e^{i phi_d Z}

    P(x) = <0| U_scalar |0> (top-left entry).

    This is what the top-left block of the QSVT unitary equals at each
    eigenvalue of A=H/alpha, so exact_polynomial_of_A() can use it
    directly.
    """
    angles = np.asarray(angles, dtype=float)
    d = len(angles) - 1
    out = np.zeros_like(x_grid, dtype=complex)
    Z   = np.array([[1, 0], [0, -1]], dtype=complex)
    I2  = np.eye(2, dtype=complex)

    def R(phi):
        return np.cos(phi) * I2 + 1j * np.sin(phi) * Z

    for i, x in enumerate(x_grid):
        s  = np.sqrt(max(0.0, 1 - x * x))
        Wx = np.array([[x, 1j * s], [1j * s, x]], dtype=complex)
        # Build U = R(phi_0) Wx R(phi_1) Wx ... Wx R(phi_d)
        # by starting from the right-most factor and pre-multiplying.
        U = R(angles[-1])
        for k in range(d - 1, -1, -1):
            U = Wx @ U
            U = R(angles[k]) @ U
        out[i] = U[0, 0]
    return out


if __name__ == "__main__":
    # Sanity check: angles [pi/4, 0, ..., 0, -pi/4] reproduce T_d exactly.
    # Verified analytically for any d >= 1:
    #   the inner Wx^d block has top-left = T_d(x) (Chebyshev recurrence),
    #   and the boundary factor e^{i pi/4} * e^{-i pi/4} = 1, so
    #   <0|U_scalar|0> = T_d(x). Lifting Wx -> U_H (8x8 BE), the same
    #   identity gives top-left 4x4 of U_QSVT = T_d(H/alpha).
    for d in [1, 2, 3, 6]:
        angles = chebyshev_qsp_angles(d)
        x = np.linspace(-1, 1, 21)
        P = polynomial_from_angles(angles, x).real
        Td = np.cos(d * np.arccos(x))
        err = np.max(np.abs(P - Td))
        print(f"d={d}  angles={np.round(angles, 4)}  "
              f"max|P(x)-T_d(x)|={err:.2e}")
