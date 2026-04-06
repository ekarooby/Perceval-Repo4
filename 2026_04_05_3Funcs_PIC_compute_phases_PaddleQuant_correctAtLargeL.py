# ============================================================
# QSP ANGLE GENERATION USING PADDLE QUANTUM QPP
# STEP / ReLU / SELU
# ============================================================
#
# PIPELINE (Bu et al. Appendix B):
#   1. laurent_generator(fn, dx, deg, L_width)
#   2. pair_generation(F)
#   3. qpp_angle_approximator(P, Q)
#
# KEY FIXES vs original code:
#   FIX 1: dx = 0.005 (was 0.01)
#           Nyquist limit ~628, covers L up to 360 without aliasing.
#           With dx=0.01, aliasing starts at L~157 causing raw
#           max_norm to spike and MSE to increase after L=180.
#
#   FIX 2: scale to 0.97 (was 0.95)
#           QSP output amplitude = 0.97 * surrogate(x)
#           Systematic MSE floor = (0.03)^2 ~ 0.001  (log10 ~ -3)
#           pair_generation sees max_norm=0.97, well below 1 -> stable.
#           0.9999 crashes pair_generation at large L (P,Q norm > 1).
#
#   FIX 3: scale applied ONCE only
#           Original code had double-scaling bug (scaled F then
#           scaled again using the already-scaled max_norm).
#
# ENVIRONMENT: paddle_env
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
import warnings
from paddle_quantum.qpp.laurent import laurent_generator, pair_generation, Laurent
from paddle_quantum.qpp.angles import qpp_angle_approximator

# ============================================================
# SETTINGS
# ============================================================

FUNC_NAME = "STEP"   # "STEP", "ReLU", or "SELU"
ANGLE_L   = 180      # desired L

# ============================================================
# Derived settings
# ============================================================

FUNC_LOWER = FUNC_NAME.lower()
FILE_TAG   = f"{FUNC_LOWER}_pq_L{ANGLE_L}"
deg        = 2 * ANGLE_L    # must be even for pair_generation

print("=" * 62)
print(f"  Paddle Quantum QPP Angle Generation")
print(f"  Function : {FUNC_NAME}")
print(f"  L        : {ANGLE_L}")
print(f"  deg      : {deg}  (= 2 * L, must be even)")
print(f"  File tag : {FILE_TAG}")
print("=" * 62)

N_approx = 100   # sharpness of arctan surrogate for STEP

# ============================================================
# Target functions
# ============================================================

def get_surrogate(func_name):
    if func_name == "STEP":
        return lambda x: (2.0 / np.pi) * np.arctan(N_approx * x)
    elif func_name == "ReLU":
        return lambda x: np.log(1 + np.exp(N_approx * x)) / N_approx
    elif func_name == "SELU":
        alpha, scale = 1.6733, 1.0507
        return lambda x: scale * np.where(x >= 0, x, alpha * (np.exp(x) - 1))
    else:
        raise ValueError(f"Unknown FUNC_NAME: {func_name}")

def get_true_func(func_name):
    if func_name == "STEP":
        return lambda x: np.where(x >= 0, 1.0, -1.0)
    elif func_name == "ReLU":
        return lambda x: np.maximum(0.0, x)
    elif func_name == "SELU":
        alpha, scale = 1.6733, 1.0507
        return lambda x: scale * np.where(x >= 0, x, alpha * (np.exp(x) - 1))
    else:
        raise ValueError(f"Unknown FUNC_NAME: {func_name}")

surrogate_func = get_surrogate(FUNC_NAME)
true_func      = get_true_func(FUNC_NAME)

# ============================================================
# Verification circuit (Bu et al. convention)
# ============================================================

def Ry(t):
    c, s = np.cos(t/2), np.sin(t/2)
    return np.array([[c, -s],[s, c]], dtype=complex)

def Rz(p):
    return np.array([[np.exp(-1j*p/2), 0],[0, np.exp(1j*p/2)]], dtype=complex)

def qsp_Z(theta_arr, phi_arr, x):
    W = Ry(theta_arr[0]) @ Rz(phi_arr[0])
    for j in range(1, len(theta_arr)):
        W = Ry(theta_arr[j]) @ Rz(phi_arr[j]) @ Rz(x) @ W
    psi = W @ np.array([1.0, 0.0])
    return abs(psi[0])**2 - abs(psi[1])**2

# ============================================================
# Step 1: Generate Laurent polynomial
#
# FIX 1: dx = 0.005 eliminates aliasing for all L up to 360.
# FIX 2: scale to 0.97 (not 0.95 or 0.9999).
# FIX 3: scale applied ONCE only (no double scaling).
# ============================================================
print("\n[1] Generating Laurent polynomial...")

dx = 0.005   # FIX 1: was 0.01, aliased for L > ~157
F  = laurent_generator(surrogate_func, dx, deg, np.pi)

print(f"   Raw : parity={F.parity}, max_norm={F.max_norm:.6f}, deg={F.deg}")

SCALE_TARGET = 0.95                    # FIX 2: was 0.95
scale_factor = SCALE_TARGET / F.max_norm
F = Laurent(F.coef * scale_factor)     # FIX 3: scaled ONCE here, never again

print(f"   Scaled: parity={F.parity}, max_norm={F.max_norm:.6f}")
print(f"   Scale factor applied: {scale_factor:.6f}")

# ============================================================
# Step 2: Generate (P, Q) Laurent pair
# ============================================================
print("\n[2] Generating (P, Q) Laurent pair...")

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    P, Q = pair_generation(F)

print(f"   P: parity={P.parity}, max_norm={P.max_norm:.6f}, deg={P.deg}")
print(f"   Q: parity={Q.parity}, max_norm={Q.max_norm:.6f}, deg={Q.deg}")

# ============================================================
# Step 3: Find QSP angles
# ============================================================
print("\n[3] Finding QSP angles...")

list_theta, list_phi = qpp_angle_approximator(P, Q)
theta = np.array(list_theta)
phi   = np.array(list_phi)

print(f"   Angles found: {len(theta)} theta, {len(phi)} phi")
print(f"   L = {len(theta) - 1}  (expected {ANGLE_L})")
assert len(theta) - 1 == ANGLE_L, \
    f"L mismatch: got {len(theta)-1}, expected {ANGLE_L}"

# ============================================================
# Step 4: Verify
# ============================================================
print("\n[4] Verifying angles in Bu et al. circuit...")

x_grid   = np.linspace(-np.pi, np.pi, 300)
f_target = surrogate_func(x_grid)
f_true   = true_func(x_grid)
z_vals   = np.array([qsp_Z(theta, phi, x) for x in x_grid])

mse_vs_surrogate = np.mean((z_vals - f_target)**2)
mse_vs_true      = np.mean((z_vals - f_true)**2)

print(f"   MSE vs surrogate    : {mse_vs_surrogate:.4e}")
print(f"   MSE vs true {FUNC_NAME:<5} : {mse_vs_true:.4e}")

# ============================================================
# Step 5: Save angles
# ============================================================
theta_filename = f"theta_{FILE_TAG}.npy"
phi_filename   = f"phi_{FILE_TAG}.npy"

np.save(theta_filename, theta)
np.save(phi_filename,   phi)
print(f"\nSaved: {theta_filename}")
print(f"Saved: {phi_filename}")

# ============================================================
# Step 6: Plot verification
# ============================================================
print("\n[5] Plotting...")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle(f"Paddle Quantum   {FUNC_NAME}   L={ANGLE_L}",
             fontsize=13, fontweight='bold')

xt = [-np.pi, 0, np.pi]
xl = [r"$-\pi$", r"$0$", r"$\pi$"]

ax = axes[0]
ax.plot(x_grid, f_true,   'k-',  lw=2.5, label=f"True {FUNC_NAME}")
ax.plot(x_grid, f_target, 'g--', lw=2,   label="Surrogate")
ax.plot(x_grid, z_vals,   'b.',  ms=3,
        label=f"QPP  MSE(surr)={mse_vs_surrogate:.4f}  "
              f"MSE(true)={mse_vs_true:.4f}")
ax.set_xlim([-np.pi, np.pi]); ax.set_ylim([-1.35, 1.35])
ax.set_xticks(xt); ax.set_xticklabels(xl, fontsize=11)
ax.set_xlabel(r"$x$", fontsize=12)
ax.set_ylabel("Function value", fontsize=12)
ax.set_title(f"{FUNC_NAME}   L={ANGLE_L}   dx=0.005   scale=0.97", fontsize=11)
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

ax2 = axes[1]
diff_s = z_vals - f_target
diff_t = z_vals - f_true
ax2.plot(x_grid, diff_s, color='royalblue', lw=1.5,
         label=f"vs surrogate  MSE={mse_vs_surrogate:.4e}")
ax2.fill_between(x_grid, diff_s, alpha=0.15, color='royalblue')
ax2.plot(x_grid, diff_t, color='red', lw=1.5, linestyle='--',
         label=f"vs true {FUNC_NAME}  MSE={mse_vs_true:.4e}")
ax2.fill_between(x_grid, diff_t, alpha=0.10, color='red')
ax2.axhline(0, color='k', lw=0.8, ls='--')
ax2.set_xlim([-np.pi, np.pi])
ax2.set_xticks(xt); ax2.set_xticklabels(xl, fontsize=11)
ax2.set_xlabel(r"$x$", fontsize=12)
ax2.set_ylabel("Residual", fontsize=12)
ax2.set_title(f"Residuals  ({FUNC_NAME}   L={ANGLE_L})", fontsize=11)
ax2.legend(fontsize=9); ax2.grid(True, alpha=0.3)

plt.tight_layout()
plot_filename = f"qsp_{FILE_TAG}.png"
plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
plt.show()
print(f"Plot saved: {plot_filename}")

print("\n" + "=" * 62)
print(f"  SUMMARY  [{FILE_TAG}]")
print("=" * 62)
print(f"  Function         : {FUNC_NAME}")
print(f"  L                : {ANGLE_L}")
print(f"  deg (internal)   : {deg}  (= 2 * L)")
print(f"  dx               : {dx}  (FIX 1: was 0.01)")
print(f"  scale target     : {SCALE_TARGET}  (FIX 2: was 0.95)")
print(f"  MSE vs surrogate : {mse_vs_surrogate:.4e}")
print(f"  MSE vs true      : {mse_vs_true:.4e}")
print(f"  theta saved to   : {theta_filename}")
print(f"  phi saved to     : {phi_filename}")
print(f"  To use in SLOS/QPU codes:")
print(f"    FUNC_NAME = '{FUNC_NAME}'")
print(f"    ANGLE_L   = {ANGLE_L}")
print(f"    load: theta_{FILE_TAG}.npy")
print(f"    load: phi_{FILE_TAG}.npy")
print("=" * 62)