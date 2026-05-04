# ============================================================
# FETCH + ANALYZE the 4 d=6 QSVT jobs that were submitted
# accidentally on 2026-04-24 ~22:12 to qpu:belenos.
#
# Mapping (by submission time, newest-on-top in Quandela UI):
#   top    -- 10:12:48 -- k=3 (lambda = +2.236)
#   2nd    -- 10:12:44 -- k=2 (lambda = +1.000)
#   3rd    -- 10:12:40 -- k=1 (lambda = -1.000)
#   bottom -- 10:12:36 -- k=0 (lambda = -2.236)
#
# Reads results only -- does NOT submit anything new.
# ============================================================

import os, re, time
import numpy as np
import matplotlib.pyplot as plt
import perceval as pcvl
from perceval import RemoteProcessor

from qsvt_angles import chebyshev_qsp_angles
from qsvt_tfim_matrix_reference import (
    build_tfim_hamiltonian, build_block_encoding,
    build_qsvt_unitary, DIM_SYS, DIM_BE,
)

TOKEN = "_T_eyJhbGciOiJIUzUxMiIsInR5cCI6IkpXVCJ9.eyJpZCI6MjU5MiwiZXhwIjoxODA2NTkwNjc1LjE4NjQ3NDN9.eCJCJNcF7BqyUtKiUSaOPK1RM0Nd42B-2UXwH0IIRiYy0TOqi9eQWcUM3049KIkcjgyKkHT-xyIxzoqzmuxyaQ"

# k -> job_id (submission order: k=0 first => bottom of UI list)
JOB_IDS = {
    0: "28cf3a16-ce34-4353-ad91-d38dcc8dbb9c",   # 10:12:36
    1: "437a7a53-4b1b-4eb2-abc6-edb82e5566e3",   # 10:12:40
    2: "0604f60b-48db-4fae-bf88-e4827a76983e",   # 10:12:44
    3: "8df96ed5-0070-45fc-a604-99bc1ddea4f6",   # 10:12:48
}

# ── Theory predictions (d=6 T_6 filter on TFIM h=1) ──
H = build_tfim_hamiltonian(1.0)
eigs_H, V_H = np.linalg.eigh(H)
U_H, alpha, A = build_block_encoding(H)
T6 = np.cos(6 * np.arccos(np.clip(eigs_H / alpha, -1, 1)))
target_p = T6 ** 2
print("Theory:")
print(f"  H eigenvalues          = {np.round(eigs_H, 4)}")
print(f"  T_6(eigvals/alpha)     = {np.round(T6, 4)}")
print(f"  |T_6|^2 (target P)     = {np.round(target_p, 4)}")

# ── Connect & fetch ──
pcvl.RemoteConfig.set_token(TOKEN)
pcvl.RemoteConfig().save()
rp = RemoteProcessor("qpu:belenos")

qpu_p_filter = np.full(4, np.nan)
qpu_total    = np.zeros(4, dtype=int)
all_counts   = {}

for k in range(4):
    jid = JOB_IDS[k]
    print(f"\n--- k={k} (lambda={eigs_H[k]:+.4f})  job_id={jid} ---")
    job = rp.resume_job(jid)
    results = job.get_results()
    raw = results["results"]
    total = sum(raw.values())
    qpu_total[k] = total

    # filter success = at least one photon detected in modes 0..3
    n_succ = 0
    for state, cnt in raw.items():
        occ = list(state)
        if any(occ[m] > 0 for m in range(DIM_SYS)):
            n_succ += cnt
    p = n_succ / total if total else 0.0
    qpu_p_filter[k] = p
    all_counts[k] = dict(raw)

    print(f"  total detected events  = {total}")
    print(f"  P(modes 0..3) measured = {p:.4f}")
    print(f"  |T_6(lambda/alpha)|^2  = {target_p[k]:.4f}")
    print(f"  |measured - target|    = {abs(p - target_p[k]):.4f}")

    # top-5 outcomes
    print(f"  top-5 outcomes:")
    for state, cnt in sorted(raw.items(), key=lambda kv: -kv[1])[:5]:
        print(f"    {str(state):<55} {cnt:>5}  ({cnt/total:.4f})")

# ── Plot ──
fig, ax = plt.subplots(figsize=(8, 5))
x = np.arange(4); w = 0.35
ax.bar(x - w/2, target_p,    w, label=r"target $|T_6(\lambda/\alpha)|^2$",
       color="#3060c0")
ax.bar(x + w/2, qpu_p_filter, w,
       label=f"Belenos QPU (~2000 shots/job, total events shown)",
       color="#a02050")
for k in range(4):
    ax.text(x[k] + w/2, qpu_p_filter[k] + 0.02,
            f"N={qpu_total[k]}", ha="center", fontsize=8)
ax.set_xticks(x)
ax.set_xticklabels([f"k={k}\n$\\lambda$={eigs_H[k]:+.3f}" for k in range(4)])
ax.set_ylabel("P(filter success | eigenstate k)")
ax.set_title(r"QSVT eigenvalue filter on Belenos -- $T_6(H/\alpha)$, h=1")
ax.set_ylim(0, 1.15)
ax.grid(True, axis="y", alpha=0.3)
ax.legend()
fig.tight_layout()
out_png = os.path.join(os.path.dirname(__file__), "results_qsvt_d6_belenos.png")
out_npz = os.path.join(os.path.dirname(__file__), "results_qsvt_d6_belenos.npz")
fig.savefig(out_png, dpi=140)
np.savez(
    out_npz,
    eigs_H=eigs_H, alpha=alpha,
    T6_eigs=T6, target_probabilities=target_p,
    qpu_p_filter=qpu_p_filter, qpu_total_events=qpu_total,
    job_ids=np.array([JOB_IDS[k] for k in range(4)]),
)
print(f"\nSaved plot: {out_png}")
print(f"Saved data: {out_npz}")
