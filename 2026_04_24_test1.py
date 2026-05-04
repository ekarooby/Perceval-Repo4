import perceval as pcvl
from perceval import RemoteProcessor
from perceval.components import PS, BS

TOKEN = "_T_eyJhbGciOiJIUzUxMiIsInR5cCI6IkpXVCJ9.eyJpZCI6MjU5MiwiZXhwIjoxODA2NTkwNjc1LjE4NjQ3NDN9.eCJCJNcF7BqyUtKiUSaOPK1RM0Nd42B-2UXwH0IIRiYy0TOqi9eQWcUM3049KIkcjgyKkHT-xyIxzoqzmuxyaQ"
remote_processor = RemoteProcessor("qpu:belenos", token=TOKEN)
arch = remote_processor.specs["architecture"]
belenos_circuit = arch.unitary_circuit()

# ── Print ONLY mode 0 PSs and (0,1) BSs in order ──────────
print("=== Mode 0 PSs and BSs on modes (0,1) in order ===")
count_ps_m0 = 0
count_bs = 0
for i, (r, c) in enumerate(belenos_circuit):
    if r == (0,) and isinstance(c, PS):
        count_ps_m0 += 1
        print(f"  {i:4d}  PS mode 0  #{count_ps_m0}  {c.describe()}")
    elif r == (0, 1):
        count_bs += 1
        print(f"  {i:4d}  BS (0,1)   #{count_bs}  {c.describe()}")
    if count_bs > 6:  # only show first 6 BSs
        break

print(f"\nTotal mode 0 PSs seen: {count_ps_m0}")
print(f"Total BSs seen: {count_bs}")

# Add this to your test code
print("\n=== Mode 1 PSs in order ===")
count_ps_m1 = 0
count_bs = 0
for i, (r, c) in enumerate(belenos_circuit):
    if r == (0, 1):
        count_bs += 1
    if r == (1,) and isinstance(c, PS):
        count_ps_m1 += 1
        print(f"  {i:4d}  PS mode 1  #{count_ps_m1}  after BS #{count_bs}  {c.describe()}")
    if count_bs > 6:
        break