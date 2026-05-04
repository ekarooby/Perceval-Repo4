import perceval as pcvl
from perceval import RemoteProcessor
from perceval.components import PS
from math import pi

TOKEN = "_T_eyJhbGciOiJIUzUxMiIsInR5cCI6IkpXVCJ9.eyJpZCI6MjU5MiwiZXhwIjoxODA2NTkwNjc1LjE4NjQ3NDN9.eCJCJNcF7BqyUtKiUSaOPK1RM0Nd42B-2UXwH0IIRiYy0TOqi9eQWcUM3049KIkcjgyKkHT-xyIxzoqzmuxyaQ"
remote_processor = RemoteProcessor("qpu:belenos", token=TOKEN)

# ── Get Belenos chip architecture ─────────────────────────
arch = remote_processor.specs["architecture"]
print(f"Architecture type: {type(arch)}")
print(f"Architecture attributes: {[x for x in dir(arch) if not x.startswith('_')]}")

# ── Try to get the circuit from architecture ───────────────
try:
    belenos_circuit = arch.unitary_circuit()
    print(f"\nCircuit type: {type(belenos_circuit)}")
    print(f"Circuit size: {belenos_circuit.m} modes")
    print("\n=== Components on modes 0 and 1 ===")
    for i, (r, c) in enumerate(belenos_circuit):
        if len(r) == 1 and r[0] in [0, 1]:
            print(f"  {i:4d}  modes={r}  {c.describe()}")
        elif len(r) == 2 and 0 in r and 1 in r:
            print(f"  {i:4d}  modes={r}  {c.describe()}")
except Exception as e:
    print(f"unitary_circuit() failed: {e}")

# ── Also try components directly ──────────────────────────
try:
    print("\n=== arch.components on modes 0 and 1 ===")
    for i, (r, c) in enumerate(arch.components):
        if len(r) == 1 and r[0] in [0, 1]:
            print(f"  {i:4d}  modes={r}  {c.describe()}")
        elif len(r) == 2 and 0 in r and 1 in r:
            print(f"  {i:4d}  modes={r}  {c.describe()}")
except Exception as e:
    print(f"arch.components failed: {e}")