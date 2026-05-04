# ============================================================
# QUANDELA TUTORIAL EXAMPLE -- RECK DECOMPOSITION OF A RANDOM
# n x n UNITARY INTO A PHOTONIC CIRCUIT OF BSs AND PSs
# ============================================================
#
# GOAL:
#   Demonstrate that ANY n x n unitary matrix can be decomposed
#   into a physical photonic circuit made of Beam Splitters (BSs)
#   and Phase Shifters (PSs) using the Reck decomposition
#   (triangle shape). This is the foundation of universal
#   linear optical quantum computing.
#
# BACKGROUND:
#   Since Reck et al. (1994), we know that any unitary matrix
#   U(n) can be implemented as a linear optical circuit using
#   only BSs and PSs arranged in a triangular mesh. Perceval
#   implements this decomposition in Circuit.decomposition().
#
# WHAT THIS CODE DOES:
#   1. Generates a random n x n unitary matrix (n=4 here)
#   2. Decomposes it into a triangular mesh of BSs and PSs
#      using BS(theta, phi_tr) as the building block
#   3. Displays the resulting photonic circuit diagram
#
# BUILDING BLOCK:
#   BS(theta=P('theta'), phi_tr=P('phi')) -- a beam splitter
#   with two free parameters: theta (reflectivity angle) and
#   phi_tr (phase on the top-right port). This is the standard
#   MZI building block for Reck decomposition.
#
# NOTE ON RANDOMNESS:
#   pcvl.Circuit.decomposition() uses a numerical search
#   starting from a random point to find the BS/PS angles.
#   Each run may give different angle values but all implement
#   the same unitary U. To get reproducible results, add:
#       pcvl.random_seed(42)
#   before running this code.
#
# NOTE ON n:
#   Change n to decompose any size unitary:
#   n=2 --> 1 BS + 2 PSs   (our QSP case)
#   n=4 --> 6 BSs + 8 PSs  (this example)
#   n=N --> N(N-1)/2 BSs   (general Reck formula)
#
# REFERENCE:
#   Reck et al., "Experimental realization of any discrete
#   unitary operator", PRL 73, 58 (1994)
#   Quandela Perceval tutorial:
#   https://perceval.quandela.net/docs/v0.13/notebooks/Tutorial.html
# ============================================================

import perceval as pcvl
from perceval.components import BS, PS

# ── Set seed for reproducibility ──────────────────────────
pcvl.random_seed(42)

# ── Generate a random n x n unitary matrix ────────────────
# Change n to decompose any size unitary
n = 4
U = pcvl.Matrix.random_unitary(n)

# ── Decompose into BS + PS circuit (Reck decomposition) ───
# Building block: BS with theta and phi_tr as free parameters
# phase_shifter_fn=PS adds a layer of PSs at the output
# shape defaults to "triangle" (Reck triangular mesh)
decomposed_circuit = pcvl.Circuit.decomposition(
    U,
    BS(theta=pcvl.P('theta'), phi_tr=pcvl.P('phi')),
    phase_shifter_fn=PS
)

# ── Display the decomposed circuit diagram ────────────────
# Renders inline in Jupyter/VS Code notebook
pcvl.pdisplay(decomposed_circuit)