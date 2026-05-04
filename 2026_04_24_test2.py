import numpy as np
import perceval as pcvl
import perceval.components as comp

theta = 3.13507
U = np.array(comp.BS.Ry(theta=theta).compute_unitary())
print(f"BS.Ry(theta={theta}) unitary:")
print(np.round(U, 6))
print(f"\nIs it close to -Identity? {np.allclose(U, -np.eye(2), atol=0.01)}")
print(f"Is it close to +Identity? {np.allclose(U, np.eye(2), atol=0.01)}")