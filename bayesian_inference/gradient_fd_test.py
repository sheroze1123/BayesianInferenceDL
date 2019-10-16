import sys
sys.path.append('../')

import matplotlib; matplotlib.use('macosx')
import time
import numpy as np
import matplotlib.pyplot as plt
import dolfin as dl; dl.set_log_level(40)

# ROMML imports
from fom.forward_solve import Fin
from fom.thermal_fin import get_space
from rom.averaged_affine_ROM import AffineROMFin 
from deep_learning.dl_model import load_parametric_model_avg
from gaussian_field import make_cov_chol

resolution = 40
V = get_space(resolution)
chol = make_cov_chol(V, length=1.2)
solver = Fin(V)

class SolverWrapper:
    def __init__(self, solver, data): 
        self.solver = solver
        self.data = data
        self.z = dl.Function(V)

    def cost_function(self, z_v):
        self.z.vector().set_local(z_v)
        w, y, A, B, C = self.solver.forward(self.z)
        y = self.solver.qoi_operator(w)
        cost = 0.5 * np.linalg.norm(y - self.data)**2 + dl.assemble(self.solver.reg)
        return cost

    def gradient(self, z_v):
        self.z.vector().set_local(z_v)
        grad = self.solver.gradient(self.z, self.data) + dl.assemble(self.solver.grad_reg)[:]
        return grad

# Setup synthetic observations
solver = Fin(V)

z_true = dl.Function(V)
norm = np.random.randn(len(chol))
nodal_vals = np.exp(0.5 * chol.T @ norm)
z_true.vector().set_local(nodal_vals)
w, y, A, B, C = solver.forward(z_true)
data = solver.qoi_operator(w)

solver_w = SolverWrapper(solver, data)

z = dl.Function(V)
norm = np.random.randn(len(chol))
eps_z = np.exp(0.5 * chol.T @ norm)
z.vector().set_local(eps_z)
eps_norm = np.sqrt(dl.assemble(dl.inner(z,z)*dl.dx))
eps_norm = np.linalg.norm(eps_z)
eps_z = eps_z/eps_norm
norm = np.random.randn(len(chol))
z_ = np.exp(0.5 * chol.T @ norm)
n_eps = 32
hs = np.power(2., -np.arange(n_eps))
dir_grad = np.dot(solver_w.gradient(z_), eps_z)

print(f"Directional gradient: {dir_grad}")

err_grads = []
grads = []
for h in hs:
    pi_h = solver_w.cost_function(z_ + h * eps_z)
    pi_0 = solver_w.cost_function(z_)
    a_g = (pi_h - pi_0)/h
    grads.append(a_g)
    err = abs(a_g - dir_grad)/abs(dir_grad)
    err_grads.append(err)

plt.loglog(hs, err_grads, "-ob", label="Error Grad")
plt.loglog(hs, (.5*err_grads[0]/hs[0])*hs, "-.k", label="First Order")
plt.savefig('grad_test.png', dpi=200)
plt.cla()
plt.clf()

print(f"FD gradients: {grads}")
print(f"Errors: {err_grads}")
plt.plot(hs, grads, "-ob")
plt.savefig('gradients.png')
