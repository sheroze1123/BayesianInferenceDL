import sys
sys.path.append('../')

import matplotlib; matplotlib.use('macosx')
import time
import numpy as np
import matplotlib.pyplot as plt
import dolfin as dl; dl.set_log_level(40)

# Tensorflow related imports
from tensorflow.keras.optimizers import Adam

# Scipy imports
from scipy.optimize import minimize, Bounds

# ROMML imports
from fom.forward_solve import Fin
from fom.thermal_fin import get_space
from rom.averaged_affine_ROM import AffineROMFin 
from deep_learning.dl_model import load_parametric_model_avg
from gaussian_field import make_cov_chol

resolution = 40
V = get_space(resolution)
chol = make_cov_chol(V, length=1.2)

# Setup DL error model
err_model = load_parametric_model_avg('elu', Adam, 0.0003, 5, 58, 200, 2000, V.dim())

# Initialize reduced order model
phi = np.loadtxt('../data/basis_five_param.txt',delimiter=",")
solver_r = AffineROMFin(V, err_model, phi)

# Setup synthetic observations
solver = Fin(V)
z_true = dl.Function(V)
norm = np.random.randn(len(chol))
nodal_vals = np.exp(0.5 * chol.T @ norm)
z_true.vector().set_local(nodal_vals)
w, y, A, B, C = solver.forward(z_true)
obs_data = solver.qoi_operator(w)
solver_r.set_data(obs_data)

z = dl.Function(V)
norm = np.random.randn(len(chol))
z_0_nodal_vals = np.exp(0.5 * chol.T @ norm)
z.vector().set_local(z_0_nodal_vals)
p = dl.plot(z)
plt.colorbar(p)
plt.savefig('z_0.png')
plt.cla()
plt.clf()

class SolverWrapper:
    def __init__(self, err_model, solver_r): 
        self.err_model = err_model
        self.solver_r = solver_r
        self.cost = None
        self.grad = None
        self.z = dl.Function(V)
        self.z_t = dl.TestFunction(V)
        self.gamma = dl.Constant(0.001)
        self.grad_reg = self.gamma * dl.inner(dl.grad(self.z), dl.grad(self.z_t)) * dl.dx
        self.reg = 0.5 * self.gamma * dl.inner(dl.grad(self.z), dl.grad(self.z)) * dl.dx

    def cost_function(self, z_v):
        self.z.vector().set_local(z_v)
        self.grad, self.cost = self.solver_r.grad_romml(self.z)
        cost  = self.cost + dl.assemble(self.reg)
        return cost

    def gradient(self, z_v):
        self.z.vector().set_local(z_v)
        self.grad, self.cost = self.solver_r.grad_romml(self.z)
        grad = self.grad + self.grad_reg_eval(z_v)
        return grad

    def grad_reg_eval(self, z_v):
        self.z.vector().set_local(z_v)
        return dl.assemble(self.grad_reg)[:]

solver_w = SolverWrapper(err_model, solver_r)

norm = np.random.randn(len(chol))
eps_z = np.exp(0.5 * chol.T @ norm)
z.vector().set_local(eps_z)
eps_norm = np.sqrt(dl.assemble(dl.inner(z,z)*dl.dx))
eps_z = eps_z/eps_norm
norm = np.random.randn(len(chol))
z_ = np.exp(0.5 * chol.T @ norm)
n_eps = 16
hs = np.power(2., -np.arange(n_eps))
dir_grad = np.dot(solver_w.gradient(z_), eps_z)
err_grads = []
for h in hs:
    a_g = (solver_w.cost_function(z_) - solver_w.cost_function(z_ + h * eps_z))/h
    err = abs(a_g - dir_grad)
    err_grads.append(err)

plt.loglog(hs, err_grads, "-ob", label="Error Grad")
plt.loglog(hs, (.5*err_grads[0]/hs[0])*hs, "-.k", label="First Order")
print(f"Directional gradient: {dir_grad}")
print(f"Errors in approx gradients: {err_grads}")
plt.savefig('grad_test.png', dpi=200)
plt.cla()
plt.clf()


bounds = Bounds(0.1, 4)
res = minimize(solver_w.cost_function, z_0_nodal_vals, 
        method='L-BFGS-B', 
        jac=solver_w.gradient,
        bounds=bounds,
        tol=1e-9)

print(f'status: {res.success}, message: {res.message}, n_it: {res.nit}, cost_min: {res.fun}')
print(f'relative error: {res.fun/np.linalg.norm(obs_data)}')
z.vector().set_local(res.x)

w,y, _, _, _ = solver.forward(z)
pred_obs = solver.qoi_operator(w)
obs_err = np.linalg.norm(pred_obs - obs_data)
print(f"True: {obs_data}\n Pred: {pred_obs}")
print(f"Observation error: {obs_err}")

p = dl.plot(z)
plt.colorbar(p)
plt.savefig("z_map.png")
plt.cla()
plt.clf()

p = dl.plot(z_true)
plt.colorbar(p)
plt.savefig("z_true.png")

reconst_err = dl.assemble(dl.inner(z - z_true, z - z_true) * dl.dx)
z_true_norm = dl.assemble(dl.inner(z_true, z_true) * dl.dx)
rel_r_err = reconst_err/z_true_norm
print(f"Relative reconstruction error: {rel_r_err * 100} %")
print(f"Reconstruction error: {reconst_err}")
