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

z_true = dl.interpolate(dl.Expression('0.3 + 0.1 * x[0] * x[1] + 0.5 * (sin(2*x[1])*cos(2*x[0]) + 1.5)', degree=2),V)

z_true = solver.nine_param_to_function([1.1, 1.11, 1.13, 1.12, 1.117, 1.1, 1.127, 1.118, 1.114])


w, y, A, B, C = solver.forward(z_true)
obs_data = solver.qoi_operator(w)
solver_r.set_data(obs_data)

v = np.linspace(0.25, 3.0, 15, endpoint=True)

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

class RSolverWrapper:
    def __init__(self, err_model, solver_r, solver): 
        self.err_model = err_model
        self.solver_r = solver_r
        self.z = dl.Function(V)
        self.solver = solver

    def cost_function(self, z_v):
        self.z.vector().set_local(z_v)
        self.grad, self.cost = self.solver_r.grad_romml(self.z)
        cost  = self.cost + dl.assemble(self.solver.reg)
        return cost

    def gradient(self, z_v):
        self.z.vector().set_local(z_v)
        self.grad, self.cost = self.solver_r.grad_romml(self.z)
        grad = self.grad + dl.assemble(self.solver.grad_reg)
        return grad

    def grad_reg_eval(self, z_v):
        self.z.vector().set_local(z_v)
        return dl.assemble(self.grad_reg)[:]

solver_w = RSolverWrapper(err_model, solver_r, solver)
#  solver_w = SolverWrapper(solver, obs_data)

bounds = Bounds(0.15, 5)
res = minimize(solver_w.cost_function, z_0_nodal_vals, 
        method='L-BFGS-B', 
        jac=solver_w.gradient,
        bounds=bounds,
        options={'ftol':1e-10, 'gtol':1e-8})

print(f'status: {res.success}, message: {res.message}, n_it: {res.nit}')
print(f'Minimum cost: {res.fun:.3F}')
z.vector().set_local(res.x)

w,y, _, _, _ = solver.forward(z)
pred_obs = solver.qoi_operator(w)
obs_err = np.linalg.norm(pred_obs - obs_data)
#  print(f"True: {obs_data}\n Pred: {pred_obs}")
print(f"Relative observation error: {obs_err/np.linalg.norm(obs_data)*100:.4f}%")

vmax=max(np.max(z_true.vector()[:]), np.max(res.x)) + 0.0005
vmin=min(np.min(z_true.vector()[:]), np.min(res.x)) - 0.0005


p = dl.plot(z, vmin=vmin, vmax=vmax)
plt.colorbar(p)
plt.savefig("z_map.png")
plt.cla()
plt.clf()

p = dl.plot(z_true, vmin=vmin, vmax=vmax)
plt.colorbar(p)
plt.savefig("z_true.png")

reconst_err = dl.assemble(dl.inner(z - z_true, z - z_true) * dl.dx)
z_true_norm = dl.assemble(dl.inner(z_true, z_true) * dl.dx)
rel_r_err = reconst_err/z_true_norm
print(f"Relative reconstruction error: {rel_r_err * 100:.4f}%")
print(f"Reconstruction error: {reconst_err:.4f}")
