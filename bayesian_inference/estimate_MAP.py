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

    def cost_function(self, z_v):
        self.z.vector().set_local(z_v)
        self.grad, self.cost = self.solver_r.grad_romml(self.z)
        return self.cost

    def gradient(self, z_v):
        self.z.vector().set_local(z_v)
        self.grad, self.cost = self.solver_r.grad_romml(self.z)
        return self.grad

solver_w = SolverWrapper(err_model, solver_r)
bounds = Bounds(0.1, 5)
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
print(f"True: {obs_data}\n Pred: {pred_obs}")

p = dl.plot(z)
plt.colorbar(p)
plt.savefig("z_map.png")
plt.cla()
plt.clf()

p = dl.plot(z_true)
plt.colorbar(p)
plt.savefig("z_true.png")
