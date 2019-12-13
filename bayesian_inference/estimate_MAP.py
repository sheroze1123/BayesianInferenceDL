import sys
sys.path.append('../')

import matplotlib; matplotlib.use('macosx')
import time
import numpy as np
import matplotlib.pyplot as plt
import dolfin as dl; dl.set_log_level(40)
from utils import nb

# Tensorflow related imports
from tensorflow.keras.optimizers import Adam

# Scipy imports
from scipy.optimize import minimize, Bounds

# ROMML imports
from fom.forward_solve import Fin
from fom.thermal_fin import get_space
from rom.averaged_affine_ROM import AffineROMFin 
from deep_learning.dl_model import load_parametric_model_avg, load_bn_model
from gaussian_field import make_cov_chol

randobs = True

resolution = 40
V = get_space(resolution)
chol = make_cov_chol(V, length=1.2)

# Setup DL error model
#  err_model = load_parametric_model_avg('elu', Adam, 0.0003, 5, 58, 200, 2000, V.dim())
err_model = load_bn_model()

# Initialize reduced order model
phi = np.loadtxt('../data/basis_nine_param.txt',delimiter=",")
solver_r = AffineROMFin(V, err_model, phi, randobs)

# Setup synthetic observations
solver = Fin(V, randobs)
z_true = dl.Function(V)

#Generate random Gaussian field
#  norm = np.random.randn(len(chol))
#  nodal_vals = np.exp(0.5 * chol.T @ norm)

#Load random Gaussian field
#  nodal_vals = np.load('res_x.npy')
#  z_true.vector().set_local(nodal_vals)
#  vmax = np.max(nodal_vals)
#  vmin = np.min(nodal_vals)

#  z_true = dl.interpolate(dl.Expression('0.3 + 0.01 * x[0] * x[1] + 0.05 * (sin(2*x[1])*cos(2*x[0]) + 1.5)', degree=2),V)
#  vmax = 0.7
#  vmin = 0.3

#Piece-wise constant field
z_true = solver.nine_param_to_function([1.1, 1.11, 1.13, 1.12, 1.117, 1.1, 1.127, 1.118, 1.114])
np.save('z_tr_pw', z_true.vector()[:])
vmax = 1.130
vmin = 1.095


w, y, A, B, C = solver.forward(z_true)
obs_data = solver.qoi_operator(w)
solver_r.set_data(obs_data)

z = dl.Function(V)
#  z = dl.interpolate(dl.Expression('0.3 + 0.01 * x[0] * x[1]', degree=2),V)
#  z_0_nodal_vals = z.vector()[:]
norm = np.random.randn(len(chol))
z_0_nodal_vals = np.exp(0.5 * chol.T @ norm)
z.vector().set_local(z_0_nodal_vals)
#  p = dl.plot(z)
#  plt.colorbar(p)
#  plt.savefig('z_0.png')
#  plt.cla()
#  plt.clf()

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

class ROMMLSolverWrapper:
    def __init__(self, err_model, solver_r, solver): 
        self.err_model = err_model
        self.solver_r = solver_r
        self.z = dl.Function(V)
        self.solver = solver
        self.data = self.solver_r.data
        self.cost = None
        self.grad = None

    def cost_function(self, z_v):
        self.z.vector().set_local(z_v)
        w_r = self.solver_r.forward_reduced(self.z)
        y_r = self.solver_r.qoi_reduced(w_r)
        e_NN = self.err_model.predict([[z_v]])[0]
        self.solver._k.assign(self.z)
        y_romml = y_r + e_NN
        self.cost = 0.5 * np.linalg.norm(y_romml - self.data)**2 + dl.assemble(self.solver.reg)
        return self.cost

    def gradient(self, z_v):
        self.z.vector().set_local(z_v)
        self.solver._k.assign(self.z)
        self.grad, self.cost = self.solver_r.grad_romml(self.z)
        self.grad = self.grad + dl.assemble(self.solver.grad_reg)
        return self.grad

class RSolverWrapper:
    def __init__(self, err_model, solver_r, solver): 
        self.err_model = err_model
        self.solver_r = solver_r
        self.z = dl.Function(V)
        self.solver = solver
        self.data = self.solver_r.data
        self.cost = None
        self.grad = None

    def cost_function(self, z_v):
        self.z.vector().set_local(z_v)
        w_r = self.solver_r.forward_reduced(self.z)
        y_r = self.solver_r.qoi_reduced(w_r)
        self.solver._k.assign(self.z)
        self.cost = 0.5 * np.linalg.norm(y_r - self.data)**2 + dl.assemble(self.solver.reg)
        return self.cost

    def gradient(self, z_v):
        self.z.vector().set_local(z_v)
        self.solver._k.assign(self.z)
        self.grad, self.cost = self.solver_r.grad_reduced(self.z)
        self.grad = self.grad + dl.assemble(self.solver.grad_reg)
        return self.grad

solver_w = RSolverWrapper(err_model, solver_r, solver)
#  solver_w = ROMMLSolverWrapper(err_model, solver_r, solver)
#  solver_w = SolverWrapper(solver, obs_data)

bounds = Bounds(0.95 * vmin, 1.05 * vmax)
#  bounds = Bounds(0.3, 0.7)
res = minimize(solver_w.cost_function, z_0_nodal_vals, 
        method='L-BFGS-B', 
        jac=solver_w.gradient,
        bounds=bounds,
        options={'ftol':1e-11, 'gtol':1e-10})

print(f'status: {res.success}, message: {res.message}, n_it: {res.nit}')
print(f'Minimum cost: {res.fun:.3F}')

####################3

z.vector().set_local(res.x)
np.save('res_ROM', res.x)
w,y, _, _, _ = solver.forward(z)
pred_obs = solver.qoi_operator(w)
obs_err = np.linalg.norm(obs_data - pred_obs)
#  print(f"True: {obs_data}\n Pred: {pred_obs}")
print(f"Relative observation error: {obs_err/np.linalg.norm(obs_data)*100:.4f}%")

p = dl.plot(z, vmax=vmax, vmin=vmin)
plt.colorbar(p)
plt.savefig("z_map.png", dpi=200)


#  plt.cla()
#  plt.clf()
#  p = dl.plot(z_true, vmax=vmax, vmin=vmin)
#  plt.colorbar(p)
#  plt.savefig("z_true.png", dpi=200)

reconst_err = dl.assemble(dl.inner(z - z_true, z - z_true) * dl.dx)
z_true_norm = dl.assemble(dl.inner(z_true, z_true) * dl.dx)
rel_r_err = np.sqrt(reconst_err/z_true_norm)
print(f"Relative reconstruction error: {rel_r_err * 100:.4f}%")
print(f"Reconstruction error: {reconst_err:.4f}")

rel_err = 100 * np.abs(z_true.vector()[:] - z.vector()[:])/np.sqrt(z_true_norm)
z_err = dl.Function(V)
z_err.vector().set_local(rel_err)
np.save('rel_err',rel_err)
