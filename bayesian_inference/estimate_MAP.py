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
#  from fom.forward_solve_exp import Fin
from fom.forward_solve import Fin
from fom.thermal_fin import get_space
from rom.averaged_affine_ROM import AffineROMFin 
from deep_learning.dl_model import load_parametric_model_avg, load_bn_model

randobs = True

resolution = 40
V = get_space(resolution)
#  chol = make_cov_chol(V, length=1.2)

# Setup DL error model
#  err_model = load_parametric_model_avg('elu', Adam, 0.0003, 5, 58, 200, 2000, V.dim())
err_model = load_bn_model()

# Initialize reduced order model
phi = np.loadtxt('../data/basis_nine_param.txt',delimiter=",")
solver_r = AffineROMFin(V, err_model, phi, randobs)

# Setup synthetic observations
solver = Fin(V, randobs)
z_true = dl.Function(V)

prior_covariance = np.load('prior_covariance.npy')
L = np.linalg.cholesky(prior_covariance)
#  draw = np.random.randn(V.dim())
#  nodal_vals = np.dot(L, draw)

#Load random Gaussian field
nodal_vals = np.load('res_x.npy')

# For exp parametrization
nodal_vals = np.log(nodal_vals)


z_true.vector().set_local(nodal_vals)
vmax = np.max(nodal_vals)
vmin = np.min(nodal_vals)
plt.cla()
plt.clf()
p = dl.plot(dl.exp(z_true))
plt.colorbar(p)
plt.savefig("z_true.png", dpi=200)

#  z_true = dl.interpolate(dl.Expression('0.3 + 0.01 * x[0] * x[1] + 0.05 * (sin(2*x[1])*cos(2*x[0]) + 1.5)', degree=2),V)
#  vmax = 0.7
#  vmin = 0.3

#Piece-wise constant field
#z_true = solver.nine_param_to_function([1.1, 1.11, 1.13, 1.12, 1.117, 1.1, 1.127, 1.118, 1.114])
#np.save('z_tr_pw', z_true.vector()[:])
#vmax = 1.130
#vmin = 1.095

w, y, A, B, C = solver.forward(z_true)
obs_data = solver.qoi_operator(w)
solver_r.set_data(obs_data)

z = dl.Function(V)
#  z = dl.interpolate(dl.Expression('0.3 + 0.01 * x[0] * x[1]', degree=2),V)
#  z_0_nodal_vals = z.vector()[:]

draw = np.load('uncorr_draw.npy')
z_0_nodal_vals = np.dot(L, draw) #For exp parametrization
z.vector().set_local(z_0_nodal_vals)


class SolverWrapper:
    def __init__(self, solver, data): 
        self.solver = solver
        self.data = data
        self.z = dl.Function(V)
        self.fwd_time = 0.0
        self.grad_time = 0.0

    def cost_function(self, z_v):
        self.z.vector().set_local(z_v)
        t_i = time.time()
        w, y, A, B, C = self.solver.forward(self.z)
        y = self.solver.qoi_operator(w)
        self.fwd_time += (time.time() - t_i)
        cost = 0.5 * np.linalg.norm(y - self.data)**2 + dl.assemble(self.solver.reg)
        return cost

    def gradient(self, z_v):
        self.z.vector().set_local(z_v)
        t_i = time.time()
        grad = self.solver.gradient(self.z, self.data)
        self.grad_time += (time.time() - t_i)
        grad += dl.assemble(self.solver.grad_reg)[:]
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
        self.fwd_time_dl = 0.0
        self.fwd_time_rom = 0.0
        self.grad_time = 0.0
        self.grad_time_dl = 0.0

    def cost_function(self, z_v):
        self.z.vector().set_local(z_v)
        t_i = time.time()
        w_r = self.solver_r.forward_reduced(self.z)
        y_r = self.solver_r.qoi_reduced(w_r)
        self.fwd_time_rom += (time.time() - t_i)
        t_i = time.time()
        e_NN = self.err_model.predict([[z_v]])[0]
        self.fwd_time_dl += (time.time() - t_i)
        self.solver._k.assign(self.z)
        y_romml = y_r + e_NN
        self.cost = 0.5 * np.linalg.norm(y_romml - self.data)**2 + dl.assemble(self.solver.reg)
        return self.cost

    def gradient(self, z_v):
        self.z.vector().set_local(z_v)
        self.solver._k.assign(self.z)
        self.grad, self.cost = self.solver_r.grad_romml(self.z)
        self.grad = self.grad + dl.assemble(self.solver.grad_reg)
        self.grad_time = solver_r.romml_grad_time
        self.grad_time_dl = solver_r.romml_grad_time_dl
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
        self.fwd_time = 0.0
        self.grad_time = 0.0

    def cost_function(self, z_v):
        self.z.vector().set_local(z_v)
        t_i = time.time()
        w_r = self.solver_r.forward_reduced(self.z)
        y_r = self.solver_r.qoi_reduced(w_r)
        self.solver._k.assign(self.z)
        self.cost = 0.5 * np.linalg.norm(y_r - self.data)**2 + dl.assemble(self.solver.reg)
        self.fwd_time += (time.time() - t_i)
        return self.cost

    def gradient(self, z_v):
        self.z.vector().set_local(z_v)
        t_i = time.time()
        self.solver._k.assign(self.z)
        self.grad, self.cost = self.solver_r.grad_reduced(self.z)
        self.grad = self.grad + dl.assemble(self.solver.grad_reg)
        self.grad_time += (time.time() - t_i)
        return self.grad

#  solver_w = RSolverWrapper(err_model, solver_r, solver)
solver_w = ROMMLSolverWrapper(err_model, solver_r, solver)
#  solver_w = SolverWrapper(solver, obs_data)

bounds = Bounds(1.05 * vmin, 1.05 * vmax)
print(f"Optimization bounds: {vmin}, {vmax}")
#  bounds = Bounds(0.3, 0.7)
res = minimize(solver_w.cost_function, z_0_nodal_vals, 
        method='L-BFGS-B', 
        jac=solver_w.gradient,
        bounds=bounds,
        options={'ftol':1e-11, 'gtol':1e-10})

print(f'status: {res.success}, message: {res.message}, n_it: {res.nit}')
print(f'Minimum cost: {res.fun:.3F}')
print(f'Running time (fwd ROM): {solver_w.fwd_time_rom} seconds')
print(f'Running time (fwd DL): {solver_w.fwd_time_dl} seconds')
print(f'Running time (grad): {solver_w.grad_time} seconds')
print(f'Running time (grad): {solver_w.grad_time_dl} seconds')
#  print(f'Running time (fwd ROM): {solver_w.fwd_time} seconds')
#  print(f'Running time (grad): {solver_w.grad_time} seconds')

####################3

z.vector().set_local(res.x)
np.save('res_ROMML', res.x)
w,y, _, _, _ = solver.forward(z)
pred_obs = solver.qoi_operator(w)
obs_err = np.linalg.norm(obs_data - pred_obs)
#  print(f"True: {obs_data}\n Pred: {pred_obs}")
print(f"Relative observation error: {obs_err/np.linalg.norm(obs_data)*100:.4f}%")


plt.cla()
plt.clf()
p = dl.plot(dl.exp(z))
plt.colorbar(p)
plt.savefig("z_map_exp.png", dpi=200)


reconst_err = dl.assemble(dl.inner(z - z_true, z - z_true) * dl.dx)
z_true_norm = dl.assemble(dl.inner(z_true, z_true) * dl.dx)
rel_r_err = np.sqrt(reconst_err/z_true_norm)
print(f"Relative reconstruction error: {rel_r_err * 100:.4f}%")
print(f"Reconstruction error: {reconst_err:.4f}")

rel_err = 100 * np.abs(z_true.vector()[:] - z.vector()[:])/np.sqrt(z_true_norm)
print(f"Relative error: {np.average(rel_err)}")
z_err = dl.Function(V)
z_err.vector().set_local(rel_err)
np.save('rel_err',rel_err)
