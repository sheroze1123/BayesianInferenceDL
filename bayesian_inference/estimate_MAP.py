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
import tensorflow as tf
from tensorflow.keras.backend import get_session, gradients

# Scipy imports
from scipy.optimize import minimize, Bounds

# ROMML imports
#  from fom.forward_solve_exp import Fin
from fom.forward_solve import Fin
from fom.thermal_fin import get_space
from rom.averaged_affine_ROM import AffineROMFin 
from deep_learning.dl_model import load_bn_model, load_surrogate_model
from bayesian_inference.gaussian_field import make_cov_chol

#  randobs = True
randobs = True

resolution = 40
V = get_space(resolution)
chol = make_cov_chol(V, length=1.6)

# Setup DL error model
#  err_model = load_parametric_model_avg('elu', Adam, 0.0003, 5, 58, 200, 2000, V.dim())
err_model = load_bn_model(randobs)
surrogate_model = load_surrogate_model(randobs)

# Initialize reduced order model
phi = np.loadtxt('../data/basis_nine_param.txt',delimiter=",")
solver_r = AffineROMFin(V, err_model, phi, randobs)

# Setup synthetic observations
solver = Fin(V, randobs)
z_true = dl.Function(V)

prior_covariance = np.load('prior_covariance_0.07_0.07.npy')
L = np.linalg.cholesky(prior_covariance)
#  draw = np.random.randn(V.dim())
#  nodal_vals = np.dot(L, draw)

#Load random Gaussian field
nodal_vals = np.load('res_x.npy')
#  nodal_vals = np.exp(nodal_vals)/np.sum(np.exp(nodal_vals)) + 1.0

# For exp parametrization
#  nodal_vals = np.log(nodal_vals)



#  z_true = dl.interpolate(dl.Expression('0.3 + 0.01 * x[0] * x[1] + 0.05 * (sin(2*x[1])*cos(2*x[0]) + 1.5)', degree=2),V)
#  vmax = 0.7
#  vmin = 0.3

#Piece-wise constant field
#  z_true = solver.nine_param_to_function([1.1, 1.11, 1.13, 1.12, 1.117, 1.1, 1.127, 1.118, 1.114])
#  nodal_vals = z_true.vector()[:]
#  np.save('z_tr_pw', z_true.vector()[:])
#  vmax = 1.130
#  vmin = 1.095

z_true.vector().set_local(nodal_vals)
vmax = np.max(nodal_vals)
vmin = np.min(nodal_vals)
plt.cla()
plt.clf()
p = dl.plot(z_true)
plt.colorbar(p)
plt.savefig("z_true.png", dpi=200)

w, y, A, B, C = solver.forward(z_true)
obs_sigma = 1e-3
obs_data = solver.qoi_operator(w)
solver_r.set_data(obs_data)

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
        w_r = self.solver_r.forward_reduced(self.z)
        y_r = self.solver_r.qoi_reduced(w_r)
        t_i = time.time()
        e_NN = self.err_model.predict([[z_v]])[0]
        self.fwd_time_dl += (time.time() - t_i)
        self.solver._k.assign(self.z)
        y_romml = y_r + e_NN
        self.cost = 0.5 * np.linalg.norm(y_romml - self.data)**2 + dl.assemble(self.solver.reg)
        self.fwd_time_rom = self.solver_r.fwd_time
        return self.cost

    def gradient(self, z_v):
        self.z.vector().set_local(z_v)
        self.solver._k.assign(self.z)
        self.grad, self.cost = self.solver_r.grad_romml(self.z)
        self.grad = self.grad + dl.assemble(self.solver.grad_reg)
        self.grad_time = solver_r.romml_grad_time
        self.grad_time_dl = solver_r.romml_grad_time_dl
        return self.grad

class MLSolverWrapper:
    def __init__(self, surrogate_model, data, solver): 
        self.model = surrogate_model
        self.z = dl.Function(V)
        self.solver = solver
        self.data = data
        self.data_ph = tf.placeholder(tf.float32, shape=(solver.n_obs,))
        self.cost = None
        self.grad = None
        self.fwd_time = 0.0
        self.grad_time = 0.0

        self.loss = tf.divide(tf.reduce_sum(tf.square(
            self.data_ph - self.model.layers[-1].output)), 2)

        self.NN_grad = gradients(self.loss, self.model.input)
        self.session = get_session()

    def cost_function(self, z_v):
        self.z.vector().set_local(z_v)
        t_i = time.time()
        y_NN = self.model.predict([[z_v]])[0]
        self.fwd_time += (time.time() - t_i)
        self.solver._k.assign(self.z)
        self.cost = 0.5 * np.linalg.norm(y_NN - self.data)**2 + dl.assemble(self.solver.reg)
        return self.cost

    def gradient(self, z_v):
        x_inp = [z_v]
        t_i = time.time()
        grad_NN = self.session.run(self.NN_grad, 
                feed_dict={self.model.input: x_inp,
                           self.data_ph: self.data})[0]
        self.grad_time += (time.time() - t_i)
        self.z.vector().set_local(z_v)
        self.solver._k.assign(self.z)
        self.grad = grad_NN + dl.assemble(self.solver.grad_reg)
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
        w_r = self.solver_r.forward_reduced(self.z)
        y_r = self.solver_r.qoi_reduced(w_r)
        self.fwd_time = self.solver_r.fwd_time
        self.solver._k.assign(self.z)
        self.cost = 0.5 * np.linalg.norm(y_r - self.data)**2 + dl.assemble(self.solver.reg)
        return self.cost

    def gradient(self, z_v):
        self.z.vector().set_local(z_v)
        self.solver._k.assign(self.z)
        self.grad, self.cost = self.solver_r.grad_reduced(self.z)
        self.grad = self.grad + dl.assemble(self.solver.grad_reg)
        self.grad_time = self.solver_r.rom_grad_time
        return self.grad

z = dl.Function(V)
#  z = dl.interpolate(dl.Expression('0.3 + 0.01 * x[0] * x[1]', degree=2),V)
#  z_0_nodal_vals = z.vector()[:]

FOM_fwd_t = []
FOM_grad_t = []
ROM_fwd_t = []
ROM_grad_t = []
ROMML_fwd_ROM_t = []
ROMML_fwd_DL_t = []
ROMML_grad_ROM_t = []
ROMML_grad_DL_t = []
ML_fwd_t = []
ML_grad_t = []
fom_best = np.inf
rom_best = np.inf
romml_best = np.inf
ml_best = np.inf
fom_errs = []
rom_errs = []
romml_errs = []
ml_errs = []
fom_errs_pw = []
rom_errs_pw = []
romml_errs_pw = []
ml_errs_pw = []
fom_iters = []
rom_iters = []
romml_iters = []
ml_iters = []

n_starting_pts = 6

solver_ML = MLSolverWrapper(surrogate_model, obs_data, solver)
solver_FOM = SolverWrapper(solver, obs_data)
solver_ROM = RSolverWrapper(err_model, solver_r, solver)
solver_ROMML = ROMMLSolverWrapper(err_model, solver_r, solver)
MAP_sol = dl.Function(V)
error_func = dl.Function(V)
z_true_norm = dl.fem.norms.norm(z_true)

t_inv_start = time.time()
for j in range(n_starting_pts):
    #  draw = np.load('uncorr_draw.npy')
    draw = np.random.randn(V.dim())
    #  z_0_nodal_vals = np.dot(L, draw) #For exp parametrization
    #  z_0_nodal_vals = np.exp(np.dot(L, draw)) #For exp parametrization

    z_0_nodal_vals = np.exp(0.5 * chol.T @ draw)

    z.vector().set_local(z_0_nodal_vals)
    plt.cla()
    plt.clf()
    p = dl.plot(z)
    plt.colorbar(p)
    plt.savefig('z_0.png')


    #  bounds = Bounds(1.05 * vmin, 1.05 * vmax)
    bounds = Bounds(0.95 * vmin, 1.05 * vmax)
    print(f"Optimization bounds: {0.95*vmin}, {1.05*vmax}")
    #  bounds = Bounds(0.3, 0.7)
    res = minimize(solver_FOM.cost_function, z_0_nodal_vals, 
            method='L-BFGS-B', 
            jac=solver_FOM.gradient,
            bounds=bounds,
            options={'ftol':1e-10, 'gtol':1e-8})

    print(f'\nstatus: {res.success}, message: {res.message}, n_it: {res.nit}')
    print(f'Minimum cost: {res.fun:.3F}')
    print(f'Running time (fwd FOM): {solver_FOM.fwd_time} seconds')
    print(f'Running time (grad FOM): {solver_FOM.grad_time} seconds')
    FOM_fwd_t.append(solver_FOM.fwd_time)
    FOM_grad_t.append(solver_FOM.grad_time)
    solver_FOM.fwd_time = 0.0
    solver_FOM.grad_time = 0.0
    fom_iters.append(res.nit)

    ####################3

    MAP_sol.vector().set_local(res.x)
    w,y, _, _, _ = solver.forward(MAP_sol)
    pred_obs = solver.qoi_operator(w)
    obs_err = np.linalg.norm(obs_data - pred_obs)
    #  print(f"True: {obs_data}\n Pred: {pred_obs}")
    print(f"Relative observation error: {obs_err/np.linalg.norm(obs_data)*100:.4f}%")

    error_func.assign(z_true - MAP_sol)
    reconst_err = dl.fem.norms.norm(error_func)
    rel_r_err = reconst_err/z_true_norm
    print(f"Relative reconstruction error: {rel_r_err * 100:.4f}%")
    fom_errs.append(rel_r_err)

    if rel_r_err < fom_best:
        fom_best = rel_r_err

        plt.cla()
        plt.clf()
        #  p = dl.plot(dl.exp(MAP_sol))
        p = dl.plot(MAP_sol)
        plt.colorbar(p)
        #  plt.savefig("z_map_exp.png", dpi=200)
        plt.savefig("z_map_FOM_smooth.png", dpi=200)
        np.save('res_FOM', res.x)

    #  print(f"Reconstruction error: {reconst_err:.4f}")

    rel_err_pw = np.linalg.norm(z_true.vector()[:] - MAP_sol.vector()[:])/np.sqrt(z_true_norm)
    fom_errs_pw.append(rel_err_pw)
    #  print(f"Relative error: {np.average(rel_err)}")
    #  z_err = dl.Function(V)
    #  z_err.vector().set_local(rel_err)
    #  np.save('rel_err',rel_err)

    #############################
    # ROM INVERSION
    #############################


    #  bounds = Bounds(1.05 * vmin, 1.05 * vmax)
    bounds = Bounds(0.95 * vmin, 1.05 * vmax)
    res = minimize(solver_ROM.cost_function, z_0_nodal_vals, 
            method='L-BFGS-B', 
            jac=solver_ROM.gradient,
            bounds=bounds,
            options={'ftol':1e-10, 'gtol':1e-8})

    print(f'\nstatus: {res.success}, message: {res.message}, n_it: {res.nit}')
    print(f'Minimum cost: {res.fun:.3F}')
    print(f'Running time (fwd ROM): {solver_ROM.fwd_time} seconds')
    print(f'Running time (grad ROM): {solver_ROM.grad_time} seconds')
    ROM_fwd_t.append(solver_ROM.fwd_time)
    ROM_grad_t.append(solver_ROM.grad_time)
    solver_ROM.fwd_time = 0.0
    solver_ROM.solver_r.fwd_time = 0.0
    solver_ROM.grad_time = 0.0
    solver_ROM.solver_r.rom_grad_time = 0.0
    rom_iters.append(res.nit)

    ####################3

    MAP_sol.vector().set_local(res.x)
    w,y, _, _, _ = solver.forward(MAP_sol)
    pred_obs = solver.qoi_operator(w)
    obs_err = np.linalg.norm(obs_data - pred_obs)
    #  print(f"True: {obs_data}\n Pred: {pred_obs}")
    print(f"Relative observation error: {obs_err/np.linalg.norm(obs_data)*100:.4f}%")

    error_func.assign(z_true - MAP_sol)
    reconst_err = dl.fem.norms.norm(error_func)
    rel_r_err = reconst_err/z_true_norm
    print(f"Relative reconstruction error: {rel_r_err * 100:.4f}%")
    rom_errs.append(rel_r_err)

    rel_err_pw = np.linalg.norm(z_true.vector()[:] - MAP_sol.vector()[:])/np.sqrt(z_true_norm)
    rom_errs_pw.append(rel_err_pw)

    if rel_r_err < rom_best:
        rom_best = rel_r_err

        plt.cla()
        plt.clf()
        #  p = dl.plot(dl.exp(MAP_sol))
        p = dl.plot(MAP_sol)
        plt.colorbar(p)
        #  plt.savefig("z_map_exp.png", dpi=200)
        plt.savefig("z_map_ROM_smooth.png", dpi=200)
        np.save('res_ROM', res.x)

    #  print(f"Reconstruction error: {reconst_err:.4f}")

    #  rel_err = 100 * np.abs(z_true.vector()[:] - MAP_sol.vector()[:])/np.sqrt(z_true_norm)
    #  print(f"Relative error: {np.average(rel_err)}")
    #  z_err = dl.Function(V)
    #  z_err.vector().set_local(rel_err)
    #  np.save('rel_err',rel_err)

    ################################
    # ROMML inversion
    ################################


    #  bounds = Bounds(1.05 * vmin, 1.05 * vmax)
    bounds = Bounds(0.95 * vmin, 1.05 * vmax)

    res = minimize(solver_ROMML.cost_function, z_0_nodal_vals, 
            method='L-BFGS-B', 
            jac=solver_ROMML.gradient,
            bounds=bounds,
            options={'ftol':1e-10, 'gtol':1e-8})

    print(f'\nstatus: {res.success}, message: {res.message}, n_it: {res.nit}')
    print(f'Minimum cost: {res.fun:.3F}')
    print(f'Running time (fwd ROM): {solver_ROMML.fwd_time_rom} seconds')
    print(f'Running time (fwd DL): {solver_ROMML.fwd_time_dl} seconds')
    print(f'Running time (grad ROM): {solver_ROMML.grad_time} seconds')
    print(f'Running time (grad DL): {solver_ROMML.grad_time_dl} seconds')
    ROMML_fwd_ROM_t.append(solver_ROMML.fwd_time_rom)
    ROMML_fwd_DL_t.append(solver_ROMML.fwd_time_dl)
    ROMML_grad_ROM_t.append(solver_ROMML.grad_time)
    ROMML_grad_DL_t.append(solver_ROMML.grad_time_dl)
    solver_ROMML.fwd_time_rom = 0.0
    solver_ROMML.fwd_time_dl = 0.0
    solver_ROMML.grad_time_rom = 0.0
    solver_ROMML.grad_time_dl = 0.0
    solver_ROM.solver_r.rom_grad_time = 0.0
    solver_ROM.solver_r.fwd_time = 0.0
    solver_ROMML.solver_r.romml_grad_time = 0.0
    solver_ROMML.solver_r.romml_grad_time_dl = 0.0
    romml_iters.append(res.nit)

    ####################3

    MAP_sol.vector().set_local(res.x)
    w,y, _, _, _ = solver.forward(MAP_sol)
    pred_obs = solver.qoi_operator(w)
    obs_err = np.linalg.norm(obs_data - pred_obs)
    #  print(f"True: {obs_data}\n Pred: {pred_obs}")
    print(f"Relative observation error: {obs_err/np.linalg.norm(obs_data)*100:.4f}%")

    error_func.assign(z_true - MAP_sol)
    reconst_err = dl.fem.norms.norm(error_func)
    rel_r_err = reconst_err/z_true_norm
    print(f"Relative reconstruction error: {rel_r_err * 100:.4f}%")
    romml_errs.append(rel_r_err)

    rel_err_pw = np.linalg.norm(z_true.vector()[:] - MAP_sol.vector()[:])/np.sqrt(z_true_norm)
    romml_errs_pw.append(rel_err_pw)

    if rel_r_err < romml_best:
        romml_best = rel_r_err

        plt.cla()
        plt.clf()
        #  p = dl.plot(dl.exp(MAP_sol))
        p = dl.plot(MAP_sol)
        plt.colorbar(p)
        #  plt.savefig("z_map_exp.png", dpi=200)
        plt.savefig("z_map_ROMML_smooth.png", dpi=200)
        np.save('res_ROMML', res.x)

    #  print(f"Reconstruction error: {reconst_err:.4f}")

    #  rel_err = 100 * np.abs(z_true.vector()[:] - MAP_sol.vector()[:])/np.sqrt(z_true_norm)
    #  print(f"Relative error: {np.average(rel_err)}")
    #  z_err = dl.Function(V)
    #  z_err.vector().set_local(rel_err)
    #  np.save('rel_err',rel_err)


    ################################
    # ML inversion
    ################################


    #  bounds = Bounds(1.05 * vmin, 1.05 * vmax)
    bounds = Bounds(0.95 * vmin, 1.05 * vmax)

    res = minimize(solver_ML.cost_function, z_0_nodal_vals, 
            method='L-BFGS-B', 
            jac=solver_ML.gradient,
            bounds=bounds,
            options={'ftol':1e-10, 'gtol':1e-8})

    print(f'\nstatus: {res.success}, message: {res.message}, n_it: {res.nit}')
    print(f'Minimum cost: {res.fun:.3F}')
    print(f'Running time (ML fwd): {solver_ML.fwd_time} seconds')
    print(f'Running time (ML grad): {solver_ML.grad_time} seconds')
    ML_fwd_t.append(solver_ML.fwd_time)
    ML_grad_t.append(solver_ML.grad_time)
    solver_ML.fwd_time = 0.0
    solver_ML.grad_time = 0.0
    ml_iters.append(res.nit)

    ####################3

    MAP_sol.vector().set_local(res.x)
    w,y, _, _, _ = solver.forward(MAP_sol)
    pred_obs = solver.qoi_operator(w)
    obs_err = np.linalg.norm(obs_data - pred_obs)
    #  print(f"True: {obs_data}\n Pred: {pred_obs}")
    print(f"Relative observation error: {obs_err/np.linalg.norm(obs_data)*100:.4f}%")

    error_func.assign(z_true - MAP_sol)
    reconst_err = dl.fem.norms.norm(error_func)
    rel_r_err = reconst_err/z_true_norm
    print(f"Relative reconstruction error: {rel_r_err * 100:.4f}%")
    ml_errs.append(rel_r_err)

    rel_err_pw = np.linalg.norm(z_true.vector()[:] - MAP_sol.vector()[:])/np.sqrt(z_true_norm)
    ml_errs_pw.append(rel_err_pw)

    if rel_r_err < ml_best:
        ml_best = rel_r_err

        plt.cla()
        plt.clf()
        #  p = dl.plot(dl.exp(MAP_sol))
        p = dl.plot(MAP_sol)
        plt.colorbar(p)
        #  plt.savefig("z_map_exp.png", dpi=200)
        plt.savefig("z_map_ML_smooth.png", dpi=200)
        np.save('res_ML', res.x)

    #  print(f"Reconstruction error: {reconst_err:.4f}")

    #  rel_err = 100 * np.abs(z_true.vector()[:] - MAP_sol.vector()[:])/np.sqrt(z_true_norm)
    #  print(f"Relative error: {np.average(rel_err)}")
    #  z_err = dl.Function(V)
    #  z_err.vector().set_local(rel_err)
    #  np.save('rel_err',rel_err)

print(f"\nFull inversion completed at {time.time() - t_inv_start} seconds")

print(f"\nFOM average fwd_time: {np.average(FOM_fwd_t)}")
print(f"FOM average grad_time: {np.average(FOM_grad_t)}")

print(f"\nROM average fwd_time: {np.average(ROM_fwd_t)}")
print(f"ROM average grad_time: {np.average(ROM_grad_t)}")

print(f"\nROMML average fwd_time (ROM): {np.average(ROMML_fwd_ROM_t)}")
print(f"ROMML average fwd_time (DL): {np.average(ROMML_fwd_DL_t)}")
print(f"ROMML average grad_time (ROM): {np.average(ROMML_grad_ROM_t)}")
print(f"ROMML average grad_time (Dl): {np.average(ROMML_grad_DL_t)}")

print(f"\nML average fwd_time: {np.average(ML_fwd_t)}")
print(f"ML average grad_time: {np.average(ML_grad_t)}")

print(f"\nBest FOM relative error: {fom_best}")
print(f"Best ROM relative error: {rom_best}")
print(f"Best ROMML relative error: {romml_best}")
print(f"Best ML relative error: {ml_best}")

print(f"\nAverage FOM relative error: {np.average(fom_errs)}")
print(f"Average ROM relative error: {np.average(rom_errs)}")
print(f"Average ROMML relative error: {np.average(romml_errs)}")
print(f"Average ML relative error: {np.average(ml_errs)}")

print(f"\nAverage FOM relative error pointw: {np.average(fom_errs_pw)}")
print(f"Average ROM relative error pointw: {np.average(rom_errs_pw)}")
print(f"Average ROMML relative error pointw: {np.average(romml_errs_pw)}")
print(f"Average ML relative error pointw: {np.average(ml_errs_pw)}")

print(f"\nAverage FOM BFGS iterations: {np.average(fom_iters)}")
print(f"Average ROM BFGS iterations: {np.average(rom_iters)}")
print(f"Average ROMML BFGS iterations: {np.average(romml_iters)}")
print(f"Average ML BFGS iterations: {np.average(ml_iters)}")
