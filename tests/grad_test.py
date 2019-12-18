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
from deep_learning.dl_model import load_parametric_model_avg, load_bn_model
from gaussian_field import make_cov_chol

# Tensorflow related imports
from tensorflow.keras.optimizers import Adam

class SolverWrapper:
    def __init__(self, solver, data): 
        self.solver = solver
        self.data = data
        self.z = dl.Function(V)

    def cost_function(self, z_v):
        self.z.vector().set_local(z_v)
        w, y, A, B, C = self.solver.forward(self.z)
        y = self.solver.qoi_operator(w)
        reg_cost = dl.assemble(self.solver.reg)
        cost = 0.5 * np.linalg.norm(y - self.data)**2
        #  cost = cost + reg_cost
        return cost

    def gradient(self, z_v):
        self.z.vector().set_local(z_v)
        grad = self.solver.gradient(self.z, self.data)
        reg_grad = dl.assemble(self.solver.grad_reg)[:]
        #  grad = grad + reg_grad
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
        #  self.cost = 0.5 * np.linalg.norm(y_romml - self.data)**2 + dl.assemble(self.solver.reg)
        self.cost = 0.5 * np.linalg.norm(y_romml - self.data)**2
        return self.cost

    def gradient(self, z_v):
        self.z.vector().set_local(z_v)
        self.solver._k.assign(self.z)
        self.grad, self.cost = self.solver_r.grad_romml(self.z)
        #  self.grad = self.grad + dl.assemble(self.solver.grad_reg)
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
        #  self.cost = 0.5 * np.linalg.norm(y_r - self.data)**2 + dl.assemble(self.solver.reg)
        self.cost = 0.5 * np.linalg.norm(y_r - self.data)**2
        return self.cost

    def gradient(self, z_v):
        self.z.vector().set_local(z_v)
        self.solver._k.assign(self.z)
        self.grad, self.cost = self.solver_r.grad_reduced(self.z)
        #  self.grad = self.grad + dl.assemble(self.solver.grad_reg)
        return self.grad

resolution = 40
V = get_space(resolution)
chol = make_cov_chol(V, length=1.2)
solver = Fin(V, True)

# Generate synthetic observations
z_true = dl.Function(V)
norm = np.random.randn(len(chol))
nodal_vals = np.exp(0.5 * chol.T @ norm)
z_true.vector().set_local(nodal_vals)
w, y, A, B, C = solver.forward(z_true)
data = solver.qoi_operator(w)

# Setup DL error model
#  err_model = load_parametric_model_avg('elu', Adam, 0.0003, 5, 58, 200, 2000, V.dim())
err_model = load_bn_model()

# Initialize reduced order model
phi = np.loadtxt('../data/basis_nine_param.txt',delimiter=",")
solver_r = AffineROMFin(V, err_model, phi, True)

solver_r.set_data(data)

solver_romml = ROMMLSolverWrapper(err_model, solver_r, solver)
solver_w = RSolverWrapper(err_model, solver_r, solver)
solver_fom = SolverWrapper(solver, data)

# Determine direction of gradient
z = dl.Function(V)
norm = np.random.randn(len(chol))
eps_z = np.exp(0.5 * chol.T @ norm)
z.vector().set_local(eps_z)
eps_norm = np.sqrt(dl.assemble(dl.inner(z,z)*dl.dx))
eps_norm = np.linalg.norm(eps_z)
eps_z = eps_z/eps_norm

# Determine location to evaluate gradient at
norm = np.random.randn(len(chol))
z_ = np.exp(0.5 * chol.T @ norm)

# Evaluate directional derivative using ROMML
dir_grad = np.dot(solver_romml.gradient(z_), eps_z)
print(f"Directional gradient ROMML: {dir_grad}")

n_eps = 32
hs = np.power(2., -np.arange(n_eps))

err_grads = []
grads = []
pi_0 = solver_romml.cost_function(z_)

for h in hs:
    pi_h = solver_romml.cost_function(z_ + h * eps_z)
    a_g = (pi_h - pi_0)/h
    grads.append(a_g)
    err = abs(a_g - dir_grad)/abs(dir_grad)
    #  err = abs(a_g - dir_grad)
    err_grads.append(err)

plt.loglog(hs, err_grads, "-ob", label="Error Grad")
plt.loglog(hs, (.5*err_grads[0]/hs[0])*hs, "-.k", label="First Order")
plt.savefig('grad_test_ROMML.png', dpi=200)
plt.cla()
plt.clf()

plt.semilogx(hs, grads, "-ob")
plt.savefig('gradients_ROMML.png')
plt.cla()
plt.clf()

err_grads = []
grads = []
pi_0 = solver_w.cost_function(z_)
dir_grad = np.dot(solver_w.gradient(z_), eps_z)

for h in hs:
    pi_h = solver_w.cost_function(z_ + h * eps_z)
    a_g = (pi_h - pi_0)/h
    grads.append(a_g)
    err = abs(a_g - dir_grad)/abs(dir_grad)
    #  err = abs(a_g - dir_grad)
    err_grads.append(err)

plt.loglog(hs, err_grads, "-ob", label="Error Grad")
plt.loglog(hs, (.5*err_grads[0]/hs[0])*hs, "-.k", label="First Order")
plt.savefig('grad_test_ROM.png', dpi=200)
plt.cla()
plt.clf()

plt.semilogx(hs, grads, "-ob")
plt.savefig('gradients_ROM.png')
plt.cla()
plt.clf()

err_grads = []
grads = []
pi_0 = solver_fom.cost_function(z_)
dir_grad = np.dot(solver_fom.gradient(z_), eps_z)

for h in hs:
    pi_h = solver_fom.cost_function(z_ + h * eps_z)
    a_g = (pi_h - pi_0)/h
    grads.append(a_g)
    err = abs(a_g - dir_grad)/abs(dir_grad)
    err_grads.append(err)

plt.loglog(hs, err_grads, "-ob", label="Error Grad")
plt.loglog(hs, (.5*err_grads[0]/hs[0])*hs, "-.k", label="First Order")
plt.savefig('grad_test_FOM.png', dpi=200)
plt.cla()
plt.clf()

plt.semilogx(hs, grads, "-ob")
plt.savefig('gradients_FOM.png')
plt.cla()
plt.clf()


#####
## Examine function behavior
####

hs = np.linspace(0, 1, 500)
pis = []
#  grads = []

for h in hs:
    pi_h = solver_w.cost_function(z_ + h * eps_z)
    pis.append(pi_h)
    #  grad = solver_w.gradient(z_ + h * eps_z)
    #  dir_grad = np.dot(grad, eps_z)
    #  grads.append(dir_grad)

pi_foms = []
#  grads_fom = []
dir_grad_fom = np.dot(solver_fom.gradient(z_), eps_z)
print(f"Direction gradient FOM: {dir_grad_fom}")
for h in hs:
    pi_h = solver_fom.cost_function(z_ + h * eps_z)
    pi_foms.append(pi_h)
    #  grad = solver_fom.gradient(z_ + h * eps_z)
    #  dir_grad = np.dot(grad, eps_z)
    #  grads_fom.append(dir_grad)


pi_rommls = []
#  grads_romml = []
for h in hs:
    pi_h = solver_romml.cost_function(z_ + h * eps_z)
    pi_rommls.append(pi_h)
    #  grad = solver_romml.gradient(z_ + h * eps_z)
    #  dir_grad = np.dot(grad, eps_z)
    #  grads_romml.append(dir_grad)

plt.plot(hs, pi_foms)
plt.savefig('func_dir_FOM.png', dpi=200)
plt.cla()
plt.clf()

plt.plot(hs, pis)
plt.savefig('func_dir_ROM.png', dpi=200)
plt.cla()
plt.clf()

plt.plot(hs, pi_rommls)
plt.savefig('func_dir_ROMML.png', dpi=200)
plt.cla()
plt.clf()

#  plt.plot(hs, grads_fom)
#  plt.plot(hs, grads)
#  plt.plot(hs, grads_romml)
#  plt.legend(["FOM", "ROM", "ROMML"])
#  plt.savefig('grad_dir.png', dpi=200)
