import pymc3 as pm
import numpy as np
import dolfin as dl; dl.set_log_level(40)
from forward_solve import Fin
from thermal_Fin import get_space
from averaged_affine_ROM import AffineROMFin 
from dl_model import load_parametric_model_avg
from tensorflow.keras.optimizers import Adam, RMSprop, Adadelta
from theano.compile.ops import as_op
import theano.tensor as tt
from gaussian_field import make_cov_chol
import theano
import tensorflow as tf
import time
import matplotlib.pyplot as plt
from keras import backend as k

class SqError:
    def __init__(self, V, chol):
        self._V = V
        self._solver = Fin(self._V)
        self._solver_r = AffineROMFin(self._V)
        self._pred_k = dl.Function(self._V)
        self.phi = np.loadtxt('data/basis_five_param.txt',delimiter=",")
        self.k_true = dl.Function(self._V)
        norm = np.random.randn(len(chol))
        nodal_vals = np.exp(0.5 * chol.T @ norm)
        self.k_true.vector().set_local(nodal_vals)
        w, y, A, B, C = self._solver.forward(self.k_true)
        self.obs_data = self._solver.qoi_operator(w)
        self._solver_r.set_reduced_basis(self.phi)
        self._solver_r.set_data(self.obs_data)
        self._err_model = load_parametric_model_avg('elu', Adam, 0.129, 3, 58, 64, 466, V.dim())

    def err_grad_FOM(self, pred_k):
        self._pred_k.vector().set_local(pred_k)
        w, y, a, b, c = self._solver.forward(self._pred_k)
        qoi = self._solver.qoi_operator(w)
        err = np.square(qoi - self.obs_data).sum()/2.0
        grad = self._solver.gradient(self._pred_k, self.obs_data)
        return err, grad

    def err_grad_ROM(self, pred_k):
        self._pred_k.vector().set_local(pred_k)
        w_r = self._solver_r.forward_reduced(self._pred_k)
        qoi_r = self._solver_r.qoi_reduced(w_r)
        err_r = np.square(qoi_r - self.obs_data).sum()/2.0
        grad_r = self._solver_r.grad_reduced(self._pred_k)
        return err_r, grad_r

    def err_grad_ROMML(self, pred_k):
        self._pred_k.vector().set_local(pred_k)
        w_r = self._solver_r.forward_reduced(self._pred_k)
        qoi_r = self._solver_r.qoi_reduced(w_r)
        err_NN = self.model.predict(pred_k)
        qoi_tilde = qoi_r + err_NN
        err_t = np.square(qoi_t - self.obs_data).sum()/2.0
        grad_t = self._solver_r.grad_romml(self._pred_k)
        return err_t, grad_t

class SqErrorOpROM(theano.Op):
    itypes = [tt.dvector]
    otypes = [tt.dscalar, tt.dvector]
    __props__ = ()

    def __init__(self, V, chol):
        self._error_op = SqError(V, chol)

    def perform(self, node, inputs, outputs):
        pred_k = inputs[0]
        value, grad = self._error_op.err_grad_ROM(pred_k)
        outputs[0][0] = value
        outputs[1][0] = grad

    def grad(self, inputs, output_gradients):
        val, grad = self(*inputs)
        return [output_gradients[0] * grad] 

class SqErrorOpFOM(theano.Op):
    itypes = [tt.dvector]
    otypes = [tt.dscalar, tt.dvector]
    __props__ = ()

    def __init__(self, V, chol):
        self._error_op = SqError(V, chol)

    def perform(self, node, inputs, outputs):
        pred_k = inputs[0]
        value, grad = self._error_op.err_grad_FOM(pred_k)
        outputs[0][0] = value
        outputs[1][0] = grad

    def grad(self, inputs, output_gradients):
        val, grad = self(*inputs)
        return [output_gradients[0] * grad] 

resolution = 40
V = get_space(resolution)
chol = make_cov_chol(V, length=1.2)
sq_err = SqErrorOpFOM(V, chol)
sq_err_r = SqErrorOpROM(V, chol)

norm = np.random.randn(len(chol))
nodal_vals_start = np.exp(0.5 * chol.T @ norm)

Wdofs_x = V.tabulate_dof_coordinates().reshape((-1, 2))
V0_dofs = V.dofmap().dofs()
points = Wdofs_x[V0_dofs, :] 
sigma = 1e-3

with pm.Model() as misfit_model:

    # Prior 
    ls = 1.2
    cov = pm.gp.cov.Matern52(2, ls=ls)
    nodal_vals = pm.gp.Latent(cov_func=cov).prior('nodal_vals', X=points)

    #  noise_var = 1e-4
    #  noise_cov = noise_var * np.eye(obs_data.size)
    #  avg_temp = fwd_model(nodal_vals)
    #  likelihood = pm.MvNormal('likelihood', mu=avg_temp, cov=noise_cov, observed=obs_data)
    #  step_method = pm.step_methods.metropolis.Metropolis()
    #  trace = pm.sample(1000,step=step_method, cores=1)

    y = pm.Potential('y', -0.5 * sq_err_r(nodal_vals)[0] / sigma / sigma)
    trace = pm.sample(200, cores=1, tune=200)

pm.traceplot(trace)
k_inv = dl.Function(V)
k_inv.vector().set_local(np.mean(trace['nodal_vals'],0))
dl.plot(k_inv)
plt.show()
dl.plot(sq_err_r._error_op.k_true)
plt.show()

import pdb; pdb.set_trace()
