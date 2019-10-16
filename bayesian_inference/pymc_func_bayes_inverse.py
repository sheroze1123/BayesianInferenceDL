import sys
sys.path.append('../')

import matplotlib; matplotlib.use('macosx')
import time
import numpy as np
import matplotlib.pyplot as plt
import dolfin as dl; dl.set_log_level(40)

# Tensorflow related imports
from tensorflow.keras.optimizers import Adam

# PyMC related imports
import pymc3 as pm
from theano.compile.ops import as_op
import theano.tensor as tt
import theano

# ROMML imports
from fom.forward_solve import Fin
from fom.thermal_fin import get_space
from rom.averaged_affine_ROM import AffineROMFin 
from deep_learning.dl_model import load_parametric_model_avg
from gaussian_field import make_cov_chol

class SqError:
    '''
    Wrapper class interfacing Theano operators and ROMML
    to compute forward solves and parameter gradients
    '''
    def __init__(self, V, chol):
        '''
        Parameters:
            V - FEniCS FunctionSpace
            chol - Covariance matrix to define Gaussian field over V
        '''
        self._V = V
        self._solver = Fin(self._V)
        self._pred_k = dl.Function(self._V)

        # Setup synthetic observations
        self.k_true = dl.Function(self._V)
        norm = np.random.randn(len(chol))
        nodal_vals = np.exp(0.5 * chol.T @ norm)
        self.k_true.vector().set_local(nodal_vals)
        w, y, A, B, C = self._solver.forward(self.k_true)
        self.obs_data = self._solver.qoi_operator(w)

        # Setup DL error model
        self._err_model = load_parametric_model_avg('elu', Adam, 0.0003, 5, 58, 200, 2000, V.dim())
        #self._solver_r.set_dl_model(self._err_model)

        # Initialize reduced order model
        self.phi = np.loadtxt('../data/basis_five_param.txt',delimiter=",")
        self._solver_r = AffineROMFin(self._V, self._err_model, self.phi)
        #  self._solver_r.set_reduced_basis(self.phi)
        self._solver_r.set_data(self.obs_data)


    def err_grad_FOM(self, pred_k):
        '''
        For a given parameter, computes the high fidelity forward solution
        and the gradient with respect to the cost function
        '''
        self._pred_k.vector().set_local(pred_k)
        w, y, a, b, c = self._solver.forward(self._pred_k)
        qoi = self._solver.qoi_operator(w)
        err = np.square(qoi - self.obs_data).sum()/2.0
        grad = self._solver.gradient(self._pred_k, self.obs_data)
        return err, grad

    def err_grad_ROM(self, pred_k):
        '''
        For a given parameter, computes the reduced-order forward solution
        and the gradient with respect to the cost function
        '''
        self._pred_k.vector().set_local(pred_k)
        w_r = self._solver_r.forward_reduced(self._pred_k)
        qoi_r = self._solver_r.qoi_reduced(w_r)
        err_r = np.square(qoi_r - self.obs_data).sum()/2.0
        grad_r = self._solver_r.grad_reduced(self._pred_k)
        return err_r, grad_r

    def err_grad_ROMML(self, pred_k):
        '''
        For a given parameter, computes the reduced-order + ML forward solution
        and the gradient with respect to the cost function
        '''
        self._pred_k.vector().set_local(pred_k)
        #  w_r = self._solver_r.forward_reduced(self._pred_k)
        #  qoi_r = self._solver_r.qoi_reduced(w_r)
        #  err_NN = self._err_model.predict([[pred_k]])[0]
        #  qoi_t = qoi_r + err_NN
        #  err_t = np.square(qoi_t - self.obs_data).sum()/2.0
        grad_t, err_t = self._solver_r.grad_romml(self._pred_k)
        return err_t, grad_t

class SqErrorOpROM(theano.Op):
    '''
    Theano operator to setup misfit functional with ROM
    '''
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
    '''
    Theano operator to setup misfit functional with FOM
    '''
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

class SqErrorOpROMML(theano.Op):
    '''
    Theano operator to setup misfit functional with ROM + ML
    '''
    itypes = [tt.dvector]
    otypes = [tt.dscalar, tt.dvector]
    __props__ = ()

    def __init__(self, V, chol):
        self._error_op = SqError(V, chol)

    def perform(self, node, inputs, outputs):
        pred_k = inputs[0]
        value, grad = self._error_op.err_grad_ROMML(pred_k)
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
sq_err_romml = SqErrorOpROMML(V, chol)

norm = np.random.randn(len(chol))
nodal_vals = np.exp(0.5 * chol.T @ norm)

sq_err_romml._error_op.err_grad_ROMML(nodal_vals)

# Start chain with random Gaussian field parameter
norm = np.random.randn(len(chol))
nodal_vals_start = np.exp(0.5 * chol.T @ norm)

Wdofs_x = V.tabulate_dof_coordinates().reshape((-1, 2))
V0_dofs = V.dofmap().dofs()
points = Wdofs_x[V0_dofs, :] 
num_pts = len(points)
sigma = 1e-3

with pm.Model() as misfit_model:

    # Prior 
    ls = 1.2
    cov = pm.gp.cov.Matern52(2, ls=ls)
    nodal_vals = pm.gp.Latent(cov_func=cov).prior('nodal_vals', X=points)

    # TODO: Missing constant term here?
    y = pm.Potential('y', sq_err_romml(nodal_vals)[0] / sigma / sigma
            + num_pts * tt.log(sigma))
    trace = pm.sample(1000, tune=500)

#  pm.plot_posterior(trace)
#  plt.show()
#  pm.traceplot(trace)

k_inv = dl.Function(V)
k_inv.vector().set_local(np.mean(trace['nodal_vals'],0))
dl.plot(k_inv)
plt.savefig("k_inv.png")
dl.plot(sq_err_r._error_op.k_true)
plt.savefig("k_true.png")