import pymc3 as pm
import numpy as np
import dolfin as dl; dl.set_log_level(40)
from forward_solve import Fin, get_space
from dl_model import load_parametric_model_avg
from tensorflow.keras.optimizers import Adam, RMSprop, Adadelta
from theano.compile.ops import as_op
import theano.tensor as tt
from gaussian_field import make_cov_chol
import theano
import time
import matplotlib.pyplot as plt

class SqError:
    def __init__(self):
        self._resolution = 40
        self._V = get_space(self._resolution)
        self._solver = Fin(self._V)
        self._pred_k = dl.Function(self._V)
        self.phi = np.loadtxt('data/basis_five_param.txt',delimiter=",")
        chol = make_cov_chol(self._V)
        self.k_true = dl.Function(self._V)
        norm = np.random.randn(len(chol))
        nodal_vals = np.exp(0.5 * chol.T @ norm)
        self.k_true.vector().set_local(nodal_vals)
        w, y, A, B, C = self._solver.forward(self.k_true)
        self.obs_data = self._solver.qoi_operator(w)

    def full_err_grad(self, pred_k):
        self._pred_k.vector().set_local(pred_k)
        w, y, a, b, c = self._solver.forward(self._pred_k)
        qoi = self._solver.qoi_operator(w)
        err = np.square(qoi - self.obs_data).sum()/2.0
        grad = self._solver.gradient(self._pred_k, self.obs_data)
        return err, grad

    def reduced_err_grad(self, pred_k):
        self._pred_k.vector().set_local(pred_k)
        w_r, g = self._solver.averaged_reduced_fwd_and_grad(self._pred_k, 
                self.phi, 
                self.obs_data)
        err = np.square(self._solver.reduced_qoi_operator(w_r) - self.obs_data).sum()/2.0
        return err, g

class SqErrorOpROM(theano.Op):
    itypes = [tt.dvector]
    otypes = [tt.dscalar, tt.dvector]
    __props__ = ()

    def __init__(self):
        self._error_op = SqError()

    def perform(self, node, inputs, outputs):
        pred_k = inputs[0]
        value, grad = self._error_op.reduced_err_grad(pred_k)
        outputs[0][0] = value
        outputs[1][0] = grad

    def grad(self, inputs, output_gradients):
        val, grad = self(*inputs)
        return [output_gradients[0] * grad] 

class SqErrorOpFOM(theano.Op):
    itypes = [tt.dvector]
    otypes = [tt.dscalar, tt.dvector]
    __props__ = ()

    def __init__(self):
        self._error_op = SqError()

    def perform(self, node, inputs, outputs):
        pred_k = inputs[0]
        value = np.array(self._error_op.full_error(pred_k))
        grad = self._error_op.full_grad(pred_k)
        outputs[0][0] = value
        outputs[1][0] = grad

    def grad(self, inputs, output_gradients):
        val, grad = self(*inputs)
        return [output_gradients[0] * grad] 

sq_err = SqErrorOpROM()

resolution = 40
V = get_space(resolution)
chol = make_cov_chol(V)
solver = Fin(V)
z_true = dl.Function(V)
norm = np.random.randn(len(chol))
nodal_vals_start = np.exp(0.5 * chol.T @ norm)
#  z_true.vector().set_local(nodal_vals)
#  w, y, A, B, C = solver.forward(z_true)
#  obs_data = solver.qoi_operator(w)
#  model = load_parametric_model_avg('relu', Adam, 0.0024, 5, 50, 130, 400, V.dim())
#  phi = np.loadtxt('data/basis_five_param.txt',delimiter=",")
#  z = dl.Function(V)

#  @as_op(itypes=[tt.dvector], otypes=[tt.dvector])
#  def fwd_model(nodal_vals):
    #  #  nodal_vals = np.exp(0.5 * chol.T @ norm)
    #  z.vector().set_local(nodal_vals)
    #  A_r, B_r, C_r, x_r, y_r = solver.averaged_forward(z, phi)
    #  qoi_r = solver.reduced_qoi_operator(x_r)
    #  qoi_e_NN = model.predict(np.array([nodal_vals]))
    #  qoi_tilde = qoi_r + qoi_e_NN
    #  return qoi_tilde.reshape((5,))

Wdofs_x = V.tabulate_dof_coordinates().reshape((-1, 2))
V0_dofs = V.dofmap().dofs()
points = Wdofs_x[V0_dofs, :] 
sigma = 1e-3

with pm.Model() as misfit_model:

    # Prior 
    ls = 0.3
    cov = pm.gp.cov.Matern32(2, ls=ls)
    nodal_vals = pm.gp.Latent(cov_func=cov).prior('nodal_vals', X=points)

    #  noise_var = 1e-4
    #  noise_cov = noise_var * np.eye(obs_data.size)
    #  avg_temp = fwd_model(nodal_vals)
    #  likelihood = pm.MvNormal('likelihood', mu=avg_temp, cov=noise_cov, observed=obs_data)
    #  step_method = pm.step_methods.metropolis.Metropolis()
    #  trace = pm.sample(1000,step=step_method, cores=1)

    y = pm.Potential('y', -0.5 * sq_err(nodal_vals)[0] / sigma / sigma)
    trace = pm.sample(200, cores=1)

pm.traceplot(trace)
k_inv = dl.Function(V)
k_inv.vector().set_local(np.mean(trace['nodal_vals'],0))
dl.plot(k_inv)
plt.show()
dl.plot(sq_err._error_op.k_true)
plt.show()
