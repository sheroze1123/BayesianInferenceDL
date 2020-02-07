import sys
sys.path.append('../')

import matplotlib; matplotlib.use('agg')
import time
import numpy as np
import matplotlib.pyplot as plt
import dolfin as dl; dl.set_log_level(40)

# PyMC related imports
import pymc3 as pm
import theano
from theano.compile.ops import as_op
import theano.tensor as tt

# ROMML imports
from fom.forward_solve_exp import Fin
from fom.thermal_fin import get_space

class ParamToObsFOM(theano.Op):
    '''
    Wrapper class to construct an operator representing the forward
    solve using the full-order model mapping from discretized thermal
    conductivity to quantities of interest.
    '''
    itypes = [tt.dvector]
    otypes = [tt.dvector, tt.dmatrix]
    __props__ = ()

    def __init__(self, V, randobs):
        '''
        Arguments:
            V - FEniCS FunctionSpace 
            randobs - Set True if random surface observations are used.
                      False is average temperature of subfins are the QoI.
        '''
        self._V = V
        self._solver = Fin(self._V, randobs)
        self._pred_k = dl.Function(V)

    def perform(self, node, inputs, outputs):
        pred_k = inputs[0]
        
        self._pred_k.vector().set_local(pred_k)
        w, y, a, b, c = self._solver.forward(self._pred_k)
        qoi = self._solver.qoi_operator(w)
        #  err = np.square(qoi - self.obs_data).sum()/2.0
        grad = self._solver.sensitivity(self._pred_k)

        outputs[0][0] = qoi
        outputs[1][0] = grad

    def grad(self, inputs, output_gradients):
        val, grad = self(*inputs)
        #  return [output_gradients[0] * grad] 
        return [tt.dot(grad.T, output_gradients[0])] 

resolution = 40
randobs = True

V = get_space(resolution)

k_true = dl.Function(V)
nodal_vals = np.log(np.load('res_x.npy'))
k_true.vector().set_local(nodal_vals)

forward_op = ParamToObsFOM(V, randobs)

w, _, _, _, _ = forward_op._solver.forward(k_true)
obs_data = forward_op._solver.qoi_operator(w)
measurement_sigma = 0.01
obs_data = obs_data + np.random.randn(len(obs_data)) * measurement_sigma

# Mass matrix
M = forward_op._solver.M

# Stiffness matrix
K = forward_op._solver.K
A = np.linalg.inv(M) @ K + np.eye(M.shape[0])
S, V_ = np.linalg.eig(A)

Wdofs_x = V.tabulate_dof_coordinates().reshape((-1, 2))
V0_dofs = V.dofmap().dofs()
points = Wdofs_x[V0_dofs, :] 
num_pts = len(points)

s = 1.2
alpha = 0.1
Cinv = alpha * M @ V_ @ np.diag(np.square(S)) @ V_.T @ M #TODO: Verify V vs V.T
cov = Cinv
mean = np.zeros(num_pts)

sigma = 1e-2
obs_covariance = sigma * sigma * np.eye(forward_op._solver.n_obs)

# Start at MAP point
mcmc_start = np.load("res_FOM.npy")
#mcmc_start = np.zeros(num_pts)

misfit_model = pm.Model()

with misfit_model:

    # Prior 
    nodal_vals = pm.distributions.multivariate.MvNormal('nodal_vals', 
            mu=mean, cov=cov, shape=(num_pts))

    qoi = forward_op(nodal_vals)[0]

    # Likelihood
    y = pm.distributions.multivariate.MvNormal('y',
            mu=qoi, cov=obs_covariance, observed=obs_data)

    #TODO: Good NUTS hyperparameters
    step = pm.NUTS(is_cov=True, scaling=M, max_treedepth=6, target_accept=0.75)
    trace = pm.sample(500, tune=100, cores=None, step=step, start={'nodal_vals':mcmc_start})

#  pm.plot_posterior(trace)
#  plt.show()
#  pm.traceplot(trace)
pm.save_trace(trace)

k_inv = dl.Function(V)
k_inv.vector().set_local(np.mean(trace['nodal_vals'],0))
p = dl.plot(k_inv)
plt.colorbar(p)
plt.savefig("k_inv.png")
plt.cla()
plt.clf()

k_inv_s = dl.Function(V)
k_inv_s.vector().set_local(np.std(trace['nodal_vals'],0))
p = dl.plot(k_inv_s)
plt.colorbar(p)
plt.savefig("k_inv_std.png")
plt.cla()
plt.clf()

nodal_vals = np.load('res_x.npy')

k_true = dl.Function(V)
k_true.vector().set_local(nodal_vals)
p = dl.plot(k_true)
plt.colorbar(p)
plt.savefig("k_true.png")
