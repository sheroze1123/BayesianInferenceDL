import sys
sys.path.append('../')

import matplotlib; matplotlib.use('agg')
import time
import numpy as np
import scipy
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

Wdofs_x = V.tabulate_dof_coordinates().reshape((-1, 2))
V0_dofs = V.dofmap().dofs()
points = Wdofs_x[V0_dofs, :] 
num_pts = len(points)

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

d_p = 0.07
g_p = 0.07

# Discretized elliptic operator representing inv. sq. root of covariance
A = g_p * np.linalg.inv(M) @ K + d_p * np.eye(num_pts)
S, V_ = np.linalg.eig(A)
#  S = np.square(S)
#  inv_prior_cov = M @ V_ @ np.diag(np.square(1./S)) @ V_.T @ M
prior_cov = M @ V_ @ np.diag(np.square(S)) @ V_.T @ M 
np.save('prior_covariance_0.07_0.07', prior_cov)

mean = np.zeros(num_pts)

sigma = measurement_sigma
obs_covariance = sigma * sigma * np.eye(forward_op._solver.n_obs)

# Start at MAP point
mcmc_start = np.load("res_FOM.npy")

S_diag = np.diag(S)
V_LAM = np.dot(V_, S_diag)
u_2 = dl.Function(V)
k_MAP = dl.Function(V)
k_MAP.vector().set_local(mcmc_start)

def H_tilde_action(x):
    '''
    Return H_tilde (x) defined by page 15 of RHMC paper by Bui and Girolami
    '''
    V_LAM_x = np.dot(V_LAM, x)
    u_2.vector().set_local(V_LAM_x)
    H_V_LAM_x = forward_op._solver.GN_hessian_action(k_MAP, u_2, obs_data)
    return np.dot(V_LAM.T, H_V_LAM_x)

r_samps = 30 # is equal to desired rank + oversampling factor
Y = np.zeros((1446, r_samps))
GAM = np.zeros((1446, r_samps))
for i in range(r_samps):
    print(f"Randomized matrix EV iteration: {i}")
    GAM[:,i] = np.random.randn(1446)
    H_u_2 = H_tilde_action(GAM[:,i])
    Y[:, i] = H_u_2

Q, R = np.linalg.qr(Y)
T = np.linalg.solve(Q.T @ GAM, Q.T @ Y)
SIG_Ts, V_Ts = np.linalg.eigh(T)

nnz_e_idx = (SIG_Ts > 1e-6)
SIG_Ts = SIG_Ts[nnz_e_idx]
V_Ts = V_Ts[:, nnz_e_idx]
V_r = Q @ V_Ts

D = np.diag(np.divide(SIG_Ts, SIG_Ts+1))
WB_INT = np.eye(1446) - (V_r @ D @ V_r.T)
G_INV = V_ @ S_diag @ WB_INT @ S_diag @ np.linalg.inv(V_)

misfit_model = pm.Model()
prior_realization = dl.Function(V)
n_samps = 400
n_tune = 100

with misfit_model:

    # Prior 
    nodal_vals = pm.distributions.multivariate.MvNormal('nodal_vals', 
            mu=mean, cov=prior_cov, shape=(num_pts))

    qoi = forward_op(nodal_vals)[0]

    # Likelihood
    y = pm.distributions.multivariate.MvNormal('y',
            mu=qoi, cov=obs_covariance, observed=obs_data)

    #  prior_realization.vector().set_local(nodal_vals.random())
    #  p = dl.plot(dl.exp(prior_realization))
    #  plt.colorbar(p)
    #  plt.savefig('random_realization.png')
    
    #TODO: Good NUTS hyperparameters
    step = pm.NUTS(scaling=G_INV, max_treedepth=7, target_accept=0.98)
    trace = pm.sample(n_samps, tune=n_tune, cores=None, step=step, 
            start={'nodal_vals':mcmc_start})
    #  trace = pm.load_trace('.pymc_7.trace')
    pm.save_trace(trace)

#  pm.plot_posterior(trace)
#  plt.show()
#  pm.traceplot(trace)

trace_len = n_samps
k_trace = dl.Function(V)
misfits = np.zeros(trace_len)
for i in range(trace_len):
    n_vals = trace['nodal_vals'][i,:]
    k_trace.vector().set_local(n_vals)
    w, _, _, _, _ = forward_op._solver.forward(k_trace)
    pred_trace = forward_op._solver.qoi_operator(w)
    misfit_trace = 0.5 * np.linalg.norm(obs_data - pred_trace)**2
    misfits[i] = misfit_trace

plt.plot(misfits)
plt.xlabel('Trace index')
plt.ylabel('Misfit functional')
plt.savefig('misfit_trace.png')
plt.cla()
plt.clf()

plt.plot(trace['nodal_vals'][trace_len,1])
#  plt.plot(trace['nodal_vals'][:trace_len,1445])
#  plt.plot(trace['nodal_vals'][:trace_len,145])
#  plt.plot(trace['nodal_vals'][:trace_len,5])
plt.savefig('trace_samples.png')
plt.cla()

k_inv = dl.Function(V)
k_inv.vector().set_local(np.mean(trace['nodal_vals'][n_samps,:],0))
p = dl.plot(k_inv)
plt.colorbar(p)
plt.savefig("k_inv.png")
plt.cla()
plt.clf()

k_inv_s = dl.Function(V)
k_inv_s.vector().set_local(np.std(trace['nodal_vals'][n_samps,:],0))
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
