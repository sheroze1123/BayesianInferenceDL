import pymc3 as pm
import numpy as np
import dolfin as dl; dl.set_log_level(40)
from forward_solve import Fin, get_space
from theano.compile.ops import as_op
import theano.tensor as tt
import theano
import matplotlib.pyplot as plt

class SqError:
    def __init__(self):
        self._resolution = 40
        self._V = get_space(resolution)
        self._solver = Fin(self._V)
        z_true = np.array([[0.41126864, 0.61789679, 0.75873243, 0.96527541, 0.22348076]])
        w, y, A, B, C = self._solver.forward_five_param(z_true[0,:])
        qoi = self._solver.qoi_operator(w)
        self._obs_data = qoi

    def error(self, pred_z):
        w, y, a, b, c = self._solver.forward_five_param(pred_z[0,:])
        qoi = self._solver.qoi_operator(w)
        return np.square(qoi - self._obs_data).sum()/2

class SquaredErrorOp(theano.Op):
    itypes = [tt.dvector]
    otypes = [tt.dvector]
    __props__ = ()

    def __init__(self):
        self._error_op = SqError()

    def perform(self, node, inputs, outputs):
        pred_z = inputs[0]
        outputs[0] = self._error_op.error(pred_z)

    def grad(self, inputs, output_gradients):
        pass

#  sq_err = SquaredErrorOp()

resolution = 40
V = get_space(resolution)
solver = Fin(V)
z_true = np.array([[0.41126864, 0.61789679, 0.75873243, 0.96527541, 0.22348076]])
w, y, A, B, C = solver.forward_five_param(z_true[0,:])
obs_data = solver.qoi_operator(w)

@as_op(itypes=[tt.dvector], otypes=[tt.dvector])
def fwd_model(pred_z):
    w, y, a, b, c = solver.forward_five_param(pred_z)
    qoi = solver.qoi_operator(w)
    return qoi

with pm.Model() as misfit_model:
    mu = np.array([0.4, 0.6, 0.7, 0.9, 0.2])
    cov = 0.02*np.eye(5)
    prior = pm.MvNormal('prior', mu=mu, cov=cov, shape=(5,))

    noise_var = 1e-4
    noise_cov = noise_var * np.eye(obs_data.size)

    fwd = pm.Deterministic('fwd', fwd_model(prior))

    likelihood = pm.MvNormal('likelihood', mu=fwd, cov=noise_cov, observed=obs_data)

    step_method = pm.step_methods.metropolis.Metropolis()
    trace = pm.sample(10000,step=step_method)

pm.traceplot(trace)
