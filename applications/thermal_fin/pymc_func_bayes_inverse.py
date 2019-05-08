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
chol = make_cov_chol(V)
solver = Fin(V)
z_true = dl.Function(V)
norm = np.random.randn(len(chol))
nodal_vals = np.exp(0.5 * chol.T @ norm)
z_true.vector().set_local(nodal_vals)
w, y, A, B, C = solver.forward(z_true)
obs_data = solver.qoi_operator(w)
print (obs_data.shape)
model = load_parametric_model_avg('relu', Adam, 0.0024, 5, 50, 130, 400, V.dim())
phi = np.loadtxt('data/basis_five_param.txt',delimiter=",")
z = dl.Function(V)

@as_op(itypes=[tt.dvector], otypes=[tt.dvector])
def fwd_model(nodal_vals):
    #  nodal_vals = np.exp(0.5 * chol.T @ norm)
    z.vector().set_local(nodal_vals)
    A_r, B_r, C_r, x_r, y_r = solver.averaged_forward(z, phi)
    qoi_r = solver.reduced_qoi_operator(x_r)
    qoi_e_NN = model.predict(np.array([nodal_vals]))
    qoi_tilde = qoi_r + qoi_e_NN
    return qoi_tilde.reshape((5,))

Wdofs_x = V.tabulate_dof_coordinates().reshape((-1, 2))
V0_dofs = V.dofmap().dofs()
points = Wdofs_x[V0_dofs, :] 

with pm.Model() as misfit_model:

    # Prior 
    ls = 0.3
    cov = pm.gp.cov.Matern32(2, ls=ls)
    nodal_vals = pm.gp.Latent(cov_func=cov).prior('nodal_vals', X=points)

    #  noise_sigma = 1e-2
    #  noise_cov = pm.gp.cov.WhiteNoise(noise_sigma)
    noise_var = 1e-4
    noise_cov = noise_var * np.eye(obs_data.size)

    #  fwd = pm.Deterministic('fwd', fwd_model(nodal_vals))
    avg_temp = fwd_model(nodal_vals)

    likelihood = pm.MvNormal('likelihood', mu=avg_temp, cov=noise_cov, observed=obs_data)

    step_method = pm.step_methods.metropolis.Metropolis()
    trace = pm.sample(1000,step=step_method, cores=1)

pm.traceplot(trace)
