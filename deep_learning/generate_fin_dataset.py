import sys
sys.path.append('../')

import numpy as np
import tensorflow as tf
from pandas import read_csv
from dolfin import *
from tqdm import tqdm

from fom.forward_solve_exp import Fin
from fom.thermal_fin import get_space
from rom.averaged_affine_ROM import AffineROMFin
from bayesian_inference.gaussian_field import make_cov_chol
from tensorflow.keras import Model, Sequential, Input
from tensorflow.keras.layers import *

# Tensorflow related imports
from tensorflow.keras.optimizers import Adam

def load_parametric_model_avg(activation,
        optimizer, lr, n_hidden_layers, n_weights, batch_size, n_epochs, input_shape):

    model = Sequential()
    for i in range(n_hidden_layers):
        if i==0:
            model.add(Dense(n_weights, activation=activation, input_shape=(input_shape,)))
        else:
            model.add(Dense(n_weights, activation=activation))
    model.add(Dense(9))
    model.compile(loss='mse', optimizer=optimizer(lr=lr), metrics=['mape'])
    return model

def residual_unit(x, activation, n_weights):
    res = x

    out = BatchNormalization()(x)
    out = activation(out)
    out = Dense(n_weights, activation=None)(out)

    out = BatchNormalization()(x)
    out = activation(out)
    out = Dense(n_weights, activation=None)(out)

    out = add([res, out])
    return out

def res_bn_fc_model(activation, optimizer, lr, n_layers, n_weights, input_shape=1446, 
        output_shape=9):
    inputs = Input(shape=(input_shape,))
    y = Dense(n_weights, input_shape=(input_shape,), activation=None)(inputs)
    out = residual_unit(y, activation, n_weights)
    for i in range(1,n_layers):
        out = residual_unit(out, activation, n_weights)
    out = BatchNormalization()(out)
    out = activation(out)
    out = Dense(output_shape)(out)
    model = Model(inputs=inputs, outputs=out)
    model.compile(loss='mse', optimizer=optimizer(lr=lr), metrics=['mape'])
    return model

def gen_affine_avg_rom_dataset(dataset_size, resolution=40, genrand=False):
    V = get_space(resolution)
    dofs = len(V.dofmap().dofs())
    #  chol = make_cov_chol(V, length=1.6)
    z = Function(V)
    solver = Fin(V, genrand)
    phi = np.loadtxt('../data/basis_nine_param.txt',delimiter=",")

    prior_cov = np.load('../bayesian_inference/prior_covariance.npy')
    L = np.linalg.cholesky(prior_cov)

    #  err_model = load_parametric_model_avg('elu', Adam, 0.0003, 5, 58, 200, 2000, V.dim())
    err_model = res_bn_fc_model(ELU(), Adam, 3e-5, 3, 200, 1446, 40)

    solver_r = AffineROMFin(V, err_model, phi, genrand)
    qoi_errors = np.zeros((dataset_size, solver_r.n_obs))
    qois = np.zeros((dataset_size, solver_r.n_obs))

    # TODO: Needs to be fixed for higher order functions
    z_s = np.zeros((dataset_size, V.dim()))

    for i in tqdm(range(dataset_size)):
        draw = np.random.randn(dofs)
        nodal_vals = np.dot(L, draw)
        z.vector().set_local(nodal_vals)
        z_s[i,:] = nodal_vals

        x, y, A, B, C = solver.forward(z)
        w_r = solver_r.forward_reduced(z)

        qoi = solver.qoi_operator(x)
        qoi_r = solver_r.qoi_reduced(w_r)

        qoi_errors[i,:] = qoi - qoi_r
        qois[i,:] = qoi

    if (dataset_size > 1000):
        np.savetxt('../data/z_aff_avg_tr_2.txt', z_s, delimiter=",")
        np.savetxt('../data/errors_aff_avg_tr_2.txt', qoi_errors, delimiter=",")
        np.save('../data/qois_avg_tr_2', qois)

    if (dataset_size < 400):
        np.savetxt('../data/z_aff_avg_eval_2.txt', z_s, delimiter=",")
        np.savetxt('../data/errors_aff_avg_eval_2.txt', qoi_errors, delimiter=",")
        np.save('../data/qois_avg_eval_2', qois)
    return (z_s, qoi_errors)

def gen_avg_rom_dataset(dataset_size, resolution=40):
    V = get_space(resolution)
    chol = make_cov_chol(V)
    z = Function(V)
    solver = Fin(V)
    phi = np.loadtxt('data/basis_five_param.txt',delimiter=",")
    qoi_errors = np.zeros((dataset_size, 5))

    # TODO: Needs to be fixed for higher order functions
    z_s = np.zeros((dataset_size, V.dim()))

    for i in range(dataset_size):
        norm = np.random.randn(len(chol))
        nodal_vals = np.exp(0.5 * chol.T @ norm)
        z.vector().set_local(nodal_vals)
        z_s[i,:] = nodal_vals
        A_r, B_r, C_r, x_r, y_r = solver.averaged_forward(z, phi)
        x, y, A, B, C = solver.forward(z)
        qoi = solver.qoi_operator(x)
        qoi_r = solver.reduced_qoi_operator(x_r)
        qoi_errors[i,:] = qoi - qoi_r

    if (dataset_size > 1000):
        np.savetxt('data/z_avg_tr.txt', z_s, delimiter=",")
        np.savetxt('data/errors_avg_tr.txt', qoi_errors, delimiter=",")

    if (dataset_size < 400):
        np.savetxt('data/z_avg_eval.txt', z_s, delimiter=",")
        np.savetxt('data/errors_avg_eval.txt', qoi_errors, delimiter=",")
    return (z_s, qoi_errors)

def generate_DL_only_dataset(dataset_size, resolution=40):
    '''
    Create dataset where the features are thermal conductivity parameters
    and the labels are the quantities of interest of the HFM

    Arguments: 
        dataset_size - number of feature-label pairs
        resolution   - finite element mesh resolution for the high fidelity model

    Returns:
        (z, qois)    - pairs of conductivity and qois
    '''

    V = get_space(resolution)
    dofs = len(V.dofmap().dofs())

    prior_cov = np.load('bayesian_inference/prior_covariance.npy')
    L = np.linalg.cholesky(prior_cov)

    # TODO: Improve this by using mass matrix covariance. Bayesian prior may work well too
    z_s = np.zeros((dataset_size, dofs))
    solver = Fin(V, True)
    qois = np.zeros((dataset_size, 40))
    k = Function(V)

    for i in range(dataset_size):
        draw = np.random.randn(dofs)
        prior_draw = np.dot(L, draw)
        k.vector().set_local(prior_draw)
        w, _, _, _, _ = solver.forward(k)
        qois[i,:] = solver.qoi_operator(w)
        z_s[i,:] = prior_draw
    
    if (dataset_size > 1000):
        np.savetxt('data/z_dlo_tr.txt', z_s, delimiter=",")
        np.savetxt('data/qois_dlo_tr.txt', qois, delimiter=",")

    if (dataset_size < 400):
        np.savetxt('data/z_dlo_eval.txt', z_s, delimiter=",")
        np.savetxt('data/qois_dlo_eval.txt', qois, delimiter=",")

    return (z_s, qois)
