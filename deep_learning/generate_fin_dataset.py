import sys
sys.path.append('../')

import numpy as np
import tensorflow as tf
from pandas import read_csv
from dolfin import *
from tqdm import tqdm

from fom.forward_solve import Fin
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
    chol = make_cov_chol(V, length=1.6)
    z = Function(V)
    solver = Fin(V, genrand)
    phi = np.loadtxt('../data/basis_nine_param.txt',delimiter=",")

    #  err_model = load_parametric_model_avg('elu', Adam, 0.0003, 5, 58, 200, 2000, V.dim())
    err_model = res_bn_fc_model(ELU(), Adam, 3e-5, 3, 200, 1446, 40)

    solver_r = AffineROMFin(V, err_model, phi, genrand)
    qoi_errors = np.zeros((dataset_size, solver_r.n_obs))

    # TODO: Needs to be fixed for higher order functions
    z_s = np.zeros((dataset_size, V.dim()))

    for i in tqdm(range(dataset_size)):
        norm = np.random.randn(len(chol))
        nodal_vals = np.exp(0.5 * chol.T @ norm)
        z.vector().set_local(nodal_vals)
        z_s[i,:] = nodal_vals

        x, y, A, B, C = solver.forward(z)
        w_r = solver_r.forward_reduced(z)

        qoi = solver.qoi_operator(x)
        qoi_r = solver_r.qoi_reduced(w_r)

        qoi_errors[i,:] = qoi - qoi_r

    if (dataset_size > 1000):
        np.savetxt('../data/z_aff_avg_tr.txt', z_s, delimiter=",")
        np.savetxt('../data/errors_aff_avg_tr.txt', qoi_errors, delimiter=",")

    if (dataset_size < 400):
        np.savetxt('../data/z_aff_avg_eval.txt', z_s, delimiter=",")
        np.savetxt('../data/errors_aff_avg_eval.txt', qoi_errors, delimiter=",")
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

def generate_five_param(dataset_size, resolution=40):
    V = get_space(resolution)
    dofs = len(V.dofmap().dofs())

    # TODO: Improve this by using mass matrix covariance. Bayesian prior may work well too
    z_s = np.random.uniform(0.1, 1, (dataset_size, 5))
    phi = np.loadtxt('data/basis_five_param.txt',delimiter=",")
    phi = phi[:,0:20]
    solver = Fin(V)
    errors = np.zeros((dataset_size, 1))

    for i in range(dataset_size):
        w, y, A, B, C = solver.forward_five_param(z_s[i,:])
        psi = np.dot(A, phi)
        A_r, B_r, C_r, x_r, y_r = solver.reduced_forward(A, B, C, psi, phi)
        errors[i][0] = y - y_r 

    #  np.savetxt('data/z_s_eval.txt', z_s, delimiter=",")
    #  np.savetxt('data/errors_eval.txt', errors, delimiter=",")
    dataset = tf.data.Dataset.from_tensor_slices((z_s,errors))

    return dataset

def generate_five_param_np(dataset_size, resolution=40):
    V = get_space(resolution)
    z_s = np.random.uniform(0.1, 1, (dataset_size, 5))
    phi = np.loadtxt('data/basis_five_param.txt',delimiter=",")
    phi = phi[:,0:10]
    solver = Fin(V)
    errors = np.zeros((dataset_size, 1))
    y_s = np.zeros((dataset_size, 1))
    y_r_s = np.zeros((dataset_size, 1))

    for i in range(dataset_size):
        w, y, A, B, C = solver.forward_five_param(z_s[i,:])
        y_s[i][0] = y
        psi = np.dot(A, phi)
        A_r, B_r, C_r, x_r, y_r = solver.reduced_forward(A, B, C, psi, phi)
        y_r_s[i][0] = y_r
        errors[i][0] = y - y_r 

    return (z_s, errors)

def gen_five_param_subfin_avg(dataset_size, resolution=40):
    V = get_space(resolution)
    z_s = np.random.uniform(0.1, 1, (dataset_size, 5))
    phi = np.loadtxt('data/basis_five_param.txt',delimiter=",")
    phi = phi[:,0:10]
    solver = Fin(V)
    errors = np.zeros((dataset_size, 5))
    avgs = np.zeros((dataset_size, 5))
    avgs_r = np.zeros((dataset_size, 5))
    
    for i in range(dataset_size):
        w, y, A, B, C = solver.forward_five_param(z_s[i,:])
        avgs[i] = solver.qoi_operator(w)
        psi = np.dot(A, phi)
        A_r, B_r, C_r, x_r, y_r = solver.reduced_forward(A, B, C, psi, phi)
        avgs_r[i] = solver.reduced_qoi_operator(x_r)
        errors[i] = avgs[i] - avgs_r[i]

    return (z_s, errors)

def generate(dataset_size, resolution=40):
    '''
    Create a tensorflow dataset where the features are thermal conductivity parameters
    and the labels are the differences in the quantity of interest between the high 
    fidelity model and the reduced order model (this is the ROM error)

    Arguments: 
        dataset_size - number of feature-label pairs
        resolution   - finite element mesh resolution for the high fidelity model

    Returns:
        dataset      - Tensorflow dataset created from tensor slices
    '''

    V = get_space(resolution)
    dofs = len(V.dofmap().dofs())

    # TODO: Improve this by using mass matrix covariance. Bayesian prior may work well too
    z_s = np.random.uniform(0.1, 1, (dataset_size, dofs))
    phi = np.loadtxt('data/basis.txt',delimiter=",")
    solver = Fin(V)
    errors = np.zeros((dataset_size, 1))

    m = Function(V)
    for i in range(dataset_size):
        m.vector().set_local(z_s[i,:])
        w, y, A, B, C = solver.forward(m)
        psi = np.dot(A, phi)
        A_r, B_r, C_r, x_r, y_r = solver.reduced_forward(A, B, C, psi, phi)
        errors[i][0] = y - y_r 

    dataset = tf.data.Dataset.from_tensor_slices((z_s,errors))

    return dataset

def load_eval_dataset():
    '''
    Loads a validation dataset from disk
    '''
    try:
        z_s = np.loadtxt('data/z_s_eval.txt', delimiter=",")
        #  z_s = read_csv('data/z_s_train.txt', delimiter=",").values 
        errors = np.loadtxt('data/errors_eval.txt', delimiter=",", ndmin=2)
    except (FileNotFoundError, OSError) as err:
        print ("Error in loading saved training dataset. Run generate_and_save_dataset.")
        raise err

    return tf.data.Dataset.from_tensor_slices((z_s, errors))
    

def load_saved_dataset():
    '''
    Loads a training dataset from disk
    '''
    try:
        z_s = np.loadtxt('../data/z_s_train.txt', delimiter=",")
        #  z_s = read_csv('data/z_s_train.txt', delimiter=",").values 
        errors = np.loadtxt('../data/errors_train.txt', delimiter=",", ndmin=2)
    except (FileNotFoundError, OSError) as err:
        print ("Error in loading saved training dataset. Run generate_and_save_dataset.")
        raise err

    return tf.data.Dataset.from_tensor_slices((z_s, errors))

def generate_and_save_dataset(dataset_size, resolution=40):
    V = get_space(resolution)
    dofs = len(V.dofmap().dofs())
    z_s = np.random.uniform(0.1, 1, (dataset_size, dofs))
    phi = np.loadtxt('data/basis.txt',delimiter=",")
    solver = Fin(V)
    errors = np.zeros((dataset_size, 1))

    m = Function(V)
    for i in range(dataset_size):
        m.vector().set_local(z_s[i,:])
        w, y, A, B, C = solver.forward(m)
        psi = np.dot(A, phi)
        A_r, B_r, C_r, x_r, y_r = solver.reduced_forward(A, B, C, psi, phi)
        errors[i][0] = y - y_r 

    np.savetxt('../data/z_s_train.txt', z_s, delimiter=",")
    np.savetxt('../data/errors_train.txt', errors, delimiter=",")

def create_initializable_iterator(buffer_size, batch_size):
    z_s = np.loadtxt('../data/z_s_train.txt', delimiter=",")
    errors = np.loadtxt('../data/errors_train.txt', delimiter=",", ndmin=2)

    features_placeholder = tf.placeholder(z_s.dtype, z_s.shape)
    labels_placeholder = tf.placeholder(errors.dtype, errors.shape)

    dataset = tf.data.Dataset.from_tensor_slices((features_placeholder, labels_placeholder))

    iterator = dataset.shuffle(buffer_size).batch(batch_size).repeat().make_initializable_iterator()

