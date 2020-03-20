import sys; sys.path.append('../')
import os
import warnings
import string
import random
import numpy as np
warnings.filterwarnings('ignore',category=FutureWarning)

from fom.forward_solve import Fin
from fom.thermal_fin import get_space
import dolfin as dl
dl.set_log_level(40)

import tensorflow as tf
from tensorflow.keras import Sequential, Model, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import *
from tensorflow.keras.backend import get_session, gradients

# Create FunctionSpace
V = get_space(40)

# Create FEniCS forward solver with surface obs as QoI
solver = Fin(V, True) 
U_func = dl.Function(V)

def _py_func_with_gradient(func, inp, Tout, stateful=True, name=None,
                           grad_func=None):
    """
    PyFunc defined as given by Tensorflow
    :param func: Custom Function
    :param inp: Function Inputs
    :param Tout: Ouput Type of out Custom Function
    :param stateful: Calculate Gradients when stateful is True
    :param name: Name of the PyFunction
    :param grad: Custom Gradient Function
    :return:
    """
    # Generate random name in order to avoid conflicts with inbuilt names
    rnd_name = 'PyFuncGrad-' + ''.join(random.choices(string.ascii_uppercase + string.digits, k=4))

    # Register Tensorflow Gradient
    tf.RegisterGradient(rnd_name)(grad_func)

    # Get current graph
    g = tf.get_default_graph()

    # Add gradient override map
    with g.gradient_override_map(
            {"PyFunc": rnd_name, "PyFuncStateless": rnd_name}):
        return tf.py_func(func, inp, Tout, stateful=stateful, name=name)

batch_size = 100
y = np.zeros((batch_size, solver.n_obs), dtype=np.float32)
grad_y = np.zeros((batch_size, solver.n_obs, solver.dofs), dtype=np.float32)

def fwd_map(U_v):
    for i in range(batch_size):
        U_func.vector().set_local(U_v[i,:])
        w, _, _, _, _ = solver.forward(U_func)
        y[i,:] = solver.qoi_operator(w)
    return y

def fwd_map_grad_impl(U_v):
    import pdb; pdb.set_trace()
    for i in range(batch_size):
        U_func.vector().set_local(U_v[i,:])
        grad_y[i,:,:] = solver.sensitivity(U_func)
    return grad_y

def fwd_map_grad(unused_op, grad):
    x = unused_op.inputs[0]
    grad_wrapper = tf.py_func(fwd_map_grad_impl, [x], [tf.float32])
    return grad * grad_wrapper

# Refer to https://www.tensorflow.org/api_docs/python/tf/custom_gradient
#  @tf.custom_gradient
#  def G(U):
    #  #  U_v = U.eval(session=tf.compat.v1.Session()) # U.numpy() in tf v2.x 
    #  U_v = U.numpy()
    #  U_func.vector().set_local(U_v)
    #  w, _, _, _, _ = solver.forward(U_func)
    #  y = solver.qoi_operator(w)
    #  y_tf = tf.convert_to_tensor(y, dtype=tf.float32)

    #  J = solver.sensitivity(U_v)
    #  J_tf = tf.convert_to_tensor(J, dtype=tf.float32)

    #  def grad_G(grad_ys):
        #  '''
        #  A vector-argument vector-valued function G's derivatives 
        #  should be its Jacobian matrix J. Here we are expressing 
        #  the Jacobian J as a function grad_G which defines how J 
        #  will transform a vector grad_ys when left-multiplied with it 
        #  (grad_ys * J). This functional representation of a matrix is 
        #  convenient to use for chain-rule calculation.
        #  '''
        #  return grad_ys * J_tf

    #  return y_tf, grad_G

# Tensorflow Keras model
n_weights = 100
lr = 3e-4
Y_input = Input(shape=(solver.n_obs,), name='Y_input')
layer1 = Dense(n_weights, input_shape=(solver.dofs,), activation='relu')(Y_input)
layer2 = Dense(n_weights, activation='relu')(layer1)
U_output = Dense(solver.dofs, name='U_output')(layer2)
model = Model(inputs=Y_input, outputs=U_output)

G = _py_func_with_gradient(fwd_map, [U_output], [tf.float32], grad_func=fwd_map_grad)[0] 


alpha = 0.1

def custom_loss(U_pred, U_true):
    return tf.reduce_mean(tf.square(U_pred - U_true)) + \
        alpha * tf.reduce_mean(tf.square(Y_input - G))

model.compile(loss=custom_loss, optimizer=Adam(lr=lr))
model.summary()

U_train = np.load('../data/parameters_training_dataset.npy')
U_val = np.load('../data/parameters_validation_dataset.npy')
Y_train = np.load('../data/fom_qois_training_dataset.npy')
Y_val = np.load('../data/fom_qois_validation_dataset.npy')

d_size = 1000 # Get subset of data due to slowness
U_train = U_train[:d_size, :]
U_val = U_val[:d_size, :]
Y_train = Y_train[:d_size, :]
Y_val = Y_val[:d_size, :]

model.fit(Y_train, U_train, batch_size=100, epochs=1000, shuffle=True, 
        validation_data=(Y_val, U_val))
        

