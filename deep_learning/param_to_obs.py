import sys; sys.path.append('../')
import os
import warnings
import string
import random
import numpy as np
warnings.filterwarnings('ignore',category=FutureWarning)

import tensorflow as tf
from tensorflow.keras import Sequential, Model, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import *
from tensorflow.python.framework import ops

tf.keras.backend.set_floatx('float64')
tf.compat.v1.disable_eager_execution()

from fom.forward_solve import Fin
from fom.thermal_fin import get_space
import dolfin as dl
dl.set_log_level(40)


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
    rnd_name = 'PyFuncGrad-' + ''.join(random.choices(string.ascii_uppercase, k=4))

    # Register Tensorflow Gradient
    tf.RegisterGradient(rnd_name)(grad_func)

    # Get current graph
    g = tf.compat.v1.get_default_graph()

    # Add gradient override map
    with g.gradient_override_map(
            {"PyFunc": rnd_name, "PyFuncStateless": rnd_name}):
        return tf.compat.v1.py_func(func, inp, Tout, stateful=stateful, name=name)

batch_size = 10
y = np.zeros((batch_size, solver.n_obs), dtype=np.float64)
grad_y = np.zeros((batch_size, solver.n_obs, solver.dofs), dtype=np.float64)

def fwd_w_grad(U_v):
    for i in range(batch_size):
        U_func.vector().set_local(U_v[i,:])
        w, _, _, _, _ = solver.forward(U_func)
        y[i,:] = solver.qoi_operator(w)
        grad_y[i,:,:] = solver.sensitivity(U_func)
    return y, grad_y

def _fwd_grad(op, grad_fwd, grad_grad):
    dL_dU = tf.reshape(tf.matmul(tf.reshape(grad_fwd, (batch_size, 1, solver.n_obs)), op.outputs[1]), (batch_size, solver.dofs))
    return dL_dU

def G(x, name=None):
    with ops.name_scope(name, "G_impl", [x]) as name:
        G_U, grad_G_U = _py_func_with_gradient(fwd_w_grad,
                              [x],
                              [tf.float64, tf.float64],
                              name=name,
                              grad_func=_fwd_grad)
        return G_U

#  @tf.custom_gradient
#  def G(x):
    #  import pdb; pdb.set_trace()
    #  U_v = x.numpy() #DOES NOT WORK AS THIS WILL FAIL.
    #  for i in range(batch_size):
        #  U_func.vector().set_local(U_v[i,:])
        #  w, _, _, _, _ = solver.forward(U_func)
        #  y[i,:] = solver.qoi_operator(w)
        #  grad_y[i,:,:] = solver.sensitivity(U_func)
    #  def grad(dy):
        #  '''
        #  A vector-argument vector-valued function G's derivatives 
        #  should be its Jacobian matrix J. Here we are expressing 
        #  the Jacobian J as a function grad_G which defines how J 
        #  will transform a vector grad_ys when left-multiplied with it 
        #  (grad_ys * J). This functional representation of a matrix is 
        #  convenient to use for chain-rule calculation.
        #  '''
        #  return dy * grad_y
    #  return y, grad

# Refer to https://www.tensorflow.org/api_docs/python/tf/custom_gradient
#  @tf.custom_gradient
#  def G(U):
    #  #  U_v = U.eval(session=tf.compat.v1.Session()) # U.numpy() in tf v2.x 
    #  U_v = U.numpy()
    #  U_func.vector().set_local(U_v)
    #  w, _, _, _, _ = solver.forward(U_func)
    #  y = solver.qoi_operator(w)
    #  y_tf = tf.convert_to_tensor(y, dtype=tf.float64)

    #  J = solver.sensitivity(U_v)
    #  J_tf = tf.convert_to_tensor(J, dtype=tf.float64)

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

# Training constants
n_weights = 80
lr = 3e-4
alpha = 0.1
train_dataset_size = 1000 # Get subset of data due to slowness
val_dataset_size = 100
n_epochs = 100

# Tensorflow Keras model
Y_input = Input(shape=(solver.n_obs,), name='Y_input')
layer1 = Dense(n_weights, input_shape=(solver.dofs,), activation='relu')(Y_input)
layer2 = Dense(n_weights, activation='relu')(layer1)
U_output = Dense(solver.dofs, name='U_output')(layer2)
model = Model(inputs=Y_input, outputs=U_output)

def custom_loss(U_true, U_pred):
    loss =  tf.reduce_mean(tf.square(U_pred - tf.math.log(U_true)))
    fwd_loss = alpha * tf.reduce_mean(tf.square(Y_input - G(tf.math.exp(U_output))))
    #  fwd_loss = K.print_tensor(fwd_loss, message='fwd_loss=')
    #  loss = K.print_tensor(loss, message='mse loss=')
    return loss + fwd_loss

model.compile(loss=custom_loss, optimizer=Adam(lr=lr))
model.summary()

U_train = np.load('../data/parameters_training_dataset.npy')
U_val = np.load('../data/parameters_validation_dataset.npy')
Y_train = np.load('../data/fom_qois_training_dataset.npy')
Y_val = np.load('../data/fom_qois_validation_dataset.npy')

U_train = U_train[:train_dataset_size, :]
U_val = U_val[:val_dataset_size, :]
Y_train = Y_train[:train_dataset_size, :]
Y_val = Y_val[:val_dataset_size, :]

model.fit(Y_train, U_train, batch_size=batch_size, epochs=n_epochs, shuffle=True, 
        validation_data=(Y_val, U_val))
        

