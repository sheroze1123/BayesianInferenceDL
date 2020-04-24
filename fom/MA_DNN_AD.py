import sys; sys.path.append('../')
import os
import warnings
import string
import random
import numpy as np
import matplotlib; matplotlib.use('macosx')
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore',category=FutureWarning)

import tensorflow as tf
from tensorflow.keras import Sequential, Model, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import *
from tensorflow.python.framework import ops

tf.keras.backend.set_floatx('float64')

#  import dolfin as dl
#  import hippylib as hl
#  from fom.model_ad_diff import *

batch_size = 2

#  mesh_fname = 'ad_10k.xml'
#  mesh = dl.Mesh(mesh_fname)
#  Vh = dl.FunctionSpace(mesh, "Lagrange", 2)
#  ic_expr = dl.Expression('min(0.5,exp(-100*(pow(x[0]-0.35,2) +  pow(x[1]-0.7,2))))', element=Vh.ufl_element())
#  true_initial_condition = dl.interpolate(ic_expr, Vh).vector()

#  gamma = 1.
#  delta = 8.
#  prior = hl.BiLaplacianPrior(Vh, gamma, delta, robin_bc=True)
#  prior.mean = dl.interpolate(dl.Constant(0.25), Vh).vector()

t_init         = 0.
t_final        = 4.
t_1            = 0.
dt             = .1
observation_dt = .1

simulation_times = np.arange(t_init, t_final+.5*dt, dt)
observation_times = np.arange(t_1, t_final+.5*dt, observation_dt)

targets = np.loadtxt('targets.txt')
n_obs = len(targets)
N_t = len(simulation_times)
dofs = 10847
#  dofs = Vh.dim()
#  print(f"Degrees of freedom: {dofs}\n")


#  misfit = SpaceTimePointwiseStateObservation(Vh, observation_times, targets)
#  wind_velocity = computeVelocityField(mesh)

#  problem = TimeDependentAD(mesh, [Vh,Vh,Vh], prior, misfit, simulation_times, wind_velocity, True)
#  L_np = problem.L.array()
#  M_stab_np = problem.M_stab.array()
#  B_np = misfit.B.array()
#  np.save('AD_L.npy', L_np)
#  np.save('AD_M_stab.npy', M_stab_np)
#  np.save('AD_B.npy', B_np)
L_np = np.load('AD_L.npy')
M_stab_np = np.load('AD_M_stab.npy')
B_np =  np.load('AD_B.npy')

L = L_np
M_stab = M_stab_np.reshape((1, dofs, dofs))
B = B_np.reshape((1,n_obs, dofs))

Y_var = tf.Variable(tf.zeros([batch_size, n_obs * N_t], dtype=tf.float64), 
        dtype=tf.float64)

@tf.custom_gradient
def PTOMap(U_inp):
    '''
    Performs batched time-stepping, and accumulates observables
    '''
    U = tf.reshape(U_inp, [batch_size, dofs, 1])
    #  Y = tf.zeros([batch_size, n_obs * N_t], dtype=tf.float64)
    for t in range(N_t):
        Y_i_t = tf.reshape(tf.linalg.matmul(B, U), [batch_size, n_obs])
        begin_idx = t*n_obs
        end_idx = (t+1)*n_obs
        Y_var[:,begin_idx:end_idx].assign(Y_i_t)
        rhs = tf.linalg.matmul(M_stab, U)
        U = tf.scan(lambda a, x: tf.linalg.solve(L, x), rhs)
    Y_i_t = tf.reshape(tf.linalg.matmul(B, U), [batch_size, n_obs])
    Y_var[:,-n_obs:].assign(Y_i_t)

    def grad(dy):
        # TODO: Adjoint method
        return dy  * tf.ones([batch_size, n_obs * N_t, dofs])
    
    return Y_var, grad

# Training constants
n_weights = 10
lr = 3e-4
alpha = 0.1
train_dataset_size = 1000 # Get subset of data due to slowness
val_dataset_size = 100
n_epochs = 100

U_train = np.load('../data/AD_parameters_training_dataset.npy')
U_val = np.load('../data/AD_parameters_validation_dataset.npy')
Y_train = np.load('../data/AD_fom_qois_training_dataset.npy')
Y_val = np.load('../data/AD_fom_qois_validation_dataset.npy')

U_train = U_train[:train_dataset_size, :]
U_val = U_val[:val_dataset_size, :]
Y_train = Y_train[:train_dataset_size, :]
Y_val = Y_val[:val_dataset_size, :]


# Tensorflow Keras model
Y_input = Input(shape=(n_obs * N_t,), name='Y_input')
layer1 = Dense(n_weights, activation='relu')
layer1out = layer1(Y_input)
layer2 = Dense(n_weights, activation='relu')
layer2out = layer2(layer1out)
U_output = Dense(dofs, name='U_output')(layer2out)
model = Model(inputs=Y_input, outputs=U_output)

print("Model defined\n")

########################################################################
# Loss function gradient check
########################################################################
#  L = tf.reduce_mean(tf.square(Y_input - G(tf.math.exp(U_output))))
#  grad_bias_1 = gradients(L, model.trainable_weights[1])[0]
#  dummy_inp = np.random.randn(batch_size, n_obs)
#  grad_bias_1_eval = get_session().run(grad_bias_1, feed_dict={Y_input:dummy_inp})
#  bias_1_vals = layer1.get_weights()[1]
#  kernel_vals = layer1.get_weights()[0]
#  L_0 = get_session().run(L, feed_dict={Y_input:dummy_inp})
#  eps_bias = np.random.randn(n_weights) #Random direction to perturn bias weights
#  eps_bias = eps_bias/np.linalg.norm(eps_bias)
#  dir_grad = np.dot(grad_bias_1_eval, eps_bias)

#  n_eps = 32
#  hs = np.power(2., -np.arange(n_eps))

#  err_grads = []
#  grads = []
#  for h in hs:
    #  b_h = bias_1_vals + h * eps_bias #Perturb bias
    #  layer1.set_weights([kernel_vals, b_h]) #Update bias in model
    #  L_h = get_session().run(L, feed_dict={Y_input:dummy_inp}) #Compute loss post bias
    #  a_g = (L_h - L_0)/h
    #  grads.append(a_g)
    #  err = abs(a_g - dir_grad)/abs(dir_grad)
    #  err_grads.append(err)

#  plt.loglog(hs, err_grads, "-ob", label="Error Grad")
#  plt.loglog(hs, (.5*err_grads[0]/hs[0])*hs, "-.k", label="First Order")
#  plt.savefig('grad_fd_check_G_wrt_bias.png', dpi=200)
#  plt.cla()
#  plt.clf()

def custom_loss(U_true, U_pred):
    loss =  tf.reduce_mean(tf.square(U_pred - tf.math.log(U_true)))
    fwd_loss = alpha * tf.reduce_mean(tf.square(Y_input - PTOMap(tf.math.exp(U_output))))
    return loss + fwd_loss

def custom_loss_unpacked(Y, U_o, Y_i):
    '''
    Custom loss with forward solve built in
    '''
    def lossF(U_true, U_pred):
        loss =  tf.reduce_mean(tf.square(tf.math.exp(U_pred) - U_true))
        U = tf.reshape(tf.math.exp(U_o), [batch_size, dofs, 1])
        for t in range(N_t):
            Y_i_t = tf.reshape(tf.linalg.matmul(B, U), [batch_size, n_obs])
            begin_idx = t*n_obs
            end_idx = (t+1)*n_obs
            Y[:,begin_idx:end_idx].assign(Y_i_t)
            rhs = tf.linalg.matmul(M_stab, U)
            U = tf.scan(lambda a, x: tf.linalg.solve(L, x), rhs)
        Y_i_t = tf.reshape(tf.linalg.matmul(B, U), [batch_size, n_obs])
        Y[:,-n_obs:].assign(Y_i_t)
        fwd_loss = alpha * tf.reduce_mean(tf.square(Y_i - Y))
        return loss + fwd_loss
    return lossF

print("Custom loss function defined\n")
model.compile(loss=custom_loss_unpacked(Y_var, U_output, Y_input), 
        optimizer=Adam(lr=lr), experimental_run_tf_function=False)
print("Model compiled\n")
model.summary()


model.fit(Y_train, U_train, batch_size=batch_size, epochs=n_epochs, shuffle=True, 
        validation_data=(Y_val, U_val))
        
