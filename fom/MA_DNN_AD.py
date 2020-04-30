import sys; sys.path.append('../')
import os
import warnings
import string
import random
import numpy as np
import matplotlib; matplotlib.use('agg')
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore',category=FutureWarning)

import tensorflow as tf
from tensorflow.keras import Sequential, Model, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import *
from tensorflow.python.framework import ops
tf.keras.backend.set_floatx('float64')

from MA_model_def import *

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
Mt_stab_np = np.load('AD_Mt_stab.npy')
Lt_np = np.load('AD_Lt.npy')

L = L_np
Lt = Lt_np
M_stab = M_stab_np.reshape((1, dofs, dofs))
Mt_stab = Mt_stab_np.reshape((1, dofs, dofs))
B = B_np.reshape((1,n_obs, dofs))

Y_var = tf.Variable(tf.zeros([batch_size, n_obs * N_t], dtype=tf.float64), 
        dtype=tf.float64)
U_var = tf.Variable(tf.zeros([batch_size, dofs, N_t], dtype=tf.float64), 
        dtype=tf.float64)
grad_state = tf.Variable(tf.zeros([batch_size, dofs, N_t], dtype=tf.float64),
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
n_weights = 100
lr = 3e-5
alpha = 1.0
train_dataset_size = 1000 # Get subset of data due to slowness
val_dataset_size = 100
n_epochs = 1000

U_train = np.load('../data/AD_parameters_training_dataset.npy')
U_val = np.load('../data/AD_parameters_validation_dataset.npy')
Y_train = np.load('../data/AD_fom_qois_training_dataset.npy')
Y_val = np.load('../data/AD_fom_qois_validation_dataset.npy')

U_train = U_train[:train_dataset_size, :]
U_val = U_val[:val_dataset_size, :]
Y_train = Y_train[:train_dataset_size, :]
Y_val = Y_val[:val_dataset_size, :]

# Tensorflow Keras model
#  Y_input = Input(shape=(n_obs * N_t,), name='Y_input')
#  layer1 = Dense(n_weights, activation='relu')
#  layer1out = layer1(Y_input)
#  layer2 = Dense(n_weights, activation='relu')
#  layer2out = layer2(layer1out)
#  U_output = Dense(dofs, name='U_output')(layer2out)
#  model = Model(inputs=Y_input, outputs=U_output)

model = res_bn_fc_model(ELU(), Adam, 3e-4, 2, 100, n_obs*N_t, dofs)
Y_input = model.inputs[0]
U_output = model.outputs[0]

print("Model defined\n")

def custom_loss_MSE(Y, U_o, Y_i):
    '''
    Custom loss with forward solve built in
    '''
    def lossF(U_true, U_pred):
        loss =  tf.reduce_mean(tf.square(tf.math.exp(U_pred) - U_true))
        return loss
    return lossF

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

def misfit_wrapper(Y, U_var, grad_state, Y_pred):
    @tf.custom_gradient
    def misfit(U_o):
        U = tf.reshape(tf.math.exp(U_o), [batch_size, dofs, 1])
        U_var[:, :, 0].assign(U)
        for t in range(N_t):
            Y_i_t = tf.reshape(tf.linalg.matmul(B, U), [batch_size, n_obs])
            begin_idx = t*n_obs
            end_idx = (t+1)*n_obs
            Y[:,begin_idx:end_idx].assign(Y_i_t)
            rhs = tf.linalg.matmul(M_stab, U)
            U = tf.scan(lambda a, x: tf.linalg.solve(L, x), rhs)
            U_var[:,:,t].assign(U)
        Y_i_t = tf.reshape(tf.linalg.matmul(B, U), [batch_size, n_obs])
        Y[:,-n_obs:].assign(Y_i_t)

        def grad(dy):
            p = tf.zeros([batch_size, dofs, 1], dtype=tf.float64) 

            for t in range(N_t-1, 0, -1):
                # When t is a time that is not observed, Bt(Bu-d) should be zero
                rhs = tf.linalg.matmul(Mt_stab, p) - grad_state[:, :, t]
                p = tf.scan(lambda a, x: tf.linalg.solve(Lt, x), rhs)

            g = tf.scan(lambda a, x: tf.linalg.solve(M, x), -tf.linalg.solve(Mt_stab, p))
            return dy * g
        
        misfit_val = tf.reduce_mean(tf.square(Y_pred - Y))
        return misfit_val, grad
    return misfit

def custom_loss_cg(Y, U_o, Y_i):
    '''
    Custom loss with forward solve built in
    '''
    forward_loss = misfit_wrapper(Y_var, U_var, grad_state, Y_i)
    def lossF(U_true, U_pred):
        loss =  tf.reduce_mean(tf.square(tf.math.exp(U_pred) - U_true))
        fwd_loss = forward_loss(U_o)
        return loss + fwd_loss
    return lossF

print("Custom loss function defined\n")
model.compile(loss=custom_loss_unpacked(Y_var, U_output, Y_input), 
        optimizer=Adam(lr=lr), experimental_run_tf_function=False, metrics=['mape'])
print("Model compiled\n")
model.summary()

cbks = [LearningRateScheduler(lr_schedule)]
history = model.fit(Y_train, U_train, batch_size=batch_size, epochs=n_epochs, shuffle=True, 
        validation_data=(Y_val, U_val), callbacks=cbks)

tr_losses = history.history['mape']
vmapes = history.history['val_mape']
plt.cla()
plt.clf()
plt.semilogy(tr_losses)
plt.semilogy(vmapes)
plt.legend(["Mean training error", "Mean validation error"], fontsize=10)
plt.xlabel("Epoch", fontsize=10)
plt.ylabel("Absolute percentage error", fontsize=10)
plt.savefig('training_error_with_MA_term.png', dpi=200)
