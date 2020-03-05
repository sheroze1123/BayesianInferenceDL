import sys; sys.path.append('../')
import numpy as np
import matplotlib; matplotlib.use('agg')
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings('ignore',category=FutureWarning)

from dolfin import set_log_level; set_log_level(40)

from tensorflow.keras import Sequential, Model, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import *
from tensorflow.keras.regularizers import l1_l2, l2, l1
from tensorflow.keras.callbacks import LearningRateScheduler

from deep_learning.generate_fin_dataset import gen_affine_avg_rom_dataset

def load_dataset_avg_rom(load_prev=True, tr_size=6000, v_size=500, genrand=False):
    '''
    Load dataset where the conductivity is parametrized as a FEniCS function
    and the QoI is the averaged temperature per subfin
    '''
    if os.path.isfile('../data/z_aff_avg_tr.npy') and load_prev:
        z_train = np.load('../data/z_aff_avg_tr.npy')
        errors_train =  np.load('../data/errors_aff_avg_tr.npy')
    else:
        (z_train, errors_train) = gen_affine_avg_rom_dataset(tr_size, genrand=genrand)

    if os.path.isfile('../data/z_aff_avg_eval.npy') and load_prev:
        z_val = np.load('../data/z_aff_avg_eval.npy')
        errors_val =  np.load('../data/errors_aff_avg_eval.npy')
    else:
        (z_val, errors_val) = gen_affine_avg_rom_dataset(v_size, genrand=genrand)

    return z_train, errors_train, z_val, errors_val

def parametric_model(activation, 
        optimizer, 
        lr, 
        n_hidden_layers, 
        n_weights, 
        batch_size, 
        n_epochs, 
        z_train=None,
        errors_train=None, 
        z_val=None, 
        errors_val=None):
    '''
    Creates a trained neural network given hyperparameters.

    Arguments:
        activation (str): Tensorflow activation function
        optimizer (tf.keras.optimizers): Tensorflow optimizer
        lr (float): Learning rate for optimizer
        n_hidden_layers (int): Number of hidden layers
        n_weights (int): Number of neurons per hidden layer
        batch_size (int): Size of a training batch
        n_epochs (int): Number of epochs to train for
        z_train (np.ndarray): Thermal conductivity values for training
        errors_train (np.ndarray): Corresponding ROM discrepancy for training
        z_val (np.ndarray): Thermal conductivity values for validation
        errors_val (np.ndarray): Corresponding ROM discrepancy for validation
    '''

    if z_train is None:
        z_train, errors_train, z_val, errors_val = load_dataset_avg_rom(False)

    input_shape = z_train.shape[1]
    model = Sequential()
    #  model.add(Dense(10, input_shape=(5,)))
    for i in range(n_hidden_layers):
        if i==0:
            model.add(Dense(n_weights, activation=activation, input_shape=(input_shape,)))
        else:
            model.add(Dense(n_weights, activation=activation))
    model.add(Dense(9))
    model.compile(loss='mse', optimizer=optimizer(lr=lr), metrics=['mape'])
    model.load_weights('../data/keras_model_avg_best')
    history = model.fit(z_train, errors_train, epochs=n_epochs, batch_size=batch_size,
            validation_data=(z_val, errors_val))

    # Mean Absolute Relative Error is the validation metric
    vmape = history.history['val_mean_absolute_percentage_error'][-1]

    # Plots the training and validation loss
    #  tr_losses = history.history['mean_absolute_percentage_error']
    #  vmapes = history.history['val_mean_absolute_percentage_error']
    #  plt.semilogy(tr_losses[200:])
    #  plt.semilogy(vmapes[200:])
    #  plt.legend(["Mean training error", "Mean validation error"], fontsize=10)
    #  plt.xlabel("Epoch", fontsize=10)
    #  plt.ylabel("Absolute percentage error", fontsize=10)
    #  plt.savefig("subfin_avg_tr_v_loss.png",dpi=250)

    # Saves Keras model (useful for Bayesian inference)
    #  save_weights = True
    #  if (save_weights):
        #  model.save_weights('data/keras_model_avg')

    # Save best model
    best_vmape = np.loadtxt('../data/best_vmape.txt').item()
    best_vmape = 50
    if (vmape < best_vmape):
        np.savetxt('../data/best_vmape.txt',  np.array([vmape]))
        model.save_weights('../data/keras_model_avg_best')

    return vmape

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

    if os.path.isfile('../data/keras_model_avg_best.index'):
        print ("Best Keras model weights loaded")
        model.load_weights('../data/keras_model_avg_best')
    elif os.path.isfile('../data/keras_model.index'):
        print ("Keras model weight loaded")
        modal.load_weights('../data/keras_model')
    else: 
        print ("Keras model not found!")

    return model

def bn_fc_model(activation, optimizer, lr, n_layers, n_weights, input_shape):
    # TODO: Add weight regularization, adaptive learning rate
    model = Sequential()
    for i in range(n_layers):
        if i==0:
            model.add(Dense(2 * n_weights, input_shape=(input_shape,), activation=None))
            model.add(BatchNormalization())
            model.add(activation())
        else:
            model.add(Dense(n_weights, activation=None))
            model.add(BatchNormalization())
            model.add(activation())
    model.add(Dense(9))
    model.compile(loss='mse', optimizer=optimizer(lr=lr), metrics=['mape'])
    return model

def residual_unit(x, activation, n_weights):
    res = x

    out = BatchNormalization()(x)
    out = activation(out)
    out = Dense(n_weights, activation=None, kernel_regularizer=l1_l2(1e-4, 1e-4))(out)

    out = BatchNormalization()(x)
    out = activation(out)
    out = Dense(n_weights, activation=None, kernel_regularizer=l1_l2(1e-4, 1e-4))(out)

    out = add([res, out])
    return out

def res_bn_fc_model(activation, optimizer, lr, n_layers, n_weights, input_shape=1446, 
        output_shape=9):
    inputs = Input(shape=(input_shape,))
    y = Dense(n_weights, input_shape=(input_shape,), activation=None, 
            kernel_regularizer=l1_l2(1e-4, 1e-4))(inputs)
    out = residual_unit(y, activation, n_weights)
    for i in range(1,n_layers):
        out = residual_unit(out, activation, n_weights)
    out = BatchNormalization()(out)
    out = activation(out)
    out = Dense(output_shape)(out)
    model = Model(inputs=inputs, outputs=out)
    model.compile(loss='mse', optimizer=optimizer(lr=lr), metrics=['mape'])
    return model

def lr_schedule(epoch):
    if epoch<=1000:
        return 3e-4
    elif epoch<=3000:
        return 1e-5
    elif epoch<=7500:
        return 5e-6
    elif epoch<=9000:
        return 1e-7
    else:
        return 1e-7

def lr_schedule_pre(epoch):
    if epoch<=500:
        return 3e-4
    elif epoch<=1000:
        return 3e-5
    elif epoch<=1500:
        return 3e-6
    elif epoch<=2000:
        return 1e-6
    else:
        return 5e-7

def load_surrogate_model(randobs=True):
    if randobs:
        model = res_bn_fc_model(ELU(), Adam, 3e-4, 3, 50, 1446, 40)
        model.summary()
        #  model.load_weights('../data/surrogate_forward_model_3_e_4_layer_2')
        model.load_weights('../data/surrogate_forward_model_3_e_4_layer_2')
        return model
    else:
        model = res_bn_fc_model(ELU(), Adam, 3e-4, 3, 50, 1446)
        model.summary()
        #  model.load_weights('../data/surrogate_forward_model_3_e_4_layer_2')
        model.load_weights('../data/surrogate_forward_model_3_e_4_layer_2_avg_obs')
        return model

def load_bn_model(randobs=True):
    if randobs:
        #  model = res_bn_fc_model(ELU(), Adam, 3e-4, 3, 100, 1446, 40)
        model = res_bn_fc_model(ELU(), Adam, 3e-4, 3, 50, 1446, 40)
        model.summary()
        #  model.load_weights('../data/error_model_lr_3e_4_layer_2')
        model.load_weights('../data/keras_model_res_bn_random')
        return model
    else:
        model = res_bn_fc_model(ELU(), Adam, 3e-4, 5, 50, 1446)
        model.summary()
        model.load_weights('../data/keras_model_res_bn')
        return model

'''
z_train, errors_train, z_val, errors_val = load_dataset_avg_rom(False, genrand=False)
qois_train = np.load('../data/qois_avg_tr.npy')
qois_eval = np.load('../data/qois_avg_eval.npy')
model = res_bn_fc_model(ELU(), Adam, 3e-4, 2, 1446, 1446, 40)
model.summary()
#  model.load_weights('../data/error_model_lr_3e_4_layer_2')

cbks = [LearningRateScheduler(lr_schedule)]
history = model.fit(z_train, errors_train, epochs=5000, batch_size=500, shuffle=True, 
        validation_data=(z_val, errors_val),
        callbacks=cbks)
model.save_weights('../data/error_model_lr_3e_4_layer_2')

#  # Plots the training and validation loss
tr_losses = history.history['mean_absolute_percentage_error']
vmapes = history.history['val_mean_absolute_percentage_error']
plt.semilogy(tr_losses)
plt.semilogy(vmapes)
plt.legend(["Mean training error", "Mean validation error"], fontsize=10)
plt.xlabel("Epoch", fontsize=10)
plt.ylabel("Absolute percentage error", fontsize=10)
plt.savefig('training_error_rom_dl.png', dpi=200)

model = res_bn_fc_model(ELU(), Adam, 3e-4, 2, 1446, 1446, 40)
model.summary()
#  model.load_weights('../data/surrogate_forward_model_3_e_4_layer_2')
cbks = [LearningRateScheduler(lr_schedule)]
history = model.fit(z_train, qois_train, epochs=5000, batch_size=500, shuffle=True, 
        validation_data=(z_val, qois_eval),
        callbacks=cbks)
model.save_weights('../data/surrogate_forward_model_3_e_4_layer_2')
#  # Plots the training and validation loss
tr_losses = history.history['mean_absolute_percentage_error']
vmapes = history.history['val_mean_absolute_percentage_error']
plt.cla()
plt.clf()
plt.semilogy(tr_losses)
plt.semilogy(vmapes)
plt.legend(["Mean training error", "Mean validation error"], fontsize=10)
plt.xlabel("Epoch", fontsize=10)
plt.ylabel("Absolute percentage error", fontsize=10)
plt.savefig('training_error_surrogate.png', dpi=200)
'''
