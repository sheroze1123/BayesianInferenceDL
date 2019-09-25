import sys; sys.path.append('../')
import numpy as np
import matplotlib; matplotlib.use('agg')
import matplotlib.pyplot as plt
import os

from dolfin import set_log_level; set_log_level(40)

from tensorflow.keras import layers, Sequential
from tensorflow.keras.optimizers import Adam, RMSprop, Adadelta
from tensorflow.keras.layers import Dropout, Dense

from deep_learning.generate_fin_dataset import gen_affine_avg_rom_dataset

def load_dataset_avg_rom(load_prev=True, tr_size=4000, v_size=200):
    '''
    Load dataset where the conductivity is parametrized as a FEniCS function
    and the QoI is the averaged temperature per subfin
    '''
    if os.path.isfile('../data/z_aff_avg_tr.txt') and load_prev:
        z_train = np.loadtxt('../data/z_aff_avg_tr.txt', delimiter=',')
        errors_train =  np.loadtxt('../data/errors_aff_avg_tr.txt', delimiter=',')
    else:
        (z_train, errors_train) = gen_affine_avg_rom_dataset(tr_size)
        np.savetxt('../data/z_aff_avg_tr.txt', z_train, delimiter=',')
        np.savetxt('../data/errors_aff_avg_tr.txt', errors_train, delimiter=',')

    if os.path.isfile('../data/z_aff_avg_eval.txt') and load_prev:
        z_val = np.loadtxt('../data/z_aff_avg_eval.txt', delimiter=',')
        errors_val =  np.loadtxt('../data/errors_aff_avg_eval.txt', delimiter=',')
    else:
        (z_val, errors_val) = gen_affine_avg_rom_dataset(v_size)
        np.savetxt('../data/z_aff_avg_eval.txt', z_val, delimiter=',')
        np.savetxt('../data/errors_aff_avg_eval.txt', errors_val, delimiter=',')

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
        z_train, errors_train, z_val, errors_val = load_dataset_avg_rom()

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

#  vmape = parametric_model('elu', Adam, 0.0003, 5, 58, 200, 2000)
#  print ('\nError: {:2.3f}%'.format(vmape))

#  z_train, errors_trainm, z_val, errors_val = gen_avg_rom_dataset()
#  vmape = parametric_model('relu', Adam, 0.001128, 4, 66, 94, 400, 
        #  z_train, errors_train, z_val, errors_val)



#  z_train = np.loadtxt('data/z_avg_tr.txt', delimiter=",")
#  errors_train = np.loadtxt('data/errors_avg_tr.txt', delimiter=",")
#  z_val = np.loadtxt('data/z_avg_eval.txt', delimiter=",")
#  errors_val = np.loadtxt('data/errors_avg_eval.txt', delimiter=",")
#  vmape = parametric_model('elu', Adam, 0.001, 4, 50, 10, 400, 
        #  z_train, errors_train, z_val, errors_val)
#  print ('\nError: {:2.3f}%'.format(vmape))

#  z_train, errors_train, z_val, errors_val = load_dataset_avg_rom()
#  model = load_parametric_model_avg('elu', Adam, 0.0003, 5, 58, 200, 2000, 1446)
#  err_predict = model.predict(z_val)
#  pred_err = np.linalg.norm(err_predict - errors_val)
#  pred_rel_err = np.linalg.norm(err_predict - errors_val) / np.linalg.norm(errors_val)
#  print(pred_err)
#  print(pred_rel_err) # best being 0.016
