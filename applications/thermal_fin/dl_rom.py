import os
from tensorflow.keras import layers, Sequential
from tensorflow.keras.layers import Dropout, Dense, Input

def load_parametric_model(activation, 
        optimizer, lr, n_hidden_layers, n_weights, batch_size, n_epochs):

    model = Sequential()
    model.add(Dense(10, input_shape=(5,)))
    for i in range(n_hidden_layers):
        model.add(Dense(n_weights, activation=activation))
    model.add(Dense(5))
    model.compile(loss='mse', optimizer=optimizer(lr=lr), metrics=['mape'])

    if os.path.isfile('data/keras_model.index'):
        print ("Keras model weights loaded")
        model.load_weights('data/keras_model')
    else: 
        print ("Keras model not found!")

    return model

def load_parametric_model_avg(activation,
        optimizer, lr, n_hidden_layers, n_weights, batch_size, n_epochs, input_shape):

    model = Sequential()
    for i in range(n_hidden_layers):
        if i==0:
            model.add(Dense(n_weights, activation=activation, input_shape=(input_shape,)))
        else:
            model.add(Dense(n_weights, activation=activation))
    model.add(Dense(5))
    model.compile(loss='mse', optimizer=optimizer(lr=lr), metrics=['mape'])

    if os.path.isfile('data/keras_model_avg.index'):
        print ("Keras model weights loaded")
        model.load_weights('data/keras_model_avg')
    else: 
        print ("Keras model not found!")

    return model
