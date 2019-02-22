import os
import matplotlib
matplotlib.use('macosx')
import matplotlib.pyplot as plt
from tensorflow.keras import layers, Sequential
from tensorflow.keras.optimizers import Adam, RMSprop, Adadelta
from tensorflow.keras.layers import Dropout, Dense
from generate_fin_dataset import generate_five_param_np
from dolfin import set_log_level; set_log_level(40)
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args
from skopt import gp_minimize

(z_train, errors_train) = generate_five_param_np(4000)
(z_val, errors_val) = generate_five_param_np(200)

space = [Categorical(['relu', 'sigmoid', 'tanh'], name='activation'),
         Categorical([Adam, RMSprop, Adadelta], name='optimizer'),
         Real(1e-4, 1, prior="log-uniform", name='lr'),
         Integer(1, 6, name='n_hidden_layers'),
         Integer(10, 100, name='n_weights'),
         Integer(10, 200, name='batch_size'),
         Integer(100, 400, name='n_epochs')]

def parametric_model(activation, optimizer, lr, n_hidden_layers, n_weights, batch_size, n_epochs):
    model = Sequential()
    for i in range(n_hidden_layers):
        model.add(Dense(n_weights, activation=activation))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer=optimizer(lr=lr), metrics=['mape'])
    history = model.fit(z_train, errors_train, epochs=n_epochs, batch_size=batch_size,
            validation_data=(z_val, errors_val))
    vmape = history.history['val_mean_absolute_percentage_error'][-1]
    tr_losses = history.history['mean_absolute_percentage_error']
    vmapes = history.history['val_mean_absolute_percentage_error']
    plt.plot(tr_losses)
    plt.plot(vmapes)
    plt.legend(["Mean training error", "Mean validation error"])
    plt.xlabel("Epoch", fontsize=14)
    plt.ylabel("Absolute percentage error", fontsize=14)
    plt.show()
    plt.plot(errors_val)
    plt.plot(model.predict(z_val), 'k-x')
    plt.legend(['True error','Predicted error'])
    plt.ylabel('Error in average temperature y - y_r', fontsize=14)
    plt.xlabel('Validation dataset index', fontsize=14)
    plt.show()
    return vmape

@use_named_args(space)
def objective(**params):
    return parametric_model(**params)

#  res_gp = gp_minimize(objective, space, n_calls=50, random_state=0)

#  print("Best score: {}".format(res_gp.fun))
#  print('''Best parameters:\n
   #  Activation function: {}
   #  Optimizer: {}
   #  Learning rate: {}
   #  Num. hidden Layers: {}
   #  Num. weights: {}
   #  Batch size: {}
   #  Num. Epochs: {}'''.format(*res_gp.x))


#  if os.path.isfile('data/keras_model.index'):
    #  print ("Keras model weights loaded")
    #  model.load_weights('./keras_model')
#  vmape = parametric_model('relu', Adam, 0.0001, 6, 100, 10, 400)
vmape = parametric_model('relu', Adam, 0.0001, 6, 100, 10, 400)
#  model.save_weights('data/keras_model')
print ('\nError: {:2.3f}%'.format(vmape))
