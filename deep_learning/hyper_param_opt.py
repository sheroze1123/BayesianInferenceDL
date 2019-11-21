import matplotlib; matplotlib.use('agg')
import matplotlib.pyplot as plt
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args
from skopt import gp_minimize
from skopt.plots import plot_convergence
from tensorflow.keras.optimizers import Adam, Adadelta
from tensorflow.keras.layers import *
from dl_model import res_bn_fc_model, parametric_model, load_dataset_avg_rom

z_train, errors_train, z_val, errors_val = load_dataset_avg_rom()

# Defines the hyperparameter space
activations = Categorical([ELU(), Activation('tanh'), Activation('sigmoid')], 
        name='activation')
optimizers = Categorical([Adam, Adadelta], name='optimizer')
learning_rates = Real(1e-8, 1e-3, prior="log-uniform", name='lr')
layer_sizes = Integer(1, 6, name='n_layers')
weight_vals = Integer(10, 200, name='n_weights')
batch_sizes = Integer(100, 500, name='batch_size')
space = [activations, optimizers, learning_rates, layer_sizes, weight_vals, batch_sizes]

@use_named_args(space)
def obj(activation, optimizer, lr, n_layers, n_weights, batch_size):
    print('''\nParameters:\n
     Activation function: {}
     Optimizer: {}
     Learning rate: {}
     Num. hidden Layers: {}
     Num. weights: {}
     Batch size: {}\n'''.format(activation, optimizer, lr, n_layers, n_weights, batch_size))
    model = res_bn_fc_model(activation, optimizer, lr, n_layers, n_weights)  
    history = model.fit(z_train, errors_train, epochs=500, batch_size=batch_size,
            validation_data=(z_val, errors_val))
    vmape = history.history['val_mean_absolute_percentage_error'][-1]
    return vmape

@use_named_args(space)
def objective(**params):
    return parametric_model(**params)

#  res_gp = gp_minimize(objective, space, n_calls=60, random_state=None)
res_gp = gp_minimize(obj, space, n_calls=60, random_state=None)

print("Best score: {}".format(res_gp.fun))
print('''Best parameters:\n
 Activation function: {}
 Optimizer: {}
 Learning rate: {}
 Num. hidden Layers: {}
 Num. weights: {}
 Batch size: {}'''.format(*res_gp.x))

plot_convergence(res_gp)
plt.savefig('../plots/res_gp_conv_avg_rom.png')
