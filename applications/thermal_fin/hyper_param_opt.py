import matplotlib; matplotlib.use('agg')
import matplotlib.pyplot as plt
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args
from skopt import gp_minimize
from skopt.plots import plot_convergence
from dl_model import load_dataset_subfin, parametric_model
from tensorflow.keras.optimizers import Adam, RMSprop, Adadelta

#  z_train, errors_train, z_val, errors_val = load_dataset_avg_rom()

# Defines the hyperparameter space
space = [Categorical(['elu', 'sigmoid', 'tanh', 'relu'], name='activation'),
         Categorical([Adam, RMSprop, Adadelta], name='optimizer'),
         Real(1e-5, 1, prior="log-uniform", name='lr'),
         Integer(1, 6, name='n_hidden_layers'),
         Integer(10, 100, name='n_weights'),
         Integer(4, 200, name='batch_size'),
         Integer(100, 600, name='n_epochs')]

@use_named_args(space)
def objective(**params):
    return parametric_model(**params)

res_gp = gp_minimize(objective, space, n_calls=60, random_state=None)

print("Best score: {}".format(res_gp.fun))
print('''Best parameters:\n
 Activation function: {}
 Optimizer: {}
 Learning rate: {}
 Num. hidden Layers: {}
 Num. weights: {}
 Batch size: {}
 Num. Epochs: {}'''.format(*res_gp.x))

plot_convergence(res_gp)
plt.savefig('plots/res_gp_conv_avg_rom.png')
