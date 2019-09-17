import sys; sys.path.append('../')
import matplotlib
matplotlib.use('macosx')
import matplotlib.pyplot as plt
import numpy as np
from dolfin import Function

from tensorflow.keras.optimizers import Adam, RMSprop, Adadelta

from deep_learning.dl_model import load_parametric_model_avg
from fom.forward_solve import Fin, get_space

z_val = np.loadtxt('../data/z_avg_eval.txt', delimiter=',')
errors_val =  np.loadtxt('../data/errors_avg_eval.txt', delimiter=',')
V = get_space(40)
model = load_parametric_model_avg('elu', Adam, 0.129, 3, 58, 64, 466, V.dim())
phi = np.loadtxt('../data/basis_five_param.txt',delimiter=",")
#  phi = phi[:,0:10]
solver = Fin(V)
avgs_f = np.zeros((len(z_val), 5))
avgs_r = np.zeros((len(z_val), 5))
avgs_d = np.zeros((len(z_val), 5))
avgs_c = np.zeros((len(z_val), 5))
err_pred = np.zeros((len(z_val), 5))
z = Function(V)
err_pred = model.predict(z_val)
print ("Average validation error in pred: {}".format(np.average(np.divide(np.abs(errors_val - err_pred), np.abs(errors_val)))))

for i in range(len(z_val)):
    nodal_vals = z_val[i,:]
    z.vector().set_local(nodal_vals)
    _, _, _, x_r, y_r = solver.averaged_forward(z, phi)
    x, y, A, B, C = solver.forward(z)
    avgs_f[i] = solver.qoi_operator(x)
    avgs_r[i] = solver.reduced_qoi_operator(x_r)
    avgs_d[i] = avgs_r[i] + err_pred[i]

print ("Average relative error in tilde: {}".format(np.average(np.divide(np.abs(avgs_f - avgs_d), np.abs(avgs_f)))))

for i in range(5):
    plt.cla()
    plt.clf()
    plt.plot(errors_val[:100,i], color='#36c7d6')
    plt.plot(err_pred[:100,i], 'k--x')
    plt.xlabel("Validation Example Index", fontsize=10)
    plt.ylabel("Error in average temperature", fontsize=10)
    plt.legend(["true error", "deep learning error"], fontsize=10)
    plt.title("True error vs learned error in avg temperature of fin {}".format(i), fontsize=14)
    plt.savefig("../plots/avg_rom_err_pred_{}".format(i), dpi=250)


for i in range(5):
    plt.cla()
    plt.clf()
    plt.plot(avgs_f[:100,i], color='#36c7d6')
    plt.plot(avgs_r[:100,i], color='#e8923c')
    plt.plot(avgs_d[:100,i], 'k--x')
    plt.xlabel("Validation Example Index", fontsize=10)
    plt.ylabel("Average temperature", fontsize=10)
    plt.legend(["true", "ROM prediction", "ROM + deep learning"], fontsize=10)
    plt.title("Average temperature prediction of fin {}".format(i), fontsize=14)
    plt.savefig("../plots/avg_rom_pred_{}".format(i), dpi=250)
