import matplotlib; matplotlib.use("agg")
import matplotlib.pyplot as plt
plt.style.use('seaborn-dark-palette')
import numpy as np
from dl_rom import load_parametric_model_avg
from tensorflow.keras.optimizers import Adam, RMSprop, Adadelta
from forward_solve import Fin, get_space
from dolfin import Function

z_val = np.loadtxt('data/z_avg_eval.txt', delimiter=',')
errors_val =  np.loadtxt('data/errors_avg_eval.txt', delimiter=',')
V = get_space(40)
model = load_parametric_model_avg('relu', Adam, 0.001, 6, 60, 90, 400, V.dim())
phi = np.loadtxt('data/basis_five_param.txt',delimiter=",")
#  phi = phi[:,0:10]
solver = Fin(V)
avgs_f = np.zeros((len(z_val), 5))
avgs_r = np.zeros((len(z_val), 5))
avgs_d = np.zeros((len(z_val), 5))
err_pred = np.zeros((len(z_val), 5))
z = Function(V)
err_pred = model.predict(z_val)
print("\nz shape:{}\n".format(z_val.shape))
print("\nerr shape:{}\n".format(errors_val.shape))
print("\nerr pred shape:{}\n".format(err_pred.shape))


for i in range(len(z_val)):
    nodal_vals = z_val[i,:]
    z.vector()[:] = nodal_vals[:]
    _, _, _, w_avg_r, y_avg_r = solver.averaged_forward(z, phi)
    w, y, A, B, C = solver.forward(z)
    avgs_f[i] = solver.qoi_operator(w)
    psi = np.dot(A, phi)
    avgs_r[i] = solver.reduced_qoi_operator(w_avg_r)
    avgs_d[i] = avgs_r[i] + err_pred[i]

for i in range(5):
    plt.cla()
    plt.clf()
    plt.plot(errors_val[:100,i], color='#36c7d6')
    plt.plot(err_pred[:100,i], 'k--x')
    plt.xlabel("Validation Example Index", fontsize=10)
    plt.ylabel("Error in average temperature", fontsize=10)
    plt.legend(["true error", "deep learning error"], fontsize=10)
    plt.title("True error vs learned error in avg temperature of fin {}".format(i), fontsize=14)
    plt.savefig("plots/avg_rom_err_pred_{}".format(i), dpi=250)


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
    plt.savefig("plots/avg_rom_pred_{}".format(i), dpi=250)
