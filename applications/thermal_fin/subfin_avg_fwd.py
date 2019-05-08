import matplotlib; matplotlib.use("agg")
import matplotlib.pyplot as plt
plt.style.use('seaborn-dark-palette')
import numpy as np
from dl_model import load_parametric_model
from tensorflow.keras.optimizers import Adam, RMSprop, Adadelta
from forward_solve import Fin, get_space

model = load_parametric_model('relu', Adam, 0.004, 6, 50, 150, 600)
z_val = np.loadtxt('data/z_avg_v.txt', delimiter=',')
errors_val =  np.loadtxt('data/err_avg_v.txt', delimiter=',')
V = get_space(40)
phi = np.loadtxt('data/basis_five_param.txt',delimiter=",")
phi = phi[:,0:10]
solver = Fin(V)
avgs_f = np.zeros((len(z_val), 5))
avgs_r = np.zeros((len(z_val), 5))
avgs_d = np.zeros((len(z_val), 5))
err_pred = np.zeros((len(z_val), 5))

for i in range(len(z_val)):
    z = z_val[i,:].reshape((1,5))
    w, y, A, B, C = solver.forward_five_param(z[0,:])
    avgs_f[i] = solver.qoi_operator(w)
    psi = np.dot(A, phi)
    A_r, B_r, C_r, x_r, y_r = solver.reduced_forward(A, B, C, psi, phi)
    avgs_r[i] = solver.reduced_qoi_operator(x_r)
    err_pred[i] = model.predict(z)[0]
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
    plt.savefig("subfin_avg_err_pred_{}".format(i), dpi=250)


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
    plt.savefig("subfin_avg_pred_{}".format(i), dpi=250)
