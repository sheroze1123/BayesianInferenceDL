import sys
sys.path.append('../')
import matplotlib; matplotlib.use("agg")
import matplotlib.pyplot as plt
plt.style.use('seaborn-dark-palette')
import numpy as np
from deep_learning.dl_model import load_bn_model, load_dataset_avg_rom
from tensorflow.keras.optimizers import Adam
from fom.forward_solve import Fin
from fom.thermal_fin import get_space
from rom.averaged_affine_ROM import AffineROMFin
from dolfin import *

#  model = load_parametric_model('relu', Adam, 0.004, 6, 50, 150, 600)
model = load_bn_model()
z_train, errors_train, z_val, errors_val = load_dataset_avg_rom()
#  z_val = np.loadtxt('data/z_avg_v.txt', delimiter=',')
#  errors_val =  np.loadtxt('data/err_avg_v.txt', delimiter=',')
V = get_space(40)
phi = np.loadtxt('../data/basis_nine_param.txt',delimiter=",")
#  phi = phi[:,0:10]
solver = Fin(V)
solver_r = AffineROMFin(V, model, phi)
avgs_f = np.zeros((len(z_val), 9))
avgs_r = np.zeros((len(z_val), 9))
avgs_d = np.zeros((len(z_val), 9))
err_pred = np.zeros((len(z_val), 9))
z = Function(V)

for i in range(len(z_val)):
    z.vector().set_local(z_val[i,:])
    z_nodal = z_val[i,:].reshape((1,1446))
    w, y, A, B, C = solver.forward(z)
    w_r = solver_r.forward_reduced(z)

    avgs_f[i] = solver.qoi_operator(w)
    avgs_r[i] = solver_r.qoi_reduced(w_r)
    err_pred[i] = model.predict(z_nodal)[0]
    avgs_d[i] = avgs_r[i] + err_pred[i]

rel_error = np.linalg.norm(np.abs(avgs_f - avgs_d), axis=1)/np.linalg.norm(avgs_f, axis=1)
print(f"Average relative error : {np.mean(rel_error)}")

for i in range(9):
    plt.cla()
    plt.clf()
    plt.plot(errors_val[:100,i], color='#36c7d6')
    plt.plot(err_pred[:100,i], 'k--x')
    plt.xlabel("Validation Example Index", fontsize=10)
    plt.ylabel("Error in average temperature", fontsize=10)
    plt.legend(["true error", "deep learning error"], fontsize=10)
    plt.title("True error vs learned error in avg temperature of fin {}".format(i), fontsize=14)
    plt.savefig("../plots/subfin_avg_err_pred_{}".format(i), dpi=250)


for i in range(9):
    plt.cla()
    plt.clf()
    plt.plot(avgs_f[:100,i], color='#36c7d6')
    plt.plot(avgs_r[:100,i], color='#e8923c')
    plt.plot(avgs_d[:100,i], 'k--x')
    plt.xlabel("Validation Example Index", fontsize=10)
    plt.ylabel("Average temperature", fontsize=10)
    plt.legend(["true", "ROM (81-dim)", "ROM + deep learning"], fontsize=10)
    plt.title("Average temperature prediction of fin {}".format(i), fontsize=14)
    plt.savefig("../plots/subfin_avg_pred_{}".format(i), dpi=250)
