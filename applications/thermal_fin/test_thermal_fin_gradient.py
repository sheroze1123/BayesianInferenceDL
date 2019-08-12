from forward_solve import Fin
from thermal_fin import get_space
from gaussian_field import make_cov_chol
import dolfin as dl
import numpy as np
import matplotlib.pyplot as plt

# Setup solver
V = get_space(40)
solver = Fin(V)
phi = np.loadtxt('data/basis_five_param.txt',delimiter=",")

# Obtain synthetic data
chol = make_cov_chol(V, length=0.8)
k_true = dl.Function(V)
norm = np.random.randn(len(chol))
nodal_vals = np.exp(0.5 * chol.T @ norm)
k_true.vector().set_local(nodal_vals)

w, y, A, B, C = solver.forward(k_true)
data = solver.qoi_operator(w)

# Comparing reduced and full
w, y, A, B, C = solver.forward(k_true)
A_r, B_r, C_r, w_r, y_r = solver.averaged_forward(k_true, phi)
k_true_averaged = solver.nine_param_to_function(solver.subfin_avg_op(k_true)) 

p = dl.plot(k_true_averaged)
plt.colorbar(p)
plt.show()

fig, (ax1, ax2) = plt.subplots(1,2)
plt.sca(ax1)
p = dl.plot(k_true)
plt.sca(ax2)
dl.plot(k_true_averaged)
plt.colorbar(p)
plt.show()

fig, (ax1, ax2) = plt.subplots(1,2)
plt.sca(ax1)
dl.plot(w)
plt.sca(ax2)
w_tilde = dl.Function(V)
w_tilde.vector().set_local(np.dot(phi, w_r))
dl.plot(w_tilde)
plt.show()

avg_rom_err = np.sqrt(np.mean(np.square(w.vector()[:] - w_tilde.vector()[:])))
print("RMSE of Averaging ROM: {}".format(avg_rom_err))

# Compute current parameter point to obtain gradients at
k_pt = dl.Function(V)
norm = np.random.randn(len(chol))
nodal_vals = np.exp(0.5 * chol.T @ norm)
k_pt.vector().set_local(nodal_vals)

# Compute full order gradient
w, y, A, B, C = solver.forward(k_pt)
qoi_0 = solver.qoi_operator(w)
Pi_0 = np.square(qoi_0 - data).sum()/2
grad = solver.gradient(k_pt, data)

# Number of successive finite difference epsilons
n_eps = 15
eps = 1e-2*np.power(2., -np.arange(n_eps))
err_grad = np.zeros(n_eps)

k = dl.Function(V)

# Compute FOM adjoint method based gradient
k_hat = dl.Function(V).vector()
k_hat.set_local(np.random.randn(V.dim()))
k_hat.apply("")
dir_grad = np.dot(grad, k_hat[:])

for i in range(n_eps):
    k.assign(k_pt)
    k.vector().axpy(eps[i], k_hat) #uh = uh + eps[i]*dir

    w, y, A, B, C = solver.forward(k)
    qoi_plus = solver.qoi_operator(w)
    Pi_plus = np.square(qoi_plus - data).sum()/2

    # Error b/w finite diff approx to gradient at given eps and adj
    err_grad[i] = abs( (Pi_plus - Pi_0)/eps[i] - dir_grad )
    fd_grad = (Pi_plus - Pi_0)/eps[i]

plt.figure()    
plt.loglog(eps, err_grad, "-ob", label="Error Grad")
plt.loglog(eps, (.5*err_grad[0]/eps[0])*eps, "-.k", label="First Order")
plt.title("Finite difference check of the first variation FOM")
plt.xlabel("eps")
plt.ylabel("Error grad")
plt.legend(loc = "upper left")
plt.show()
print("FD grad: {}, Adj grad: {}".format(fd_grad, dir_grad))

w_r, red_grad = solver.averaged_reduced_fwd_and_grad(k_pt, phi, data)
qoi_0 = solver.reduced_qoi_operator(w_r)
Pi_0 = np.square(qoi_0 - data).sum()/2
dir_grad = np.dot(red_grad, k_hat[:])
FD_grad = np.zeros(n_eps)

for i in range(n_eps):
    k.assign(k_pt)
    k.vector().axpy(eps[i], k_hat) #uh = uh + eps[i]*dir

    A_r, B_r, C_r, w_r, y_r = solver.averaged_forward(k, phi)
    qoi_plus = solver.reduced_qoi_operator(w_r)
    Pi_plus = np.square(qoi_plus - data).sum()/2
    err_grad[i] = abs( (Pi_plus - Pi_0)/eps[i] - dir_grad )
    FD_grad[i] = (Pi_plus - Pi_0)/eps[i]

plt.figure()    
plt.loglog(eps, err_grad, "-ob", label="Error Grad")
plt.loglog(eps, (.5*err_grad[0]/eps[0])*eps, "-.k", label="First Order")
plt.title("Finite difference check of the first variation (ROM)")
plt.xlabel("eps")
plt.ylabel("Error grad")
plt.legend(loc = "upper left")
plt.show()

print("FD: {}, compute: {}".format(FD_grad[4],dir_grad))
plt.loglog(eps, np.abs(FD_grad), "r-x")
plt.title('FD grad as a func of eps, adj grad: {}'.format(dir_grad))
plt.ylabel('grad')
plt.show()

plt.plot(eps, err_grad, "-ob", label="Error grad")
plt.title('Error compared to FD')
plt.show()


