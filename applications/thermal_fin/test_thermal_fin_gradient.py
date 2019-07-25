from forward_solve import Fin, get_space
from gaussian_field import make_cov_chol
import dolfin as dl
import numpy as np
import matplotlib.pyplot as plt

V = get_space(40)
solver = Fin(V)

chol = make_cov_chol(V)
k_true = dl.Function(V)
norm = np.random.randn(len(chol))
nodal_vals = np.exp(0.5 * chol.T @ norm)
k_true.vector().set_local(nodal_vals)
w, y, A, B, C = solver.forward(k_true)
data = solver.qoi_operator(w)

k_pt = dl.Function(V)
norm = np.random.randn(len(chol))
nodal_vals = np.exp(0.5 * chol.T @ norm)
k_pt.vector().set_local(nodal_vals)

w, y, A, B, C = solver.forward(k_pt)
qoi_0 = solver.qoi_operator(w)
Pi_0 = np.square(qoi_0 - data).sum()/2
grad = solver.gradient(k_pt, data)

n_eps = 32
eps = 1e-2*np.power(2., -np.arange(n_eps))
err_grad = np.zeros(n_eps)

k = dl.Function(V)

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
    err_grad[i] = abs( (Pi_plus - Pi_0)/eps[i] - dir_grad )

plt.figure()    
plt.loglog(eps, err_grad, "-ob", label="Error Grad")
plt.loglog(eps, (.5*err_grad[0]/eps[0])*eps, "-.k", label="First Order")
plt.title("Finite difference check of the first variation (gradient)")
plt.xlabel("eps")
plt.ylabel("Error grad")
plt.legend(loc = "upper left")
plt.show()

phi = np.loadtxt('data/basis_five_param.txt',delimiter=",")
w_r, red_grad = solver.averaged_reduced_fwd_and_grad(k_pt, phi, data)
qoi_0 = solver.reduced_qoi_operator(w_r)
Pi_0 = np.square(qoi_0 - data).sum()/2
dir_grad = np.dot(red_grad, k_hat[:])

for i in range(n_eps):
    k.assign(k_pt)
    k.vector().axpy(eps[i], k_hat) #uh = uh + eps[i]*dir

    A_r, B_r, C_r, w_r, y_r = solver.averaged_forward(k, phi)
    qoi_plus = solver.reduced_qoi_operator(w_r)
    Pi_plus = np.square(qoi_plus - data).sum()/2
    err_grad[i] = abs( (Pi_plus - Pi_0)/eps[i] - dir_grad )

plt.figure()    
plt.loglog(eps, err_grad, "-ob", label="Error Grad")
plt.loglog(eps, (.5*err_grad[0]/eps[0])*eps, "-.k", label="First Order")
plt.title("Finite difference check of the first variation (gradient)")
plt.xlabel("eps")
plt.ylabel("Error grad")
plt.legend(loc = "upper left")
plt.show()

