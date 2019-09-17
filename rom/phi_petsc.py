from petsc4py import PETSc
import numpy as np
from forward_solve_petsc import Fin, get_space
from forward_solve import Fin as F
from gaussian_field import make_cov_chol
from dolfin import *
#  import time

phi = np.loadtxt('data/basis_five_param.txt',delimiter=",")
(n,n_r) = phi.shape
phi_mat = PETSc.Mat().createDense([n,n_r], array=phi)
phi_mat.assemblyBegin()
phi_mat.assemblyEnd()

V = get_space(40)
chol = make_cov_chol(V)

z = Function(V)
norm = np.random.randn(len(chol))
nodal_vals = np.exp(0.5 * chol.T @ norm)
z.vector().set_local(nodal_vals)

z2 = Function(V)
norm2 = np.random.randn(len(chol))
nodal_vals2 = np.exp(0.5 * chol.T @ norm2)
z2.vector().set_local(nodal_vals2)

z3 = Function(V)
norm3 = np.random.randn(len(chol))
nodal_vals3 = np.exp(0.5 * chol.T @ norm3)
z3.vector().set_local(nodal_vals3)

#  t_i_s = time.time()
solver_slow = F(V)
x_s, y_s, A_s, B_s, C_s = solver_slow.forward(z)
A_r, B_r, C_r, x_r, y_r = solver_slow.averaged_forward(z, phi)
x_s, y_s2, A_s, B_s, C_s = solver_slow.forward(z2)
A_r, B_r, C_r, x_r, y_r2 = solver_slow.averaged_forward(z2, phi)
x_s, y_s3, A_s, B_s, C_s = solver_slow.forward(z3)
A_r, B_r, C_r, x_r, y_r3 = solver_slow.averaged_forward(z3, phi)
#  t_f_s = time.time()

#  t_i = time.time()
solver = Fin(V, phi_mat)
x, y, A, B, C = solver.forward(z)
x_r, y_rf, A_r, B_r, C_r = solver.averaged_forward(z)
x, y2, A, B, C = solver.forward(z2)
x_r, y_rf2, A_r, B_r, C_r = solver.averaged_forward(z2)
x, y3, A, B, C = solver.forward(z3)
x_r, y_rf3, A_r, B_r, C_r = solver.averaged_forward(z3)
#  t_f = time.time()
#  print ("y={:.5f}, y2={:.5f}, y3={:.5f}, time={:.5f}".format(y_s, y_s2, y_s3, t_f_s-t_i_s))
#  print ("y={:.5f}, y2={:.5f}, y3={:.5f},time={:.5f}".format(y, y2, y3, t_f-t_i))
#  print ("y_r={:.5f}, y_r2={:.5f}, y_r3={:.5f}, time={:.5f}".format(y_r, y_r2, y_r3, t_f_s-t_i_s))
#  print ("y_r={:.5f}, y_r2={:.5f}, y_r3={:.5f}, time={:.5f}".format(y_rf, y_rf2, y_rf3, t_f-t_i))

dataset_size = 1000
qoi_errors = np.zeros((dataset_size, 5))
qoi_errors_s = np.zeros((dataset_size, 5))

# TODO: Needs to be fixed for higher order functions
z_s = np.zeros((dataset_size, V.dim()))

#  t_i = time.time()
for i in range(dataset_size):
    norm = np.random.randn(len(chol))
    nodal_vals = np.exp(0.5 * chol.T @ norm)
    z.vector().set_local(nodal_vals)
    z_s[i,:] = nodal_vals
    x, y, A, B, C = solver_slow.forward(z)
    qoi = solver_slow.qoi_operator(x)
    A_r, B_r, C_r, x_r, y_r = solver_slow.averaged_forward(z, phi)
    qoi_r = solver_slow.reduced_qoi_operator(x_r)
    qoi_errors_s[i,:] = qoi - qoi_r
#  t_f = time.time()
#  print("Slow solver completed in {:.5f} seconds".format(t_f-t_i))

#  t_i = time.time()
for i in range(dataset_size):
    norm = np.random.randn(len(chol))
    nodal_vals = np.exp(0.5 * chol.T @ norm)
    z.vector().set_local(nodal_vals)
    z_s[i,:] = nodal_vals
    x, y, A, B, C = solver.forward(z)
    qoi = solver.qoi_operator(x)
    A_r, B_r, C_r, x_r, y_r = solver.averaged_forward(z)
    qoi_r = solver.reduced_qoi_operator(x_r)
    qoi_errors[i,:] = qoi - qoi_r
#  t_f = time.time()
#  print("Fast solver completed in {:.5f} seconds".format(t_f-t_i))
