from sys import platform
if platform == 'darwin'
    import matplotlib
    matplotlib.use('macosx')
import time
import numpy as np
import matplotlib.pyplot as plt
import mshr
from mshr import Rectangle, generate_mesh
from dolfin import *
from forward_solve import Fin
from error_optimization import optimize
from model_constr_adaptive_sampling import sample, enrich

set_log_level(40)

# Create a fin geometry
geometry = Rectangle(Point(2.5, 0.0), Point(3.5, 4.0)) \
        + Rectangle(Point(0.0, 0.75), Point(2.5, 1.0)) \
        + Rectangle(Point(0.0, 1.75), Point(2.5, 2.0)) \
        + Rectangle(Point(0.0, 2.75), Point(2.5, 3.0)) \
        + Rectangle(Point(0.0, 3.75), Point(2.5, 4.0)) \
        + Rectangle(Point(3.5, 0.75), Point(6.0, 1.0)) \
        + Rectangle(Point(3.5, 1.75), Point(6.0, 2.0)) \
        + Rectangle(Point(3.5, 2.75), Point(6.0, 3.0)) \
        + Rectangle(Point(3.5, 3.75), Point(6.0, 4.0)) \

mesh = generate_mesh(geometry, 40)

V = FunctionSpace(mesh, 'CG', 1)
dofs = len(V.dofmap().dofs())
solver = Fin(V)

##########################################################3
# Basis initialization with dummy solves and POD
##########################################################3
samples = 10
Y = np.zeros((samples, dofs))
for i in range(0,samples):

    if i == 0:
        m = interpolate(Expression("0.1 + exp(-(pow(x[0] - 0.5, 2) + pow(x[1], 2)) / 0.01)", degree=2),V)
    elif i == 1:
        m = interpolate(Expression("2*x[0] + 0.1", degree=2), V)
    elif i == 2:
        m = interpolate(Expression("1 + sin(x[0])* sin(x[0])", degree=2), V)
    elif i == 3:
        m = interpolate(Expression("1 + sin(x[1])* sin(x[1])", degree=2), V)
    elif i == 4:
        m = interpolate(Expression("1 + sin(x[0])* sin(x[1])", degree=2), V)
    else:
        m = Function(V)
        m.vector().set_local(np.random.uniform(0.1, 10.0, dofs))

    w = solver.forward(m)[0]
    Y[i,:] = w.vector()[:]

K = np.dot(Y, Y.T)

# Initial basis vectors computed using proper orthogonal decomposition
e,v = np.linalg.eig(K)

basis_size = 5
U = np.zeros((basis_size, dofs))
for i in range(basis_size):
    e_i = v[:,i].real
    U[i,:] = np.sum(np.dot(np.diag(e_i), Y),0)

basis = U.T

t_i = time.time()
def random_initial():
    z_0 = Function(V)
    z_0.vector().set_local(np.random.uniform(0.1, 10.0, dofs))
    return z_0

basis = sample(basis, random_initial, optimize, solver)
t_f = time.time()
print("Sampling time taken: {}".format(t_f - t_i))
print("Computed basis with shape {}".format(basis.shape))

m = interpolate(Expression("2*x[1] + 1.0", degree=2), V)
w, y, A, B, C  = solver.forward(m)
p = plot(m, title="Conductivity")
plt.colorbar(p)
plt.show()
p = plot(w, title="Temperature")
plt.colorbar(p)
plt.show()
A_r, B_r, C_r, x_r, y_r = solver.reduced_forward(A, B, C, np.dot(A, basis), basis) 
x_tilde = np.dot(basis, x_r)
x_tilde_f = Function(V)
x_tilde_f.vector().set_local(x_tilde)
p = plot(x_tilde_f, title="Temperature reduced")
plt.colorbar(p)
plt.show()

print("Reduced system relative error: {}".format(np.linalg.norm(y-y_r)/np.abs(y)))
np.savetxt("basis.txt", basis, delimiter=",")

#Modify basis
w = w.vector()[:]
w = w.reshape(w.shape[0], 1)
basis = enrich(basis, w)

A_r, B_r, C_r, x_r, y_r = solver.reduced_forward(A, B, C, np.dot(A, basis), basis) 
x_tilde = np.dot(basis, x_r)
x_tilde_f = Function(V)
x_tilde_f.vector().set_local(x_tilde)
p = plot(x_tilde_f, title="Temperature reduced")
plt.colorbar(p)
plt.show()

print("Reduced system relative error with added snapshot: {}".format(np.linalg.norm(y-y_r)/np.abs(y)))
np.savetxt("basis.txt", basis, delimiter=",")

