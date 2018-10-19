import time
import matplotlib
matplotlib.use('macosx')
import matplotlib.pyplot as plt
from dolfin import *
import mshr
import numpy as np
from forward_solve import forward, reduced_forward
from error_optimization import optimize
from model_constr_adaptive_sampling import sample

# Create a fin geometry
geometry = mshr.Rectangle(Point(2.5, 0.0), Point(3.5, 4.0)) \
        + mshr.Rectangle(Point(0.0, 0.75), Point(2.5, 1.0)) \
        + mshr.Rectangle(Point(0.0, 1.75), Point(2.5, 2.0)) \
        + mshr.Rectangle(Point(0.0, 2.75), Point(2.5, 3.0)) \
        + mshr.Rectangle(Point(0.0, 3.75), Point(2.5, 4.0)) \
        + mshr.Rectangle(Point(3.5, 0.75), Point(6.0, 1.0)) \
        + mshr.Rectangle(Point(3.5, 1.75), Point(6.0, 2.0)) \
        + mshr.Rectangle(Point(3.5, 2.75), Point(6.0, 3.0)) \
        + mshr.Rectangle(Point(3.5, 3.75), Point(6.0, 4.0)) \

mesh = mshr.generate_mesh(geometry, 40)

V = FunctionSpace(mesh, 'CG', 1)
dofs = len(V.dofmap().dofs())

##########################################################3
# Basis initialization with dummy solves and POD
##########################################################3
samples = 10
Y = np.zeros((samples, dofs))
for i in range(0,samples):

    if i == 0:
        m = interpolate(Expression("0.1 +s*exp(-(pow(x[0] - c_x, 2) + pow(x[1]-c, 2)) / 0.02)", degree=2, s=2.0, c=0.03*i, c_x =0.5 + 0.01*i), V)
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

    w = forward(m, V)[0]
    Y[i,:] = w.vector()[:]

K = np.dot(Y, Y.T)
e,v = np.linalg.eig(K)
#  plt.plot(e[:10])
#  plt.show()

basis_size = 5
U = np.zeros((basis_size, dofs))
for i in range(basis_size):
    e_i = v[:,i].real
    U[i,:] = np.sum(np.dot(np.diag(e_i), Y),0)

basis = U.T
z_0 = Function(V)
z_0.vector().set_local(np.random.uniform(0.1, 10, dofs))

t_i = time.time()
basis = sample(basis, z_0, optimize, forward, V)
t_f = time.time()
print("Sampling time taken: {}".format(t_f - t_i))

print("Computed basis with shape {}".format(basis.shape))
m = interpolate(Expression("2*x[1] + 1.0", degree=2), V)
w, y, A, B, C, dA_dz  = forward(m, V)
p = plot(m, title="Conductivity")
plt.colorbar(p)
plt.show()
p = plot(w, title="Temperature")
plt.colorbar(p)
plt.show()
A_r, B_r, C_r, x_r, y_r = reduced_forward(A, B, C, np.dot(A, basis), basis) 
x_tilde = np.dot(basis, x_r)
x_tilde_f = Function(V)
x_tilde_f.vector().set_local(x_tilde)
p = plot(x_tilde_f, title="Temperature reduced")
plt.colorbar(p)
plt.show()

print("Reduced system error: {}".format(np.linalg.norm(y-y_r)))
np.savetxt("basis.txt", basis, delimiter=",")
