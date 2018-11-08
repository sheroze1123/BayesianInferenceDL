import sys
sys.path.append('../')

if sys.platform == 'darwin'
    import matplotlib
    matplotlib.use('macosx')

import matplotlib.pyplot as plt
from dolfin import *
import mshr
import numpy as np

from forward_solve import Fin 

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
plot(mesh)
plt.show()

V = FunctionSpace(mesh, 'CG', 1)
dofs = len(V.dofmap().dofs())
solver = Fin(V)

samples = 5
Y = np.zeros((samples, dofs))
for i in range(0,samples):

    if i%2 == 0:
        m = interpolate(Expression("0.1 +s*exp(-(pow(x[0] - c_x, 2) + pow(x[1]-c, 2)) / 0.02)", degree=2, s=2.0, c=0.03*i, c_x =0.5 + 0.01*i), V)
    else:
        m = interpolate(Expression("c*x[0] + 0.1", degree=2, c=2.0*i), V)

    w = solver.forward(m)[0]

    #  if i%45 == 0:
        #  p = plot(w, title="Temperature")
        #  plt.colorbar(p)
        #  plt.show()
    Y[i,:] = w.vector()[:]

K = np.dot(Y, Y.T)
e,v = np.linalg.eig(K)
plt.semilogy(e.real, 'b.' )
plt.title("Eigenvalues")
plt.show()

U = np.zeros((3, dofs))
for i in range(3):
    e_i = v[:,i].real
    U[i,:] = np.sum(np.dot(np.diag(e_i), Y),0)

m = Function(V)
m.vector().set_local(U[0,:])

fig = plt.figure()
p = plot(m, title="First eigenvector with eigenvalue {}".format(e[0].real))
plt.colorbar(p)
plt.show()

m = Function(V)
m.vector().set_local(U[1,:])

fig = plt.figure()
p = plot(m, title="Second eigenvector with eigenvalue {}".format(e[1].real))
plt.colorbar(p)
plt.show()

m = Function(V)
m.vector().set_local(U[2,:])

fig = plt.figure()
p = plot(m, title="Third eigenvector with eigenvalue {}".format(e[2].real))
plt.colorbar(p)
plt.show()

plt.figure()
p = plot(w, title="Temperature")
plt.colorbar(p)
plt.show()
