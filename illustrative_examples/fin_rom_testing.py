import sys
sys.path.append('../')
import matplotlib.pyplot as plt
from dolfin import *
from mshr import Rectangle, generate_mesh
import numpy as np

from forward_solve import Fin
from error_optimization import optimize
from model_constr_adaptive_sampling import sample

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
f = Fin(V)

basis = np.loadtxt('data/basis.txt', delimiter=",")
m = interpolate(Expression("2*x[1] + 1.0", degree=2), V)
w, y, A, B, C  = f.forward(m, V)
p = plot(m, title="Conductivity")
plt.colorbar(p)
plt.show()
p = plot(w, title="Temperature")
plt.colorbar(p)
plt.show()
A_r, B_r, C_r, x_r, y_r = f.reduced_forward(A, B, C, np.dot(A, basis), basis) 
x_tilde = np.dot(basis, x_r)
x_tilde_f = Function(V)
x_tilde_f.vector().set_local(x_tilde)
p = plot(x_tilde_f, title="Temperature reduced")
plt.colorbar(p)
plt.show()

print("Reduced system error: {}".format(np.linalg.norm(y-y_r)))
