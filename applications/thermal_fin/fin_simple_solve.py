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
print("DOFS: {}".format(dofs))

# Pick a more interesting conductivity to see what happens
#  m = Function(V)

#  m = interpolate(Expression("2.0*exp(-(pow(x[0] - 0.5, 2) + pow(x[1]-0.5, 2)) / 0.02)", degree=2), V)
m = interpolate(Expression("5- x[1]", degree=2), V)
solver = Fin(V)
w = solver.forward(m)[0]
fig = plt.figure()
p = plot(m, title="Conductivity")
plt.colorbar(p)
plt.show()
plt.figure()
p = plot(w, title="Temperature")
plt.colorbar(p)
plt.show()
