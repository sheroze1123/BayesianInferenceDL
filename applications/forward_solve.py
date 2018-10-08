import matplotlib
matplotlib.use('macosx')
import matplotlib.pyplot as plt
from dolfin import *
import mshr


def forward(m, V):
    '''
    Performs a forward solve to obtain temperature distribution
    given the conductivity field m and FunctionSpace V.
    '''

    mesh = V.mesh()
    w = TrialFunction(V)
    v = TestFunction(V)


    domains = MeshFunction("size_t", mesh, mesh.topology().dim())
    domains.set_all(0)

    # boundary conditions
    left =  CompiledSubDomain("near(x[0], side) && on_boundary", side = 0.0)
    right = CompiledSubDomain("near(x[0], side) && on_boundary", side = 1.0)
    boundaries = MeshFunction("size_t", mesh, mesh.topology().dim()-1)
    boundaries.set_all(0)
    left.mark(boundaries, 1)
    right.mark(boundaries, 2)

    dx = Measure('dx', domain=mesh, subdomain_data=domains)
    ds = Measure('ds', domain=mesh, subdomain_data=boundaries)

    # Setting up the variational form
    Bi = Constant(0.1)
    F = inner(exp(m) * grad(w), grad(v)) * dx(0) + v * Bi * w * ds(1)
    a = v * ds(2)

    w = Function(V)
    solve(F == a, w) 

    return w

# Create a fin geometry
geometry = mshr.Rectangle(Point(0.42, 0.0), Point(0.58, 1.0)) \
        + mshr.Rectangle(Point(0.0, 0.17), Point(0.42, 0.25)) \
        + mshr.Rectangle(Point(0.0, 0.42), Point(0.42, 0.5)) \
        + mshr.Rectangle(Point(0.0, 0.67), Point(0.42, 0.75)) \
        + mshr.Rectangle(Point(0.0, 0.92), Point(0.42, 1)) \
        + mshr.Rectangle(Point(0.58, 0.17), Point(1.0, 0.25)) \
        + mshr.Rectangle(Point(0.58, 0.42), Point(1.0, 0.5)) \
        + mshr.Rectangle(Point(0.58, 0.67), Point(1.0, 0.75)) \
        + mshr.Rectangle(Point(0.58, 0.92), Point(1.0, 1)) 

mesh = mshr.generate_mesh(geometry, 20)
#  plot(mesh)
#  plt.show()

V = FunctionSpace(mesh, 'CG', 1)

# Pick a more interesting conductivity to see what happens
#  m = Function(V)
m = interpolate(Expression("100*exp(-(pow(x[0] - 0.5, 2) + pow(x[1], 2)) / 0.02)", degree=2), V)
w = forward(m, V)

fig = plt.figure()
p = plot(m, title="Conductivity")
plt.colorbar(p)
plt.show()

plt.figure()
p = plot(w, title="Temperature")
plt.colorbar(p)
plt.show()
