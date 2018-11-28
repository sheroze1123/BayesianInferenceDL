import time
import numpy as np
from mshr import Rectangle, generate_mesh
from dolfin import *
from forward_solve import Fin
from error_optimization import optimize_five_param
from model_constr_adaptive_sampling import sample

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
    k = np.random.uniform(0.1, 1.0, 5)
    w = solver.forward_five_param(k)[0]
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

def random_initial_five_param():
    '''
    Currently uses a simple random initialization. 
    Eventually replace with a more sophisticated function
    with Bayesian prior sampling
    '''
    return np.random.uniform(0.1,1.0,5)

##########################################################3
# Create reduced basis with adaptive sampling
##########################################################3

t_i = time.time()
basis = sample(basis, random_initial_five_param, optimize_five_param, solver)
t_f = time.time()
print("Sampling time taken: {}".format(t_f - t_i))
print("Computed basis with shape {}".format(basis.shape))

np.savetxt("data/basis_five_param.txt", basis, delimiter=",")
