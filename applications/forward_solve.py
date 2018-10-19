from dolfin import *
import numpy as np

def averaging_operator(V):
    # Kinda hacky way to get C as the averaging operator over the domain. Probably a better way
    v = TestFunction(V)
    d_omega_f = interpolate(Expression("1.0", degree=2), V)
    domain_integral = assemble(v*dx)
    domain_measure = np.dot(np.array(d_omega_f.vector()[:]), np.array(domain_integral[:]))
    C = domain_integral/domain_measure
    C = C[:]
    return C

def get_weak_forms(m, V):
    mesh = V.mesh()
    w = TrialFunction(V)
    v = TestFunction(V)

    domains = MeshFunction("size_t", mesh, mesh.topology().dim())
    domains.set_all(0)

    # boundary conditions
    bottom = CompiledSubDomain("near(x[1], side) && on_boundary", side = 0.0)
    exterior = CompiledSubDomain("!near(x[1], side) && on_boundary", side = 0.0)
    boundaries = MeshFunction("size_t", mesh, mesh.topology().dim()-1)
    boundaries.set_all(0)
    exterior.mark(boundaries, 1)
    bottom.mark(boundaries, 2)

    dx = Measure('dx', domain=mesh, subdomain_data=domains)
    ds = Measure('ds', domain=mesh, subdomain_data=boundaries)

    # Setting up the variational form
    Bi = Constant(0.1)
    F = inner(m * grad(w), grad(v)) * dx(0) + v * Bi * w * ds(1)
    a = v * ds(2)

    dF_dm = inner(grad(w), grad(v)) * dx(0)
    return F, a, dF_dm

def forward(m, V):
    '''
    Performs a forward solve to obtain temperature distribution
    given the conductivity field m and FunctionSpace V.
    This solve assumes Biot number to be a constant.
    Returns:
     w - Temperature field 
     y - Average temperature
     A - Mass matrix
     B - Discretized RHS
     C - Averaging operator
     dA_dz - Partial derivative of the mass matrix w.r.t. the parameters
    '''

    F, a, dF_dm = get_weak_forms(m, V)
    A = assemble(F).array()
    B = assemble(a)[:]

    #TODO: This has to be a tensor so this is incorrect
    dA_dz = assemble(dF_dm).array()

    w = Function(V)
    solve(F == a, w) 

    # TODO: Compute quantity of interest
    w_nodal_values = np.array(w.vector()[:]) 
    y = np.mean(w_nodal_values)
    print("Average temperature: {}".format(y))

    C = averaging_operator(V)

    return w, y, A, B, C, dA_dz

def reduced_forward(A, B, C, psi, phi):
    '''
    Returns the reduced matrices given a reduced trial and test basis
    to solve reduced system of A(z)x = B(z)
    Arguments:
    A   - LHS forward operator in A(z)x = B(z)
    B   - RHS in A(z)x = B(z)
    psi - Trial basis
    phi - Test basis
    '''
    A_r = np.dot(psi.T, np.dot(A, phi))
    B_r = np.dot(psi.T, B)
    C_r = np.dot(C, phi)

    #Solve reduced system to obtain x_r
    x_r = np.linalg.solve(A_r, B_r)
    y_r = np.dot(C_r, x_r)

    return A_r, B_r, C_r, x_r, y_r

def reduced_forward_no_full(m, V, psi, phi):
    F, a, dF_dm = get_weak_forms(m, V)
    A = assemble(F).array()
    B = assemble(a)[:]
    C = averaging_operator(V)

    return reduced_forward(A,B,C, psi, phi)
