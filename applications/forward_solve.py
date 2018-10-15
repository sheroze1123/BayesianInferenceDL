from dolfin import *
import numpy as np

def forward(m, V):
    '''
    Performs a forward solve to obtain temperature distribution
    given the conductivity field m and FunctionSpace V.
    This solve assumes Biot number to be a constant
    '''

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

    w = Function(V)

    dF_dm = inner(grad(w), grad(v)) * dx(0)
    A = assemble(F)
    B = assemble(a)
    dA_dz = assemble(dF_dm)

    solve(F == a, w) 

    # TODO: Compute quantity of interest
    w_nodal_values = np.array(w.vector()[:]) 
    y = np.mean(w_nodal_values)
    print("Average temperature: {}".format(y))

    # Kinda hacky way to get C as the averaging operator over the domain. Probably a better way
    d_omega_f = interpolate(Expression("1.0", degree=2), V)
    domain_integral = assemble(v*dx)
    domain_measure = np.dot(np.array(d_omega_f.vector()[:]), np.array(domain_integral[:]))
    C = domain_integral/domain_measure

    return w, y, A, B, C, dA_dz

def get_reduced_system(A, B, C, psi, phi):
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

def gradient(z, psi, phi, V):
    '''
    Computes the gradient of the objective function G with respect to 
    the parameters z using the method of Lagrange multipliers
    '''
    #Computes \nabla G(z)
    x, y, A, B, C, dA_dz = forward(z, V)
    A_r, B_r, C_r, x_r, y_r = get_reduced_system(A, B, C, psi, phi)
    lambda_f = np.linalg.solve(A.T, C.T * (y_r - y))
    lambda_r = np.linalg.solve(A_r.T, C_r.T * (y - y_r))

    dA_r_dz = np.dot(psi, np.dot(dA_dz, phi))

    grad = np.dot(lambda_f.T, np.dot(dA_dz, x)) + np.dot(lambda_r.T, np.dot(dA_r_dz, x_r))
    return grad

def cost_functional(z, psi, phi, V):
    '''
    Computes the error between the QoI produced by the full vs the 
    reduced order model (0.5 * || y - y_r ||_2^2)
    '''
    x, y, A, B, C, dA_dz = forward(z, V)
    A_r, B_r, C_r, x_r, y_r = get_reduced_system(A, B, C, psi, phi)
    return 0.5 * (y - y_r)**2

def optimizer(z_0, phi, V):
    '''
    Finds the parameter with the maximum ROM error given a starting point z_0
    '''
    bounds = scipy.optimize.Bounds(0.1, 10)
    optimize_result = scipy.optimize.minimize(-cost_functional, 
                            z_0, 
                            args=(psi, phi, V), 
                            method='L-BFGS-B', 
                            jac=gradient,
                            bounds=bounds)
    z_star = optimize_result.x
    g_z_star = optimize_result.fun
    return z_star, g_z_star
