from dolfin import *
import numpy as np
from forward_solve import forward
import scipy.optimize
import time

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

def gradient(z_vec, psi, phi, V):
    '''
    Computes the gradient of the objective function G with respect to 
    the parameters z using the method of Lagrange multipliers
    '''
    #Computes \nabla G(z)
    z = Function(V)
    z.vector().set_local(z_vec)
    x_f, y, A, B, C, dA_dz = forward(z, V)
    x = x_f.vector()[:]
    A_r, B_r, C_r, x_r, y_r = get_reduced_system(A, B, C, psi, phi)
    lambda_f = np.linalg.solve(A.T, C.T * (y_r - y))
    lambda_r = np.linalg.solve(A_r.T, C_r.T * (y - y_r))

    grad_G = np.zeros(x.shape)
    w = TrialFunction(V)
    v = TestFunction(V)

    t_i = time.time()
    # Reimplement this in tensor format to vastly improve perforamnce
    for i in range(x.shape[0]):
        m = Function(V)
        m.vector()[i] = 1
        dA_dz_weak = inner(m * grad(w), grad(v)) * dx
        dA_dz = assemble(dA_dz_weak).array()
        dA_r_dz = np.dot(psi.T, np.dot(dA_dz, phi))
        grad_G[i] = np.dot(lambda_f.T, np.dot(dA_dz, x)) + np.dot(lambda_r.T, np.dot(dA_r_dz, x_r))

    t_f = time.time()
    print("Time taken for gradient computation: {}".format(t_f - t_i))

    #  import pdb; pdb.set_trace()
    return grad_G

def cost_functional(z_vec, psi, phi, V):
    '''
    Computes the error between the QoI produced by the full vs the 
    reduced order model (0.5 * || y - y_r ||_2^2)
    '''
    z = Function(V)
    z.vector().set_local(z_vec)
    x, y, A, B, C, dA_dz = forward(z, V)
    A_r, B_r, C_r, x_r, y_r = get_reduced_system(A, B, C, psi, phi)

    # Since we are using a minimizer, cost function has to be negated
    
    cost = -0.5 * (y - y_r)**2
    print("\nCost at given basis: {}\n".format(cost))
    return cost

def optimize(z_0, phi, V):
    '''
    Finds the parameter with the maximum ROM error given a starting point z_0
    '''
    w, y, A, B, C, dA_dz = forward(z_0, V)
    psi = np.dot(A, phi)

    z_bounds = scipy.optimize.Bounds(0.1, 10)
    optimize_result = scipy.optimize.minimize(cost_functional, 
                            z_0.vector()[:], 
                            args=(psi, phi, V), 
                            method='L-BFGS-B', 
                            jac=gradient,
                            options={'maxiter':3}, #Hilariously low
                            bounds=z_bounds)

    print("\nOptimization run complete\n")
    z_star = Function(V)
    z_star.vector().set_local(optimize_result.x)
    g_z_star = optimize_result.fun
    return z_star, g_z_star
