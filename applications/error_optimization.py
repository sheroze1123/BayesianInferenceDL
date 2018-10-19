from dolfin import *
import numpy as np
from forward_solve import forward, reduced_forward
import scipy.optimize
#  import time


def gradient(z_vec, psi, phi, V):
    '''
    Computes the gradient of the objective function G with respect to 
    the parameters z using the method of Lagrange multipliers
    '''

    z = Function(V)
    z.vector().set_local(z_vec)
    x_f, y, A, B, C, dA_dz = forward(z, V)
    x = x_f.vector()[:]
    A_r, B_r, C_r, x_r, y_r = reduced_forward(A, B, C, psi, phi)
    lambda_f = np.linalg.solve(A.T, C.T * (y_r - y))

    #TODO: Fix the singular matrix issues
    try:
        lambda_r = np.linalg.solve(A_r.T, C_r.T * (y - y_r))
    except np.linalg.linalg.LinAlgError as err:
        print ("Singular matrix in reduced adjoint solve")
        import pdb; pdb.set_trace()

    x_shape = x.shape
    grad_G = np.zeros(x.shape)
    x = x_f
    z_hat = TrialFunction(V)
    v = TestFunction(V)

    #TODO: Verify gradient
    dR_dz = assemble(z_hat * inner(grad(x), grad(v)) * dx).array()
    dR_r_dz = np.dot(psi.T, dR_dz)
    grad_G = dR_dz @ lambda_f + lambda_r @ dR_r_dz

    return -grad_G

def cost_functional(z_vec, psi, phi, V):
    '''
    Computes the error between the QoI produced by the full vs the 
    reduced order model (0.5 * || y - y_r ||_2^2)
    '''
    z = Function(V)
    z.vector().set_local(z_vec)
    x, y, A, B, C, dA_dz = forward(z, V)
    A_r, B_r, C_r, x_r, y_r = reduced_forward(A, B, C, psi, phi)

    # Since we are using a minimizer, cost function has to be negated
    cost = -0.5 * (y - y_r)**2

    #  print("\nCost at given basis: {}\n".format(-cost))
    return cost

def optimize(z_0, phi, V):
    '''
    Finds the parameter with the maximum ROM error given a starting point z_0
    '''
    w, y, A, B, C, dA_dz = forward(z_0, V)
    psi = np.dot(A, phi)
    init_cost = -cost_functional(z_0.vector()[:], psi, phi, V)

    z_bounds = scipy.optimize.Bounds(0.1, 10)
    optimize_result = scipy.optimize.minimize(cost_functional, 
                            z_0.vector()[:], 
                            args=(psi, phi, V), 
                            method='L-BFGS-B', 
                            jac=gradient,
                            options={'maxiter':20}, #Hilariously low
                            bounds=z_bounds)

    print("Optimizer return with message: {}".format(optimize_result.message))
    print("Initial G: {}, Final G: {}".format(init_cost, -optimize_result.fun))
    z_star = Function(V)
    z_star.vector().set_local(optimize_result.x)
    g_z_star = optimize_result.fun
    return z_star, g_z_star
