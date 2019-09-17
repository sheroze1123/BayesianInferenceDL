import sys
sys.path.append('../')

from dolfin import *
import numpy as np
from fom.forward_solve import Fin
import scipy.optimize
#  import time

def gradient_five_param(k_vec, psi, phi, solver):
    z_vec = solver.five_param_to_function(k_vec).vector()[:]
    return solver.dz_dk_T @ gradient(z_vec, psi, phi, solver)

def cost_functional_five_param(k_vec, psi, phi, solver):
    z_vec = solver.five_param_to_function(k_vec).vector()[:]
    return cost_functional(z_vec, psi, phi, solver)

def optimize_five_param(k_0, phi, solver):
    '''
    Finds the parameter with the maximum ROM error given a starting point z_0
    '''
    w, y, A, B, C = solver.forward_five_param(k_0)
    psi = np.dot(A, phi)
    init_cost = -cost_functional_five_param(k_0, psi, phi, solver)

    z_bounds = scipy.optimize.Bounds(0.1, 10)
    res = scipy.optimize.minimize(cost_functional_five_param, 
                            k_0, 
                            args=(psi, phi, solver), 
                            method='L-BFGS-B', 
                            jac=gradient_five_param,
                            options={'maxiter':200}, 
                            bounds=z_bounds)

    print("Optimizer message: {}".format(res.message))
    print("Cost functional evaluations: {}, Opt. Iterations: {}".format(res.nfev, res.nit))
    print("Initial G: {}, Final G: {}".format(init_cost, -res.fun))
    z_star = solver.five_param_to_function(res.x)
    g_z_star = -res.fun
    return z_star, g_z_star

def gradient(z_vec, psi, phi, solver):
    '''
    Computes the gradient of the objective function G with respect to 
    the parameters z using the method of Lagrange multipliers
    '''

    #TODO: Rewrite this in full PETSc for improved performance and standardization
    z = Function(solver.V)
    z.vector().set_local(z_vec)
    x, y, A, B, C = solver.forward(z)
    A_r, B_r, C_r, x_r, y_r = solver.reduced_forward(A, B, C, psi, phi)
    lambda_f = np.linalg.solve(A.T, C.T * (y_r - y))

    #TODO: Fix the singular matrix issues
    try:
        lambda_r = np.linalg.solve(A_r.T, C_r.T * (y - y_r))
    except np.linalg.linalg.LinAlgError as err:
        print ("Singular matrix in reduced adjoint solve")
        import pdb; pdb.set_trace()

    z_hat = TrialFunction(solver.V)
    v = TestFunction(solver.V)

    #TODO: Verify gradient
    dR_dz = assemble(z_hat * inner(grad(x), grad(v)) * solver.dx).array()
    dR_r_dz = np.dot(psi.T, dR_dz)
    grad_G = dR_dz @ lambda_f + lambda_r @ dR_r_dz

    return -grad_G

def cost_functional(z_vec, psi, phi, solver):
    '''
    Computes the error between the QoI produced by the full vs the 
    reduced order model (0.5 * || y - y_r ||_2^2)
    '''
    z = Function(solver.V)
    z.vector().set_local(z_vec)
    x, y, A, B, C = solver.forward(z)
    A_r, B_r, C_r, x_r, y_r = solver.reduced_forward(A, B, C, psi, phi)

    # Since we are using a minimizer, cost function has to be negated
    cost = -0.5 * (y - y_r)**2

    #  print("\nCost at given basis: {}\n".format(-cost))
    return cost

def optimize(z_0, phi, solver):
    '''
    Finds the parameter with the maximum ROM error given a starting point z_0
    '''
    w, y, A, B, C = solver.forward(z_0)
    psi = np.dot(A, phi)
    init_cost = -cost_functional(z_0.vector()[:], psi, phi, solver)

    z_bounds = scipy.optimize.Bounds(0.1, 10)
    res = scipy.optimize.minimize(cost_functional, 
                            z_0.vector()[:], 
                            args=(psi, phi, solver), 
                            method='L-BFGS-B', 
                            jac=gradient,
                            options={'maxiter':200}, 
                            bounds=z_bounds)

    print("Optimizer message: {}".format(res.message))
    print("Cost functional evaluations: {}, Opt. Iterations: {}".format(res.nfev, res.nit))
    print("Initial G: {}, Final G: {}".format(init_cost, -res.fun))
    z_star = Function(solver.V)
    z_star.vector().set_local(res.x)
    g_z_star = -res.fun
    return z_star, g_z_star
