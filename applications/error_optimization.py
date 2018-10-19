from dolfin import *
import numpy as np
from forward_solve import forward, reduced_forward
import scipy.optimize
import time


class G:
    def __init__(self, psi, phi, V):
        self.psi = psi
        self.phi = phi
        self.V = V
        self.dofs = len(V.dofmap().dofs())
        w = TrialFunction(V)
        v = TestFunction(V)
        self.dA_dzs = []
        self.dA_r_dzs = []

        # Precompute dA_dz terms to speed things up. Still not the fastest method probably.
        for i in range(self.dofs):
            m = Function(V)
            m.vector()[i] = 1
            dA_dz_weak = inner(m * grad(w), grad(v)) * dx
            dA_dz = assemble(dA_dz_weak)
            self.dA_dzs.append(dA_dz)
            dA_r_dz = np.dot(psi.T, np.dot(dA_dz.array(), phi))
            self.dA_r_dzs.append(dA_r_dz)

    def gradient(self, z_vec, psi, phi, V):
        '''
        Computes the gradient of the objective function G with respect to 
        the parameters z using the method of Lagrange multipliers
        '''
        #Computes \nabla G(z)
        z = Function(V)
        z.vector().set_local(z_vec)
        x_f, y, A, B, C, dA_dz = forward(z, V)
        x = x_f.vector()[:]
        A_r, B_r, C_r, x_r, y_r = reduced_forward(A, B, C, psi, phi)
        lambda_f = np.linalg.solve(A.T, C.T * (y_r - y))
        lambda_r = np.linalg.solve(A_r.T, C_r.T * (y - y_r))

        x_shape = x.shape
        grad_G = np.zeros(x.shape)
        x = x_f
        w = TrialFunction(V)
        v = TestFunction(V)

        t_i = time.time()
        # Reimplement this in tensor format to vastly improve perforamnce
        #TODO Precompute dA_dz and store them sparsely
        #TODO Only use PEtscVector
        #TODO Can we do this better analytically?
        dR_dz = np.zeros((lambda_f.shape[0], lambda_f.shape[0]))
        dR_r_dz = np.zeros((lambda_f.shape[0], lambda_r.shape[0]))
        for i in range(x_shape[0]):
            dR_dz[i][:] = (self.dA_dzs[i] * x.vector())[:]
            dR_r_dz[i][:] = np.dot(self.dA_r_dzs[i], x_r)

        grad_G = dR_dz @ lambda_f + dR_r_dz @ lambda_r
        t_f = time.time()
        print("Time taken for gradient computation: {}".format(t_f - t_i))

        #  import pdb; pdb.set_trace()
        return -grad_G

    def cost_functional(self, z_vec, psi, phi, V):
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
        print("\nCost at given basis: {}\n".format(-cost))
        return cost

def optimize(z_0, phi, V):
    '''
    Finds the parameter with the maximum ROM error given a starting point z_0
    '''
    w, y, A, B, C, dA_dz = forward(z_0, V)
    psi = np.dot(A, phi)
    G_instance = G(psi, phi, V)

    z_bounds = scipy.optimize.Bounds(0.1, 10)
    optimize_result = scipy.optimize.minimize(G_instance.cost_functional, 
                            z_0.vector()[:], 
                            args=(psi, phi, V), 
                            method='L-BFGS-B', 
                            jac=G_instance.gradient,
                            options={'maxiter':3}, #Hilariously low
                            bounds=z_bounds)

    print("\nOptimization run complete\n")
    z_star = Function(V)
    z_star.vector().set_local(optimize_result.x)
    g_z_star = optimize_result.fun
    return z_star, g_z_star
