from error_optimization import optimize
import numpy as np
import time

def sample(basis, random_initial, optimizer, solver, tol=1.0e-14, maxiter=80):
    '''
    Model constrained adaptive sampler to create trial basis for ROM

    Parameters
    ----------
        basis : numpy.ndarray 
            Initial reduced basis
        z_0 : dolfin.Function
            Initial parameter guess
        optimizer : function
            Function to find param with maximum error
        solver : Solver class
            Solver with forward and reduced forward functions
        tol : tol
            Misfit error tolerance
        maxiter : int
            Maximum sampling iterations
    '''
    iterations = 0
    g_z_star = 1e30

    while g_z_star > tol and iterations < maxiter:
        # Perform optimization to find parameter with the maximum error
        z_0 = random_initial()

        t_i = time.time()
        prev_g_z_star = g_z_star
        z_star, g_z_star = optimizer(z_0, basis, solver)
        t_f = time.time()
        iterations += 1
        print("Optimizer iteration {} time taken: {}".format(iterations, t_f - t_i))

        #Solve full system with z_star and obtain state vector x(z_star)
        w, y, A, B, C  = solver.forward(z_star)
        w = w.vector()[:]
        w = w.reshape(w.shape[0], 1)

        #Enrich basis with generated snapshots
        basis = enrich(basis,w)
        
        print("Current error: {}, Improv: {}\n".format(g_z_star, prev_g_z_star - g_z_star))
        
    print("Sampling completed after {} iterations".format(iterations))
    return basis

def enrich(basis, w):
    '''
    Enrich basis given a new snapshot with Gram-Schmidt Orthogonalization
    TODO: I am throwing away samples during the optimization phase as I am
    using a blackbox scipy optimization tool. Implementing a bound-constrained
    optimizer will expose those to me and I can use those snapshots to produce
    a POD basis.

    basis - existing basis
    w     - new snapshot
    '''
    (n,k) = basis.shape
    U = np.hstack((basis, w))

    for j in range(0,k-1):
      U[:,-1] = U[:,-1] - ( U[:,-1].T @ U[:,j] )/( U[:,j].T @ U[:,j] ) * U[:,j]
    
    U[:,-1] = U[:,-1] / np.sqrt(U[:,-1].T @ U[:,-1])
    return U
