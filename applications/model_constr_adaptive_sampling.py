from error_optimization import optimize
import numpy as np
import time

def sample(basis, z_0, optimizer, forward, V, tol=1.0e-14, maxiter=500):
    '''
    Model constrained adaptive sampler to create trial basis for ROM
    Arguments:
        basis     - initial reduced basis
        z_0       - initial parameter guess
        optimizer - callable function to find param with maximum error
        forward   - callable function to solve forward problem
        V         - function space for forward solve
        tol       - misfit error tolerance
        maxiter   - maximum sampling iterations
    '''
    iterations = 0
    g_z_star = 1e30

    while abs(g_z_star) > tol and iterations < maxiter:
        # Perform optimization to find parameter with the maximum error

        t_i = time.time()
        z_star, g_z_star = optimizer(z_0, basis, V) 
        t_f = time.time()
        print("Single optimizer run time taken: {}".format(t_f - t_i))

        #Solve full system with z_star and obtain state vector x(z_star)
        #  import pdb; pdb.set_trace()
        w = forward(z_star, V)[0].vector()[:]
        w = w.reshape(w.shape[0], 1)
        #Enrich basis with generated snapshots
        basis = enrich(basis,w)
        iterations += 1
        
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
