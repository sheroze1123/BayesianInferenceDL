from scipy.optimize import minimize

def sample(basis, z_0, tol=1.0e-14, maxiter=500, optimizer):
    iterations = 1
    while g_z_star > tol or iterations > maxiter:
        # Perform optimization to find parameter with the maximum error
        z_star, g_z_star = optimizer(z_0, basis, V) 

        #Solve full system with z_star and obtain state vector x(z_star)
        w = forward(z_star, V)
        #Enrich basis with generated snapshots
        basis = enrich(basis,w)
        iterations += 1
    return basis

