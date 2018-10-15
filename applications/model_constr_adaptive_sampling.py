from scipy.optimize import minimize

def sample(basis, z_0, tol=1.0e-14, optimizer):
    z_star, g_z_star = optimizer() # z_star = arg max G(z)
    # Use scipy.optimize.minimize to get z_star. What's left is to use the Lagrangian method to
    # compute gradients (and maybe Hessians). One forward and one adjoint solve is all we need 
    # for this. 
    while min_val > tol:
        #Solve full system with z_star and obtain state vector x(z_star)
        #Enrich basis with generated snapshots
    return basis

