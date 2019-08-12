from thermal_fin import get_space
import numpy as np
from scipy import linalg, spatial
from fenics import Function, plot
import matplotlib.pyplot as plt

def make_cov_chol(V, kern_type='m52', length=1.6):
    Wdofs_x = V.tabulate_dof_coordinates().reshape((-1, 2))
    V0_dofs = V.dofmap().dofs()
    points = Wdofs_x[V0_dofs, :] 
    dists = spatial.distance.pdist(points)
    dists = spatial.distance.squareform(dists)

    if kern_type=='sq_exp':
        # Squared Exponential / Radial Basis Function
        alpha = 1 / (2 * length ** 2)
        noise_var = 1e-5
        cov = np.exp(-alpha * dists ** 2) + np.eye(len(points)) * noise_var
    elif kern_type=='m52':
        # Matern52
        tmp = np.sqrt(5) * dists / length
        cov = (1 + tmp + tmp * tmp / 3) * np.exp(-tmp)
    else:
        # Matern32
        tmp = np.sqrt(3) * dists / length
        cov = (1 + tmp) * np.exp(-tmp)

    chol = linalg.cholesky(cov)
    return chol

#  V = get_space(40)
#  chol = make_cov_chol(V)
#  norm = np.random.randn(len(chol))
#  q = Function(V)
#  q.vector().set_local(np.exp(0.5 * chol.T @ norm))

#  f = plot(q)
#  plt.colorbar(f)
#  plt.show()
#  plt.savefig('fin.png')
#  f.write_png('fin.png')
