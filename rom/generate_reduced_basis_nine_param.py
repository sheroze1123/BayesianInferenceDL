import sys; sys.path.append('../')
import matplotlib; matplotlib.use('macosx')
import matplotlib.pyplot as plt
import numpy as np
from fom.thermal_fin import get_space
from dolfin import *
from fom.forward_solve import Fin
from bayesian_inference.gaussian_field import make_cov_chol

class SubFin(SubDomain):
    def __init__(self, subfin_bdry, **kwargs):
        self.y_b = subfin_bdry[0]
        self.is_left = subfin_bdry[1]
        super(SubFin, self).__init__(**kwargs)

    def inside(self, x, on_boundary):
        if self.is_left:
            return (between(x[1], (self.y_b, self.y_b+0.75)) and between(x[0], (0.0, 2.5)))
        else:
            return (between(x[1], (self.y_b, self.y_b+0.75)) and between(x[0], (3.5, 6.0)))

class SubFinBoundary(SubDomain):
    def __init__(self, subfin_bdry, **kwargs):
        self.y_b = subfin_bdry[0]
        self.is_left = subfin_bdry[1]
        super(SubFinBoundary, self).__init__(**kwargs)

    def inside(self, x, on_boundary):
        if self.is_left:
            return (on_boundary and between(x[1], (self.y_b, self.y_b+0.75)) 
                    and between(x[0], (0.0, 2.5)))
        else:
            return (on_boundary and between(x[1], (self.y_b, self.y_b+0.75)) 
                    and between(x[0], (3.5, 6.0)))

class CenterFin(SubDomain):
    def inside(self, x, on_boundary):
        return between(x[0], (2.5, 3.5))

class CenterFinBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return (on_boundary and between(x[0], (2.5, 3.5)) and not (near(x[1], 0.0)))

class AffineROMFin:
    '''
    A class the implements the heat conduction problem for a thermal fin
    '''

    def __init__(self, V):
        '''
        Initializes a thermal fin instance for a given function space

        Arguments:
            V - dolfin FunctionSpace
        '''

        self.num_params = 9

        self.phi = None
        self.V = V
        self.n = len(V.dofmap().dofs()) 
        self.data = None

        # Currently uses a fixed Biot number
        self.Bi = Constant(0.1)

        # Trial and test functions for the weak forms
        self.w = TrialFunction(V)
        self.v = TestFunction(V)

        self.w_hat = TestFunction(V)
        self.v_trial = TrialFunction(V)

        self.fin1 = SubFin([0.75, True])
        self.fin2 = SubFin([1.75, True])
        self.fin3 = SubFin([2.75, True])
        self.fin4 = SubFin([3.75, True])
        self.fin5 = CenterFin()
        self.fin6 = SubFin([3.75, False])
        self.fin7 = SubFin([2.75, False])
        self.fin8 = SubFin([1.75, False])
        self.fin9 = SubFin([0.75, False])

        mesh = V.mesh()
        domains = MeshFunction("size_t", mesh, mesh.topology().dim())
        domains.set_all(0)
        self.fin1.mark(domains, 1)
        self.fin2.mark(domains, 2)
        self.fin3.mark(domains, 3)
        self.fin4.mark(domains, 4)
        self.fin5.mark(domains, 5)
        self.fin6.mark(domains, 6)
        self.fin7.mark(domains, 7)
        self.fin8.mark(domains, 8)
        self.fin9.mark(domains, 9)

        # Marking boundaries for boundary conditions
        bottom = CompiledSubDomain("near(x[1], side) && on_boundary", side = 0.0)
        fin1_b = SubFinBoundary([0.75, True])
        fin2_b = SubFin([1.75, True])
        fin3_b = SubFin([2.75, True])
        fin4_b = SubFin([3.75, True])
        fin5_b = CenterFinBoundary()
        fin6_b = SubFin([3.75, False])
        fin7_b = SubFin([2.75, False])
        fin8_b = SubFin([1.75, False])
        fin9_b = SubFin([0.75, False])

        boundaries = MeshFunction("size_t", mesh, mesh.topology().dim()-1)
        boundaries.set_all(0)
        fin1_b.mark(boundaries, 1)
        fin2_b.mark(boundaries, 2)
        fin3_b.mark(boundaries, 3)
        fin4_b.mark(boundaries, 4)
        fin5_b.mark(boundaries, 5)
        fin6_b.mark(boundaries, 6)
        fin7_b.mark(boundaries, 7)
        fin8_b.mark(boundaries, 8)
        fin9_b.mark(boundaries, 9)
        bottom.mark(boundaries, 10)

        self.dx = Measure('dx', domain=mesh, subdomain_data=domains)
        self.ds = Measure('ds', domain=mesh, subdomain_data=boundaries)

        self.fin1_A = assemble(Constant(1.0) * self.dx(1))
        self.fin2_A = assemble(Constant(1.0) * self.dx(2))
        self.fin3_A = assemble(Constant(1.0) * self.dx(3))
        self.fin4_A = assemble(Constant(1.0) * self.dx(4))
        self.fin5_A = assemble(Constant(1.0) * self.dx(5))
        self.fin6_A = assemble(Constant(1.0) * self.dx(6))
        self.fin7_A = assemble(Constant(1.0) * self.dx(7))
        self.fin8_A = assemble(Constant(1.0) * self.dx(8))
        self.fin9_A = assemble(Constant(1.0) * self.dx(9))

        # Dummy init for averaged subfin values
        self.averaged_k_s = [Constant(1.0) for i in range(self.num_params)]

        self._F = self.averaged_k_s[0] * inner(grad(self.w), grad(self.v)) \
                                                            * self.dx(1) + \
                   self.Bi *  self.v * self.w * self.ds(1)
        for i in range(1,9):
            self._F += self.averaged_k_s[i] * inner(grad(self.w), grad(self.v)) \
                                                                * self.dx(i+1) + \
                       self.Bi *  self.v * self.w * self.ds(i+1)
        self._a = self.v * self.ds(10)

        self.B = assemble(self._a)[:]
        self._A = PETScMatrix()
        self.B_obs = self.observation_operator()

    def forward(self, k):
        '''
        Computes the forward solution given a conductivity field
        by averaging over subfins.

        Arguments:
            k : dolfin function - thermal conductivity

        Returns:
            w : dolfin function - temperature distribution over the fin 
        '''

        k_s = self.subfin_avg_op(k)

        for i in range(len(k_s)):
            self.averaged_k_s[i].assign(k_s[i])

        w = Function(self.V)

        solve(self._F == self._a, w) 

        return w

    def forward_reduced(self, k):
        k_s = self.subfin_avg_op(k)
        return self.forward_nine_param_r(k_s, self.phi)

    def forward_nine_param(self, k_s):
        for i in range(len(k_s)):
            self.averaged_k_s[i].assign(k_s[i])
        w = Function(self.V)
        solve(self._F == self._a, w) 
        return w

    def forward_nine_param_r(self, k_s, phi):
        '''
        Given average thermal conductivity values of each subfin,
        returns the reduced forward solve
        '''
        for i in range(len(k_s)):
            self.averaged_k_s[i].assign(k_s[i])

        self.phi = phi

        assemble(self._F, tensor=self._A)

        A = self._A.array()
        psi = np.dot(A, self.phi)
        A_r = np.dot(psi.T, psi)
        B_r = np.dot(psi.T, self.B)

        w_r = np.linalg.solve(A_r, B_r)

        return w_r

    def qoi_reduced(self, w_r):
        return np.dot(self.B_obs, np.dot(self.phi, w_r))

    def subfin_avg_op(self, k):
        # Subfin averages
        fin1_avg = assemble(k * self.dx(1))/self.fin1_A 
        fin2_avg = assemble(k * self.dx(2))/self.fin2_A 
        fin3_avg = assemble(k * self.dx(3))/self.fin3_A 
        fin4_avg = assemble(k * self.dx(4))/self.fin4_A 
        fin5_avg = assemble(k * self.dx(5))/self.fin5_A
        fin6_avg = assemble(k * self.dx(6))/self.fin6_A 
        fin7_avg = assemble(k * self.dx(7))/self.fin7_A 
        fin8_avg = assemble(k * self.dx(8))/self.fin8_A 
        fin9_avg = assemble(k * self.dx(9))/self.fin9_A 
        subfin_avgs = np.array([fin1_avg, fin2_avg, fin3_avg, fin4_avg, fin5_avg, 
            fin6_avg, fin7_avg, fin8_avg, fin9_avg])
        #  print("Subfin averages: {}".format(subfin_avgs))
        return subfin_avgs

    def observation_operator(self):
        z = TestFunction(self.V)
        fin1_avg = assemble(z * self.dx(1))/self.fin1_A 
        fin2_avg = assemble(z * self.dx(2))/self.fin2_A 
        fin3_avg = assemble(z * self.dx(3))/self.fin3_A 
        fin4_avg = assemble(z * self.dx(4))/self.fin4_A 
        fin5_avg = assemble(z * self.dx(5))/self.fin5_A
        fin6_avg = assemble(z * self.dx(6))/self.fin6_A 
        fin7_avg = assemble(z * self.dx(7))/self.fin7_A 
        fin8_avg = assemble(z * self.dx(8))/self.fin8_A 
        fin9_avg = assemble(z * self.dx(9))/self.fin9_A 

        B = np.vstack((
            fin1_avg[:],
            fin2_avg[:],
            fin3_avg[:],
            fin4_avg[:],
            fin5_avg[:],
            fin6_avg[:],
            fin7_avg[:],
            fin8_avg[:],
            fin9_avg[:]))

        return B

    def set_phi(self, phi):
        self.phi = phi

    def set_data(self, data):
        self.data = data

class SolverWrapper:
    def __init__(self, solver, data): 
        self.solver = solver
        self.data = data
        self.z = Function(V)

    def cost_function(self, z_v):
        self.z.vector().set_local(z_v)
        w, y, A, B, C = self.solver.forward(self.z)
        y = self.solver.qoi_operator(w)
        cost = 0.5 * np.linalg.norm(y - self.data)**2 + assemble(self.solver.reg)
        return cost

    def gradient(self, z_v):
        self.z.vector().set_local(z_v)
        grad = self.solver.gradient(self.z, self.data) + assemble(self.solver.grad_reg)[:]
        return grad

class RSolverWrapper:
    def __init__(self, solver_r, data): 
        self.solver_r = solver_r
        self.z = Function(V)
        self.data = data
        self.cost = None
        self.grad = None

    def cost_function(self, z_v):
        self.z.vector().set_local(z_v)
        w_r = self.solver_r.forward_reduced(self.z)
        y_r = self.solver_r.qoi_reduced(w_r)
        self.cost = 0.5 * np.linalg.norm(y_r - self.data)**2
        return self.cost


#  resolution = 40
#  V = get_space(resolution)
#  chol = make_cov_chol(V, length=1.2)
#  solver_r = AffineROMFin(V)
#  dofs = len(V.dofmap().dofs())

#  #  samples = 200
#  #  Y = np.zeros((samples, dofs))
#  #  for i in range(0,samples):
    #  #  k = np.random.uniform(0.1, 3.5, 9)
    #  #  w = solver.forward_nine_param(k)
    #  #  Y[i,:] = w.vector()[:]

#  #  K = np.dot(Y, Y.T)

#  #  # Initial basis vectors computed using proper orthogonal decomposition
#  #  e,v = np.linalg.eig(K)

#  #  plt.semilogy(e.real)
#  #  plt.savefig('eigs.png', dpi=200)

#  #  basis_size = 81
#  #  U = np.zeros((basis_size, dofs))
#  #  for i in range(basis_size):
    #  #  e_i = v[:,i].real
    #  #  U[i,:] = np.sum(np.dot(np.diag(e_i), Y),0)

#  #  basis = U.T
#  #  np.savetxt('../data/basis_nine_param.txt', basis, delimiter=',')

#  basis = np.loadtxt('../data/basis_nine_param.txt', delimiter=',')

#  solver = Fin(V)


#  z_true = Function(V)
#  norm = np.random.randn(len(chol))
#  nodal_vals = np.exp(0.5 * chol.T @ norm)
#  z_true.vector().set_local(nodal_vals)
#  w, y, A, B, C = solver.forward(z_true)
#  data = solver.qoi_operator(w)

#  solver_fom = SolverWrapper(solver, data)
#  solver_r = AffineROMFin(V)
#  solver_r.set_phi(basis)
#  solver_rom = RSolverWrapper(solver_r, data)

#  z = Function(V)
#  norm = np.random.randn(len(chol))
#  eps_z = np.exp(0.5 * chol.T @ norm)
#  z.vector().set_local(eps_z)
#  eps_norm = np.sqrt(assemble(inner(z,z)*dx))
#  eps_norm = np.linalg.norm(eps_z)
#  eps_z = eps_z/eps_norm
#  norm = np.random.randn(len(chol))
#  z_ = np.exp(0.5 * chol.T @ norm)

#  hs = np.linspace(0, 1, 500)
#  pis = []

#  for h in hs:
    #  pi_h = solver_fom.cost_function(z_ + h * eps_z)
    #  pis.append(pi_h)

#  plt.plot(hs, pis)
#  plt.savefig('func_dir_fom.png', dpi=200)
#  plt.cla()
#  plt.clf()

#  pi_roms = []

#  for h in hs:
    #  pi_h = solver_rom.cost_function(z_ + h * eps_z)
    #  pi_roms.append(pi_h)

#  plt.plot(hs, pi_roms)
#  plt.savefig('func_dir_rom.png', dpi=200)
#  plt.cla()
#  plt.clf()
