import sys
sys.path.append('../')
import time
import numpy as np
import matplotlib.pyplot as plt
from utils import nb

from dolfin import *
from mshr import Rectangle, generate_mesh
from petsc4py import PETSc

from tensorflow.keras.backend import get_session, gradients
#  from tensorflow.keras.backend import gradients
import tensorflow as tf

from fom.thermal_fin import get_space

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

    def __init__(self, V, err_model, phi, external_obs=False):
        '''
        Initializes a thermal fin instance for a given function space

        Arguments:
            V - dolfin FunctionSpace
        '''
        self.fwd_time = 0.0
        self.rom_grad_time = 0.0
        self.romml_grad_time = 0.0
        self.romml_grad_time_dl = 0.0

        self.num_params = 9

        self.phi = phi
        (self.n,self.n_r) = self.phi.shape

        self.phi_p = PETSc.Mat().createDense([self.n,self.n_r], array=phi)
        self.phi_p.assemblyBegin()
        self.phi_p.assemblyEnd()

        self.V = V
        self.dofs = len(V.dofmap().dofs()) 

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


        # Reduced variables (PETSc)
        self._w_r = PETSc.Vec().createSeq(self.n_r)
        self._A_r = PETSc.Mat()
        self._B_r = PETSc.Vec().createSeq(self.n_r)
        self._C_r = PETSc.Vec().createSeq(self.n_r)
        self.B_p = as_backend_type(assemble(self._a))
        self.ksp = PETSc.KSP().create()
        #  self.ksp.setType('cg')
        #  self.ksp.setType('gmres')
        self.psi_p = PETSc.Mat()

        self.A = PETScMatrix()
        self.B = self.B_p[:]

        self._adj_F = self.averaged_k_s[0] * inner(grad(self.w_hat), grad(self.v_trial)) \
                                                                    * self.dx(1) + \
                        self.Bi * self.w_hat * self.v_trial * self.ds(1)
        for i in range(1,9):
            self._adj_F += self.averaged_k_s[i] * inner(grad(self.w_hat), grad(self.v_trial)) \
                                                                * self.dx(i+1) + \
                       self.Bi *  self.w_hat * self.v_trial * self.ds(i+1)
        self.A_adj = PETScMatrix()

        self.psi = None
        self.data = None

        if external_obs:
            # Exterior boundary for measurements
            exterior_domain = CompiledSubDomain("!near(x[1], 0.0) && on_boundary")
            exterior_bc = DirichletBC(V, 1, exterior_domain)
            u = Function(V)
            exterior_bc.apply(u.vector())
            self.boundary_indices = (u.vector() == 1)
            self.n_obs = 40
            #  np.random.seed(32)
            #  b_vals = np.random.choice(np.nonzero(self.boundary_indices)[0], self.n_obs)
            #  np.save("rand_boundary_indices", b_vals)
            b_vals = np.load("../bayesian_inference/rand_boundary_indices.npy")
            self.B_obs = np.zeros((self.n_obs, self.dofs))
            self.B_obs[np.arange(self.n_obs), b_vals] = 1
        else:
            self.n_obs = 9
            self.B_obs = self.observation_operator()

        self.dsigma_dk = self.observation_operator()
        self.data_ph = tf.placeholder(tf.float32, shape=(self.n_obs,))
        self.B_obs_phi = np.dot(self.B_obs, self.phi)
        #  import pdb; pdb.set_trace()

        self.dA_dsigmak = np.zeros((self.num_params, self.n, self.n))
        self.dA_dsigmak_phi = np.zeros((self.num_params, self.n, self.n_r))
        for i in range(self.num_params):
            A_i = assemble(inner(grad(self.w), grad(self.v))  * self.dx(i+1)).array()
            self.dA_dsigmak[i, :, :] = A_i
            self.dA_dsigmak_phi[i, :, :] = np.dot(A_i, self.phi)

        self.dl_model = err_model

        # Placeholders and precomputed values required for setting up gradients
        self.reduced_fwd_obs = tf.placeholder(tf.float32, shape=(self.n_obs,))
        self.loss = tf.divide(tf.reduce_sum(tf.square(
            self.data_ph - self.reduced_fwd_obs - self.dl_model.layers[-1].output)), 2)
        self.NN_grad = gradients(self.loss, self.dl_model.input)

        #  self.dA_dsigmak_phi = np.zeros((self.num_params, self.dofs, self.phi.shape[1]))
        #  for i in range(self.num_params):
            #  self.dA_dsigmak_phi[i, :, :] = np.dot(self.dA_dsigmak[i], self.phi)

        self.NN_grad_val = None
        self.session = get_session()

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
        '''
        Computes the forward solution given a conductivity field
        by averaging over subfins and then using a ROM on the resulting system.

        Arguments:
            k : dolfin function - thermal conductivity
            phi: numpy array    - reduced basis

        Returns:
            w : numpy array     - reduced solution
        '''

        t_i = time.time()
        k_s = self.subfin_avg_op(k)
        self.fwd_time += (time.time() - t_i)
        return self.forward_nine_param_reduced(k_s)

    def forward_nine_param_reduced(self, k_s):
        '''
        Given average thermal conductivity values of each subfin,
        returns the reduced forward solve

        Args:
            k_s : Average conductivity of each subfin

        Returns:
            w_r : Temperature in reduced dimensions
        '''
        for i in range(len(k_s)):
            self.averaged_k_s[i].assign(k_s[i])

        assemble(self._F, tensor=self.A) #Precalculate this>?

        t_i = time.time()
        self.A.mat().matMult(self.phi_p, self.psi_p)
        self.psi_p.transposeMatMult(self.psi_p, self._A_r)
        self.psi_p.multTranspose(self.B_p.vec(), self._B_r)
        self.fwd_time += (time.time() - t_i)
        #w_r = np.linalg.solve(self._A_r.getDenseArray(), self._B_r.getArray())
        
        A_r_dense = self._A_r.getDenseArray()
        B_r_dense = self._B_r.getArray()
        t_i = time.time()
        w_r = np.linalg.solve(A_r_dense, B_r_dense)
        self.fwd_time += (time.time() - t_i)

        #  self.ksp.setOperators(self._A_r)
        #  self.ksp.solve(self._B_r, self._w_r)
        #  w_r = self._w_r[:]
        return w_r

    def qoi(self, w):
        '''
        Computes the quantity of interest

        Args:
            w : Temperature distribution over the fin
        '''
        qoi_vals =  np.dot(self.B_obs, w.vector()[:])
        return qoi_vals
        #  return self.subfin_avg_op(w) 

    def qoi_reduced(self, w_r):
        '''
        Computes the quantity of interest

        Args:
            w_r : Temperature distribution over the fin in reduced dimensions
        '''
        t_i = time.time()
        qoi_vals =  np.dot(self.B_obs_phi, w_r)
        self.fwd_time += (time.time() - t_i)
        return qoi_vals

    def grad_reduced(self, k):
        w_r = self.forward_reduced(k)
        t_i = time.time()
        reduced_fwd_obs = np.dot(self.B_obs_phi, w_r)
        reduced_adj_rhs = np.dot(self.B_obs_phi.T, self.data - reduced_fwd_obs)
        self.rom_grad_time += (time.time() - t_i)

        A_r = self._A_r.getDenseArray()
        psi = self.psi_p.getDenseArray()

        t_i = time.time()
        v_r = np.linalg.solve(A_r.T, reduced_adj_rhs)
        psi_v_r = np.dot(psi, v_r)

        A_phi_w_r = np.dot(self.dA_dsigmak_phi, w_r).T
        dROM_dk = np.dot(A_phi_w_r, self.dsigma_dk)
        dJ_dk = np.dot(psi_v_r.T, dROM_dk)
        self.rom_grad_time += (time.time() - t_i)

        J = 0.5 * np.linalg.norm(self.data - reduced_fwd_obs)**2

        return dJ_dk, J

    def grad_romml(self, k):
        w_r = self.forward_reduced(k)
        e_NN = self.dl_model.predict([[k.vector()[:]]])[0]
        t_i = time.time()

        reduced_fwd_obs = np.dot(self.B_obs_phi, w_r)
        romml_fwd_obs = reduced_fwd_obs + e_NN
        reduced_adj_rhs = np.dot(self.B_obs_phi.T, self.data - romml_fwd_obs)

        self.romml_grad_time += (time.time() - t_i)

        A_r = self._A_r.getDenseArray()
        psi = self.psi_p.getDenseArray()

        t_i = time.time()
        v_r = np.linalg.solve(A_r.T, reduced_adj_rhs)
        psi_v_r = np.dot(psi, v_r)

        A_phi_w_r = np.dot(self.dA_dsigmak_phi, w_r).T
        dROM_dk = np.dot(A_phi_w_r, self.dsigma_dk)
        f_x_dp_x = np.dot(psi_v_r.T, dROM_dk)
        self.romml_grad_time += (time.time() - t_i)

        x_inp = [k.vector()[:]]
        t_i = time.time()
        f_eps_dp_eps = self.session.run(self.NN_grad, 
                feed_dict={self.dl_model.input: x_inp,
                           self.reduced_fwd_obs: reduced_fwd_obs,
                           self.data_ph: self.data})[0]

        self.NN_grad_val = f_eps_dp_eps.reshape(f_eps_dp_eps.size)
        self.romml_grad_time_dl += (time.time() - t_i)
        grad =  f_x_dp_x + self.NN_grad_val

        loss = self.session.run(self.loss,
                feed_dict={self.dl_model.input: x_inp,
                           self.reduced_fwd_obs: reduced_fwd_obs,
                           self.data_ph: self.data})
        return grad, loss

    def set_data(self, data):
        self.data = data

    def set_dl_model(self, model):
        self.dl_model = model

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

        #TODO: If random surface obs, change this

        return B
