import sys
sys.path.append('../')
import time
import numpy as np
import matplotlib.pyplot as plt

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

    def __init__(self, V, err_model, phi):
        '''
        Initializes a thermal fin instance for a given function space

        Arguments:
            V - dolfin FunctionSpace
        '''

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
        self.Aphi = PETSc.Mat()

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
        self.data_ph = tf.placeholder(tf.float32, shape=(9,))
        self.B_obs = self.observation_operator()
        self.B_obs_phi = np.dot(self.B_obs, self.phi)

        self.dA_dsigmak = np.zeros((self.num_params, self.n, self.n))
        self.dA_dsigmak_phi = np.zeros((self.num_params, self.n, self.n_r))
        for i in range(self.num_params):
            A_i = assemble(inner(grad(self.w), grad(self.v))  * self.dx(i+1)).array()
            self.dA_dsigmak[i, :, :] = A_i
            self.dA_dsigmak_phi[i, :, :] = np.dot(A_i, self.phi)

        self.dl_model = err_model

        # Placeholders and precomputed values required for setting up gradients
        self.reduced_fwd_obs = tf.placeholder(tf.float32, shape=(9,))
        self.loss = tf.divide(tf.reduce_sum(tf.square(
            self.data_ph - self.reduced_fwd_obs - self.dl_model.layers[-1].output)), 2)
        self.NN_grad = gradients(self.loss, self.dl_model.input)

        self.dL_dsigmak = np.zeros(self.num_params)


        self.dA_dsigmak_phi = np.zeros((self.num_params, self.dofs, self.phi.shape[1]))
        for i in range(self.num_params):
            self.dA_dsigmak_phi[i, :, :] = np.dot(self.dA_dsigmak[i], self.phi)

        self.NN_grad_val = None

    #  @tf.function
    #  def cost_function(self, x_inp, data, y_ROM):
        #  y_NN = self.dl_model.predict([x_inp])[0]
        #  return tf.divide(tf.reduce_sum(tf.square(data - y_ROM - y_NN)), 2)


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

        k_s = self.subfin_avg_op(k)

        for i in range(len(k_s)):
            self.averaged_k_s[i].assign(k_s[i])

        assemble(self._F, tensor=self.A)
        A = self.A.array()
        self.psi = np.dot(A, self.phi)
        A_r = np.dot(self.psi.T, np.dot(A, self.phi))
        B_r = np.dot(self.psi.T, self.B)
        w_r = np.linalg.solve(A_r, B_r)

#####################
## PETSc version
####################

        self.A.mat().matMult(self.phi_p, self.psi_p)
        self.A.mat().matMult(self.phi_p, self.Aphi)
        self.psi_p.transposeMatMult(self.Aphi, self._A_r)
        self.psi_p.multTranspose(self.B_p.vec(), self._B_r)
        #  self.ksp.setOperators(self._A_r)
        #  self.ksp.solve(self._B_r, self._w_r)
        #  w_r = self._w_r[:]

        return w_r

    def forward_nine_param(self, k_s):
        '''
        Given average thermal conductivity values of each subfin,
        returns the reduced forward solve
        '''
        for i in range(len(k_s)):
            self.averaged_k_s[i].assign(k_s[i])

        assemble(self._F, tensor=self.A)

        A = self.A.array()
        self.psi = np.dot(A, self.phi)
        A_r = np.dot(self.psi.T, np.dot(A, self.phi))
        B_r = np.dot(self.psi.T, self.B)
        w_r = np.linalg.solve(A_r, B_r)

        self.A.mat().matMult(self.phi_p, self.psi_p)
        self.A.mat().matMult(self.phi_p, self.Aphi)
        self.psi_p.transposeMatMult(self.Aphi, self._A_r)
        self.psi_p.multTranspose(self.B_p.vec(), self._B_r)
        #  self.ksp.setOperators(self._A_r)
        #  self.ksp.solve(self._B_r, self._w_r)
        #  w_r = self._w_r[:]

        return w_r

    def qoi(self, w):
        return self.subfin_avg_op(w) 

    def qoi_reduced(self, w_r):
        #  w = Function(self.V)
        #  w.vector().set_local(np.dot(self.phi, w_r))
        #  return self.subfin_avg_op(w)
        return np.dot(self.B_obs_phi, w_r)

    #  def grad(self, k):
        #  k_s = self.subfin_avg_op(k)
        #  for i in range(len(k_s)):
            #  self.averaged_k_s[i].assign(k_s[i])

        #  z = Function(self.V)
        #  solve(self._F == self._a, z) 
        #  pred_obs = self.qoi(z)

        #  v = Function(self.V)
        #  adj_RHS = -np.dot(self.B_obs.T, pred_obs - self.data)
        #  assemble(self._adj_F, tensor=self.A_adj)
        #  v_nodal_vals = np.linalg.solve(self.A_adj.array(), adj_RHS)
        #  v.vector().set_local(v_nodal_vals)

        #  dL_dsigmak = np.zeros(len(self.averaged_k_s))
        #  for i in range(len(dL_dsigmak)):
            #  dL_dsigmak[i] = assemble(inner(grad(z), grad(v)) * self.dx(i+1))

        #  return np.dot(self.B_obs.T, dL_dsigmak)

    def grad_reduced(self, k):
        w_r = self.forward_reduced(k)
        reduced_fwd_obs = np.dot(self.B_obs_phi, w_r)

        reduced_adj_rhs = np.dot(self.B_obs_phi.T, self.data - reduced_fwd_obs)

        psi = self.psi_p.getDenseArray()
        A_r = self._A_r.getDenseArray()
        v_r = np.linalg.solve(A_r.T, reduced_adj_rhs)

        psi_v_r = np.dot(psi, v_r)

        A_phi_w_r = np.dot(self.dA_dsigmak_phi, w_r).T
        dROM_dk = np.dot(A_phi_w_r, self.B_obs)
        dJ_dk = np.dot(psi_v_r.T, dROM_dk)

        J = 0.5 * np.linalg.norm(self.data - reduced_fwd_obs)**2

        return dJ_dk, J

    def grad_romml(self, k):
        w_r = self.forward_reduced(k)
        e_NN = self.dl_model.predict([[k.vector()[:]]])[0]

        reduced_fwd_obs = np.dot(self.B_obs_phi, w_r)
        romml_fwd_obs = reduced_fwd_obs + e_NN

        reduced_adj_rhs = np.dot(self.B_obs_phi.T, self.data - romml_fwd_obs)

        psi = self.psi_p.getDenseArray()

        A_r = self._A_r.getDenseArray()
        v_r = np.linalg.solve(A_r.T, reduced_adj_rhs)

        psi_v_r = np.dot(psi, v_r)

        A_phi_w_r = np.dot(self.dA_dsigmak_phi, w_r).T
        dROM_dk = np.dot(A_phi_w_r, self.B_obs)
        f_x_dp_x = np.dot(psi_v_r.T, dROM_dk)

        x_inp = [k.vector()[:]]
        session = get_session()
        f_eps_dp_eps = session.run(self.NN_grad, 
                feed_dict={self.dl_model.input: x_inp,
                           self.reduced_fwd_obs: reduced_fwd_obs,
                           self.data_ph: self.data})[0]

        loss = session.run(self.loss,
                feed_dict={self.dl_model.input: x_inp,
                           self.reduced_fwd_obs: reduced_fwd_obs,
                           self.data_ph: self.data})

        self.NN_grad_val = f_eps_dp_eps.reshape(f_eps_dp_eps.size)
        grad =  f_x_dp_x + self.NN_grad_val
        return grad, loss

    def set_reduced_basis(self, phi):
        '''
        Sets the computed trial basis for the ROM system

        Arguments:
            phi: numpy array    - reduced basis
        '''
        self.phi = phi

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

        return B
