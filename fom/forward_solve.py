from dolfin import *
import numpy as np
from mshr import Rectangle, generate_mesh

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

class SubfinExpr(UserExpression):
    def __init__(self, subfin_bdry, **kwargs):
        self.y_b = subfin_bdry[0]
        self.isLeft = subfin_bdry[1]
        super(SubfinExpr, self).__init__(**kwargs)

    def eval(self, value, x):
        y_t = self.y_b + 0.25
        if self.isLeft:
            if (x[1] >= self.y_b) and (x[1] <= y_t) and (x[0] < 2.5):
                value[0] = 1.0
            else:
                value[0] = 0.0
        else:
            if (x[1] >= self.y_b) and (x[1] <= y_t) and (x[0] > 3.5):
                value[0] = 1.0
            else:
                value[0] = 0.0

    def value_shape(self):
        return ()

class SubfinValExpr(UserExpression):
    def __init__(self, k_s, **kwargs):
        self.k1, self.k2, self.k3, self.k4, self.k5, self.k6, self.k7, self.k8, self.k9 = k_s
        super(SubfinValExpr, self).__init__(**kwargs)
    def eval(self, value, x):
        if between(x[0], (2.5, 3.5)):
            value[0] = self.k5
        elif (x[0] <= 2.5):
            if between(x[1], (0.75, 1.0)):
                value[0] = self.k1
            elif between(x[1], (1.75, 2.0)):
                value[0] = self.k2
            elif between(x[1], (2.75, 3.0)):
                value[0] = self.k3
            elif between(x[1], (3.75, 4.0)):
                value[0] = self.k4
            else:
                value[0] = 0.0
        else:
            if between(x[1], (0.75, 1.0)):
                value[0] = self.k9
            elif between(x[1], (1.75, 2.0)):
                value[0] = self.k8
            elif between(x[1], (2.75, 3.0)):
                value[0] = self.k7
            elif between(x[1], (3.75, 4.0)):
                value[0] = self.k6
            else:
                value[0] = 0.0
    def value_shape(self):
        return ()

class Fin:
    '''
    A class the implements the heat conduction problem for a thermal fin
    '''

    def __init__(self, V, external_obs=False):
        '''
        Initializes a thermal fin instance for a given function space

        Arguments:
            V - dolfin FunctionSpace
        '''

        self.phi = None

        self.V = V
        self.dofs = len(V.dofmap().dofs()) 

        # Currently uses a fixed Biot number
        self.Bi = Constant(0.1)

        # Trial and test functions for the weak forms
        self.w = TrialFunction(V)
        self.v = TestFunction(V)

        self.w_hat = TestFunction(V)
        self.v_trial = TrialFunction(V)

        mesh = V.mesh()
        domains = MeshFunction("size_t", mesh, mesh.topology().dim())
        domains.set_all(0)

        self.fin1 = SubFin([0.75, True])
        self.fin2 = SubFin([1.75, True])
        self.fin3 = SubFin([2.75, True])
        self.fin4 = SubFin([3.75, True])
        self.fin5 = CenterFin()
        self.fin6 = SubFin([3.75, False])
        self.fin7 = SubFin([2.75, False])
        self.fin8 = SubFin([1.75, False])
        self.fin9 = SubFin([0.75, False])
        domains_sub = MeshFunction("size_t", mesh, mesh.topology().dim())
        domains_sub.set_all(0)
        self.fin1.mark(domains_sub, 1)
        self.fin2.mark(domains_sub, 2)
        self.fin3.mark(domains_sub, 3)
        self.fin4.mark(domains_sub, 4)
        self.fin5.mark(domains_sub, 5)
        self.fin6.mark(domains_sub, 6)
        self.fin7.mark(domains_sub, 7)
        self.fin8.mark(domains_sub, 8)
        self.fin9.mark(domains_sub, 9)

        # Marking boundaries for boundary conditions
        bottom = CompiledSubDomain("near(x[1], side) && on_boundary", side = 0.0)
        exterior = CompiledSubDomain("!near(x[1], side) && on_boundary", side = 0.0)
        boundaries = MeshFunction("size_t", mesh, mesh.topology().dim()-1)
        boundaries.set_all(0)
        exterior.mark(boundaries, 1)
        bottom.mark(boundaries, 2)

        self.dx = Measure('dx', domain=mesh, subdomain_data=domains)
        self.ds = Measure('ds', domain=mesh, subdomain_data=boundaries)
        self.dx_s = Measure('dx', domain=mesh, subdomain_data=domains_sub)

        self._k = Function(V)

        self._F = inner(self._k * grad(self.w), grad(self.v)) * self.dx(0) + \
            self.v * self.Bi * self.w * self.ds(1)
        self._a = self.v * self.ds(2)
        self.B = assemble(self._a)[:]
        self.C, self.domain_measure = self.averaging_operator()
        self.A = PETScMatrix()

        self._adj_F = inner(self._k * grad(self.w_hat), grad(self.v_trial)) * self.dx(0) + \
                self.Bi * self.w_hat * self.v_trial * self.ds(1)
        self.A_adj = PETScMatrix()

        self.gamma = Constant(1e-4)

        self.M = assemble(inner(self.w, self.v) * self.dx(0)).array() 
        self.K = assemble(inner(grad(self.w), grad(self.v)) * self.dx(0)).array()

        # Tikhonov Regularization
        #  self.grad_reg = self.gamma * inner(grad(self._k), grad(self.w_hat)) * self.dx(0)
        #  self.reg = 0.5 * self.gamma * inner(grad(self._k), grad(self._k)) * self.dx(0)

        #  Total Variation Regularization
        def TV(u, eps=Constant(1e-7)):
            return sqrt( inner(grad(u), grad(u)) + eps)
            
        self.reg = 0.5 * self.gamma * TV(self._k) * self.dx(0)
        self.grad_reg = self.gamma / TV(self._k) \
                * inner(grad(self._k), grad(self.w_hat))* self.dx(0)

        self.fin1_A = assemble(Constant(1.0) * self.dx_s(1))
        self.fin2_A = assemble(Constant(1.0) * self.dx_s(2))
        self.fin3_A = assemble(Constant(1.0) * self.dx_s(3))
        self.fin4_A = assemble(Constant(1.0) * self.dx_s(4))
        self.fin5_A = assemble(Constant(1.0) * self.dx_s(5))
        self.fin6_A = assemble(Constant(1.0) * self.dx_s(6))
        self.fin7_A = assemble(Constant(1.0) * self.dx_s(7))
        self.fin8_A = assemble(Constant(1.0) * self.dx_s(8))
        self.fin9_A = assemble(Constant(1.0) * self.dx_s(9))

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

        # second-order forward
        self.u_tilde = TestFunction(self.V)
        self.w_h = TestFunction(self.V)
        self.lam_h = TestFunction(self.V)
        self.u_sq = Function(self.V)
        self.z = Function(self.V)
        self.w_sq_f = Function(self.V)
        self.lam_sq = Function(self.V)
        self.lam = Function(self.V)
        self.w_sq = TrialFunction(self.V)
        self.l_sq = TrialFunction(self.V)

        self.incr_F = exp(self._k) * inner(grad(self.w_sq), grad(self.lam_h)) * self.dx(0) \
            + self.lam_h * self.Bi * self.w_sq * self.ds(1)
        self.incr_a = -self.u_sq * exp(self._k) * inner(grad(self.z), grad(self.lam_h)) * self.dx(0) #TODO: Not exp version!

        self.incr_F_adj = exp(self._k) * inner(grad(self.l_sq), grad(self.w_h)) * self.dx(0)\
                + self.Bi * self.l_sq * self.w_h * self.ds(1)
        self.incr_a_adj = -self.u_sq * exp(self._k) * inner(grad(self.lam), grad(self.w_h)) * self.dx(0)

        self.hessian_action_form = self.u_tilde * exp(self._k) * \
                inner(grad(self.z), grad(self.lam_sq)) * self.dx(0) \
                + self.u_tilde * exp(self._k) * inner(grad(self.w_sq_f), grad(self.lam)) \
                * self.dx(0) + self.u_tilde * self.u_sq * exp(self._k) * \
                inner(grad(self.z), grad(self.lam)) * self.dx(0)

        self.Fisher_Inf_action = self.u_tilde * exp(self._k) * \
                inner(grad(self.z), grad(self.lam_sq)) * self.dx(0)

        self.A_incr_adj = PETScMatrix()
        # Randomly sampling state vector for inverse problems
        #  self.n_samples = 3
        #  self.samp_idx = np.random.randint(0, self.dofs, self.n_samples)    

    def forward_five_param(self, k_s):
        return self.forward(self.five_param_to_function(k_s))

    def forward(self, k):
        '''
        Performs a forward solve to obtain temperature distribution
        given the conductivity field m and FunctionSpace V.
        This solve assumes Biot number to be a constant.
        Returns:
         z - Temperature field 
         y - Average temperature (quantity of interest)
         A - Mass matrix
         B - Discretized RHS
         C - Averaging operator
        '''

        z = Function(self.V)

        self._k.assign(k)
        solve(self._F == self._a, z) 
        #  y = assemble(z * self.dx)/self.domain_measure
        #  assemble(self._F, tensor=self.A)

        #  return z, y, self.A.array(), self.B, self.C
        return z, None, None, None, None

    def gradient(self, k, data):
        '''
        Computes the gradient of the cost function with respect to k
        using the adjoint method.
        '''

        if (np.allclose((np.linalg.norm(k.vector()[:])), 0.0)):
            print("parameter vector norm close to zero!")

        z = Function(self.V)

        self._k.assign(k)
        solve(self._F == self._a, z) 
        pred_obs = np.dot(self.B_obs, z.vector()[:])

        adj_RHS = -np.dot((pred_obs - data).T, self.B_obs)
        assemble(self._adj_F, tensor=self.A_adj)
        v_nodal_vals = np.linalg.solve(self.A_adj.array(), adj_RHS)

        v = Function(self.V)
        v.vector().set_local(v_nodal_vals)

        k_hat = TestFunction(self.V)
        grad_w_form = k_hat * inner(grad(z), grad(v)) * self.dx(0)
        grad_vec = assemble(grad_w_form)[:]

        if np.allclose(np.linalg.norm(grad_vec), 0.0):
            print("Gradient norm is zero!")

        return grad_vec

    def sensitivity(self, k):
        '''
        Computes the gradient of the parameter-to-observable w.r.t k
        using the adjoint method
        '''
        z = Function(self.V)

        self._k.assign(k)
        solve(self._F == self._a, z)
        pred_obs = np.dot(self.B_obs, z.vector()[:])

        adj_RHS = -self.B_obs.T
        assemble(self._adj_F, tensor=self.A_adj)
        v_nodal_vals = np.linalg.solve(self.A_adj.array(), adj_RHS)

        #  v = Function(self.V)
        #  v.vector().set_local(v_nodal_vals)

        k_hat = TestFunction(self.V)
        v_hat = TrialFunction(self.V)
        #  grad_w_form = k_hat * inner(grad(z), grad(v)) * self.dx(0)
        grad_w_form = k_hat * inner(grad(z), grad(v_hat)) * self.dx(0)
        #  grad_vec = assemble(grad_w_form)[:]
        grad_vec = np.dot(assemble(grad_w_form).array(), v_nodal_vals)

        return grad_vec.T

    def hessian_action(self, k, u_2, data):
        '''
        Computes the Hessian (w.r.t. objective function) action 
        given a second variation u_2
        and a parameter location k #TODO: fix the exponentiation
        '''

        self._k.assign(k)
        solve(self._F == self._a, self.z) 
        pred_obs = np.dot(self.B_obs, self.z.vector()[:])

        adj_RHS = -np.dot((pred_obs - data).T, self.B_obs)
        assemble(self._adj_F, tensor=self.A_adj)
        lam_nodal_vals = np.linalg.solve(self.A_adj.array(), adj_RHS)
        self.lam.vector().set_local(lam_nodal_vals)

        self.u_sq.assign(u_2)
        solve(self.incr_F == self.incr_a, self.w_sq_f)
        assemble(self.incr_F_adj, tensor=self.A_incr_adj)
        b_incr_adj_RHS = assemble(self.incr_a_adj)
        l_sq_np = np.linalg.solve(self.A_incr_adj.array(), b_incr_adj_RHS[:]  + \
                -np.dot(np.dot(self.B_obs, self.w_sq_f.vector()[:]).T, self.B_obs))
        self.lam_sq.vector().set_local(l_sq_np)

        return assemble(self.hessian_action_form)[:]

    def GN_hessian_action(self, k, u_2, data):
        '''
        Computes the Gauss-Newton Hessian (w.r.t. objective function) action 
        given a second variation u_2
        and a parameter location k
        '''

        self._k.assign(k)
        solve(self._F == self._a, self.z) 
        pred_obs = np.dot(self.B_obs, self.z.vector()[:])

        adj_RHS = -np.dot((pred_obs - data).T, self.B_obs)
        assemble(self._adj_F, tensor=self.A_adj)
        lam_nodal_vals = np.linalg.solve(self.A_adj.array(), adj_RHS)
        self.lam.vector().set_local(lam_nodal_vals)

        self.u_sq.assign(u_2)
        solve(self.incr_F == self.incr_a, self.w_sq_f)
        assemble(self.incr_F_adj, tensor=self.A_incr_adj)
        b_incr_adj_RHS = assemble(self.incr_a_adj)
        l_sq_np = np.linalg.solve(self.A_incr_adj.array(), \
                -np.dot(np.dot(self.B_obs, self.w_sq_f.vector()[:]).T, self.B_obs))
        self.lam_sq.vector().set_local(l_sq_np)

        return assemble(self.Fisher_Inf_action)[:]

    def averaging_operator(self):
        '''
        Returns an operator that when applied to a function in V, gives the average.
        '''
        v = TestFunction(self.V)
        d_omega_f = interpolate(Expression("1.0", degree=2), self.V)
        domain_integral = assemble(v * self.dx)
        domain_measure = assemble(d_omega_f * self.dx)
        C = domain_integral/domain_measure
        C = C[:]
        return C, domain_measure

    def qoi_operator(self, x):
        '''
        Returns the quantities of interest given the state variable
        '''
        return np.dot(self.B_obs, x.vector()[:])
        #  return self.subfin_avg_op(x)

    def reduced_qoi_operator(self, z_r):
        z_nodal_vals = np.dot(self.phi, z_r)
        z_tilde = Function(self.V)
        z_tilde.vector().set_local(z_nodal_vals)
        return self.qoi_operator(z_tilde)

    def reduced_forward(self, A, B, C, psi, phi):
        '''
        Returns the reduced matrices given a reduced trial and test basis
        to solve reduced system: A_r(z)x_r = B_r(z)
        where x = phi x_r + e (small error)

        TODO: Rewrite everything in PETScMatrix

        Arguments:
            A   - LHS forward operator in A(z)x = B(z)
            B   - RHS in A(z)x = B(z)
            psi - Trial basis
            phi - Test basis

        Returns:
            A_r - Reduced LHS
            B_r - Reduced forcing
            C_r - Reduced averaging operator
            x_r - Reduced state variable
            y_r - Reduced QoI
        '''

        self.phi = phi
        A_r = np.dot(psi.T, np.dot(A, phi))
        B_r = np.dot(psi.T, B)
        C_r = np.dot(C, phi)

        #Solve reduced system to obtain x_r
        x_r = np.linalg.solve(A_r, B_r)
        y_r = np.dot(C_r, x_r)

        return A_r, B_r, C_r, x_r, y_r

    def r_fwd_no_full(self, k, phi):
        '''
        Solves the reduced system without solving the full system
        '''
        
        self._k.assign(k)
        assemble(self._F, tensor=self.A)

        A_m = self.A.array()
        psi = np.dot(A_m, phi)
        return self.reduced_forward(A_m, self.B,self.C, psi, phi)

    def subfin_avg_op(self, k):
        # Subfin averages
        fin1_avg = assemble(k * self.dx_s(1))/self.fin1_A 
        fin2_avg = assemble(k * self.dx_s(2))/self.fin2_A 
        fin3_avg = assemble(k * self.dx_s(3))/self.fin3_A 
        fin4_avg = assemble(k * self.dx_s(4))/self.fin4_A 
        fin5_avg = assemble(k * self.dx_s(5))/self.fin5_A
        fin6_avg = assemble(k * self.dx_s(6))/self.fin6_A 
        fin7_avg = assemble(k * self.dx_s(7))/self.fin7_A 
        fin8_avg = assemble(k * self.dx_s(8))/self.fin8_A 
        fin9_avg = assemble(k * self.dx_s(9))/self.fin9_A 
        subfin_avgs = np.array([fin1_avg, fin2_avg, fin3_avg, fin4_avg, fin5_avg, 
            fin6_avg, fin7_avg, fin8_avg, fin9_avg])
        #  print("Subfin averages: {}".format(subfin_avgs))
        return subfin_avgs

    def nine_param_to_function(self, k_s):
        '''
        Same as five_param_to_function but does not assume symmetry.
        '''
        return interpolate(SubfinValExpr(k_s, degree=1), self.V)

    def observation_operator(self):
        z = TestFunction(self.V)
        fin1_avg = assemble(z * self.dx_s(1))/self.fin1_A 
        fin2_avg = assemble(z * self.dx_s(2))/self.fin2_A 
        fin3_avg = assemble(z * self.dx_s(3))/self.fin3_A 
        fin4_avg = assemble(z * self.dx_s(4))/self.fin4_A 
        fin5_avg = assemble(z * self.dx_s(5))/self.fin5_A
        fin6_avg = assemble(z * self.dx_s(6))/self.fin6_A 
        fin7_avg = assemble(z * self.dx_s(7))/self.fin7_A 
        fin8_avg = assemble(z * self.dx_s(8))/self.fin8_A 
        fin9_avg = assemble(z * self.dx_s(9))/self.fin9_A 

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
