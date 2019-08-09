from dolfin import *
import numpy as np
from mshr import Rectangle, generate_mesh

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
        self.k5, self.k1, self.k2, self.k3, self.k4, self.k6, self.k7, self.k8, self.k9 = k_s
        super(SubfinValExpr, self).__init__(**kwargs)
    def eval(self, value, x):
        if (x[0] >= 2.5) and (x[0] <= 3.5):
            value[0] = self.k5
        elif (x[0] < 2.5):
            if (x[1] >= 0.75) and (x[1] <= 1.0):
                value[0] = self.k1
            elif (x[1] >= 1.75) and (x[1] <= 2.0):
                value[0] = self.k2
            elif (x[1] >= 2.75) and (x[1] <= 3.0):
                value[0] = self.k3
            elif (x[1] >= 3.75) and (x[1] <= 4.0):
                value[0] = self.k4
            else:
                value[0] = 0.0
        else:
            if (x[1] >= 0.75) and (x[1] <= 1.0):
                value[0] = self.k9
            elif (x[1] >= 1.75) and (x[1] <= 2.0):
                value[0] = self.k8
            elif (x[1] >= 2.75) and (x[1] <= 3.0):
                value[0] = self.k7
            elif (x[1] >= 3.75) and (x[1] <= 4.0):
                value[0] = self.k6
            else:
                value[0] = 0.0
    def value_shape(self):
        return ()

class Fin:
    '''
    A class the implements the heat conduction problem for a thermal fin
    '''

    def __init__(self, V):
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

        # Marking boundaries for boundary conditions
        bottom = CompiledSubDomain("near(x[1], side) && on_boundary", side = 0.0)
        exterior = CompiledSubDomain("!near(x[1], side) && on_boundary", side = 0.0)
        boundaries = MeshFunction("size_t", mesh, mesh.topology().dim()-1)
        boundaries.set_all(0)
        exterior.mark(boundaries, 1)
        bottom.mark(boundaries, 2)

        self.dx = Measure('dx', domain=mesh, subdomain_data=domains)
        self.ds = Measure('ds', domain=mesh, subdomain_data=boundaries)

        self._k = Function(V)

        self._F = inner(self._k * grad(self.w), grad(self.v)) * self.dx(0) + \
            self.v * self.Bi * self.w * self.ds(1)
        self._a = self.v * self.ds(2)
        self.B = assemble(self._a)[:]
        self.C, self.domain_measure = self.averaging_operator()
        self.A = PETScMatrix()
        self.dz_dk_T = self.full_grad_to_five_param()

        self._adj_F = inner(self._k * grad(self.w_hat), grad(self.v_trial)) * self.dx(0) + \
                self.Bi * self.w_hat * self.v_trial * self.ds(1)
        self.A_adj = PETScMatrix()

        self.middle = Expression("((x[0] >= 2.5) && (x[0] <= 3.5))", degree=1)


        self.fin1 = SubfinExpr([0.75, True], degree=1)
        self.fin2 = SubfinExpr([1.75, True], degree=1)
        self.fin3 = SubfinExpr([2.75, True], degree=1)
        self.fin4 = SubfinExpr([3.75, True], degree=1)
        self.fin6 = SubfinExpr([3.75, False], degree=1)
        self.fin7 = SubfinExpr([2.75, False], degree=1)
        self.fin8 = SubfinExpr([1.75, False], degree=1)
        self.fin9 = SubfinExpr([0.75, False], degree=1)

        #  self.fin6   = Expression("((x[1] >=3.75) && (x[1] <= 4.0)) && (x[0] > 3.5)", degree=2)
        #  self.fin7   = Expression("((x[1] >=2.75) && (x[1] <= 3.0)) && (x[0] > 3.5)", degree=2)
        #  self.fin8   = Expression("((x[1] >=1.75) && (x[1] <= 2.0)) && (x[0] > 3.5)", degree=2)
        #  self.fin9   = Expression("((x[1] >=0.75) && (x[1] <= 1.0)) && (x[0] > 3.5)", degree=2)
        #  self.fin1   = Expression("((x[1] >=0.75) && (x[1] <= 1.0)) && (x[0] < 2.5)", degree=2)
        #  self.fin2   = Expression("((x[1] >=1.75) && (x[1] <= 2.0)) && (x[0] < 2.5)", degree=2)
        #  self.fin3   = Expression("((x[1] >=2.75) && (x[1] <= 3.0)) && (x[0] < 2.5)", degree=2)
        #  self.fin4   = Expression("((x[1] >=3.75) && (x[1] <= 4.0)) && (x[0] < 2.5)", degree=2)

        self.B_obs = self.observation_operator()
        self.ds_dk = self.dsigma_dk()

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
        y = assemble(z * self.dx)/self.domain_measure
        assemble(self._F, tensor=self.A)

        return z, y, self.A.array(), self.B, self.C

    def gradient(self, k, data):

        if (np.allclose((np.linalg.norm(k.vector()[:])), 0.0)):
            return 100 * np.ones(k.vector()[:].shape)

        z = Function(self.V)

        self._k.assign(k)
        solve(self._F == self._a, z) 
        pred_obs = np.dot(self.B_obs, z.vector()[:])

        adj_RHS = -np.dot(self.B_obs.T, pred_obs - data)
        assemble(self._adj_F, tensor=self.A_adj)
        v_nodal_vals = np.linalg.solve(self.A_adj.array(), adj_RHS)

        v = Function(self.V)
        v.vector().set_local(v_nodal_vals)

        k_hat = TestFunction(self.V)
        grad_w_form = k_hat * inner(grad(z), grad(v)) * self.dx(0)
        return assemble(grad_w_form)[:]

    def averaged_reduced_fwd_and_grad(self, k, phi, data):

        self._k.assign(k)
        assemble(self._F, tensor=self.A)
        A_arr = self.A.array()
        A_r, B_r, C_r, w_r, y_r = self.averaged_forward(k, phi)

        reduced_fwd_obs = np.dot(np.dot(self.B_obs, phi), w_r)
        
        reduced_adj_rhs = - np.dot(np.dot(self.B_obs, phi).T, reduced_fwd_obs - data)

        #  v_r = np.linalg.solve(A_r.T, reduced_adj_rhs)

        psi = np.dot(A_arr, phi) 
        v_r = np.linalg.solve(np.dot(psi.T, psi), reduced_adj_rhs)
        #  v_r = np.linalg.solve(A_r.T, reduced_adj_rhs)

        k_hat = TestFunction(self.V)
        phi_w_r = Function(self.V)
        phi_w_r.vector().set_local(np.dot(phi, w_r))
        phi_v_r = Function(self.V)
        phi_v_r.vector().set_local(np.dot(phi, v_r))

        # TODO: Verify this!
        #  reduced_grad_w_form = k_hat * self.ds_dk *  inner(grad(phi_w_r), grad(phi_v_r)) * dx
        reduced_grad_w_form = k_hat *  inner(grad(phi_w_r), grad(phi_v_r)) * dx
        dsdk_v = assemble(reduced_grad_w_form)[:]
        g = np.dot(self.ds_dk.T, dsdk_v)

        return w_r, g

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
        #  average = assemble(z * self.dx)/self.domain_measure

        #  z_vec = z.vector()[:] #TODO: Very inefficient
        #  rand_sample = z_vec[self.samp_idx]

        #TODO: External surface sampling. Most physically realistic
        return self.subfin_avg_op(x)

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

    def averaged_forward(self, m, phi):
        '''
        Given thermal conductivity as a FEniCS function, uses subfin averages
        to reduce the parameter dimension and performs a ROM solve given 
        the reduced basis phi
        '''

        #TODO: INCORRECT! Use the affine decomposition to make this far simpler
        return self.r_fwd_no_full_9_param(self.subfin_avg_op(m), phi)

    def r_fwd_no_full_9_param(self, k_s, phi):
        return self.r_fwd_no_full(self.nine_param_to_function(k_s), phi)

    def r_fwd_no_full_5_param(self, k_s, phi):
        return self.r_fwd_no_full(self.five_param_to_function(k_s), phi)

    def r_fwd_no_full(self, k, phi):
        '''
        Solves the reduced system without solving the full system
        '''
        
        self._k.assign(k)
        #  F, a = self.get_weak_forms(m)
        assemble(self._F, tensor=self.A)

        A_m = self.A.array()
        psi = np.dot(A_m, phi)
        return self.reduced_forward(A_m, self.B,self.C, psi, phi)

    def subfin_avg_op(self, z):
        # Subfin averages
        middle_avg = assemble(self.middle * z * self.dx)
        fin1_avg = assemble(self.fin1 * z * self.dx)/0.25 #0.25 is the area of the subfin
        fin2_avg = assemble(self.fin2 * z * self.dx)/0.25 
        fin3_avg = assemble(self.fin3 * z * self.dx)/0.25 
        fin4_avg = assemble(self.fin4 * z * self.dx)/0.25 
        fin6_avg = assemble(self.fin6 * z * self.dx)/0.25 #0.25 is the area of the subfin
        fin7_avg = assemble(self.fin7 * z * self.dx)/0.25 
        fin8_avg = assemble(self.fin8 * z * self.dx)/0.25 
        fin9_avg = assemble(self.fin9 * z * self.dx)/0.25 

        subfin_avgs = np.array([middle_avg, fin1_avg, fin2_avg, fin3_avg, fin4_avg, 
            fin6_avg, fin7_avg, fin8_avg, fin9_avg])

        return subfin_avgs

    def five_param_to_function(self, k_s):
        '''
        A simpler thermal conductivity problem is the case where each fin has a constant 
        conductivity instead of having a spatially varying conductivity. This function takes
        in conductivity for each fin and returns a FEniCS function on the space. 
        Assumes symmetry of the fin.

        More info on Tan's thesis Pg. 80
        '''

        k1, k2, k3, k4, k5 = k_s
        
        k = interpolate(Expression("k_5 * ((x[0] >= 2.5) && (x[0] <= 3.5)) \
           + k_1 * (((x[1] >=0.75) && (x[1] <= 1.0))  && ((x[0] < 2.5) || (x[0] > 3.5)))  \
           + k_2 * (((x[1] >=1.75) && (x[1] <= 2.0))  && ((x[0] < 2.5) || (x[0] > 3.5)))  \
           + k_3 * (((x[1] >=2.75) && (x[1] <= 3.0))  && ((x[0] < 2.5) || (x[0] > 3.5)))  \
           + k_4 * (((x[1] >=3.75) && (x[1] <= 4.0))  && ((x[0] < 2.5) || (x[0] > 3.5)))",\
                  degree=1, k_1=k1, k_2=k2, k_3=k3, k_4=k4, k_5=k5), self.V)
        return k

    def nine_param_to_function(self, k_s):
        '''
        Same as five_param_to_function but does not assume symmetry.
        '''
        return interpolate(SubfinValExpr(k_s, degree=1), self.V)

    def full_grad_to_five_param(self):
        dz_dk_T = np.zeros((5,self.dofs))

        for i in range(5):
            impulse = np.zeros((5,))
            impulse[i] = 1.0
            k = self.five_param_to_function(impulse)
            dz_dk_T[i,:] = k.vector()[:]

        return dz_dk_T

    def observation_operator(self):
        B = np.loadtxt("B_obs.txt", delimiter=",")

        return B

    def dsigma_dk(self):
        ds_dk = interpolate(Expression("k_5 * ((x[0] >= 2.5) && (x[0] <= 3.5)) \
           + k_1 * (((x[1] >=0.75) && (x[1] <= 1.0))  && ((x[0] < 2.5) || (x[0] > 3.5)))  \
           + k_2 * (((x[1] >=1.75) && (x[1] <= 2.0))  && ((x[0] < 2.5) || (x[0] > 3.5)))  \
           + k_3 * (((x[1] >=2.75) && (x[1] <= 3.0))  && ((x[0] < 2.5) || (x[0] > 3.5)))  \
           + k_4 * (((x[1] >=3.75) && (x[1] <= 4.0))  && ((x[0] < 2.5) || (x[0] > 3.5)))",\
                  degree=1, k_1=4.0, k_2=4.0, k_3=4.0, k_4=4.0, k_5=1.0), self.V)

        B = self.B_obs

        dsdk = np.zeros((B.shape[1], B.shape[1]))

        for i in range(B.shape[0]):
            nnz = np.nonzero(B[i,:])
            if i == 0:
                mult = 1.0
            else:
                mult = 4.0
            for j in range(len(nnz)):
                dsdk[j, nnz] = mult
        return dsdk


def get_space(resolution):
    # Create the thermal fin geometry as referenced in Tan's thesis

    geometry = Rectangle(Point(2.5, 0.0), Point(3.5, 4.0)) \
            + Rectangle(Point(0.0, 0.75), Point(2.5, 1.0)) \
            + Rectangle(Point(0.0, 1.75), Point(2.5, 2.0)) \
            + Rectangle(Point(0.0, 2.75), Point(2.5, 3.0)) \
            + Rectangle(Point(0.0, 3.75), Point(2.5, 4.0)) \
            + Rectangle(Point(3.5, 0.75), Point(6.0, 1.0)) \
            + Rectangle(Point(3.5, 1.75), Point(6.0, 2.0)) \
            + Rectangle(Point(3.5, 2.75), Point(6.0, 3.0)) \
            + Rectangle(Point(3.5, 3.75), Point(6.0, 4.0)) \

    mesh = generate_mesh(geometry, resolution)

    V = FunctionSpace(mesh, 'CG', 1)
    return V
