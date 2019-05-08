from dolfin import *
from mshr import Rectangle, generate_mesh

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
        self._a = self.v * self.dx
        self.B = assemble(self._a)
        self.C, self.domain_measure = self.averaging_operator()
        self.A = PETScMatrix()
        self.dz_dk_T = self.full_grad_to_five_param()

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

        self._k.assign(k)
        #  F, a = self.get_weak_forms(m)

        z = Function(self.V)
        solve(self._F == self._a, z) 

        y = assemble(z * self.dx)/self.domain_measure

        assemble(self._F, tensor=self.A)
        return z, y, self.A, self.B, self.C

    def averaging_operator(self):
        '''
        Returns an operator that when applied to a function in V, gives the average.
        '''
        v = TestFunction(self.V)
        d_omega_f = interpolate(Expression("1.0", degree=2), self.V)
        domain_integral = assemble(v * self.dx)
        domain_measure = assemble(d_omega_f * self.dx)
        C = domain_integral/domain_measure
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
        #  z_nodal_vals = np.dot(self.phi, z_r)
        z_nodal_vals = self.phi.mult(z_r)
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
        #  A_r = np.dot(psi.T, np.dot(A, phi))
        A_r = psi.transposeMatMult(A.matMult(phi))
        #  B_r = np.dot(psi.T, B)
        B_r = psi.transposeMatMult(B)
        #  C_r = np.dot(C, phi)
        C_r = C.matMult(phi)

        #Solve reduced system to obtain x_r
        #  x_r = np.linalg.solve(A_r, B_r)
        x_r = PETScVector()
        A_r.solve(B_r, x_r)
        #  y_r = np.dot(C_r, x_r)
        y_r = PETScVector()
        C_r.mult(x_r, y_r)

        return x_r, y_r, A_r, B_r, C_r 

    def averaged_forward(self, m, phi):
        '''
        Given thermal conductivity as a FEniCS function, uses subfin averages
        to reduce the parameter dimension and performs a ROM solve given 
        the reduced basis phi
        '''
        return self.r_fwd_no_full_5_param(self.subfin_avg_op(m), phi)

    def r_fwd_no_full_5_param(self, k_s, phi):
        return self.r_fwd_no_full(self.five_param_to_function(k_s), phi)

    def r_fwd_no_full(self, k, phi):
        '''
        Solves the reduced system without solving the full system
        '''
        
        self._k.assign(k)
        assemble(self._F, tensor=self.A)

        #  A_m = A.array()
        #  psi = np.dot(A_m, phi)
        psi = PETScMatrix()
        self.A.matMult(phi, psi)
        return self.reduced_forward(A_m, self.B,self.C, psi, phi)

    def subfin_avg_op(self, z):
        # Subfin averages
        middle = Expression("((x[0] >= 2.5) && (x[0] <= 3.5))", degree=2)
        fin1 = Expression("(((x[1] >=0.75) && (x[1] <= 1.0)) && ((x[0] < 2.5) || (x[0] > 3.5)))", degree=2)
        fin2 = Expression("(((x[1] >=1.75) && (x[1] <= 2.0)) && ((x[0] < 2.5) || (x[0] > 3.5)))", degree=2)
        fin3 = Expression("(((x[1] >=2.75) && (x[1] <= 3.0)) && ((x[0] < 2.5) || (x[0] > 3.5)))", degree=2)
        fin4 = Expression("(((x[1] >=3.75) && (x[1] <= 4.0))  && ((x[0] < 2.5) || (x[0] > 3.5)))", degree=2)

        middle_avg = assemble(middle * z * self.dx)
        fin1_avg = assemble(fin1 * z * self.dx)/0.25 #0.25 is the area of the subfin
        fin2_avg = assemble(fin2 * z * self.dx)/0.25 
        fin3_avg = assemble(fin3 * z * self.dx)/0.25 
        fin4_avg = assemble(fin4 * z * self.dx)/0.25 

        #  subfin_avgs = np.array([middle_avg, fin1_avg, fin2_avg, fin3_avg, fin4_avg])
        # Better way to create all of this at once with assemble?
        subfin_avgs = PETScVector()
        subfin_avgs.init(5)
        subfin_avgs.set_local([middle_avg, fin1_avg, fin2_avg, fin3_avg, fin4_avg])

        return subfin_avgs

    def five_param_to_function(self, k_s):
        '''
        A simpler thermal conductivity problem is the case where each fin has a constant 
        conductivity instead of having a spatially varying conductivity. This function takes
        in conductivity for each fin and returns a FEniCS function on the space

        More info on Tan's thesis Pg. 80
        '''

        k1, k2, k3, k4, k5 = k_s
        
        k = interpolate(Expression("k_5 * ((x[0] >= 2.5) && (x[0] <= 3.5)) \
           + k_1 * (((x[1] >=0.75) && (x[1] <= 1.0))  && ((x[0] < 2.5) || (x[0] > 3.5)))  \
           + k_2 * (((x[1] >=1.75) && (x[1] <= 2.0))  && ((x[0] < 2.5) || (x[0] > 3.5)))  \
           + k_3 * (((x[1] >=2.75) && (x[1] <= 3.0))  && ((x[0] < 2.5) || (x[0] > 3.5)))  \
           + k_4 * (((x[1] >=3.75) && (x[1] <= 4.0))  && ((x[0] < 2.5) || (x[0] > 3.5)))",\
                  degree=2, k_1=k1, k_2=k2, k_3=k3, k_4=k4, k_5=k5), self.V)
        return k

    def full_grad_to_five_param(self):
        dz_dk_T = PETScMatrix()
        #  dz_dk_T = np.zeros((5,self.dofs))

        #  for i in range(5):
            #  impulse = np.zeros((5,))
            #  impulse[i] = 1.0
            #  k = self.five_param_to_function(impulse)
            #  dz_dk_T[i,:] = k.vector()[:]

        return dz_dk_T

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