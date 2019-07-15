from dolfin import *
from mshr import Rectangle, generate_mesh
from petsc4py import PETSc
set_log_level(40)

class Fin:
    '''
    A class the implements the heat conduction problem for a thermal fin
    '''

    def __init__(self, V, phi):
        '''
        Initializes a thermal fin instance for a given function space

        Arguments:
            V - dolfin FunctionSpace
        '''

        self.phi = phi
        (self.n, self.n_r) = phi.getSize()

        # Set boundary conditions for domain 
        self.V = V
        self.dofs = len(V.dofmap().dofs()) 

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

        # Currently uses a fixed Biot number
        self.Bi = Constant(0.1)

        # Trial and test functions for the weak forms
        self.w = TrialFunction(V)
        self.v = TestFunction(V)

        # Thermal conductivity parameter
        self._k = Function(V)

        # Reduced variables
        self._w_r = PETSc.Vec().createSeq(self.n_r)
        self._A_r = PETSc.Mat()
        self._B_r = PETSc.Vec().createSeq(self.n_r)
        self._C_r = PETSc.Vec().createSeq(self.n_r)

        self.ksp = PETSc.KSP().create()

        # Output temperature field
        self._w = Function(self.V)

        # Temperature field corrected prediction
        self._w_tilde =  Function(self.V)

        # Setup weak forms
        self._F = inner(self._k * grad(self.w), grad(self.v)) * self.dx(0) + \
            self.v * self.Bi * self.w * self.ds(1)
        self._a = self.v * self.dx

        # Setup matrices for solves
        self.A = PETScMatrix()
        self.B = as_backend_type(assemble(self._a))
        self.C, self.domain_measure = self.averaging_operator()
        self.dz_dk_T = self.full_grad_to_five_param()

        # Setup QoI calculation
        middle = Expression("((x[0] >= 2.5) && (x[0] <= 3.5))", degree=2)
        fin1 = Expression("(((x[1] >=0.75) && (x[1] <= 1.0)) && ((x[0] < 2.5) || (x[0] > 3.5)))", degree=2)
        fin2 = Expression("(((x[1] >=1.75) && (x[1] <= 2.0)) && ((x[0] < 2.5) || (x[0] > 3.5)))", degree=2)
        fin3 = Expression("(((x[1] >=2.75) && (x[1] <= 3.0)) && ((x[0] < 2.5) || (x[0] > 3.5)))", degree=2)
        fin4 = Expression("(((x[1] >=3.75) && (x[1] <= 4.0))  && ((x[0] < 2.5) || (x[0] > 3.5)))", degree=2)

        self.middle_avg = middle * self._w * self.dx
        self.fin1_avg = 4.0 * fin1 * self._w * self.dx # 0.25 is the area of the subfin
        self.fin2_avg = 4.0 * fin2 * self._w * self.dx
        self.fin3_avg = 4.0 * fin3 * self._w * self.dx
        self.fin4_avg = 4.0 * fin4 * self._w * self.dx

        # Setup randomly sampling state vector for inverse problems
        #  self.n_samples = 3
        #  self.samp_idx = np.random.randint(0, self.dofs, self.n_samples)    

    def forward_five_param(self, k_s):
        '''
        Given a numpy array of conductivity values, (a real value per subfin)
        performs a forward solves and returns temperature of fin.
        '''
        return self.forward(self.five_param_to_function(k_s))

    def forward(self, k):
        '''
        Performs a forward solve to obtain temperature distribution
        given the conductivity field m and FunctionSpace V.
        This solve assumes Biot number to be a constant.

        Arguments:
         k - Thermal conductivity as a numpy array of nodal values

        Returns:
         z - Temperature field 
         y - Average temperature (quantity of interest)
         A - Mass matrix
         B - Discretized RHS
         C - Averaging operator
        '''

        self._k.assign(k)
        solve(self._F == self._a, self._w) 

        # Compute QoI (average temperature)
        y = self._w.vector().inner(self.C)

        # Assemble LHS for ROM
        assemble(self._F, tensor=self.A)

        return self._w, y, self.A, self.B, self.C

    def averaging_operator(self):
        '''
        Returns an operator that when applied to a function in V, gives the average.
        '''
        v = TestFunction(self.V)
        d_omega_f = interpolate(Expression("1.0", degree=2), self.V)
        domain_measure = assemble(d_omega_f * self.dx)
        C = as_backend_type(assemble(v/domain_measure * self.dx))
        return C, domain_measure

    def qoi_operator(self, w):
        '''
        Returns the quantities of interest given the state variable
        '''
        #  qoi = assemble(z * self.dx)/self.domain_measure

        # Random sample QoI
        #  z_vec = z.vector()[:] #TODO: Very inefficient
        #  qoi = z_vec[self.samp_idx]

        #TODO: External surface sampling. Most physically realistic
        
        # Subfin temperature average (default QoI)
        qoi = self.subfin_avg_op(w)

        return qoi

    def reduced_qoi_operator(self, w_r):
        '''
        Arguments:
         w_r : PETSc.Vec - Reduced state vector
        '''
        #  z_nodal_vals = np.dot(self.phi, z_r)
        w_tilde_vec = as_backend_type(self._w_tilde.vector()).vec()
        self.phi.mult(w_r, w_tilde_vec)
        return self.qoi_operator(self._w_tilde)

    def reduced_forward(self, A, B, C):
        '''
        Returns the reduced matrices given a reduced trial and test basis
        to solve reduced system: A_r(z)x_r = B_r(z)
        where x = phi x_r + e (small error)

        TODO: Rewrite everything in PETScMatrix

        Arguments:
            A : PETScMatrix - LHS forward operator in A(z)x = B(z)
            B : PETScVector - RHS in A(z)x = B(z)
            psi : PETSc.Mat - Trial basis
            phi : PETSc.Mat - Test basis

        Returns:
            A_r - Reduced LHS
            B_r - Reduced forcing
            C_r - Reduced averaging operator
            x_r - Reduced state variable
            y_r - Reduced QoI
        '''

        psi = PETSc.Mat()
        self.A.mat().matMult(self.phi, psi)
        #  A_r = np.dot(psi.T, np.dot(A, phi))
        
        Aphi = PETSc.Mat()
        A.mat().matMult(self.phi, Aphi)
        A_r = PETSc.Mat()
        #  import pdb; pdb.set_trace()
        psi.transposeMatMult(Aphi, A_r)
        #  psi.transposeMatMult(Aphi, A_r)
        #  B_r = np.dot(psi.T, B)
        psi.multTranspose(B.vec(), self._B_r)
        #  C_r = np.dot(C, phi)
        self.phi.multTranspose(C.vec(), self._C_r)

        self.ksp.setOperators(A_r)
        self.ksp.solve(self._B_r, self._w_r)
        y_r = self._C_r.dot(self._w_r)

        return self._w_r, y_r, self._A_r, self._B_r, self._C_r 

    def averaged_forward(self, m):
        '''
        Given thermal conductivity as a FEniCS function, uses subfin averages
        to reduce the parameter dimension and performs a ROM solve given 
        the reduced basis phi
        '''
        return self.r_fwd_no_full_5_param(self.subfin_avg_op(m))

    def r_fwd_no_full_5_param(self, k_s):
        return self.r_fwd_no_full(self.five_param_to_function(k_s))

    def r_fwd_no_full(self, k):
        '''
        Solves the reduced system without solving the full system
        '''
        
        self._k.assign(k)
        assemble(self._F, tensor=self.A)
        return self.reduced_forward(self.A, self.B,self.C)

    def subfin_avg_op(self, w):
        # Subfin averages
        self._w.assign(w)

        # Better way to create all of this at once with assemble?
        subfin_avgs = PETScVector()
        subfin_avgs.init(5)
        subfin_avgs.set_local([assemble(self.middle_avg), 
            assemble(self.fin1_avg), 
            assemble(self.fin2_avg), 
            assemble(self.fin3_avg), 
            assemble(self.fin4_avg)])

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
