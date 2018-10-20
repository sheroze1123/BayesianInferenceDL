from dolfin import *
import numpy as np

class Fin:
    def __init__(self, V):
        self.V = V
        self.Bi = Constant(0.1)

        mesh = V.mesh()
        self.w = TrialFunction(V)
        self.v = TestFunction(V)

        domains = MeshFunction("size_t", mesh, mesh.topology().dim())
        domains.set_all(0)

        # boundary conditions
        bottom = CompiledSubDomain("near(x[1], side) && on_boundary", side = 0.0)
        exterior = CompiledSubDomain("!near(x[1], side) && on_boundary", side = 0.0)
        boundaries = MeshFunction("size_t", mesh, mesh.topology().dim()-1)
        boundaries.set_all(0)
        exterior.mark(boundaries, 1)
        bottom.mark(boundaries, 2)

        self.dx = Measure('dx', domain=mesh, subdomain_data=domains)
        self.ds = Measure('ds', domain=mesh, subdomain_data=boundaries)
        self.a = self.v * self.dx
        self.B = assemble(self.a)[:]
        self.C, self.domain_measure = self.averaging_operator()

    def averaging_operator(self):
        # Kinda hacky way to get C as the averaging operator over the domain. Probably a better way
        v = TestFunction(self.V)
        d_omega_f = interpolate(Expression("1.0", degree=2), self.V)
        domain_integral = assemble(v * self.dx)
        domain_measure = assemble(d_omega_f * self.dx)
        C = domain_integral/domain_measure
        C = C[:]
        return C, domain_measure

    def get_weak_forms(self, m):
        # Setting up the variational form
        F = inner(m * grad(self.w), grad(self.v)) * self.dx(0) + self.v * self.Bi * self.w * self.ds(1)
        return F, self.a

    def forward(self, m):
        '''
        Performs a forward solve to obtain temperature distribution
        given the conductivity field m and FunctionSpace V.
        This solve assumes Biot number to be a constant.
        Returns:
         w - Temperature field 
         y - Average temperature
         A - Mass matrix
         B - Discretized RHS
         C - Averaging operator
         dA_dz - Partial derivative of the mass matrix w.r.t. the parameters
        '''

        F, a = self.get_weak_forms(m)
        A = assemble(F).array()

        z = Function(self.V)
        solve(F == self.a, z) 

        # TODO: Compute quantity of interest
        #  w_nodal_values = np.array(w.vector()[:]) 
        #  y = np.mean(w_nodal_values)
        y = assemble(z * self.dx)/self.domain_measure
        #  print("Average temperature: {}".format(y))

        return z, y, A, self.B, self.C

    def reduced_forward(self, A, B, C, psi, phi):
        '''
        Returns the reduced matrices given a reduced trial and test basis
        to solve reduced system of A(z)x = B(z)
        Arguments:
        A   - LHS forward operator in A(z)x = B(z)
        B   - RHS in A(z)x = B(z)
        psi - Trial basis
        phi - Test basis
        '''
        A_r = np.dot(psi.T, np.dot(A, phi))
        B_r = np.dot(psi.T, B)
        C_r = np.dot(C, phi)

        #Solve reduced system to obtain x_r
        x_r = np.linalg.solve(A_r, B_r)
        y_r = np.dot(C_r, x_r)

        return A_r, B_r, C_r, x_r, y_r

    def reduced_forward_no_full(self, m, psi, phi):
        F, a = self.get_weak_forms(m)
        A = assemble(F).array()

        return reduced_forward(A, self.B,self.C, psi, phi)
