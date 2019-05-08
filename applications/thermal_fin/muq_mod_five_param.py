import sys
sys.path.insert(0,'/home/fenics/Installations/MUQ_INSTALL/lib')
import pymuqModeling as mm
import numpy as np
from dl_model import load_parametric_model
from tensorflow.keras.optimizers import Adam, RMSprop, Adadelta
from forward_solve import Fin, get_space

class FOM_forward(mm.PyModPiece):
    """
    Solves the thermal fin steady state problem with
    a full order model
    """
    def __init__(self, resolution=40, out_type="total_avg"):
        """ 
        INPUTS:
     
        """
        V = get_space(resolution)
        dofs = len(V.dofmap().dofs())
        self.solver = Fin(V)
        self.out_type=out_type

        if out_type == "total_avg":
            out_dim = 1
        elif out_type == "subfin_avg":
            out_dim = 5
        elif out_type == "rand_pt":
            out_dim = 1
        elif out_type == "rand_pts":
            out_dim = 5
        mm.PyModPiece.__init__(self, [5],[out_dim])

    def EvaluateImpl(self, inputs):
        """
        Performs the forward solve and returns observations.
        
        """
        z = inputs[0]

        x, y, A, B, C = self.solver.forward_five_param(z)
        output = self.solver.qoi_operator(x)
        self.outputs = [output]

class ROM_forward(mm.PyModPiece):
    """
    Solves the thermal fin steady state problem with 
    projection based ROM with a given basis
    """
    
    def __init__(self, resolution=40, out_type="total_avg"):
        """ 
        INPUTS:
     
        """
        V = get_space(resolution)
        dofs = len(V.dofmap().dofs())
        self.solver = Fin(V)
        self.phi = np.loadtxt('data/basis_five_param.txt',delimiter=",")
        self.phi = self.phi[:,0:10]
        self.out_type=out_type

        if out_type == "total_avg":
            out_dim = 1
        elif out_type == "subfin_avg":
            out_dim = 5
        elif out_type == "rand_pt":
            out_dim = 1
        elif out_type == "rand_pts":
            out_dim = 5
        mm.PyModPiece.__init__(self, [5],[out_dim])
            
    def EvaluateImpl(self, inputs):
        """
        Performs the forward solve and returns observations.
        
        """
        z = inputs[0]

        A_r, B_r, C_r, x_r, y_r = self.solver.r_fwd_no_full_5_param(z, self.phi)
        if self.out_type == "total_avg":
            output = np.array([y_r])
        else:
            # The QoI operator determines whether we look at subfin averages
            # or random points on the boundary or domain
            output = self.solver.reduced_qoi_operator(x_r)
        
        self.outputs = [output]

class DL_ROM_forward(mm.PyModPiece):
    """
    Solves the thermal fin steady state problem with 
    projection based ROM with a given basis and augments
    QoI prediction with deep learning prediciton.
    """
    
    def __init__(self, resolution=40, out_type="total_avg"):
        """ 
        INPUTS:
     
        """
        V = get_space(resolution)
        dofs = len(V.dofmap().dofs())
        self.solver = Fin(V)
        self.phi = np.loadtxt('data/basis_five_param.txt',delimiter=",")
        self.phi = self.phi[:,0:10]
        self.model = load_parametric_model('relu', Adam, 0.004, 6, 50, 150, 600)

        self.out_type=out_type

        if out_type == "total_avg":
            out_dim = 1
        elif out_type == "subfin_avg":
            out_dim = 5
        elif out_type == "rand_pt":
            out_dim = 1
        elif out_type == "rand_pts":
            out_dim = 5
            
        mm.PyModPiece.__init__(self, [5],[out_dim])
        
    def EvaluateImpl(self, inputs):
        """
        Performs the forward solve and returns observations.
        
        """
        z = inputs[0]
        A_r, B_r, C_r, x_r, y_r = self.solver.r_fwd_no_full_5_param(z, self.phi)
        e_NN = self.model.predict(z.reshape((1,5)))

        if self.out_type == "total_avg":
            output = np.array([y_r + e_NN[0,0]])
        else:
            # The QoI operator determines whether we look at subfin averages
            # or random points on the boundary or domain
            output = self.solver.reduced_qoi_operator(x_r) + e_NN[0]
                
        self.outputs = [output]
