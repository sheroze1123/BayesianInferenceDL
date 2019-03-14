import sys
sys.path.insert(0,'/home/fenics/Installations/MUQ_INSTALL/lib')
import pymuqModeling as mm
import numpy as np
from dl_rom import load_parametric_model
from tensorflow.keras.optimizers import Adam, RMSprop, Adadelta
from forward_solve import Fin, get_space

class ROM_forward(mm.PyModPiece):
    """
    Solves the thermal fin steady state problem with 
    projection based ROM with a given basis
    """
    
    def __init__(self, resolution=40):
        """ 
        INPUTS:
     
        """
        mm.PyModPiece.__init__(self, [5],[1])
        V = get_space(resolution)
        dofs = len(V.dofmap().dofs())
        self.solver = Fin(V)
        self.phi = np.loadtxt('data/basis_five_param.txt',delimiter=",")
        self.phi = self.phi[:,0:10]
            
    def EvaluateImpl(self, inputs):
        """
        Performs the forward solve and returns observations.
        
        """
        z = inputs[0]
        A_r, B_r, C_r, x_r, y_r = self.solver.reduced_forward_no_full_5_param(z, self.phi)
        output = np.array([y_r])
        self.outputs = [output]

class DL_ROM_forward(mm.PyModPiece):
    """
    Solves the thermal fin steady state problem with 
    projection based ROM with a given basis and augments
    QoI prediction with deep learning prediciton.
    """
    
    def __init__(self, resolution=40):
        """ 
        INPUTS:
     
        """
        mm.PyModPiece.__init__(self, [5],[1])
        V = get_space(resolution)
        dofs = len(V.dofmap().dofs())
        self.solver = Fin(V)
        self.phi = np.loadtxt('data/basis_five_param.txt',delimiter=",")
        self.phi = self.phi[:,0:10]
        self.model = load_parametric_model('relu', Adam, 0.0001, 6, 100, 10, 400)
        
    def EvaluateImpl(self, inputs):
        """
        Performs the forward solve and returns observations.
        
        """
        z = inputs[0]
        A_r, B_r, C_r, x_r, y_r = self.solver.reduced_forward_no_full_5_param(z, self.phi)
        e_NN = self.model.predict(z.reshape((1,5)))
        output = np.array([y_r + e_NN[0,0]])
        self.outputs = [output]
