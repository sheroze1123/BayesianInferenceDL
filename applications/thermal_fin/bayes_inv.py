import numpy as np
import logging
logging.getLogger('FFC').setLevel(logging.ERROR)
logging.getLogger('UFC').setLevel(logging.ERROR)
import dolfin as dl
dl.set_log_level(40)
from forward_solve import Fin, get_space
from bayesinv_five_param import ROM_forward, DL_ROM_forward

# MUQ Includes
import sys
sys.path.insert(0,'/home/fenics/Installations/MUQ_INSTALL/lib')
import pymuqModeling as mm # Needed for Gaussian distribution
import pymuqApproximation as ma # Needed for Gaussian processes
import pymuqSamplingAlgorithms as ms # Needed for MCMC


r_fwd = ROM_forward(40)
d_fwd = DL_ROM_forward(40)

z_true = np.random.uniform(0.1,1, (1,5))
print ("z_true: {}".format(z_true))

V = get_space(resolution)
full_solver = Fin(V)
w, y, A, B, C = solver.forward_five_param(z_true)
obsData = y

# Define prior
logPriorMu = 0.5*np.ones(5)
logPriorCov = 1.0*np.eye(5)

logPrior = mm.Gaussian(logPriorMu, logPriorCov).AsDensity()

# Likelihood
noiseVar = 1e-4
noiseCov = noiseVar*np.eye(1)
likelihood = mm.Gaussian(obsData, noiseCov).AsDensity()

# Posterior
posteriorPiece = mm.DensityProduct(2)
zPiece = mm.IdentityOperator(5)

# Define graph
graph = mm.WorkGraph()

# Forward model nodes and edges
graph.AddNode(zPiece, "z")
graph.AddNode(r_fwd, "r_fwd")
graph.AddEdge("z", 0, "r_fwd", 0)

# Other nodes and edges
graph.AddNode(likelihood, "Likelihood")
graph.AddNode(logPrior, "Prior")
graph.AddNode(posteriorPiece,"Posterior")

graph.AddEdge("r_fwd", 0, "Likelihood", 0)
graph.AddEdge("z", 0, "Prior", 0)
graph.AddEdge("Prior",0,"Posterior",0)
graph.AddEdge("Likelihood",0, "Posterior",1)

problem = ms.SamplingProblem(graph.CreateModPiece("Posterior"))

proposalOptions = dict()
proposalOptions['Method'] = 'AMProposal'
proposalOptions['ProposalVariance'] = 1e-2
proposalOptions['AdaptSteps'] = 100
proposalOptions['AdaptStart'] = 1000
proposalOptions['AdaptScale'] = 0.1

kernelOptions = dict()
kernelOptions['Method'] = 'MHKernel'
kernelOptions['Proposal'] = 'ProposalBlock'
kernelOptions['ProposalBlock'] = proposalOptions

options = dict()
options['NumSamples'] = 5000
options['ThinIncrement'] = 1
options['BurnIn'] = 100
options['KernelList'] = 'Kernel1'
options['PrintLevel'] = 3
options['Kernel1'] = kernelOptions

mcmc = ms.SingleChainMCMC(options,problem)

startPt = 1.0*np.ones(5)
samps = mcmc.Run(startPt)

ess = samps.ESS()
print('Effective Sample Size = \n', ess)

sampMean = samps.Mean()
print('\nSample mean = \n', sampMean)

sampCov = samps.Covariance()
print('\nSample Covariance = \n', sampCov)

mcErr = np.sqrt( samps.Variance() / ess)
print('\nEstimated MC error in mean = \n', mcErr)
