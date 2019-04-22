import numpy as np
import logging
logging.getLogger('FFC').setLevel(logging.ERROR)
logging.getLogger('UFC').setLevel(logging.ERROR)
import dolfin as dl
dl.set_log_level(40)
from forward_solve import Fin, get_space
from muq_mod_five_param import ROM_forward, DL_ROM_forward, FOM_forward
import matplotlib.pyplot as plt

# MUQ Includes
import sys
sys.path.insert(0,'/home/fenics/Installations/MUQ_INSTALL/lib')
import pymuqModeling as mm # Needed for Gaussian distribution
import pymuqApproximation as ma # Needed for Gaussian processes
import pymuqSamplingAlgorithms as ms # Needed for MCMC

resolution = 40
r_fwd = ROM_forward(resolution, out_type="subfin_avg")
d_fwd = DL_ROM_forward(resolution, out_type="subfin_avg")
f_fwd = FOM_forward(resolution, out_type="subfin_avg")

#z_true = np.random.uniform(0.1,1, (1,5))
z_true = np.array([[0.41126864, 0.61789679, 0.75873243, 0.96527541, 0.22348076]])

V = get_space(resolution)
full_solver = Fin(V)
w, y, A, B, C = full_solver.forward_five_param(z_true[0,:])
qoi = full_solver.qoi_operator(w)
obsData = qoi

def MCMC_sample(fwd):
    # Define prior
    logPriorMu = 0.5*np.ones(5)
    logPriorCov = 0.5*np.eye(5)

    logPrior = mm.Gaussian(logPriorMu, logPriorCov).AsDensity()

    # Likelihood
    noiseVar = 1e-4
    noiseCov = noiseVar*np.eye(obsData.size)
    likelihood = mm.Gaussian(obsData, noiseCov).AsDensity()

    # Posterior
    posteriorPiece = mm.DensityProduct(2)
    zPiece = mm.IdentityOperator(5)

    # Define graph
    graph = mm.WorkGraph()

    # Forward model nodes and edges
    graph.AddNode(zPiece, "z")
    graph.AddNode(fwd, "fwd")
    graph.AddEdge("z", 0, "fwd", 0)

    # Other nodes and edges
    graph.AddNode(likelihood, "Likelihood")
    graph.AddNode(logPrior, "Prior")
    graph.AddNode(posteriorPiece,"Posterior")

    graph.AddEdge("fwd", 0, "Likelihood", 0)
    graph.AddEdge("z", 0, "Prior", 0)
    graph.AddEdge("Prior",0,"Posterior",0)
    graph.AddEdge("Likelihood",0, "Posterior",1)

    problem = ms.SamplingProblem(graph.CreateModPiece("Posterior"))

    proposalOptions = dict()
    proposalOptions['Method'] = 'AMProposal'
    proposalOptions['ProposalVariance'] = 1e-4
    proposalOptions['AdaptSteps'] = 1000
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

    startPt = 0.5*np.ones(5)
    samps = mcmc.Run(startPt)


    sampMean = samps.Mean()
    print ("z_mean: {}".format(sampMean))
    print ("z_true: {}".format(z_true))

    sampCov = samps.Covariance()
    print('\nSample Covariance = \n', sampCov)

    ess = samps.ESS()
    print('Effective Sample Size = \n', ess)

    mcErr = np.sqrt( samps.Variance() / ess)
    print('\nEstimated MC error in mean = \n', mcErr)

MCMC_sample(f_fwd)
#MCMC_sample(r_fwd)
#MCMC_sample(d_fwd)
