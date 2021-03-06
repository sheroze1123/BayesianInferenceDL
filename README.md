# Bayesian Inference with Deep Learning and Reduced Order Models

Ongoing work on improving reduced order models by better characterizing their error using deep learning methods.

Currently implemented work characterizes the error of a reduced order model solving the heat conduction problem 
in a thermal fin. The problem statement is defined in Prof. Tan Bui's [thesis.](http://users.ices.utexas.edu/~tanbui/PublishedPapers/TanBuiPhDthesis.pdf)

A high fidelity finite element solver is implemented using [FEniCS](https://fenicsproject.org).

A reduced basis is formed using a greedy [model-constrained adaptive sampling method](http://hdl.handle.net/1721.1/40305) suggested by Prof. Bui.

Tensorflow estimators and Keras models are used to learn the error between the high fidelity model and the reduced model. 

PyMC is used to perform Bayesian Inference with Hamiltonian Dynamics

### Dependencies:
* FEniCS 2018.1
* python 3.7.3
* tensorflow 1.13
* pandas 0.24.2
* pymc3 3.6
* scikit-learn 0.20
* scikit-optimize 0.5

## Introduction
- This directory contains routines to solve the Bayesian inverse problem to 
predict thermal conductivity in a thermal fin.
- The forward problem (solved with finite element methods in FEniCS) 
solves for the temperature distribution in a thermal fin given conductivity.
- The reduced order model attempts to simplify the complexity of the forward 
problem by projecting state space to a reduced basis. An adaptive 
model-constrained sampling method is used to discover a reduced basis. 
More information about this algorithm can be found in Prof. Tan Bui's Ph.D. 
[thesis.](http://users.ices.utexas.edu/~tanbui/PublishedPapers/TanBuiPhDthesis.pdf)
- Although this reduced order model improves on computational complexity, 
it introduces errors compared to the high fidelity forward solve done using 
finite element methods. 
- The goal of this research project is to capture this nonlinear error introduced 
by the reduced order models using deep learning models.
- The `five_param` versions of the code corresponds to the parametrization of 
the heat conductivity of the thermal fin by assigning a single real value per 
subfin. 

## Code Layout
- The `fom/forward_solve.py` file provides functions to perform high fidelity forward
solves using FEniCS and reduced order forward solves given a precomputed reduced 
basis.
- The `rom/generate_reduced_basis{_five_param}.py` file performs adaptive 
model-constrained sampling to create a reduced basis. Saves a 
`basis_{five_param}.txt` in the `data` folder to be used by the forward solver.
- `rom/model_constr_adaptive_sampling.py` performs adaptive sampling to construct
the reduced basis.
- `rom/error_optimziation.py` provides routines to solve the optimization problem 
in the adaptive sampling method.
- The `rom/rom_error_predict{_five_param}.py` file trains a neural network to fit 
the error between the high fidelity model and the reduced order model given 
training examples using Tensorflow Estimators (deprecated. Refer to 
and `deep_learing/hyper_param_opt.py`.
- The `models` folder contains a growing collection of deep neural network models 
created using Tensorflow Estimators.
- `deep_learing/generate_fin_dataset.py` creates Tensorflow-friendly datasets by solving the 
forward problem with random thermal conductivity parameters.
- `deep_learning/dl_model.py` provides a parametric initialization of the neural network
used to create a discrepancy function between the FOM and the ROM.
- `deep_learning/hyper_param_opt.py` provides routines to perform Bayesian optimization to pick 
the appropriate hyperparameters given a metric to assess the accuracy of the 
neural network.
- `deep_learning/bayes_inv.py` uses the forward solvers with the reduced order model and with 
the neural network correction to perform the Bayesian inference of the thermal 
conductivities. NOTE: Currently uses [MUQ](http://muq.mit.edu). Deprecated in
favor of the PyMC implementation. `muq_mod_five_param.py` provides helper 
routines to create MUQ mod pieces.
- `deep_learning/pymc_bayes_inverse.py` performs Bayesian inference using the reduced-order
model and the deep learning discrepancy function to predict thermal conductivity.
