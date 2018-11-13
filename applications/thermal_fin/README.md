## Introduction
This directory contains routines to solve the heat conduction problem in a thermal fin. 
The forward problem (solved with finite element methods in FEniCS) solves for the temperature
distribution in a thermal fin given conductivity.
The reduced order model attempts to simplify the complexity of the forward problem by projecting
state space to a reduced basis. An adaptive model-constrained sampling method is used to discover
a reduced basis.
Although this reduced order model improves on computational complexity, it introduces errors compared to 
the high fidelity forward solve done using finite element methods. 
The goal of this research project is to capture this nonlinear error introduced by the reduced order models
using deep learning models.


## Code Layout
The `forward_solve.py` file provides functions to perform high fidelity forward solves and reduced order 
forward solves.
The `generate_reduced_basis.py` file performs adaptive model-constrained sampling to create a reduced basis.
The `rom_error_predict.py` file trains a neural network to fit the error between the high fidelity model
and the reduced order model given training examples.
The `five_param` variants of the above files assume constant thermal conductivities in each of the subfins,
thus simplifying the problem. These variants act as fast prototypes for the harder research problem.
The `models` folder contains a growing collection of deep neural network models created using Tensorflow Estimators.

