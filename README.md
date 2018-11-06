# HROM_BIDL
Bayesian Inversion Problems with Hierarchical Reduced Order Models augmented by Deep Learning

Ongoing work on improving reduced order models by better characterizing their error using deep learning methods.

Currently implemented work characterizes the error of a reduced order model solving the heat conduction problem 
in a thermal fin. The problem statement is defined in Prof. Tan Bui's [thesis.](http://users.ices.utexas.edu/~tanbui/PublishedPapers/TanBuiPhDthesis.pdf)

A high fidelity finite element solver is implemented using [FEniCS](https://fenicsproject.org).

A reduced basis is formed using a greedy [model-constrained adaptive sampling method](http://hdl.handle.net/1721.1/40305) suggested by Prof. Bui.

A Tensorflow estimator is used to learn the error between the high fidelity model and the reduced model. 

### Dependencies:
FEniCS 2018.1
hippylib 2.1.0
python 3.6.6
tensorflow 1.11.0
pandas 0.23.4
