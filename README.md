# HROM_BIDL
**H**ierarchical **R**educed **O**rder **M**odels in **B**ayesian **I**nversion augmented by **D**eep **L**earning

Ongoing work on improving reduced order models by better characterizing their error using deep learning methods.

Currently implemented work characterizes the error of a reduced order model solving the heat conduction problem 
in a thermal fin. The problem statement is defined in Prof. Tan Bui's [thesis.](http://users.ices.utexas.edu/~tanbui/PublishedPapers/TanBuiPhDthesis.pdf)

A high fidelity finite element solver is implemented using [FEniCS](https://fenicsproject.org).

A reduced basis is formed using a greedy [model-constrained adaptive sampling method](http://hdl.handle.net/1721.1/40305) suggested by Prof. Bui.

Tensorflow estimators and Keras models are used to learn the error between the high fidelity model and the reduced model. 

### Dependencies:
* FEniCS 2018.1
* python 3.7.3
* tensorflow 1.13
* pandas 0.24.2
* pymc3 3.6
* scikit-learn 0.20
* scikit-optimize 0.5
