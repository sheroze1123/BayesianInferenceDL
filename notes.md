# Bayesian Inference

- Extend PyMC3 to have DRAM
- Compute gradient of ROM + error model to use NUTS, HMC
- Gaussian field to model conductivity. Do log conductivity as it cond. is positive.
- Gaussian field to generate random conductivity fields
- Add noisy measurements

# Forward Problem 

Implement model contrained adaptive sampling
    - Reimplement in PETScVector format
    - PETSc Optimization tools
    - Save intermediate computations of state -> perform POD to improve basis

Improve performance
    - Use PETScMatrix only!

# Reduced Order Modeling

Learning ROM error
    - Architecture for regresion
    - Precompute dataset and use shuffle -COMPLETE-
    - CNNs on finite element nodes

No affine decomposition case:
    If the parameter is a field, average per subfin. Gives you an affine decomposition.
    Get snapshots of each parameters, then take average k

# Machine Learning

Universal Approximator Theorem
    - Verify 5-parameter UAT (R^5 -> R continuous operator on a compact set. So increasing width should help)

