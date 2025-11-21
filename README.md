# **Finite-size Quantum Key Distribution Rates from Rényi Entropies Using Conic Optimization**

This repository contains the code used to reproduce the results presented in *Finite-size Quantum Key Distribution Rates from Rényi Entropies Using Conic Optimization* ([arXiv:2511.10584 ](https://arxiv.org/abs/2511.10584)).

The files starting with `Instance` include the main functions to compute the key rates for their respective protocols, while `Utils_data` provides the datasets required to reproduce the figures and numerical results for the parameter settings specified in the manuscript.

All code is written in [Julia](https://docs.julialang.org/en/v1/manual/getting-started/). To install the required packages, enter the Julia package manager by typing `]` and then run:

```julia
pkg> add PackageName
```

In particular, to implement the cones introduced in our work, you need to install ConicQKD.jl by typing

```julia
pkg> add https://github.com/araujoms/ConicQKD.jl
```
For more details on how to use this package, visit the repository [araujoms/ConicQKD.jl](https://github.com/araujoms/ConicQKD.jl?tab=readme-ov-file).
