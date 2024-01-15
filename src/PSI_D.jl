module PSI_D
println("PSI_D: Plateform for Seismic Imaging - Deterministic")

using TauP
using StaticArrays
using SparseArrays
using IterativeSolvers
using Distributions
using FFTW
using LinearAlgebra
using TOML
using WriteVTK
using Dates

include("psi_coordinate_systems.jl")
include("psi_forward.jl")
include("psi_forward_elastic_tensor.jl")
include("psi_forward_splitting_parameters.jl")
include("psi_inverse.jl")
include("psi_output.jl")
include("psi_buildinputs.jl")
include("utilities.jl")

export build_inputs, psi_forward, psi_inverse, psi_inverse!

end
