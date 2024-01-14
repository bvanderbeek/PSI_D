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

include("/Users/bvanderbeek/research/software/GitRepos/PSI_D/src/psi_coordinate_systems.jl")
include("/Users/bvanderbeek/research/software/GitRepos/PSI_D/src/psi_forward.jl")
include("/Users/bvanderbeek/research/software/GitRepos/PSI_D/src/psi_forward_elastic_tensor.jl")
include("/Users/bvanderbeek/research/software/GitRepos/PSI_D/src/psi_forward_splitting_parameters.jl")
include("/Users/bvanderbeek/research/software/GitRepos/PSI_D/src/psi_inverse.jl")
include("/Users/bvanderbeek/research/software/GitRepos/PSI_D/src/psi_output.jl")
include("/Users/bvanderbeek/research/software/GitRepos/PSI_D/src/psi_buildinputs.jl")
include("/Users/bvanderbeek/research/software/GitRepos/PSI_D/src/utilities.jl")

export build_inputs, psi_forward, psi_inverse, psi_inverse!

end
