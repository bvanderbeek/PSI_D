using TauP
using StaticArrays
using SparseArrays
using IterativeSolvers
using Distributions
using TOML
using WriteVTK
using Plots

include("/Users/bvanderbeek/research/software/GitRepos/PSI_D/src/psi_coordinate_systems.jl"); # Coordinate system functions
include("/Users/bvanderbeek/research/software/GitRepos/PSI_D/src/psi_forward.jl"); # Forward problem routines
include("/Users/bvanderbeek/research/software/GitRepos/PSI_D/src/psi_inverse.jl"); # Inversion routines
include("/Users/bvanderbeek/research/software/GitRepos/PSI_D/src/psi_output.jl"); # Function for writing results
include("/Users/bvanderbeek/research/software/GitRepos/PSI_D/src/psi_buildinputs.jl"); # Inversion routines
include("/Users/bvanderbeek/research/software/GitRepos/PSI_D/src/utilities.jl"); # Random utilities

# The parameter file
the_parameters = "/Users/bvanderbeek/research/software/GitRepos/PSI_D/examples/VF2021_Subduction/psi_parameter_file_AniInvP.toml"

# Build Input Parameters
PsiParameters, Observations, ForwardModel, PerturbationModel, Solver = build_inputs(the_parameters);

# Call the forward problem
predictions, relative_residuals, Kernels = psi_forward(Observations, ForwardModel);

# Call the inverse problem
psi_inverse!(Observations, ForwardModel, PerturbationModel, Solver);
# @time psi_inverse!(Model, InvParam, Obs, Solv);

# Write model to a VTK file
write_model(ForwardModel, "PsiModel");