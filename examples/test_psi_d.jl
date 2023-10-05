using TauP
using StaticArrays
using SparseArrays
using IterativeSolvers
using Distributions
using TOML
using Plots

include("/Users/bvanderbeek/research/software/JuliaProjects/PSI/utilities.jl"); # Random utilities
include("/Users/bvanderbeek/research/software/JuliaProjects/PSI/CoordinateSystems.jl"); # Coordinate system functions
include("/Users/bvanderbeek/research/software/JuliaProjects/PSI/psi_structures.jl"); # Main structures
include("/Users/bvanderbeek/research/software/JuliaProjects/PSI/psi_buildinputs.jl"); # Inversion routines
include("/Users/bvanderbeek/research/software/JuliaProjects/PSI/psi_forward.jl"); # Forward problem routines
include("/Users/bvanderbeek/research/software/JuliaProjects/PSI/psi_inverse.jl"); # Inversion routines
include("/Users/bvanderbeek/research/software/JuliaProjects/PSI/psi_interpolations.jl"); # Interpolation routines

# Build Input Parameters
Obs, Model, InvParam, Solv, Dp = build_inputs("/Users/bvanderbeek/research/CASCADIA/Joint/PSI_Inputs/deterministic/psi_parameters.toml");

# Calling just the forward problem
K, r, q = psi_forward(Obs, Model);

# Compute Demeaned Residuals
Δt, _ = mean_source_delay(r, Obs, Model.Sources)
println(1000*std(Δt))

# Calling the inversion
psi_inverse!(Model, InvParam, Obs, Solv);
# @time psi_inverse!(Model, InvParam, Obs, Solv);

# Plot Solution
if ~isnothing(InvParam.Velocity.up)
    dlnV, _ = compute_dlnV(Model)
    clim = (-0.02, 0.02)
elseif ~isnothing(InvParam.Velocity.us)
    _, dlnV = compute_dlnV(Model)
    clim = (-0.03, 0.03)
end
heatmap(transpose(dlnV[:,:,16]), color=:roma, aspect_ratio=:equal, clims=clim)
heatmap(transpose(dlnV[:,:,31]), color=:roma, aspect_ratio=:equal, clims=clim)
heatmap(transpose(dlnV[:,151,end:-1:1]), color=:roma, aspect_ratio=:equal, clims=clim)

# Plot Sensitivities
heatmap(transpose(sqrt.(InvParam.Velocity.up.RSJS[:,5,end:-1:1])), color=:viridis, aspect_ratio=:equal)
