# EXAMPLE SKS SPLITTING PARAMETERS
# An interactive approach for making and plotting SKS splitting parameters.
# Path names reference are relative to the 'SinkingSlab' example directory.
using Plots
using PSI_D
# Extract path to SinkingSlab example directory
path, _ = splitdir(@__DIR__)

###############################
### ADD SKS EVENTS TO MODEL ###
###############################

# Build a PsiModel from an existing parameter file with existing receiver geometry
# for computing SKS splitting parameters
_, _, Model, _ = build_inputs(path*"/psi_input/psi_parameters_synthetic.toml");

Δ = 110.0 # SKS range
baz = 0:30:330 # SKS event backazimuths
src_elv = -50.0 # Constant source elevation (km)
# Compute geographic source coordinates
sid = Vector{Int}(undef, length(baz));
xsrc = Vector{NTuple{3, Float64}}(undef, length(baz));
for (i, α) in enumerate(baz)
    src_lat, src_lon = PSI_D.direct_geodesic(Model.Mesh.Geometry.ϕ₀, Model.Mesh.Geometry.λ₀, Δ, α; tf_degrees = true)
    sid[i] = 100 + i
    xsrc[i] = (src_lon, src_lat, src_elv)
end
# Build SeismicSources structure
Sources_SKS = PSI_D.SeismicSources(sid, xsrc);
# Optionally, write these sources to file for future use
# PSI_D.write_psi_structure(path*"/psi_input/Sources_SKS.dat", Sources_SKS);

# Re-build the model using these SKS sources
Model = PSI_D.PsiModel(Model.Mesh, Sources_SKS, Model.Receivers, Model.Parameters, Model.Methods);

####################################
### PREDICT SPLITTING PARAMETERS ###
####################################

# Create a dummy SplittingParameter observation
s_phase = "SKS" # Phase name
period = 10.0 # Dominant period of reference Ricker wavelet to split (s; must be > 0)
s_polarization = 0.0 # Polarization azimuth (0 for SKS phase)
dummy_value = (0.0, 0.0) # Place-holder (split_time, fast_azimuth)
dummy_error = (0.1, π/180.0) # Place-holder error
# SplittingParameter observation structure
a_SP = PSI_D.SplittingParameters(PSI_D.ShearWave(s_phase, period, s_polarization), PSI_D.ForwardTauP(),
0, "?", dummy_value, dummy_error) # Uses dummy source ID = 0 and dummy station ID = "?"
# Create array of dummy SplittingParameters
SP = Vector{typeof(a_SP)}(undef, length(Model.Sources.id)*length(Model.Receivers.id))
n = 0
for id_src in eachindex(Model.Sources.id)
    for id_rcv in eachindex(Model.Receivers.id)
        n += 1
        SP[n] =PSI_D.SplittingParameters(PSI_D.ShearWave(s_phase, period, s_polarization), PSI_D.ForwardTauP(),
        id_src, id_rcv, dummy_value, dummy_error)
    end
end

# Forward model SKS Splits
predictions, _ = psi_forward(SP, Model);

################
### PLOTTING ###
################

# Simple quiver plotting function (Plots.jl has built-in quiver function but cannot remove arrowheads that obscure splitting trends)
function plot_splits(x, y, dt, azm; scale = 1.0)
    H = plot()
    u, v = (zeros(2), zeros(2))
    for i in eachindex(x)
        dy, dx = scale*dt[i].*sincos(azm[i])
        u[1], u[2] = (x[i] - dx, x[i] + dx)
        v[1], v[2] = (y[i] - dy, y[i] + dy)
        plot!(H, u, v, color = :black, label = "", aspect_ratio = :equal)
    end

    return H
end

# Compute Station-Averaged Splits
rcv_lon = zeros(length(Model.Receivers.id));
rcv_lat = zeros(length(Model.Receivers.id));
u = zeros(length(Model.Receivers.id));
v = zeros(length(Model.Receivers.id));
num = zeros(length(Model.Receivers.id));
for (n, B) in enumerate(SP)
    jsta = Model.Receivers.id[B.receiver_id]
    rcv_lon[jsta], rcv_lat[jsta], _ = Model.Receivers.coordinates[jsta]
    Δt, ϕ = predictions[n] # Parse prediction tuple (delay time, fast azimuth)
    # Use cos2θ-average
    u[jsta] += Δt*cos(2.0*ϕ)
    v[jsta] += Δt*sin(2.0*ϕ)
    num[jsta] += 1
end
u ./= num;
v ./= num;
fast_azimuths, delay_times = ( 0.5*atan.(v,u), sqrt.((u.^2) + (v.^2)) );

push!(rcv_lon, 80.0)
push!(rcv_lat, -6.0)
push!(delay_times, 2.0)
push!(fast_azimuths, 0.0)
plot_splits(rcv_lon, rcv_lat, delay_times, fast_azimuths; scale = 0.25)