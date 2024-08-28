# MAKE INPUTS
# Build input data for SinkingSlab example. The following outputs are generated:
# 1) SeismicReceivers structure and data file
# 2) SeismicSources structure and data file
# 3) Dummy TravelTime (P and S) and Splitting Intensity (S) observations and data files
using PSI_D
# Extract path to SinkingSlab example directory
path, _ = splitdir(@__DIR__)

########################
### MODEL BOUNDARIES ###
########################

# Extrema of SinkingSlab model domain for reference in building the inputs
min_lon, min_lat, min_elv = (80.0, -10.0, -660.0)
max_lon, max_lat, max_elv = (100.0, 10.0, 0.0)

#########################
### SEISMIC RECEIVERS ###
#########################

# Define a regular array of seismic receivers
rcv_lon = range(start = min_lon + 1.8, stop = max_lon - 1.8, length = 25) # Receiver longitude coordinate vector (deg.); includes 1° buffer
rcv_lat = range(start = min_lat + 1.8, stop = max_lat - 1.8, length = 25) # Receiver latitude coordinate vector (deg.); includes 1° buffer
rcv_elv = 0.0 # Constant receiver elevation (placing all receivers at sea-level)
nrcv = length(rcv_lon)*length(rcv_lat) # Number of receivers
rid = Vector{String}(undef, nrcv) # Unique STRING receiver ID
xrcv = Vector{NTuple{3, Float64}}(undef, nrcv) # Receiver coordinates
# Fill receiver data
k = 0
for ϕ in rcv_lat
    for λ in rcv_lon
        global k += 1 # Allows script to be run from command line
        rid[k] = "SYN"*string(k; pad = 4)
        xrcv[k] = (λ, ϕ, rcv_elv)
    end
end
# Build Seismic Receivers Structure
Receivers = PSI_D.SeismicReceivers(rid, xrcv)

#######################
### SEISMIC SOURCES ###
#######################

# Define teleseismic sources regularly distributed in range and back-azimuth
ref_lon, ref_lat = (mean(rcv_lon), mean(rcv_lat)) # Use center of array as reference coordinate for computing source locations
dlt = 50.0:30.0:80.0 # Arc distances (deg.) of sources from model origin
baz = 0.0:30.0:330.0 # Back-azimuth of sources (counter-clockwise angle from East; deg.) from model origin
src_elv = -50.0 # Constant source elevation (i.e. negative source depth; km)
nsrc = length(dlt)*length(baz) # Number of sources
sid = Vector{Int}(undef, nsrc) # Unique INTEGER source IDs
xsrc = Vector{NTuple{3, Float64}}(undef, nsrc) # Source coordinates
# Fill source data
k = 0
for Δ in dlt
    for θ in baz
        global k += 1 # Allows script to be run from command line
        src_lat, src_lon = PSI_D.direct_geodesic(ref_lat, ref_lon, Δ, θ; tf_degrees = true)
        sid[k] = k
        xsrc[k] = (src_lon, src_lat, src_elv)
    end
end
# Build the SeismicSources Structure
Sources = PSI_D.SeismicSources(sid, xsrc)

###########################
### SEISMIC OBSERVABLES ###
###########################

# Create dummy observations for specific observable and phase types
p_phase, s_phase = ("P", "S") # Phase names for observations
period, dummy_obs, obs_error = (0.0, 0.0, 0.1) # Dominant period (s), place-holder observation value (s), and error (> 0!; s)
s_polarization = 0.25*π # Polarization azimuth for S-wave (measured in from Q-axis in ray-aligned QTL coordinates)
# Dummy P-wave travel-time
a_TTp = PSI_D.TravelTime(PSI_D.CompressionalWave(p_phase, period), PSI_D.ForwardTauP(),
sid[1], rid[1], dummy_obs, obs_error)
# Dummy S-wave travel-time
a_TTs = PSI_D.TravelTime(PSI_D.ShearWave(s_phase, period, s_polarization), PSI_D.ForwardTauP(),
sid[1], rid[1], dummy_obs, obs_error)
# Dummy S-wave  splitting intensity
a_SIs = PSI_D.SplittingIntensity(PSI_D.ShearWave(s_phase, period, s_polarization), PSI_D.ForwardTauP(),
sid[1], rid[1], dummy_obs, obs_error) # S-wave splitting intensity

# Create dummy observables for every source-receiver pair
nobs = length(Sources.id)*length(Receivers.id)
TTp = Vector{typeof(a_TTp)}(undef, nobs)
TTs = Vector{typeof(a_TTs)}(undef, nobs)
SIs = Vector{typeof(a_SIs)}(undef, nobs)
n = 0
for id_src in eachindex(Sources.id)
    for id_rcv in eachindex(Receivers.id)
        global n += 1 # Allows script to be run from command line
        TTp[n] = PSI_D.TravelTime(PSI_D.CompressionalWave(p_phase, period), PSI_D.ForwardTauP(),
        id_src, id_rcv, dummy_obs, obs_error)
        TTs[n] = PSI_D.TravelTime(PSI_D.ShearWave(s_phase, period, s_polarization), PSI_D.ForwardTauP(),
        id_src, id_rcv, dummy_obs, obs_error)
        SIs[n] = PSI_D.SplittingIntensity(PSI_D.ShearWave(s_phase, period, s_polarization), PSI_D.ForwardTauP(),
        id_src, id_rcv, dummy_obs, obs_error)
    end
end

####################################
### WRITE PSI STRUCTURES TO FILE ###
####################################

# Write the Sources and Receivers structures to data files
# PSI_D.write_psi_structure(path*"/psi_input/Sources.dat", Sources)
# PSI_D.write_psi_structure(path*"/psi_input/Receivers.dat", Receivers)

# Writing observations. By default, observation filenames have the format, 'Observable_Phase.dat' where
# 'Observable' is the type of observation (e.g. TravelTime) and 'Phase' is the type of phase (e.g. ShearWave).
# There is also the option to prepend any string to the beginning of the file name.
# PSI_D.write_observations(path*"/psi_input/", TTp; prepend = "DUMMY")
# PSI_D.write_observations(path*"/psi_input/", TTs; prepend = "DUMMY")
# PSI_D.write_observations(path*"/psi_input/", SIs; prepend = "DUMMY")