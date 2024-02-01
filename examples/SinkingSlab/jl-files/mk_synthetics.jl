# MAKE SYNTHETIC OBSERVATIONS
# Calls the psi_forward problem and generates the following synthetic datasets:
# 1) psi_output/SYN_SinkingBlock/SYN_TravelTime_CompressionalWave.dat
# 2) psi_output/SYN_SinkingBlock/SYN_TravelTime_ShearWave.dat
# 3) psi_output/SYN_SinkingBlock/SYN_SplittingIntensity_ShearWave.dat
# Extract path to SinkingSlab example directory
path, _ = splitdir(@__DIR__)

# PSI_D uses JavaCall to compute 1D rays with the TauP Toolkit. Non-Windows users must define the
# environment variable "JULIA_COPY_STACKS" as below for JavaCall to function properly. Additionally,
# Mac users must start Julia with the flag "--handle-signals=no" to avoid Java segmentation faults.
# Note that this may cause multi-threading in Julia to crash.
ENV["JULIA_COPY_STACKS"] = "yes"

# Optionally, define the TAUP_JAR environment variable that points to the TauP jar-file. If not defined,
# the version of the TauP Toolkit that ships with PSI_D is used.
# ENV["TAUP_JAR"] = "TauP-2.4.5/lib/TauP-2.4.5.jar" # Specify path to alternative TauP jar-file
delete!(ENV, "TAUP_JAR") # Revert to PSI_D TauP version

using PSI_D

# Compute synthetic data
# ~6 minutes to compute 52,488 observations in fully elastic model
psi_forward(path*"/psi_input/psi_parameters_synthetic.toml");