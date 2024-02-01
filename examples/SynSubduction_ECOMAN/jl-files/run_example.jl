# Run Example

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

# Run the anisotropic P + S travel-time inversion by passing the parameter file
# Paths in parameter file assume the code is being run from inside the SynSubduction_ECOMAN example directory
# Once the inversion is finished, an output is stored in the directory 'psi_output/ANI_TTP_TTS'
# To visualize the result in Paraview, load the state file, 'visualization/render_model.pvsm'. When prompted, update
# the path names in the state file to reference 'psi_output/ANI_TTP_TTS/FinalModel.vts'.
# Note, this is a big model and will take 1.9 Gb of storage.
psi_inverse(path*"/psi_input/psi_parameter_file_inversion.toml");