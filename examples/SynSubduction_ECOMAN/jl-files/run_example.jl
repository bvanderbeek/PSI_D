# Run Example

# PSI_D uses JavaCall to compute 1D rays with the TauP Toolkit. Non-Windows users must define the
# environment variable "JULIA_COPY_STACKS" as below for JavaCall to function properly. Additionally,
# Mac users must start Julia with the flag "--handle-signals=no" to avoid Java segmentation faults.
ENV["JULIA_COPY_STACKS"] = "yes"

# Install the PSI_D package
using Pkg
Pkg.add(url="https://github.com/bvanderbeek/PSI_D.git")

using PSI_D

# Run the anisotropic P + S travel-time inversion by passing the parameter file
# Paths in parameter file assume the code is being run from inside the SynSubduction_ECOMAN example directory
# Once the inversion is finished, an output is stored in the directory 'psi_output/ANI_TTP_TTS'
# To visualize the result in Paraview, load the state file, 'visualization/render_model.pvsm'. When prompted, update
# the path names in the state file to reference 'psi_output/ANI_TTP_TTS/FinalModel.vts'.
# Note, this is a big model and will take 1.9 Gb of storage.
psi_inverse("psi_input/psi_parameter_file_inversion.toml");