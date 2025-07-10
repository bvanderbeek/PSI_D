#################### SOLVERS ####################

# Solver: Structures for storing parameters relevant to the inverse problem solver
abstract type InverseMethod end
abstract type LinearisedSolver <: InverseMethod end

# Solver LSQR
struct SolverLSQR{T} <: LinearisedSolver
    damp::T # Solver internal damping parameter; should generally be zero
    atol::T # Stopping tolerance; see lsqr documentation (1e-6)
    conlim::T # Stopping tolerance; see lsqr documentation (1e8)
    maxiter::Int # Maximum number of solver iterations (maximum(size(A)))
    verbose::Bool # Output solver iteration info
    # Custom options
    tf_jac_scale::Bool # Apply Jacobian scaling to system
    nonlinit::Int # Maximum number of non-linear iterations for iterative linearised approach
end

struct DataResiduals{T}
    n::Int # Number of observations
    f_stat::Vector{T} # F-statistic, ratios of variances between iterations (i.e., wssrₖ/wssrₖ₊₁)
    f_critical::NTuple{2, T} # Critical valued at α-level significance
    chi_squared::Vector{T} # Chi-squared values, i.e., (∑rᵢ²)/n
    # subfit::Dict{NTuple{2, String}, T} # Dictionary of data fit metrics sub-divided by observation type
end
function DataResiduals(n; α = 0.95)
    return DataResiduals(n, zeros(niter), (quantile(FDist(n - 1, n - 1), α), α), zeros(niter))
end

#################### INVERSION PARAMETERS ####################

abstract type InversionParameter end

# Parameter Field: Structure to hold a parameter discretised in space
struct ParameterField{T,N} <: InversionParameter
    Mesh::AbstractMesh # Mesh defining discretisation of parameters
    dm::Array{T,N} # cumulative perturbations to parameters
    ddm::Array{T,N} # Incremental perturbations to parameters
    m0::Array{T,N} # Starting model values
    uncertainty::Array{T,N} # A priori fractional uncertainty
    # Regularisation parameters
    wdamp::Vector{T} # Damping Weight (square-root of Lagrangian Multiplier)
    tf_damp_cumulative::Vector{Bool} # Penalize cumulative (true) or incremental (false) perturbations
    wsmooth::Vector{T} # Smoothing Weight (square-root of Lagrangian Multiplier)
    tf_smooth_cumulative::Vector{Bool} # Penalize cumulative (true) or incremental (false) roughness
    # Jacobian info
    jcol::Vector{Int} # Jacobian column index for parameter 
    RSJS::Array{T,N} # Row-sum of the Jacobian-squared (i.e. a proxy for parameter sensitivity)
end
# Initialises a ParameterField
function ParameterField(Mesh; dm = zeros(Float64, size(Mesh)), ddm = zeros(Float64, size(Mesh)), m0 = zeros(Float64, size(Mesh)),
    uncertainty = ones(Float64, size(Mesh)), wdamp = zeros(Float64, 1), tf_damp_cumulative = [false], wsmooth = zeros(Float64, ndims(Mesh)),
    tf_smooth_cumulative = [false], jcol = [1, length(Mesh)], RSJS = zeros(Float64, size(Mesh)))

    return ParameterField(Mesh, dm, ddm, m0, uncertainty, wdamp, tf_damp_cumulative, wsmooth, tf_smooth_cumulative, jcol, RSJS)
end
# Custom display of structure
function Base.show(io::IO, obj::ParameterField)
    S = split(string(typeof(obj.Mesh)),"{")
    println(io, "ParameterField")
    println(io, "--> Mesh:             ", S[1])
    println(io, "--> Size:             ", size(obj.Mesh))
    println(io, "--> Coordinates:      ", S[2])
    println(io, "--> Parameterisation: ", typeof(obj.m0))

    return nothing
end

# Statics. Even though these structures are the same, need to break them up for dispatch.
abstract type SeismicStatics <: InversionParameter end

# Seismic Sources
struct SeismicSourceStatics{K, T, L} <: SeismicStatics
    dm::Dict{K,T} # Dictionary of (ID, Static Type, Static Phase) *keys* and cumulative static perturbations
    ddm::Dict{K,T} # Dictionary of (ID, Static Type, Static Phase) *keys* and incremental static perturbations
    uncertainty::Dict{K,T} # Dictionary of (ID, Static Type, Static Phase) *keys* and static uncertainty
    wdamp::Dict{L,T} # Dictionary of (Static Type, Static Phase) *keys* and their damping weights
    tf_damp_cumulative::Dict{L,Bool} # Dictionary of (Static Type, Static Phase) *keys* and their damping constraint
    jcol::Dict{K,Int} # Dictionary of (ID, Static Type, Static Phase) *keys* and Jacobian column index
    RSJS::Dict{K,T} # Dictionary of (ID, Static Type, Static Phase) *keys* and row-sum of Jacobian squared value
end
# Seismic Receivers
struct SeismicReceiverStatics{K, T, L} <: SeismicStatics
    dm::Dict{K,T} # Dictionary of (ID, Static Type, Static Phase) *keys* and cumulative static perturbations
    ddm::Dict{K,T} # Dictionary of (ID, Static Type, Static Phase) *keys* and incremental static perturbations
    uncertainty::Dict{K,T} # Dictionary of (ID, Static Type, Static Phase) *keys* and static uncertainty
    wdamp::Dict{L,T} # Dictionary of (Static Type, Static Phase) *keys* and their damping weight
    tf_damp_cumulative::Dict{L,Bool} # Dictionary of (Static Type, Static Phase) *keys* and their damping constraint
    jcol::Dict{K,Int} # Dictionary of (ID, Static Type, Static Phase) *keys* and Jacobian column index
    RSJS::Dict{K,T} # Dictionary of (ID, Static Type, Static Phase) *keys* and row-sum of Jacobian squared value
end

struct InverseSeismicVelocity{I, A} <: InversionParameter
    Isotropic::I
    Anisotropic::A
end
struct InverseIsotropicSlowness <: InversionParameter
    Up::Union{<:ParameterField, Nothing}
    Us::Union{<:ParameterField, Nothing}
    coupling_option::Int # 0 = independent, 1 = solve for us/up ratio instead of us
end
struct InverseAnisotropicVector{T, U} <: InversionParameter
    Orientations::T
    Fractions::U
end
struct InverseAzRadVector <: InversionParameter
    A::Union{<:ParameterField, Nothing} # f*cos²θ*cos2ϕ
    B::Union{<:ParameterField, Nothing} # f*cos²θ*sin2ϕ
    C::Union{<:ParameterField, Nothing} # √f*sinθ
end

# Seismic Perturbation Model: A container for all possible inversion parameters
# for which one can invert seismic observables
struct SeismicPerturbationModel{T1, T2, T3, T4, T5} <: InversionParameter
    Velocity::T1
    Interface::T2
    Hypocenter::T3
    SourceStatics::T4
    ReceiverStatics::T5
end
function SeismicPerturbationModel(; Velocity = nothing, Interface = nothing, Hypocenter = nothing,
    SourceStatics = nothing, ReceiverStatics = nothing)

    return SeismicPerturbationModel(Velocity, Interface, Hypocenter, SourceStatics, ReceiverStatics)
end



# MAIN INVERSION FUNCTION

function psi_inverse(parameter_file::String)
    # Load inputs
    PsiParameters, Observations, ForwardModel, PerturbationModel, Solver = build_inputs(parameter_file);

    # Create output directory
    if !isempty(PsiParameters["Output"]["output_directory"])
        # Create time-stamped directory?
        if PsiParameters["Output"]["tf_time_stamp"]
            date_now, time_now = split(string(now()), "T")
            date_now, time_now = (split(date_now, "-"), split(time_now, ":"))
            PsiParameters["Output"]["output_directory"] *= "/"*date_now[1][3:4]*prod(date_now[2:end])*"_"*prod(time_now[1:2])*time_now[3][1:2]
        end
        mkpath(PsiParameters["Output"]["output_directory"])
        # Save copy of parameter file
        path_filename = splitdir(parameter_file)
        cp(parameter_file, PsiParameters["Output"]["output_directory"]*"/"*path_filename[2])
    end

    # Call inverse problem
    psi_inverse!(Observations, ForwardModel, PerturbationModel, Solver; output_directory = PsiParameters["Output"]["output_directory"])

    return PsiParameters, Observations, ForwardModel, PerturbationModel, Solver
end
# Solve linearized inverse problem of the form Ax = b using the LSQR solver
function psi_inverse!(Observations::Vector{<:Observable}, ForwardModel::PsiModel, PerturbationModel::SeismicPerturbationModel,
    Solver::SolverLSQR; output_directory = "")
    # Assign Jacobian indices to parameters. Returns total number of parameters (i.e. Jacobian columns).
    npar = assign_jacobian_indices!(PerturbationModel)
    nobs = length(Observations)
    # Allocate solution vector
    x = zeros(npar)
    # Critical value for F-distribution (95%)
    fcrit = quantile(FDist(nobs - 1, nobs - 1), 0.95)
    fstat = 10.0*fcrit # Initialise an initial F-statisitic > fcrit

    # Compute kernels and relative residuals (b-vector) for all observations
    println("Calling forward problem for all observations. \n")
    _, b, Kernels = psi_forward(Observations, ForwardModel)
    # Initial sum of squared (relative) residuals
    wssr_k = sum(b -> b^2, b; init = 0.0)
    wsum = sum(K -> (1.0/K.Observation.error)^2, Kernels; init = 0.0)
    ssr_k = 0.0
    [ssr_k += (b[i]*K_i.Observation.error)^2 for (i, K_i) in enumerate(Kernels)]

    # Create file to store data fit at each iteration
    if !isempty(output_directory)
        fid_fit = open(output_directory*"/chi_squared.txt", "w")
        println(fid_fit, "F-crit (n = ", nobs,", alpha = 0.95): ",fcrit)
        println(fid_fit, "iteration, chi-squared, RMS, weighted RMS")
        println(fid_fit, "0, ",wssr_k/nobs,", ",sqrt(ssr_k/nobs),", ",sqrt(wssr_k/wsum))
    end

    # Inner inversion iterations
    IncPerturbationModel = deepcopy(PerturbationModel) # Incremental perturbations
    vp_ref, vs_ref = return_isotropic_velocities(ForwardModel.Parameters) # Save starting model velocities
    kiter, tf_run_forward = 0, isa(Observations[1].Forward, ForwardShortestPath)
    while (fstat > fcrit) && (kiter < Solver.nonlinit)
        println("Starting iteration ", string(kiter + 1), ".")

        # Build Jacobian Matrix
        Aᵢ, Aⱼ, Aᵥ = psi_build_jacobian(PerturbationModel, Kernels; row_offset = 0)

        # Update row counter when finished building Jacobian for all observations in a set
        jrow = maximum(Aᵢ) # Should equal 'nobs'
        println("Number of observations: ", string(jrow), ".")
        println("Number of parameters: ", string(npar), ".")
 
        # Compute row-sum of the Jacobian squared (RSJS)
        accumarray!(x, Aⱼ, Aᵥ, x -> x^2)
        # Assign to parameters
        fill_rsjs!(PerturbationModel, x)
        # Reset solution vector
        fill!(x, 0.0)

        # Build Constraint Equations
        build_constraints!(Aᵢ, Aⱼ, Aᵥ, b, PerturbationModel; row_offset = jrow, tf_jac_scale = Solver.tf_jac_scale)

        # Update row counter when finished building Jacobian for all observations in a set
        nrows = maximum(Aᵢ)
        println("Number of constraints: ", string(nrows - nobs), ".")

        # Build and solve sparse linearised system
        A = sparse(Aᵢ, Aⱼ, Aᵥ, nrows, npar, +) # Sum repeated indices
        lsqr!(x, A, b; damp = Solver.damp, atol = Solver.atol, conlim = Solver.conlim, maxiter = Solver.maxiter, verbose = Solver.verbose, log = false)

        # Update Parameters
        update_parameters!(PerturbationModel, x)

        # Update kernels
        if tf_run_forward
            # Update model with incremental perturbations
            update_parameters!(IncPerturbationModel, x)
            update_model!(ForwardModel, IncPerturbationModel)
            update_parameters!(IncPerturbationModel, -x)
            _, b, Kernels = psi_forward(Observations, ForwardModel)
        else
            update_kernel!(Kernels, PerturbationModel)
            _, b = evaluate_kernel(Kernels)
        end
        
        # Compute sum of squared residuals
        wssr_l = sum(b -> b^2, b)
        ssr_l = 0.0
        [ssr_l += (b[i]*K_i.Observation.error)^2 for (i, K_i) in enumerate(Kernels)]
        !isempty(output_directory) ? println(fid_fit, kiter + 1,", ",wssr_l/nobs,", ",sqrt(ssr_l/nobs),", ",sqrt(wssr_l/wsum)) : nothing
        # Compute new fstat
        fstat = wssr_k/wssr_l
        # Check for divergence. For non-linear inversions the solution can bounce around once it reaches a minimum.
        if fstat < 1.0
            println("The LSQR solution diverged. Reverting to previous iteration.")
            update_parameters!(PerturbationModel, -x)
            if tf_run_forward
                update_parameters!(IncPerturbationModel, -x)
                update_model!(ForwardModel, IncPerturbationModel)
                update_parameters!(IncPerturbationModel, x)
                _, b, Kernels = psi_forward(Observations, ForwardModel)
            else
                update_kernel!(Kernels, PerturbationModel)
                _, b = evaluate_kernel(Kernels)
            end

            wssr_l = sum(b -> b^2, b)
            ssr_l = 0.0
            [ssr_l += (b[i]*K_i.Observation.error)^2 for (i, K_i) in enumerate(Kernels)]
            fstat = wssr_k/wssr_l
            # This line search tends to not converge suggesting that the regularisation is controlling the gradient when the
            # LSQR solution diverges. In this case, the regularization weights need to be reduced for the solution to improve.
            # fstat, wssr_l, b = line_search!(x, PerturbationModel, Kernels, wssr_k, wssr_l; maxiter = 10, step_size = 0.5)
        end

        # Reset solution vector
        fill!(x, 0.0)

        # Display fit summary for this iteration
        println("F-stat = ", string(fstat), " | F-crit = ", string(fcrit))
        println("rmsₖ₋₁ = ", string(sqrt(ssr_k/nobs)), " | rmsₖ = ", string(sqrt(ssr_l/nobs))," (",string(sqrt(wssr_l/wsum)),")")
        println("χ²ₖ₋₁ = ", string(wssr_k/nobs), " | ", "χ²ₖ = ", string(wssr_l/nobs), "\n")
        # Update prior fit
        wssr_k, ssr_k = wssr_l, ssr_l

        # Update iteration counter
        kiter += 1
    end

    # Add total perturbation to forwward model if not already done so
    if !tf_run_forward
        update_model!(ForwardModel, PerturbationModel)
    end

    # Write results
    if !isempty(output_directory)
        # Close chi-squared file
        close(fid_fit)
        # Write absolute residuals from final model
        _, res  = evaluate_kernel(Kernels)
        for i in eachindex(res)
            res[i] *= Observations[i].error
        end
        write_observations(output_directory, Observations; alt_data = res, prepend = "RES")
        # Write parameter sampling VTK
        write_sampling_to_vtk(output_directory, PerturbationModel)
        # Write the model VTK file (with starting model)
        write_model_to_vtk(output_directory*"/FinalModel", ForwardModel; vp_ref = vp_ref, vs_ref = vs_ref)
        # Write PsiModel files
        tf_write_sources = !isnothing(PerturbationModel.Hypocenter) || !isnothing(PerturbationModel.SourceStatics)
        tf_write_receivers = !isnothing(PerturbationModel.SourceStatics)
        write_psi_model(output_directory, ForwardModel; tf_write_sources = tf_write_sources,
        tf_write_receivers = tf_write_receivers, tf_write_parameters = true)
    end

    return nothing
end
function line_search!(x, PerturbationModel, Kernels, wssr_k, wssr_l; maxiter = 10, step_size = 0.5)
    niter = 0
    fstat = wssr_k/wssr_l
    while (fstat < 1.0) && (niter < maxiter)
        # Basic line search to find a new minimum
        x .*= step_size # Iteratively reduce perturbation vector by scalar
        update_parameters!(PerturbationModel, -x) # Remove perturbations 
        update_kernel!(Kernels, PerturbationModel)
        _, b = evaluate_kernel(Kernels)
        wssr_l = sum(b -> b^2, b)
        fstat = wssr_k/wssr_l
        niter += 1

        println("F-stat = ", string(fstat))
    end
    if fstat < 1.0
        println("Line search failed to converge. Reverting to previous iteration solution.")
        # Remove remaining perturbations
        update_parameters!(PerturbationModel, -x)
        update_kernel!(Kernels, PerturbationModel)
        _, b = evaluate_kernel(Kernels)
        wssr_l = sum(b -> b^2, b)
        fstat = wssr_k/wssr_l
    end

    return fstat, wssr_l, b
end


# JACOBIAN INDEXING ASSIGNMENTS

# Assign Jacobian column indices to SeismicPerturbationalModel fields
function assign_jacobian_indices!(PerturbationModel::SeismicPerturbationModel; jacobian_column = 0)
    # Assign Jacobian indices to all fields in a Seismic Perturbation Model
    # Each call to assign_jacobian_indices! updates and returns the Jacobian row counter
    if ~isnothing(PerturbationModel.Velocity)
        jacobian_column = assign_jacobian_indices!(PerturbationModel.Velocity; jacobian_column = jacobian_column)
    end
    if ~isnothing(PerturbationModel.Interface)
        jacobian_column = assign_jacobian_indices!(PerturbationModel.Interface; jacobian_column = jacobian_column)
    end
    if ~isnothing(PerturbationModel.Hypocenter)
        jacobian_column = assign_jacobian_indices!(PerturbationModel.Hypocenter; jacobian_column = jacobian_column)
    end
    if ~isnothing(PerturbationModel.SourceStatics)
        jacobian_column = assign_jacobian_indices!(PerturbationModel.SourceStatics; jacobian_column = jacobian_column)
    end
    if ~isnothing(PerturbationModel.ReceiverStatics)
        jacobian_column = assign_jacobian_indices!(PerturbationModel.ReceiverStatics; jacobian_column = jacobian_column)
    end

    # Return the total number of inversion parameters
    return jacobian_column
end
# Assign Jacobian column indices to InverseSeismicVelocity fields
function assign_jacobian_indices!(PerturbationModel::InverseSeismicVelocity; jacobian_column = 0)
    if ~isnothing(PerturbationModel.Isotropic)
        jacobian_column = assign_jacobian_indices!(PerturbationModel.Isotropic, jacobian_column = jacobian_column)
    end
    if ~isnothing(PerturbationModel.Anisotropic)
        jacobian_column = assign_jacobian_indices!(PerturbationModel.Anisotropic, jacobian_column = jacobian_column)
    end
    # Return the total number of inversion parameters
    return jacobian_column
end
# Assign Jacobian column indices to IsotropicSlowness fields
function assign_jacobian_indices!(PerturbationModel::InverseIsotropicSlowness; jacobian_column = 0)
    # Indices of P-slowness parameters
    if ~isnothing(PerturbationModel.Up)
        jacobian_column = assign_jacobian_indices!(PerturbationModel.Up; jacobian_column = jacobian_column)
    end
    # Indices of S-slowness parameters
    if ~isnothing(PerturbationModel.Us)
        jacobian_column = assign_jacobian_indices!(PerturbationModel.Us; jacobian_column = jacobian_column)
    end

    # Return the total number of inversion parameters
    return jacobian_column
end
# Assign Jacobian column indices to InverseAnisotropicVector fields
function assign_jacobian_indices!(PerturbationModel::InverseAnisotropicVector; jacobian_column = 0)
    if ~isnothing(PerturbationModel.Orientations)
        jacobian_column = assign_jacobian_indices!(PerturbationModel.Orientations; jacobian_column = jacobian_column)
    end
    if ~isnothing(PerturbationModel.Fractions)
        jacobian_column = assign_jacobian_indices!(PerturbationModel.Fractions; jacobian_column = jacobian_column)
    end

    return jacobian_column
end
# Assign Jacobian column indices to InverseAzRadVector fields
function assign_jacobian_indices!(PerturbationModel::InverseAzRadVector; jacobian_column = 0)
    # Inidces for the vectoral A-parameters
    if ~isnothing(PerturbationModel.A)
        jacobian_column = assign_jacobian_indices!(PerturbationModel.A; jacobian_column = jacobian_column)
    end
    # Inidces for the vectoral B-parameters
    if ~isnothing(PerturbationModel.B)
        jacobian_column = assign_jacobian_indices!(PerturbationModel.B; jacobian_column = jacobian_column)
    end
    # Inidces for the vectoral C-parameters
    if ~isnothing(PerturbationModel.C)
        jacobian_column = assign_jacobian_indices!(PerturbationModel.C; jacobian_column = jacobian_column)
    end
    # Return the total number of inversion parameters
    return jacobian_column
end
# Assign Jacobian column indices to a ParameterField
function assign_jacobian_indices!(PerturbationModel::ParameterField; jacobian_column = 0)
    # Number of parameters is simply the number of nodes/elements in the mesh
    num_params = length(PerturbationModel.Mesh)
    PerturbationModel.jcol[1] = 1 + jacobian_column
    PerturbationModel.jcol[2] = num_params + jacobian_column
    # Add number of parameters in this field to the total
    jacobian_column += num_params

    # Return the total number of inversion parameters
    return jacobian_column
end
# Assign Jacobian column indices to SeismicStatics
function assign_jacobian_indices!(PerturbationModel::SeismicStatics; jacobian_column = 0)
    # Loop over static keys
    for k in eachindex(PerturbationModel.jcol)
        # Increment jacobian index
        jacobian_column += 1
        # Assign index to key
        PerturbationModel.jcol[k] = jacobian_column
    end

    # Return the total number of inversion parameters
    return jacobian_column
end



# JACOBIAN BUILDING FUNCTIONS

# Build Jacobian for all parameters in a SeismicPerturbationModel for a Vector of ObservableKernel
function psi_build_jacobian(PerturbationModel::SeismicPerturbationModel, Kernels::Vector{<:ObservableKernel}; row_offset = 0)
    # Initialise storage arrays for linear system
    Aᵢ = Vector{Int}()
    Aⱼ = Vector{Int}()
    Aᵥ = Vector{Float64}()

    if ~isnothing(PerturbationModel.Velocity)
        psi_jacobian_block!(Aᵢ, Aⱼ, Aᵥ, PerturbationModel.Velocity, Kernels; row_offset = row_offset)
    end
    if ~isnothing(PerturbationModel.Interface)
        error("Interface inversion not yet implemented.")
    end
    if ~isnothing(PerturbationModel.Hypocenter)
        error("Hypocenter inversion not yet implemented.")
    end
    if ~isnothing(PerturbationModel.SourceStatics)
        psi_jacobian_block!(Aᵢ, Aⱼ, Aᵥ, PerturbationModel.SourceStatics, Kernels; row_offset = row_offset)
    end
    if ~isnothing(PerturbationModel.ReceiverStatics)
        psi_jacobian_block!(Aᵢ, Aⱼ, Aᵥ, PerturbationModel.ReceiverStatics, Kernels; row_offset = row_offset)
    end

    return Aᵢ, Aⱼ, Aᵥ
end
# Build a block of the Jacobian for a specific InversionParameter and a Vector of ObservableKernel
function psi_jacobian_block!(Aᵢ, Aⱼ, Aᵥ, PerturbationModel::InversionParameter, Kernels::Vector{<:ObservableKernel}; row_offset = 0)
    # Loop over kernels
    for (i, iKernel) in enumerate(Kernels)
        # Differentiate observable kernel with respect to parameter field
        j, dbdm = psi_jacobian_row(PerturbationModel, iKernel)
        # Append to Jacobian
        foreach(_ -> push!(Aᵢ, i + row_offset), 1:length(j))
        append!(Aⱼ, j)
        append!(Aᵥ, dbdm)
    end

    return nothing
end
# Build a row of the Jacobian for a specific InversionParameter and a single ObservableKernel
function psi_jacobian_row(PerturbationModel::InversionParameter, Kernel::ObservableKernel)
    # Loop over elements in kernel
    Wj = Vector{Int}()
    Wv = Vector{Float64}()
    for i in eachindex(Kernel.coordinates)
        # Differentiate the kernel
        psi_differentiate_kernel!(Wj, Wv, PerturbationModel, Kernel, i)
    end
    # Accumulate the weights
    jcol, dbdm = accumlate_weights(Wj, Wv)
    # Scale sensitivity by observation error
    dbdm ./= Kernel.Observation.error

    return jcol, dbdm
end
# Build a row of the Jacobian for SeismicStatics and a single ObservableKernel
function psi_jacobian_row(PerturbationModel::SeismicStatics, Kernel::ObservableKernel)
    # Dictionary key for static
    static_type = typeof(PerturbationModel)
    if static_type <: SeismicSourceStatics
        k = (Kernel.Observation.source_id, nameof(typeof(Kernel.Observation)), nameof(typeof(Kernel.Observation.Phase)))
    elseif static_type <: SeismicReceiverStatics
        k = (Kernel.Observation.receiver_id, nameof(typeof(Kernel.Observation)), nameof(typeof(Kernel.Observation.Phase)))
    else
        error("Unrecognized static type!")
    end

    if haskey(PerturbationModel.jcol, k)
        # Get Jacobian index
        jcol = PerturbationModel.jcol[k]
        # Static sensitivity is always 1
        dbdm = 1.0/Kernel.Observation.error
    else
        println("No source static for key, ", k)
    end

    return jcol, dbdm
end



# PARTIAL DERIVATIVE FUNCTIONS

# Differentiate an ObservableKernel with respect to InverseSeismicVelocity
function psi_differentiate_kernel!(Wj, Wv, PerturbationModel::InverseSeismicVelocity, Kernel::ObservableKernel, index)
    if ~isnothing(PerturbationModel.Isotropic)
        psi_differentiate_kernel!(Wj, Wv, PerturbationModel.Isotropic, Kernel, index)
    end

    if ~isnothing(PerturbationModel.Anisotropic)
        psi_differentiate_kernel!(Wj, Wv, PerturbationModel.Anisotropic, Kernel, index)
    end

    return nothing
end
# Differentiate an ObservableKernel with respect to InverseAnisotropicVector
function psi_differentiate_kernel!(Wj, Wv, PerturbationModel::InverseAnisotropicVector, Kernel::ObservableKernel, index)
    if ~isnothing(PerturbationModel.Orientations)
        psi_differentiate_kernel!(Wj, Wv, PerturbationModel.Orientations, Kernel, index)
    end
    if ~isnothing(PerturbationModel.Fractions)
        psi_differentiate_kernel!(Wj, Wv, PerturbationModel.Fractions, Kernel, index)
    end

    return nothing
end

# Differentiate IsotropicVelocity P-wave Travel-Time with respect to IsotropicSlowness
function psi_differentiate_kernel!(Wj, Wv, PerturbationModel::InverseIsotropicSlowness, Kernel::ObservableKernel{<:TravelTime{<:CompressionalWave}, <:IsotropicVelocity}, index)
    if ~isnothing(PerturbationModel.Up)
        # Kernel coordinates in perturbational mesh coordinate system
        x_local = global_to_local(Kernel.coordinates[index][1], Kernel.coordinates[index][2], Kernel.coordinates[index][3], PerturbationModel.Up.Mesh.Geometry)
        # Map partial derivatives to inversion parameters. Travel-time slowness partial is simply the kernel weight distribued amongst the parameters.
        map_partial_to_jacobian!(Wj, Wv, PerturbationModel.Up.Mesh, x_local, Kernel.weights[index].dr, PerturbationModel.Up.jcol[1])
    end

    return nothing
end
# Differentiate IsotropicVelocity S-wave Travel-Time with respect to IsotropicSlowness
function psi_differentiate_kernel!(Wj, Wv, PerturbationModel::InverseIsotropicSlowness, Kernel::ObservableKernel{<:TravelTime{<:ShearWave}, <:IsotropicVelocity}, index)
    if PerturbationModel.coupling_option == 0
        # Inversion for S-slowness
        dtdm = Kernel.weights[index].dr
    elseif PerturbationModel.coupling_option == 1
        # Sensitivity to P-slowness
        # Kernel coordinates in P-slowness mesh
        x_local = global_to_local(Kernel.coordinates[index][1], Kernel.coordinates[index][2], Kernel.coordinates[index][3], PerturbationModel.Up.Mesh.Geometry)
        # Estimate P-slowness at kernel coordinate for current iteration
        up = linearly_interpolate(PerturbationModel.Up.Mesh.x, PerturbationModel.Up.m0, x_local; tf_extrapolate = false, tf_harmonic = false)
        up += linearly_interpolate(PerturbationModel.Up.Mesh.x, PerturbationModel.Up.dm, x_local; tf_extrapolate = false, tf_harmonic = false)
        # Compute sensitivity to P-slowness
        dtdm = Kernel.weights[index].dr/(up*Kernel.Parameters.vs[index])
        map_partial_to_jacobian!(Wj, Wv, PerturbationModel.Up.Mesh, x_local, dtdm, PerturbationModel.Up.jcol[1])

        # Sensitivity to S-to-P slowness ratio
        dtdm = up*Kernel.weights[index].dr
    else
        error("Unrecognized couplinf option!")
    end

    # Sensitivity to S-slowness or S-to-P slowness ratio
    if ~isnothing(PerturbationModel.Us)
        # Kernel coordinates in perturbational mesh coordinate system
        x_local = global_to_local(Kernel.coordinates[index][1], Kernel.coordinates[index][2], Kernel.coordinates[index][3], PerturbationModel.Us.Mesh.Geometry)
        # Map partial derivatives to inversion parameters. Travel-time slowness partial is simply the kernel weight distribued amongst the parameters.
        map_partial_to_jacobian!(Wj, Wv, PerturbationModel.Us.Mesh, x_local, dtdm, PerturbationModel.Us.jcol[1])
    end

    return nothing
end

# Differentiate HexagonalVectoralVelocity P-wave Travel-Time with respect to IsotropicSlowness
function psi_differentiate_kernel!(Wj, Wv, PerturbationModel::InverseIsotropicSlowness, Kernel::ObservableKernel{<:TravelTime{<:CompressionalWave}, <:HexagonalVectoralVelocity}, index)
    if ~isnothing(PerturbationModel.Up)
        # qP-phase velocity
        vqp = kernel_phase_velocity(Kernel.Observation.Phase, Kernel, index)
        # Isotropic P-velocity
        vip, _ = return_isotropic_velocities(Kernel.Parameters, index)
        # Isotropic slowness derivative
        dtdu = Kernel.weights[index].dr*(vip/vqp)
        # Kernel coordinates in perturbational mesh coordinate system
        x_local = global_to_local(Kernel.coordinates[index][1], Kernel.coordinates[index][2], Kernel.coordinates[index][3], PerturbationModel.Up.Mesh.Geometry)
        # Map partial derivatives to inversion parameters
        map_partial_to_jacobian!(Wj, Wv, PerturbationModel.Up.Mesh, x_local, dtdu, PerturbationModel.Up.jcol[1])
    end

    return nothing
end
# Differentiate HexagonalVectoralVelocity S-wave Travel-Time with respect to IsotropicSlowness
function psi_differentiate_kernel!(Wj, Wv, PerturbationModel::InverseIsotropicSlowness, Kernel::ObservableKernel{<:TravelTime{<:ShearWave}, <:HexagonalVectoralVelocity}, index)
    if PerturbationModel.coupling_option == 0
        # Invarient Isotropic S-velocity
        _, β₀ = return_isotropic_velocities(Kernel.Parameters, index)
        # Extract Thomsen Parameters
        α, β, ϵ, δ, γ = return_thomsen_parameters(Kernel.Parameters, index)
        # Compute qS phase velocities
        if Kernel.Parameters.tf_exact
            vqs1, vqs2, _, ζ = qs_phase_velocities_thomsen(Kernel.weights[index].azimuth, Kernel.weights[index].elevation,
            Kernel.Observation.Phase.paz, Kernel.Parameters.azimuth[index], Kernel.Parameters.elevation[index], α, β, ϵ, δ, γ)
        else
            vqs1, vqs2, _, ζ = qs_phase_velocities_thomsen(Kernel.weights[index].azimuth, Kernel.weights[index].elevation,
            Kernel.Observation.Phase.paz, Kernel.Parameters.azimuth[index], Kernel.Parameters.elevation[index], α, β, ϵ - δ, γ)
        end
        # Isotropic slowness derivative
        dtdu = Kernel.weights[index].dr*( (β₀/vqs2) + ( (2.0*β*β₀/(vqs1^2)) - (β₀/vqs1) - (β₀/vqs2) )*(cos(ζ)^2) )
        # Kernel coordinates in perturbational mesh coordinate system
        x_local = global_to_local(Kernel.coordinates[index][1], Kernel.coordinates[index][2], Kernel.coordinates[index][3], PerturbationModel.Us.Mesh.Geometry)
        # Map partial derivatives to inversion parameters
        map_partial_to_jacobian!(Wj, Wv, PerturbationModel.Us.Mesh, x_local, dtdu, PerturbationModel.Us.jcol[1])
    else
        error("Unrecognized coupling option!")
    end

    return nothing
end
# Differentiate HexagonalVectoralVelocity S-wave SplittingIntensity with respect to IsotropicSlowness
function psi_differentiate_kernel!(Wj, Wv, PerturbationModel::InverseIsotropicSlowness, Kernel::ObservableKernel{<:SplittingIntensity{<:ShearWave}, <:HexagonalVectoralVelocity}, index)
    # Ignoring weak sensitivity of splitting intensity to isotropic velocity
    return nothing
end
# Differentiate HexagonalVectoralVelocity P-wave Travel-Time with respect to InverseAzRadVector
function psi_differentiate_kernel!(Wj, Wv, PerturbationModel::InverseAzRadVector, Kernel::ObservableKernel{<:TravelTime{<:CompressionalWave}, <:HexagonalVectoralVelocity}, index)
    # Currently assuming anisotropic parameters are all defined on the same mesh
    if (PerturbationModel.A.Mesh.Geometry != PerturbationModel.B.Mesh.Geometry) ||
        (PerturbationModel.A.Mesh.Geometry != PerturbationModel.C.Mesh.Geometry)
        error("Mesh Geometry for InverseHexagonalVector_VF2021 must be equivalent")
    end
    # Coordinate System Conversions
    x_global = Kernel.coordinates[index]
    x_local = global_to_local(x_global[1], x_global[2], x_global[3], PerturbationModel.A.Mesh.Geometry)
    # Propagation and symmetry axis orientations in the local geographic coordinate system
    pazm, pelv = (Kernel.weights[index].azimuth, Kernel.weights[index].elevation)
    sazm, selv = (Kernel.Parameters.azimuth[index], Kernel.Parameters.elevation[index])

    # Extract relevant parameters
    α = Kernel.Parameters.α[index] # Reference symmetry axis velocity
    f = Kernel.Parameters.f[index] # Anisotropic strength
    vqp = kernel_phase_velocity(Kernel.Observation.Phase, Kernel, index) # qP-phase velocity
    vip, _ = return_isotropic_velocities(Kernel.Parameters, index) # Isotropic P-velocity
    rϵ, rη, _ = return_anisotropic_ratios(Kernel.Parameters, index) # Thomsen parameter ratios
    # Cosine of angle between propagation direction annd symmetry axis
    cosθ = symmetry_axis_cosine(sazm, selv, pazm, pelv)
    cosθ_2, cosθ_4 = (cosθ^2, cosθ^4)

    # qP-phase velocity partial derivatives
    dαdf = -(2/15)*((α^3)/(vip^2))*(5.0*rϵ - rη)
    dsdA, dsdB, dsdC = differentiate_symmetry_axis_product(pazm, pelv, sazm, selv, f)
    dfdA, dfdB, dfdC = differentiate_anisotropic_strength(sazm, selv, f)
    dvdA = α*( (rϵ - rη*cosθ_4)*dfdA - (rϵ + rη - 2.0*rη*cosθ_2)*dsdA ) + dαdf*dfdA*(vqp/α)
    dvdB = α*( (rϵ - rη*cosθ_4)*dfdB - (rϵ + rη - 2.0*rη*cosθ_2)*dsdB ) + dαdf*dfdB*(vqp/α)
    dvdC = α*( (rϵ - rη*cosθ_4)*dfdC - (rϵ + rη - 2.0*rη*cosθ_2)*dsdC ) + dαdf*dfdC*(vqp/α)
    # Use second-order perturbation to approximate ∂v/∂C when anisotropy is near zero
    # Δv/Δc ≈ (∂v/∂C) + 0.5*(∂²v/∂C²)*ΔC
    if abs(f*rϵ) < 0.0001
        Δf = abs(0.001/rϵ)
        ΔC = sqrt(Δf)
        dvdC = ΔC*( α*( (rϵ - rη*cosθ_4) - (rϵ + rη - 2.0*rη*cosθ_2)*(sin(pelv)^2) ) + dαdf )
    end
    # Chain rule to get travel-time partial derivative; ∂t/∂m = (∂t/∂u)*(∂u/∂v)*(∂v/∂m)
    dtdu = Kernel.weights[index].dr
    dtdv = -dtdu/(vqp^2)
    dtdA, dtdB, dtdC = (dtdv*dvdA, dtdv*dvdB, dtdv*dvdC)

    # Map partial derivatives to inversion parameters
    if sqrt(sum(x -> x^2, x_global)) > (PerturbationModel.A.Mesh.Geometry.R₀ - 500.0) # Squeezing!
        map_partial_to_jacobian!(Wj, Wv, PerturbationModel.A.Mesh, x_local, dtdA, PerturbationModel.A.jcol[1])
        map_partial_to_jacobian!(Wj, Wv, PerturbationModel.B.Mesh, x_local, dtdB, PerturbationModel.B.jcol[1])
        map_partial_to_jacobian!(Wj, Wv, PerturbationModel.C.Mesh, x_local, dtdC, PerturbationModel.C.jcol[1])
    end

    return nothing
end
# Differentiate HexagonalVectoralVelocity S-wave Travel-Time with respect to InverseAzRadVector
function psi_differentiate_kernel!(Wj, Wv, PerturbationModel::InverseAzRadVector, Kernel::ObservableKernel{<:TravelTime{<:ShearWave}, <:HexagonalVectoralVelocity}, index)
    # Local perturbational mesh coordinates
    x_global = Kernel.coordinates[index]
    x_local = global_to_local(x_global[1], x_global[2], x_global[3], PerturbationModel.A.Mesh.Geometry)
    # Sensitivity of qS-phase velocities to anisotropic parameters
    du1dABC, du2dABC, dζdABC_Δu₁₂, ζ = differentiate_qs_velocities(PerturbationModel, Kernel, index)
    # The travel-time partial derivatives; Eq. A23 of VanderBeek et al. (GJI 2023)
    sin2ζ, cos2ζ = sincos(2.0*ζ)
    cosζ_2 = 0.5*(cos2ζ + 1.0)
    dtdA = Kernel.weights[index].dr*( du2dABC[1] + (du1dABC[1] - du2dABC[1])*cosζ_2 - dζdABC_Δu₁₂[1]*sin2ζ )
    dtdB = Kernel.weights[index].dr*( du2dABC[2] + (du1dABC[2] - du2dABC[2])*cosζ_2 - dζdABC_Δu₁₂[2]*sin2ζ )
    dtdC = Kernel.weights[index].dr*( du2dABC[3] + (du1dABC[3] - du2dABC[3])*cosζ_2 - dζdABC_Δu₁₂[3]*sin2ζ )
    # Map partial derivatives to inversion parameters
    if sqrt(sum(x -> x^2, x_global)) > (PerturbationModel.A.Mesh.Geometry.R₀ - 500.0) # Squeezing!
        map_partial_to_jacobian!(Wj, Wv, PerturbationModel.A.Mesh, x_local, dtdA, PerturbationModel.A.jcol[1])
        map_partial_to_jacobian!(Wj, Wv, PerturbationModel.B.Mesh, x_local, dtdB, PerturbationModel.B.jcol[1])
        map_partial_to_jacobian!(Wj, Wv, PerturbationModel.C.Mesh, x_local, dtdC, PerturbationModel.C.jcol[1])
    end

    return nothing
end
# Differentiate HexagonalVectoralVelocity S-wave SplittingIntensity with respect to InverseAzRadVector
function psi_differentiate_kernel!(Wj, Wv, PerturbationModel::InverseAzRadVector, Kernel::ObservableKernel{<:SplittingIntensity{<:ShearWave}, <:HexagonalVectoralVelocity}, index)
    # Local perturbational mesh coordinates
    x_global = Kernel.coordinates[index]
    x_local = global_to_local(x_global[1], x_global[2], x_global[3], PerturbationModel.A.Mesh.Geometry)
    # Sensitivity of qS-phase velocities to anisotropic parameters
    du1dABC, du2dABC, dζdABC_Δu₁₂, ζ = differentiate_qs_velocities(PerturbationModel, Kernel, index)
    # Splitting intensity partial derivatives; Eq. A24 of VanderBeek et al. (GJI 2023)
    sin2ζ, cos2ζ = sincos(2.0*ζ)
    dtdA = 0.5*Kernel.weights[index].dr*( (du2dABC[1] - du1dABC[1])*sin2ζ - 2.0*dζdABC_Δu₁₂[1]*cos2ζ )
    dtdB = 0.5*Kernel.weights[index].dr*( (du2dABC[2] - du1dABC[2])*sin2ζ - 2.0*dζdABC_Δu₁₂[2]*cos2ζ )
    dtdC = 0.5*Kernel.weights[index].dr*( (du2dABC[3] - du1dABC[3])*sin2ζ - 2.0*dζdABC_Δu₁₂[3]*cos2ζ )
    # Map partial derivatives to inversion parameters
    if sqrt(sum(x -> x^2, x_global)) > (PerturbationModel.A.Mesh.Geometry.R₀ - 500.0) # Squeezing!
        map_partial_to_jacobian!(Wj, Wv, PerturbationModel.A.Mesh, x_local, dtdA, PerturbationModel.A.jcol[1])
        map_partial_to_jacobian!(Wj, Wv, PerturbationModel.B.Mesh, x_local, dtdB, PerturbationModel.B.jcol[1])
        map_partial_to_jacobian!(Wj, Wv, PerturbationModel.C.Mesh, x_local, dtdC, PerturbationModel.C.jcol[1])
    end

    return nothing
end
function differentiate_qs_velocities(PerturbationModel::InverseAzRadVector, Kernel::ObservableKernel{<:SeismicObservable, <:HexagonalVectoralVelocity}, index)
    # Currently assuming anisotropic parameters are all defined on the same mesh
    if (PerturbationModel.A.Mesh.Geometry != PerturbationModel.B.Mesh.Geometry) ||
        (PerturbationModel.A.Mesh.Geometry != PerturbationModel.C.Mesh.Geometry)
        error("Mesh Geometry for InverseHexagonalVector_VF2021 must be equivalent")
    end
    # Propagation and symmetry axis orientations in the local geographic coordinate system
    pazm, pelv = (Kernel.weights[index].azimuth, Kernel.weights[index].elevation)
    sazm, selv = (Kernel.Parameters.azimuth[index], Kernel.Parameters.elevation[index])
    # Invarient Isotropic S-velocity
    α₀, β₀ = return_isotropic_velocities(Kernel.Parameters, index)
    # Extract Thomsen Parameters
    α, β, ϵ, δ, γ = return_thomsen_parameters(Kernel.Parameters, index)
    rϵ, rη, rγ = return_anisotropic_ratios(Kernel.Parameters, index)
    f = Kernel.Parameters.f[index] # Anisotropic strength
    # Compute qS phase velocities
    if Kernel.Parameters.tf_exact
        v1, v2, cosΔ, ζ = qs_phase_velocities_thomsen(pazm, pelv, Kernel.Observation.Phase.paz, sazm, selv, α, β, ϵ, δ, γ)
    else
        v1, v2, cosΔ, ζ = qs_phase_velocities_thomsen(pazm, pelv, Kernel.Observation.Phase.paz, sazm, selv, α, β, ϵ - δ, γ)
    end
    # Symmetry axis partial derivatives
    dαdf = -(2/15)*((α^3)/(α₀^2))*(5.0*rϵ - rη)
    dβdf = -rγ*(β^3)/(3.0*(β₀^2))
    dsdA, dsdB, dsdC = differentiate_symmetry_axis_product(pazm, pelv, sazm, selv, f)
    dfdA, dfdB, dfdC = differentiate_anisotropic_strength(sazm, selv, f)
    # The qS'-slowness partial derivatives
    cosΔ_2, cosΔ_4 = (cosΔ^2, cosΔ^4)
    du1dv1 = -1.0/(v1^2)
    du1dA = du1dv1*( ((α^2)/β)*rη*(dsdA - 2.0*cosΔ_2*dsdA + cosΔ_4*dfdA) + (2.0 - (v1/β))*dβdf*dfdA + 2.0*((v1 - β)/α)*dαdf*dfdA )
    du1dB = du1dv1*( ((α^2)/β)*rη*(dsdB - 2.0*cosΔ_2*dsdB + cosΔ_4*dfdB) + (2.0 - (v1/β))*dβdf*dfdB + 2.0*((v1 - β)/α)*dαdf*dfdB )
    du1dC = du1dv1*( ((α^2)/β)*rη*(dsdC - 2.0*cosΔ_2*dsdC + cosΔ_4*dfdC) + (2.0 - (v1/β))*dβdf*dfdC + 2.0*((v1 - β)/α)*dαdf*dfdC )
    # The qS''-slowness partials derivatives
    du2dv2 = -1.0/(v2^2)
    du2dA = du2dv2*( β*rγ*(dfdA - dsdA) + (v2/β)*dβdf*dfdA )
    du2dB = du2dv2*( β*rγ*(dfdB - dsdB) + (v2/β)*dβdf*dfdB )
    du2dC = du2dv2*( β*rγ*(dfdC - dsdC) + (v2/β)*dβdf*dfdC )
    # Polarization sensitivity
    sinϕ, cosϕ = sincos(pazm)
    sinθ, cosθ = sincos(pelv)
    sinψ, cosψ = sincos(sazm)
    sinλ, cosλ = sincos(selv)
    sin2ψ, cos2ψ = (2.0*sinψ*cosψ, 2.0*(cosψ^2) - 1.0)
    tanλ = conditioned_tangent(sinλ, cosλ; e = 64)
    # The terms (∂ζ/∂m)*(u' - u''); Eq. A43 - A45 of VanderBeek et al. (GJI 2023)
    dζdA_Δu₁₂ = 0.5*(cosθ*tanλ*(sinψ*cosϕ + cosψ*sinϕ) - sinθ*sin2ψ)*(rγ - ((α/β)^2)*rη*cosΔ_2)/β
    dζdB_Δu₁₂ = 0.5*(sinθ*cos2ψ - cosθ*tanλ*(cosψ*cosϕ - sinψ*sinϕ))*(rγ - ((α/β)^2)*rη*cosΔ_2)/β
    dζdC_Δu₁₂ = sqrt(f)*cosθ*cosλ*(sinψ*cosϕ - cosψ*sinϕ)*(rγ - ((α/β)^2)*rη*cosΔ_2)/β

    return (du1dA, du1dB, du1dC), (du2dA, du2dB, du2dC), (dζdA_Δu₁₂, dζdB_Δu₁₂, dζdC_Δu₁₂), ζ
end
# Compute the partial derivatives ∂(fcos²x)/∂m for m = (A, B, C) where f is the anisotropic magnitude and x is
# the angle between the propagation direction and symmetry axis; Equations A37-A39 of VanderBeek et al. (GJI 2023)
function differentiate_symmetry_axis_product(ray_azm, ray_elv, sym_azm, sym_elv, f)
    # Compute angle cosines and sines
    sinϕ, cosϕ = sincos(ray_azm)
    sinθ, cosθ = sincos(ray_elv)
    sinψ, cosψ = sincos(sym_azm)
    sinλ, cosλ = sincos(sym_elv)
    # Trigonometric identities
    cos2ϕ = 2.0*(cosϕ^2) - 1.0
    sin2ϕ = 2.0*sinϕ*cosϕ
    sin2θ = 2.0*sinθ*cosθ
    cos2ψ = 2.0*(cosψ^2) - 1.0
    sin2ψ = 2.0*sinψ*cosψ
    # Define tanλ such that it does not go to infinity
    tanλ = conditioned_tangent(sinλ, cosλ; e = 64)
    # Partial derivatives with respect to the azimuthal and vertical components
    # Equations A37-A39 of VanderBeek et al. (GJI 2023)
    dsdA = (cos2ϕ + cos2ψ)*(cosθ^2) + (cosϕ*cosψ - sinϕ*sinψ)*tanλ*sin2θ
    dsdB = (sin2ϕ + sin2ψ)*(cosθ^2) + (sinϕ*cosψ + cosϕ*sinψ)*tanλ*sin2θ
    dsdC = sqrt(f)*( 4.0*sinλ*(sinθ^2) + 2.0*(cosϕ*cosψ + sinϕ*sinψ)*cosλ*sin2θ )
    # Note the factor of 0.5! A factor of 2 was included in Eq. A37-A39 that we do not want here
    return 0.5*dsdA, 0.5*dsdB, 0.5*dsdC
end
function conditioned_tangent(y, x; e = 64)
    k = 0.5*π
    λ = atan(y/x) # Place on interval [-pi/2,pi/2]
    λ = (2*((k^e)/((k^e) + (λ^e))) - 1)*λ
   return tan(λ)
end
# Compute the partial derivatives ∂f/∂m for m = (A, B, C) where the anisotropic magnitude, f = √(A² + B²) + C²
# Equations A33-A34 of VanderBeek et al. (GJI 2023)
function differentiate_anisotropic_strength(sym_azm, sym_elv, f)
    # Equations A33-A34 of VanderBeek et al. (GJI 2023)
    sin2ψ, cos2ψ = sincos(2.0*sym_azm)
    return cos2ψ, sin2ψ, 2.0*sqrt(f)*sin(sym_elv)
end

# Maps Sensitivity (i.e. observation partial derivative) to the appropriate Jacobian columns
function map_partial_to_jacobian!(Wj, Wv, Mesh::RegularGrid{G, 3}, x::NTuple{3, T}, dbdm, col_offset) where {G,T}
    mesh_index, weights = trilinear_weights(Mesh.x[1], Mesh.x[2], Mesh.x[3], x[1], x[2], x[3]; tf_extrapolate = false, scale = dbdm)
    for n in eachindex(mesh_index)
        push!(Wj, mesh_index[n] + col_offset - 1)
        push!(Wv, weights[n])
    end

    return nothing
end
# Accumulates weights for repeated indices
function accumlate_weights(Wn, Wv)
    # Unique indices in Wn
    n = sort(vec(Wn))
    unique!(n)
    # Create a dictionary that maps n-value to the n-index 
    D = Dict([(x, i) for (i, x) in enumerate(n)])
    # Accumulate weights
    v = zeros(eltype(Wv), length(n))
    for (i, k) in enumerate(Wn)
        j = D[k]
        v[j] += Wv[i]
    end

    return n, v
end



# ASSIGN PARAMETER SENSITIVITIES
# This is a realtive measure how well-samped is a parameter and is defined as
# the row-sum of the Jacobian-squared (RJSJ), specifically,
#   xⱼ = (∂b₁/∂mⱼ)² + (∂b₂/∂mⱼ)² + ... + (∂bₙ/∂mⱼ)²
# where mⱼ is the jᵗʰ-parameter and bᵢ is the iᵗʰ-observation.

# Assign RSJS for all parameters in a SeismicPerturbationModel
function fill_rsjs!(PerturbationModel::SeismicPerturbationModel, x)

    if ~isnothing(PerturbationModel.Velocity)
        fill_rsjs!(PerturbationModel.Velocity, x)
    end
    if ~isnothing(PerturbationModel.Interface)
        error("Interface RSJS not yet defined.")
    end
    if ~isnothing(PerturbationModel.Hypocenter)
        error("Hypocenter RSJS not yet defined.")
    end
    if ~isnothing(PerturbationModel.SourceStatics)
        fill_rsjs!(PerturbationModel.SourceStatics, x)
    end
    if ~isnothing(PerturbationModel.ReceiverStatics)
        fill_rsjs!(PerturbationModel.ReceiverStatics, x)
    end

    return nothing
end
# Assign RSJS for all parameters in a InverseSeismicVelocity
function fill_rsjs!(PerturbationModel::InverseSeismicVelocity, x)
    if ~isnothing(PerturbationModel.Isotropic)
        fill_rsjs!(PerturbationModel.Isotropic, x)
    end
    if ~isnothing(PerturbationModel.Anisotropic)
        fill_rsjs!(PerturbationModel.Anisotropic, x)
    end

    return nothing
end
# Assign RSJS for all parameters in a InverseIsotropicSlowness
function fill_rsjs!(PerturbationModel::InverseIsotropicSlowness, x)
    # Update P-wave Slowness Sensitivities (if not empty)
    if ~isnothing(PerturbationModel.Up)
        fill_rsjs!(PerturbationModel.Up, x)
    end
    # Update S-wave Slowness Sensitivities (if not empty)
    if ~isnothing(PerturbationModel.Us)
        fill_rsjs!(PerturbationModel.Us, x)
    end

    return nothing
end
# Assign RSJS for all parameters in a InverseAnisotropicVector
function fill_rsjs!(PerturbationModel::InverseAnisotropicVector, x)
    if ~isnothing(PerturbationModel.Orientations)
        fill_rsjs!(PerturbationModel.Orientations, x)
    end
    if ~isnothing(PerturbationModel.Fractions)
        fill_rsjs!(PerturbationModel.Fractions, x)
    end

    return nothing
end
# Assign RSJS for all parameters in a InverseIsotropicSlowness
function fill_rsjs!(PerturbationModel::InverseAzRadVector, x)
    if ~isnothing(PerturbationModel.A)
        fill_rsjs!(PerturbationModel.A, x)
    end
    if ~isnothing(PerturbationModel.B)
        fill_rsjs!(PerturbationModel.B, x)
    end
    if ~isnothing(PerturbationModel.C)
        fill_rsjs!(PerturbationModel.C, x)
    end

    return nothing
end
# Assign RSJS for a ParameterField
function fill_rsjs!(PerturbationModel::ParameterField, x)
    N = 1 + PerturbationModel.jcol[2] - PerturbationModel.jcol[1]
    copyto!(PerturbationModel.RSJS, 1, x, PerturbationModel.jcol[1], N)

    return nothing
end
# Assign RSJS for all SeismicStatics
function fill_rsjs!(PerturbationModel::SeismicStatics, x)
    # Loop over keys and extract Jacobian element
    for k in eachindex(PerturbationModel.jcol)
        PerturbationModel.RSJS[k] = x[PerturbationModel.jcol[k]]
    end

    return nothing
end



# BUILD CONSTRAINT EQUATIONS

# Build constraint equations for all parameters in a SeismicPerturbationModel
function build_constraints!(Aᵢ, Aⱼ, Aᵥ, b, PerturbationModel::SeismicPerturbationModel; row_offset = 0, tf_jac_scale = false)
    if ~isnothing(PerturbationModel.Velocity)
        row_offset = build_constraints!(Aᵢ, Aⱼ, Aᵥ, b, PerturbationModel.Velocity; row_offset = row_offset, tf_jac_scale = tf_jac_scale)
    end
    if ~isnothing(PerturbationModel.Interface)
        error("Interface regularisation not yet defined.")
    end
    if ~isnothing(PerturbationModel.Hypocenter)
        error("Hypocenter regularisation not yet defined.")
    end
    if ~isnothing(PerturbationModel.SourceStatics)
        row_offset = build_constraints!(Aᵢ, Aⱼ, Aᵥ, b, PerturbationModel.SourceStatics; row_offset = row_offset) # No Jacobian scaling for statics
    end
    if ~isnothing(PerturbationModel.ReceiverStatics)
        row_offset = build_constraints!(Aᵢ, Aⱼ, Aᵥ, b, PerturbationModel.ReceiverStatics; row_offset = row_offset) # No Jacobian scaling for statics
    end

    # Return row index of the last constraint equation
    return row_offset
end
# Build constraint equations for all parameters in a InverseSeismicVelocity
function build_constraints!(Aᵢ, Aⱼ, Aᵥ, b, PerturbationModel::InverseSeismicVelocity; row_offset = 0, tf_jac_scale = false)
    if ~isnothing(PerturbationModel.Isotropic)
        row_offset = build_constraints!(Aᵢ, Aⱼ, Aᵥ, b, PerturbationModel.Isotropic; row_offset = row_offset, tf_jac_scale = tf_jac_scale)
    end
    if ~isnothing(PerturbationModel.Anisotropic)
        row_offset = build_constraints!(Aᵢ, Aⱼ, Aᵥ, b, PerturbationModel.Anisotropic; row_offset = row_offset, tf_jac_scale = tf_jac_scale)
    end
    # Return row index of the last constraint equation
    return row_offset
end
# Build constraint equations for all parameters in a InverseIsotropicSlowness
function build_constraints!(Aᵢ, Aⱼ, Aᵥ, b, PerturbationModel::InverseIsotropicSlowness; row_offset = 0, tf_jac_scale = false)

    # P-slowness
    if ~isnothing(PerturbationModel.Up) && (PerturbationModel.coupling_option < 2)
        row_offset = build_constraints!(Aᵢ, Aⱼ, Aᵥ, b, PerturbationModel.Up; row_offset = row_offset, tf_jac_scale = tf_jac_scale)
    end
    # S-slowness
    if ~isnothing(PerturbationModel.Us) && (PerturbationModel.coupling_option < 2)
        row_offset = build_constraints!(Aᵢ, Aⱼ, Aᵥ, b, PerturbationModel.Us; row_offset = row_offset, tf_jac_scale = tf_jac_scale)
    end
    # Joint Inversion
    if (PerturbationModel.coupling_option > 1) && ~isnothing(ΔU.up) && ~isnothing(ΔU.us)
        error("Missing Coupling Constraints.")
        # row_offset = build_constraints!(Aᵢ, Aⱼ, Aᵥ, b, PerturbationModel.Up, PerturbationModel.Us; row_offset = row_offset, tf_jac_scale = tf_jac_scale)
    end

    # Return row index of the last constraint equation
    return row_offset
end
# Build constraint equations for all parameters in a InverseAnisotropicVector
function build_constraints!(Aᵢ, Aⱼ, Aᵥ, b, PerturbationModel::InverseAnisotropicVector; row_offset = 0, tf_jac_scale = false)
    if ~isnothing(PerturbationModel.Orientations)
        row_offset = build_constraints!(Aᵢ, Aⱼ, Aᵥ, b, PerturbationModel.Orientations; row_offset = row_offset, tf_jac_scale = tf_jac_scale)
    end
    if ~isnothing(PerturbationModel.Fractions)
        row_offset = build_constraints!(Aᵢ, Aⱼ, Aᵥ, b, PerturbationModel.Fractions; row_offset = row_offset, tf_jac_scale = tf_jac_scale)
    end

    # Return row index of the last constraint equation
    return row_offset
end
# Build constraint equations for all parameters in a InverseAzRadVector
function build_constraints!(Aᵢ, Aⱼ, Aᵥ, b, PerturbationModel::InverseAzRadVector; row_offset = 0, tf_jac_scale = false)
    # Some questions
    # 1. Individual or composit Jacobian scaling?
    # 1.1. Does composite scaling work when meshes for each component are different?
    # 2. Are seperate damping and smoothing weights for each component really necessary?
    if tf_jac_scale
        # Mean-squared sensitivity
        wa, _ = conditional_mean(PerturbationModel.A.RSJS; f = (x -> x > 0.0))
        wb, _ = conditional_mean(PerturbationModel.B.RSJS; f = (x -> x > 0.0))
        wc, _ = conditional_mean(PerturbationModel.C.RSJS; f = (x -> x > 0.0))
        # Composite squared-sensitivity to anisotropic strength
        wf = sqrt(wa + wb) + wc
        # Scale by mean uncertainty
        wa = wf*mean(PerturbationModel.A.uncertainty)
        wb = wf*mean(PerturbationModel.B.uncertainty)
        wc = sqrt(wf)*mean(PerturbationModel.C.uncertainty)
    else
        wa, wb, wc = (1.0, 1.0, 1.0)
    end

    if ~isnothing(PerturbationModel.A)
        # Kludge to prevent weighting contraints by prior model (do not want to do this for anisotropy)
        m0 = range(start = 1.0, stop = 1.0, length = length(PerturbationModel.A.m0))
        # Weights
        μ = wa*PerturbationModel.A.wdamp[1]
        println("A Damping Weight = "*string(μ/PerturbationModel.A.uncertainty[1]))
        λ1 = wf*PerturbationModel.A.wsmooth[1]
        λ2 = wf*PerturbationModel.A.wsmooth[2]
        λ3 = wf*PerturbationModel.A.wsmooth[3]
        println("A Smoothing Weight = "*string(λ1))
        # Build damping constraints--always penalize incremental since this is non-linear
        row_offset = damping_constraint!(Aᵢ, Aⱼ, Aᵥ, b, μ, m0, PerturbationModel.A.uncertainty, PerturbationModel.A.dm,
        false; row_offset = row_offset, col_offset = PerturbationModel.A.jcol[1] - 1)
        # Build smoothing constraints
        row_offset = smoothing_constraint!(Aᵢ, Aⱼ, Aᵥ, b, (λ1/3.0, λ2/3.0, λ3/3.0), PerturbationModel.A.Mesh, m0, PerturbationModel.A.dm,
        PerturbationModel.A.tf_smooth_cumulative[1]; row_offset = row_offset, col_offset = PerturbationModel.A.jcol[1] - 1)
    end
    if ~isnothing(PerturbationModel.B)
        # Kludge to prevent weighting contraints by prior model (do not want to do this for anisotropy)
        m0 = range(start = 1.0, stop = 1.0, length = length(PerturbationModel.B.m0))
        # Weights
        μ = wb*PerturbationModel.B.wdamp[1]
        println("B Damping Weight = "*string(μ/PerturbationModel.B.uncertainty[1]))
        λ1 = wf*PerturbationModel.B.wsmooth[1]
        λ2 = wf*PerturbationModel.B.wsmooth[2]
        λ3 = wf*PerturbationModel.B.wsmooth[3]
        println("B Smoothing Weight = "*string(λ1))
        # Build damping constraints--always penalize incremental since this is non-linear
        row_offset = damping_constraint!(Aᵢ, Aⱼ, Aᵥ, b, μ, m0, PerturbationModel.B.uncertainty, PerturbationModel.B.dm,
        false; row_offset = row_offset, col_offset = PerturbationModel.B.jcol[1] - 1)
        # Build smoothing constraints
        row_offset = smoothing_constraint!(Aᵢ, Aⱼ, Aᵥ, b, (λ1/3.0, λ2/3.0, λ3/3.0), PerturbationModel.B.Mesh, m0, PerturbationModel.B.dm,
        PerturbationModel.B.tf_smooth_cumulative[1]; row_offset = row_offset, col_offset = PerturbationModel.B.jcol[1] - 1)
    end
    if ~isnothing(PerturbationModel.C)
        # Kludge to prevent weighting contraints by prior model (do not want to do this for anisotropy)
        m0 = range(start = 1.0, stop = 1.0, length = length(PerturbationModel.C.m0))
        # Weights
        μ = wc*PerturbationModel.C.wdamp[1]
        println("C Damping Weight = "*string(μ/PerturbationModel.C.uncertainty[1]))
        λ1 = sqrt(wf)*PerturbationModel.C.wsmooth[1]
        λ2 = sqrt(wf)*PerturbationModel.C.wsmooth[2]
        λ3 = sqrt(wf)*PerturbationModel.C.wsmooth[3]
        println("C Smoothing Weight = "*string(λ1))
        # Build damping constraints--always penalize incremental since this is non-linear
        row_offset = damping_constraint!(Aᵢ, Aⱼ, Aᵥ, b, μ, m0, PerturbationModel.C.uncertainty, PerturbationModel.C.dm,
        false; row_offset = row_offset, col_offset = PerturbationModel.C.jcol[1] - 1)
        # Build smoothing constraints
        row_offset = smoothing_constraint!(Aᵢ, Aⱼ, Aᵥ, b, (λ1/3.0, λ2/3.0, λ3/3.0), PerturbationModel.C.Mesh, m0, PerturbationModel.C.dm,
        PerturbationModel.C.tf_smooth_cumulative[1]; row_offset = row_offset, col_offset = PerturbationModel.C.jcol[1] - 1)
    end

    # Total anisotropic strength
    if PerturbationModel.A.tf_damp_cumulative[1]
        μ = (wa + wb + (wc^2))/3.0 # For uniform fraction uncertainties, μ = wf*σf
        μ = μ*((PerturbationModel.A.wdamp[1] + PerturbationModel.B.wdamp[1] + PerturbationModel.C.wdamp[1])/3.0)
        println("Cumulative Anisotropic Damping Weight = "*string(μ/PerturbationModel.A.uncertainty[1]))
        row_offset = min_total_anisotropic_fraction!(Aᵢ, Aⱼ, Aᵥ, b, μ, PerturbationModel; row_offset = row_offset)
    end

    # Return row index of the last constraint equation
    return row_offset
end
# Build constraint equations to minimise the cumulative perturbation to the anisotropic magnitude
function min_total_anisotropic_fraction!(Aᵢ, Aⱼ, Aᵥ, b, μ, PerturbationModel::InverseAzRadVector; row_offset = 0)
    if PerturbationModel.A.Mesh.Geometry != PerturbationModel.B.Mesh.Geometry ||
        PerturbationModel.A.Mesh.Geometry != PerturbationModel.C.Mesh.Geometry ||
        PerturbationModel.A.Mesh.x != PerturbationModel.B.Mesh.x ||
        PerturbationModel.A.Mesh.x != PerturbationModel.C.Mesh.x
        error("Inconsistent meshes!")
    end

    npar = length(PerturbationModel.A.Mesh)
    col_offset = (PerturbationModel.A.jcol[1] - 1, PerturbationModel.B.jcol[1] - 1, PerturbationModel.C.jcol[1] - 1)
    for n = 1:npar
        # Increment row counter
        row_offset += 1
        # Current symmetry axis parameters
        sazm = 0.5*atan(PerturbationModel.B.m0[n] + PerturbationModel.B.dm[n],
        PerturbationModel.A.m0[n] + PerturbationModel.A.dm[n])
        sin2ψ, cos2ψ = sincos(2.0*sazm)
        c = PerturbationModel.C.m0[n] + PerturbationModel.C.dm[n]
        # Total damping weights
        σf = (PerturbationModel.A.uncertainty[n] + PerturbationModel.B.uncertainty[n] + (PerturbationModel.C.uncertainty[n]^2))/3.0
        wa = μ*cos2ψ/σf
        wb = μ*sin2ψ/σf
        wc = 2.0*μ*c/σf
        # Add constraint
        append!(Aᵢ, (row_offset, row_offset, row_offset))
        append!(Aⱼ, (n + col_offset[1], n + col_offset[2], n + col_offset[3]))
        append!(Aᵥ, (wa, wb, wc))
        # Add condition
        df = sqrt( (PerturbationModel.A.dm[n]^2) + (PerturbationModel.B.dm[n]^2) ) + (PerturbationModel.C.dm[n]^2)
        push!(b, -μ*df/σf)
    end

    return row_offset
end
# Build constraint equations for a ParameterField
function build_constraints!(Aᵢ, Aⱼ, Aᵥ, b, PerturbationModel::ParameterField; row_offset = 0, tf_jac_scale = false)
    # Weight adjustment for Jacobian scaling
    if tf_jac_scale
        # Compute the RMS sensitivity of the sampled parameters
        w, _ = conditional_mean(PerturbationModel.RSJS; f = (x -> x > 0.0))
        w = sqrt(w)
        m̅ = mean(PerturbationModel.m0)
        σ̅ = mean(PerturbationModel.uncertainty)
    else
        w = 1.0
        m̅ = 1.0
        σ̅ = 1.0
    end
    # Scale damping and smoothing multipliers
    μ = w*σ̅*m̅*PerturbationModel.wdamp[1]
    λ1 = w*m̅*PerturbationModel.wsmooth[1]
    λ2 = w*m̅*PerturbationModel.wsmooth[2]
    λ3 = w*m̅*PerturbationModel.wsmooth[3]

    # Build damping constraints
    row_offset = damping_constraint!(Aᵢ, Aⱼ, Aᵥ, b, μ, PerturbationModel.m0, PerturbationModel.uncertainty, PerturbationModel.dm,
    PerturbationModel.tf_damp_cumulative[1]; row_offset = row_offset, col_offset = PerturbationModel.jcol[1] - 1)

    # Build smoothing constraints
    row_offset = smoothing_constraint!(Aᵢ, Aⱼ, Aᵥ, b, (λ1/3.0, λ2/3.0, λ3/3.0), PerturbationModel.Mesh, PerturbationModel.m0, PerturbationModel.dm,
    PerturbationModel.tf_smooth_cumulative[1]; row_offset = row_offset, col_offset = PerturbationModel.jcol[1] - 1)

    return row_offset
end
# Build constraint equations for all SeismicStatics
function build_constraints!(Aᵢ, Aⱼ, Aᵥ, b, PerturbationModel::SeismicStatics; row_offset = 0)
    # Loop over static keys
    for kj in eachindex(PerturbationModel.jcol) # Returns each dictionary key
        # Define the damping parameter key (i.e. Observation-Phase pair)
        ki = (kj[2], kj[3])
        # Define the constraint if the damping is > 0
        if PerturbationModel.wdamp[ki] > 0.0
            row_offset += 1 # Increment row counter
            j = PerturbationModel.jcol[kj] # Jacobian index
            w = PerturbationModel.wdamp[ki]/PerturbationModel.uncertainty[kj] # Define the weight
            # Append constraint
            push!(Aᵢ, row_offset)
            push!(Aⱼ, j)
            push!(Aᵥ, w)
            # Constraint value
            if PerturbationModel.tf_damp_cumulative[ki]
                dm = PerturbationModel.wdamp[ki]*PerturbationModel.dm[kj]/PerturbationModel.uncertainty[kj]
                push!(b, -dm)
            else
                push!(b, 0.0)
            end
        end
    end

    return row_offset
end

# Build constraint equation to minimise the perturbation to a single parameter
function damping_constraint!(Aᵢ, Aⱼ, Aᵥ, b, μ, m₀, σₘ, Δm, tf_damp_cumulative; row_offset = 0, col_offset = 0)
    # Damping constraint--minimise sum of squared parameter perturbations
    npar = length(m₀)
    for n in 1:npar
        # Increment row counter
        row_offset += 1
        # Add constraint
        push!(Aᵢ, row_offset)
        push!(Aⱼ, n + col_offset)
        push!(Aᵥ, μ/(σₘ[n].*m₀[n]))
        if tf_damp_cumulative
            push!(b, -μ*Δm[n]/(σₘ[n].*m₀[n]))
        else
            push!(b, 0.0)
        end
    end

    return row_offset
end

# Build constraint equations to enforce spatially smooth perturbations
function smoothing_constraint!(Aᵢ, Aⱼ, Aᵥ, b, λ, Mesh::RegularGrid{G, 3, R}, m₀, dm, tf_smooth_cumulative; row_offset = 0, col_offset = 0) where {G,R}
    (ni, nj, nk) = size(Mesh)
    for i in 1:ni, j in 1:nj, k in 1:nk
        # Increment row counter
        row_offset += 1
        # Laplacian smoothing weights (3 per node in each dimension)
        i_index, i_coeff = laplacian_weights(ni, i)
        j_index, j_coeff = laplacian_weights(nj, j)
        k_index, k_coeff = laplacian_weights(nk, k)

        # First-dimension
        jcol_1 = subscripts_to_index((ni, nj, nk), (i_index[1], j, k))
        jcol_2 = subscripts_to_index((ni, nj, nk), (i_index[2], j, k))
        jcol_3 = subscripts_to_index((ni, nj, nk), (i_index[3], j, k))
        # Second-dimension
        jcol_4 = subscripts_to_index((ni, nj, nk), (i, j_index[1], k))
        jcol_5 = subscripts_to_index((ni, nj, nk), (i, j_index[2], k))
        jcol_6 = subscripts_to_index((ni, nj, nk), (i, j_index[3], k))
        # Third-dimension
        jcol_7 = subscripts_to_index((ni, nj, nk), (i, j, k_index[1]))
        jcol_8 = subscripts_to_index((ni, nj, nk), (i, j, k_index[2]))
        jcol_9 = subscripts_to_index((ni, nj, nk), (i, j, k_index[3]))
        # Append laplacian indices and weights for the n'th row (constraint)
        append!(Aᵢ, (row_offset, row_offset, row_offset, row_offset, row_offset, row_offset, row_offset, row_offset, row_offset))
        append!(Aⱼ, (col_offset + jcol_1, col_offset + jcol_2, col_offset + jcol_3,
        col_offset + jcol_4, col_offset + jcol_5, col_offset + jcol_6,
        col_offset + jcol_7, col_offset + jcol_8, col_offset + jcol_9))
        append!(Aᵥ, (λ[1]*i_coeff[1]/m₀[jcol_1], λ[1]*i_coeff[2]/m₀[jcol_2], λ[1]*i_coeff[3]/m₀[jcol_3],
        λ[2]*j_coeff[1]/m₀[jcol_4], λ[2]*j_coeff[2]/m₀[jcol_5], λ[2]*j_coeff[3]/m₀[jcol_6],
        λ[3]*k_coeff[1]/m₀[jcol_7], λ[3]*k_coeff[2]/m₀[jcol_8], λ[3]*k_coeff[3]/m₀[jcol_9]))

        if tf_smooth_cumulative
            b_jrow = (λ[1]*i_coeff[1]*dm[jcol_1]/m₀[jcol_1]) + (λ[1]*i_coeff[2]*dm[jcol_2]/m₀[jcol_2]) + (λ[1]*i_coeff[3]*dm[jcol_3]/m₀[jcol_3]) +
            (λ[2]*j_coeff[1]*dm[jcol_4]/m₀[jcol_4]) + (λ[2]*j_coeff[2]*dm[jcol_5]/m₀[jcol_5]) + (λ[2]*j_coeff[3]*dm[jcol_6]/m₀[jcol_6]) +
            (λ[3]*k_coeff[1]*dm[jcol_7]/m₀[jcol_7]) + (λ[3]*k_coeff[2]*dm[jcol_8]/m₀[jcol_8]) + (λ[3]*k_coeff[3]*dm[jcol_9]/m₀[jcol_9])
            push!(b, -b_jrow)
        else
            push!(b, 0.0)
        end
    end

    return row_offset
end

# Return finite-difference coefficients for the 1D Laplacian
function laplacian_weights(n, i)
    if (i > 1) && (i < n)
        j = (i-1, i, i+1)
    else
        if i == 1
            j = (i, i+1, i+2)
        elseif i == n
            j = (i, i-1, i-2)
        else
            error("Index out of range!")
        end
    end
    w = (0.5, -1.0, 0.5)

    return j, w
end

# Compute mean in which only values satisfying some condition are considered
function conditional_mean(x; f = (x -> x > 0.0))
    sum_x = zero(eltype(x))
    n = 0
    for xi in x
        if f(xi)
            sum_x += xi
            n += 1
        end
    end
    if n > 0
        sum_x /= n
    end

    return sum_x, n
end



# PERTURBATION MODEL UPDATE

# Update all perturbation fields in a SeismicPerturbationModel
function update_parameters!(PerturbationModel::SeismicPerturbationModel, x)
    if ~isnothing(PerturbationModel.Velocity)
        update_parameters!(PerturbationModel.Velocity, x)
    end
    if ~isnothing(PerturbationModel.Interface)
        error("Interface parameter update not yet defined.")
    end
    if ~isnothing(PerturbationModel.Hypocenter)
        error("Hypocenter parameter update not yet defined.")
    end
    if ~isnothing(PerturbationModel.SourceStatics)
        update_parameters!(PerturbationModel.SourceStatics, x)
    end
    if ~isnothing(PerturbationModel.ReceiverStatics)
        update_parameters!(PerturbationModel.ReceiverStatics, x)
    end

    return nothing
end
# Update all perturbation fields in a InverseSeismicVelocity
function update_parameters!(PerturbationModel::InverseSeismicVelocity, x)
    if ~isnothing(PerturbationModel.Isotropic)
        update_parameters!(PerturbationModel.Isotropic, x)
    end
    if ~isnothing(PerturbationModel.Anisotropic)
        update_parameters!(PerturbationModel.Anisotropic, x)
    end

    return nothing
end
# Update all perturbation fields in a InverseIsotropicSlowness
function update_parameters!(PerturbationModel::InverseIsotropicSlowness, x)
    if ~isnothing(PerturbationModel.Up)
        update_parameters!(PerturbationModel.Up, x)
    end

    if ~isnothing(PerturbationModel.Us)
        update_parameters!(PerturbationModel.Us, x)
    end

    return nothing
end
# Update all perturbation fields in a InverseAnisotropicVector
function update_parameters!(PerturbationModel::InverseAnisotropicVector, x)
    if ~isnothing(PerturbationModel.Orientations)
        update_parameters!(PerturbationModel.Orientations, x)
    end
    if ~isnothing(PerturbationModel.Fractions)
        update_parameters!(PerturbationModel.Fractions, x)
    end

    return nothing
end
# Update all perturbation fields in a InverseAzRadVector
function update_parameters!(PerturbationModel::InverseAzRadVector, x)
    if ~isnothing(PerturbationModel.A)
        update_parameters!(PerturbationModel.A, x)
    end
    if ~isnothing(PerturbationModel.B)
        update_parameters!(PerturbationModel.B, x)
    end
    if ~isnothing(PerturbationModel.C)
        update_parameters!(PerturbationModel.C, x)
    end

    return nothing
end
# Update all perturbation fields in a ParameterField
function update_parameters!(PerturbationModel::ParameterField, x)
    # Copy the columns of the solution vector to the appropriate incremental perturbations field
    N = 1 + PerturbationModel.jcol[2] - PerturbationModel.jcol[1]
    copyto!(PerturbationModel.ddm, 1, x, PerturbationModel.jcol[1], N)
    # Update the cumulative perturbation field
    PerturbationModel.dm .+= PerturbationModel.ddm

    return nothing
end
# Update all SeismicStatics
function update_parameters!(PerturbationModel::SeismicStatics, x)
    for k in eachindex(PerturbationModel.jcol)
        PerturbationModel.ddm[k] = x[PerturbationModel.jcol[k]]
        PerturbationModel.dm[k] += x[PerturbationModel.jcol[k]]
    end

    return nothing
end



# UPDATE KERNEL PARAMETERS

# Update all SeismicPerturbationModel parameters in a Vector of ObservableKernel
function update_kernel!(Kernels::Vector{<:ObservableKernel}, PerturbationModel::SeismicPerturbationModel)
    if ~isnothing(PerturbationModel.Velocity)
        println("Updating Kernels -> Velocity")
        update_kernel!(Kernels, PerturbationModel.Velocity)
    end
    if ~isnothing(PerturbationModel.Interface)
        error("Interface kernel update not yet defined.")
    end
    if ~isnothing(PerturbationModel.Hypocenter)
        error("Hypocenter kernel update not yet defined.")
    end
    if ~isnothing(PerturbationModel.ReceiverStatics)
        println("Updating Kernels -> ReceiverStatics")
        update_kernel!(Kernels, PerturbationModel.ReceiverStatics)
    end
    if ~isnothing(PerturbationModel.SourceStatics)
        println("Updating Kernels -> SourceStatics")
        update_kernel!(Kernels, PerturbationModel.SourceStatics)
    end

    return nothing
end
# Update a InversionParameter in a Vector of ObservableKernel
function update_kernel!(Kernels::Vector{<:ObservableKernel}, PerturbationModel::InversionParameter)
    println("Updating Kernels -> "*string(typeof(PerturbationModel)))
    for a_Kernel in Kernels
        update_kernel!(a_Kernel, PerturbationModel)
    end

    return nothing
end
# Update InverseSeismicVelocity parameters in an ObservableKernel
function update_kernel!(Kernel::ObservableKernel, PerturbationModel::InverseSeismicVelocity)
    # Convert kernel parameterisation to be consistent with inversion parameterisation
    convert_parameters!(Kernel.Parameters, PerturbationModel)
    # Update converted kernel parameters
    if ~isnothing(PerturbationModel.Isotropic)
        update_kernel!(Kernel, PerturbationModel.Isotropic)
    end
    if ~isnothing(PerturbationModel.Anisotropic)
        update_kernel!(Kernel, PerturbationModel.Anisotropic)
    end
    # Revert back to original kernel parameterisation
    revert_parameters!(Kernel.Parameters, PerturbationModel)

    return nothing
end
function convert_parameters!(::IsotropicVelocity, ::InverseSeismicVelocity)
    return nothing
end
function revert_parameters!(::IsotropicVelocity, ::InverseSeismicVelocity)
    return nothing
end
function convert_parameters!(Parameters::HexagonalVectoralVelocity, ::InverseSeismicVelocity)
    # Convert symmetry axis (i.e. Thomsen's reference velocities) to isotropic velocities
    for i in eachindex(Parameters.f)
        vip, vis = return_isotropic_velocities(Parameters, i)
        if ~isnothing(Parameters.α)
            Parameters.α[i] = vip
        end
        if ~isnothing(Parameters.β)
            Parameters.β[i] = vis
        end
    end

    return nothing
end
function revert_parameters!(Parameters::HexagonalVectoralVelocity, ::InverseSeismicVelocity)
    # Convert symmetry axis (i.e. Thomsen's reference velocities) to isotropic velocities
    for i in eachindex(Parameters.f)
        # Assumed that the α and β fields are currently storing the isotropic P- and S-velocity
        vip, vis, _ = return_thomsen_parameters(Parameters, i)
        α, β = return_reference_velocities(Parameters, vip, vis, i)
        if ~isnothing(Parameters.α)
            Parameters.α[i] = α
        end
        if ~isnothing(Parameters.β)
            Parameters.β[i] = β
        end
    end

    return nothing
end
# Update InverseIsotropicSlowness parameters in an ObservableKernel
function update_kernel!(Kernel::ObservableKernel, PerturbationModel::InverseIsotropicSlowness)
    # Get pointers to velocity fields
    vp_field, vs_field = return_velocity_fields(Kernel.Parameters)

    # Update P-Velocities
    if ~isnothing(vp_field) && ~isnothing(PerturbationModel.Up)
        # Convert velocity to slowness
        vp_field .= inv.(vp_field)
        # Interpolate and add slowness perturbations
        update_kernel_field!(vp_field, Kernel, PerturbationModel.Up; tf_extrapolate = false, tf_harmonic = false)
        # Convert back to velocity
        vp_field .= inv.(vp_field)
    end

    # Update S-velocities
    if ~isnothing(vs_field) && ~isnothing(PerturbationModel.Us)
        if PerturbationModel.coupling_option == 0
            # Convert velocity to slowness
            vs_field .= inv.(vs_field)
            # Interpolate and add slowness perturbations
            update_kernel_field!(vs_field, Kernel, PerturbationModel.Us; tf_extrapolate = false, tf_harmonic = false)
            # Convert back to velocity
            vs_field .= inv.(vs_field)
        elseif PerturbationModel.coupling_option == 1
            # Inversion for S-to-P slowness ratio. How to do this update?
            #   us = r*up -> Dus = r*(dus/dup) + (dus/dr)*up
            #   us = (r + dr)*(up + dup) = r*up + r*dup + dr*up + dr*dup = us + r*dup + dr*up + dr*dup
            for (i, xq) in enumerate(Kernel.coordinates)
                # Convert global kernel coordinates to the local mesh coordinates
                xl, yl, zl = global_to_local(xq[1], xq[2], xq[3], PerturbationModel.Us.Mesh.Geometry)
                # Interpolate starting S-to-P slowness ratio
                r = linearly_interpolate(PerturbationModel.Us.Mesh.x, PerturbationModel.Us.m0, (xl, yl, zl);
                tf_extrapolate = false, tf_harmonic = false)
                # Interpolate the cumulative S-to-P slowness ratio perturbation
                dr = linearly_interpolate(PerturbationModel.Us.Mesh.x, PerturbationModel.Us.dm, (xl, yl, zl);
                tf_extrapolate = false, tf_harmonic = false)
                # Interpolate the incremental S-to-P slowness ratio perturbation
                ddr = linearly_interpolate(PerturbationModel.Us.Mesh.x, PerturbationModel.Us.ddm, (xl, yl, zl);
                tf_extrapolate = false, tf_harmonic = false)
                # Interpolate incremental perturbation to P-slowness
                xl, yl, zl = global_to_local(xq[1], xq[2], xq[3], PerturbationModel.Up.Mesh.Geometry)
                ddup = linearly_interpolate(PerturbationModel.Up.Mesh.x, PerturbationModel.Up.ddm, (xl, yl, zl);
                tf_extrapolate = false, tf_harmonic = false)
                # P-slowness at previous iteration
                r = r + dr - ddr
                up = 1.0/(r*vs_field[i])
                # S-velocity update
                vs_field[i] = 1.0/((1.0/vs_field[i]) + r*ddup + ddr*up + ddr*ddup)
            end
        else
            error("Coupling option not implemented!")
        end
    end

    return nothing
end
# Update InverseAnisotropicVector parameters in an ObservableKernel
function update_kernel!(Kernel::ObservableKernel, PerturbationModel::InverseAnisotropicVector)
    if ~isnothing(PerturbationModel.Orientations)
        update_kernel!(Kernel, PerturbationModel.Orientations)
    end
    if ~isnothing(PerturbationModel.Fractions)
        update_kernel!(Kernel, PerturbationModel.Fractions)
    end

    return nothing
end
# Update InverseAzRadVector parameters in a HexagonalVectoralVelocity-Parameterized ObservableKernel
function update_kernel!(Kernel::ObservableKernel{B, P}, PerturbationModel::InverseAzRadVector) where {B, P <: HexagonalVectoralVelocity}
    # Check that Mesh geometries are consistent
    if (PerturbationModel.A.Mesh.Geometry != PerturbationModel.B.Mesh.Geometry) ||
        (PerturbationModel.A.Mesh.Geometry != PerturbationModel.C.Mesh.Geometry)
        error("Inconsistent mesh geometries!")
    end

    # Update kernel elements
    for (i, x_global) in enumerate(Kernel.coordinates)
        # Coordinate System Conversions
        x_local = global_to_local(x_global[1], x_global[2], x_global[3], PerturbationModel.A.Mesh.Geometry)
        # Symmetry axis orientations in the local geographic coordinate system
        sazm, selv = (Kernel.Parameters.azimuth[i], Kernel.Parameters.elevation[i])

        # Interpolate incremental perturbations
        dda = linearly_interpolate(PerturbationModel.A.Mesh.x, PerturbationModel.A.ddm, x_local;
        tf_extrapolate = false, tf_harmonic = false)
        ddb = linearly_interpolate(PerturbationModel.B.Mesh.x, PerturbationModel.B.ddm, x_local;
        tf_extrapolate = false, tf_harmonic = false)
        ddc = linearly_interpolate(PerturbationModel.C.Mesh.x, PerturbationModel.C.ddm, x_local;
        tf_extrapolate = false, tf_harmonic = false)
        # Compute new AzRad vector parameters
        f, sazm, selv = update_azrad_vector(Kernel.Parameters.f[i], sazm, selv, dda, ddb, ddc)

        # Update kernel anisotropy parameters
        Kernel.Parameters.f[i] = f
        Kernel.Parameters.azimuth[i] = sazm
        Kernel.Parameters.elevation[i] = selv
    end

    return nothing
end
function update_azrad_vector(f, sazm, selv, da, db, dc)
    # Current symmetry axis trigonometric factors
    sin2ψ, cos2ψ = sincos(2.0*sazm)
    sinλ, cosλ = sincos(selv)
    # Compute perturbed AzRad vector components
    a = f*cos2ψ*(cosλ^2) + da
    b = f*sin2ψ*(cosλ^2) + db
    c = sqrt(f)*sinλ + dc
    # Compute perturbed anisotropic strength and orientations 
    f = sqrt((a^2) + (b^2)) + (c^2)
    ψ = 0.5*atan(b, a)
    λ = atan(c, sqrt(sqrt((a^2) + (b^2))))

    return f, ψ, λ
end
# Update SeismicStatics parameters in an ObservableKernel
function update_kernel!(Kernel::ObservableKernel, PerturbationModel::SeismicStatics)
    static_type = typeof(PerturbationModel)
    if static_type <: SeismicSourceStatics
        kiop = (Kernel.Observation.source_id, nameof(typeof(Kernel.Observation)), nameof(typeof(Kernel.Observation.Phase)))
    elseif static_type <: SeismicReceiverStatics
        kiop = (Kernel.Observation.receiver_id, nameof(typeof(Kernel.Observation)), nameof(typeof(Kernel.Observation.Phase)))
    else
        error("Unrecognized static type!")
    end
    
    if haskey(PerturbationModel.ddm, kiop)
        Kernel.static[1] += PerturbationModel.ddm[kiop]
    end

    return nothing
end

# Adds incremental perturbations in a ParameterField to an ObservableKernel
function update_kernel_field!(outfield, Kernel::ObservableKernel, PerturbationModel::ParameterField; tf_extrapolate = false, tf_harmonic = false)
    for (i, xq) in enumerate(Kernel.coordinates)
        # Convert global kernel coordinates to the local mesh coordinates
        xl, yl, zl = global_to_local(xq[1], xq[2], xq[3], PerturbationModel.Mesh.Geometry)
        # Interpolate perturbation
        ddm_q = linearly_interpolate(PerturbationModel.Mesh.x, PerturbationModel.ddm, (xl, yl, zl);
        tf_extrapolate = tf_extrapolate, tf_harmonic = tf_harmonic)
        # Add perturbation to kernel parameter
        outfield[i] += ddm_q
    end

    return nothing
end


# UPDATE FORWARD MODEL
# NOTE! As implemented, cumulative perturbations are being added to the forward model.
# Consequently, this update is assumed to be performed only once. However, if multiple
# iterations involving re-computing kerenls are performed, this update will be incorrect!

# Update all SeismicPerturbationModel parameters in a PsiModel
function update_model!(Model::PsiModel, PerturbationModel::SeismicPerturbationModel)
    if ~isnothing(PerturbationModel.Velocity)
        update_model!(Model, PerturbationModel.Velocity)
    end
    if ~isnothing(PerturbationModel.Interface)
        error("Interface model update not yet defined.")
    end
    if ~isnothing(PerturbationModel.Hypocenter)
        error("Hypocenter model update not yet defined.")
    end
    if ~isnothing(PerturbationModel.SourceStatics)
        update_model!(Model, PerturbationModel.SourceStatics)
    end
    if ~isnothing(PerturbationModel.ReceiverStatics)
        update_model!(Model, PerturbationModel.ReceiverStatics)
    end

    return nothing
end
# Update all InverseSeismicVelocity parameters in a PsiModel
function update_model!(Model::PsiModel, PerturbationModel::InverseSeismicVelocity)
    # Convert model parameterisation to be consistent with inversion parameterisation
    convert_parameters!(Model.Parameters, PerturbationModel)
    # Update converted kernel parameters
    if ~isnothing(PerturbationModel.Isotropic)
        update_model!(Model, PerturbationModel.Isotropic)
    end
    if ~isnothing(PerturbationModel.Anisotropic)
        update_model!(Model, PerturbationModel.Anisotropic)
    end
    # Revert back to original kernel parameterisation
    revert_parameters!(Model.Parameters, PerturbationModel)

    return nothing
end
# Update all InverseIsotropicSlowness parameters in a PsiModel
function update_model!(Model::PsiModel, PerturbationModel::InverseIsotropicSlowness)
    # Get pointers to velocity fields
    vp_field, vs_field = return_velocity_fields(Model.Parameters)

    if ~isnothing(PerturbationModel.Up)
        # Convert to slowness
        vp_field .= inv.(vp_field)
        # Add slowness perturbations
        add_perturbations!(vp_field, PerturbationModel.Up.Mesh, PerturbationModel.Up.dm, Model.Mesh;
        tf_extrapolate = false, tf_harmonic = false)
        # Convert back to velocity
        vp_field .= inv.(vp_field)
    end

    if ~isnothing(PerturbationModel.Us)
        if PerturbationModel.coupling_option == 0
            # Convert to slowness
            vs_field .= inv.(vs_field)
            # Add slowness perturbations
            add_perturbations!(vs_field, PerturbationModel.Us.Mesh, PerturbationModel.Us.dm, Model.Mesh;
            tf_extrapolate = false, tf_harmonic = false)
            # Convert back to velocity
            vs_field .= inv.(vs_field)
        else
            error("Model update not implemented!")
        end
    end

    return nothing
end
# Update InverseAnisotropicVector parameters in an PsiModel
function update_model!(Model::PsiModel, PerturbationModel::InverseAnisotropicVector)
    if ~isnothing(PerturbationModel.Orientations)
        update_model!(Model, PerturbationModel.Orientations)
    end
    if ~isnothing(PerturbationModel.Fractions)
        update_model!(Model, PerturbationModel.Fractions)
    end

    return nothing
end
# Update all InverseAzRadVector parameters in a PsiModel with HexagonalVectoralVelocity parameters
function update_model!(Model::PsiModel{<:HexagonalVectoralVelocity}, PerturbationModel::InverseAzRadVector)
    # Check that Mesh geometries are consistent
    if (PerturbationModel.A.Mesh.Geometry != PerturbationModel.B.Mesh.Geometry) ||
        (PerturbationModel.A.Mesh.Geometry != PerturbationModel.C.Mesh.Geometry)
        error("Inconsistent mesh geometries!")
    end

    # Update model elements
    for (k, qz) in enumerate(Model.Mesh.x[3])
        for (j, qy) in enumerate(Model.Mesh.x[2])
            for (i, qx) in enumerate(Model.Mesh.x[1])
                # Interpolate cumulative perturbations
                da = linearly_interpolate(PerturbationModel.A.Mesh.x, PerturbationModel.A.dm, (qx, qy, qz);
                tf_extrapolate = false, tf_harmonic = false)
                db = linearly_interpolate(PerturbationModel.B.Mesh.x, PerturbationModel.B.dm, (qx, qy, qz);
                tf_extrapolate = false, tf_harmonic = false)
                dc = linearly_interpolate(PerturbationModel.C.Mesh.x, PerturbationModel.C.dm, (qx, qy, qz);
                tf_extrapolate = false, tf_harmonic = false)
                # Compute new AzRad vector parameters
                f, sazm, selv = update_azrad_vector(Model.Parameters.f[i,j,k], Model.Parameters.azimuth[i,j,k],
                Model.Parameters.elevation[i,j,k], da, db, dc)

                # Update kernel anisotropy parameters
                Model.Parameters.f[i,j,k] = f
                Model.Parameters.azimuth[i,j,k] = sazm
                Model.Parameters.elevation[i,j,k] = selv
            end
        end
    end

    return nothing
end
# Update SeismicSourceStatics in a PsiModel
function update_model!(Model::PsiModel, PerturbationModel::SeismicSourceStatics)
    for k in eachindex(PerturbationModel.dm)
        if haskey(Model.Sources.statics, k)
            # Accumulate static
            Model.Sources.statics[k] += PerturbationModel.dm[k]
        else
            # Add static
            Model.Sources.statics[k] = PerturbationModel.dm[k]
        end
    end

    return nothing
end
# Update SeismicReceiverStatics in a PsiModel
function update_model!(Model::PsiModel, PerturbationModel::SeismicReceiverStatics)
    for k in eachindex(PerturbationModel.dm)
        if haskey(Model.Receivers.statics, k)
            Model.Receivers.statics[k] += PerturbationModel.dm[k]
        else
            Model.Receivers.statics[k] = PerturbationModel.dm[k]
        end
    end

    return nothing
end



# NEW INTERPOLATION ROUTINES

# Interpolate between meshes -- Same as interpolate_field! but adds interpolated value to outfield instead of replacing
function add_perturbations!(outfield, inMesh::RegularGrid{G}, infield, outMesh::RegularGrid{G, 3}; tf_extrapolate = false, tf_harmonic = false) where {G}
    # Check that coordinate systems are equivalent
    if (inMesh.Geometry.λ₀ != outMesh.Geometry.λ₀) || (inMesh.Geometry.ϕ₀ != outMesh.Geometry.ϕ₀) ||
        (inMesh.Geometry.R₀ != outMesh.Geometry.R₀) || (inMesh.Geometry.β != outMesh.Geometry.β)
        error("Need to update interpolate_field! for meshes with different origins!")
    end

    for (k, qz) in enumerate(outMesh.x[3])
        for (j, qy) in enumerate(outMesh.x[2])
            for (i, qx) in enumerate(outMesh.x[1])
                outfield[i,j,k] += linearly_interpolate(inMesh.x, infield, (qx,qy,qz); tf_extrapolate = tf_extrapolate, tf_harmonic = tf_harmonic)
            end
        end
    end

    return nothing
end

# USED IN BUILDING INPUTS <- THESE NEED TO BE UPDATED
# Interpolating values between models
function interpolate_parameters!(OutField::InverseIsotropicSlowness, InField::PsiModel{<:IsotropicVelocity}; tf_extrapolate = false, tf_harmonic = false)
    if ~isnothing(OutField.Up)
        interpolate_field!(OutField.Up.m0, InField.Mesh, InField.Parameters.vp, OutField.Up.Mesh; tf_extrapolate = tf_extrapolate, tf_harmonic = tf_harmonic)
        # Convert back to slowness
        OutField.Up.m0 .= inv.(OutField.Up.m0)
    end
    if ~isnothing(OutField.Us)
        interpolate_field!(OutField.Us.m0, InField.Mesh, InField.Parameters.vs, OutField.Us.Mesh; tf_extrapolate = tf_extrapolate, tf_harmonic = tf_harmonic)
        # Convert back to slowness
        OutField.Us.m0 .= inv.(OutField.Us.m0)
        if OutField.coupling_option == 1
            # Convert S-Slownes to S-to-P slowness ratio (i.e. Vp/Vs ratio)
            # This conversion could be done without allocation but I'm being lazy here...
            vp_field = zeros(eltype(OutField.Us), size(OutField.Us.Mesh))
            # Interpolate P-velocity field to the S-slowness inversion mesh
            interpolate_field!(vp_field, InField.Mesh, InField.Parameters.vp, OutField.Us.Mesh; tf_extrapolate = tf_extrapolate, tf_harmonic = tf_harmonic)
            # Compute ratio
            OutField.Us.m0 .*= vp_field
        end
    end

    return nothing
end
function interpolate_parameters!(OutField::InverseIsotropicSlowness, InField::PsiModel{<:HexagonalVectoralVelocity}; tf_extrapolate = false, tf_harmonic = false)
    
    # Convert reference velocities to isotropic velocities before interpolation
    # convert_parameters!(OutField, InField.Parameters) # <---- NOT NEEDED IF STARTING MODEL IS ASSUMED ISOTROPIC!

    if ~isnothing(OutField.Up)
        # Interpolate the isotropic velocities
        interpolate_field!(OutField.Up.m0, InField.Mesh, InField.Parameters.α, OutField.Up.Mesh; tf_extrapolate = tf_extrapolate, tf_harmonic = tf_harmonic)
        # Convert back to slowness
        OutField.Up.m0 .= inv.(OutField.Up.m0)
    end
    if ~isnothing(OutField.Us)
        interpolate_field!(OutField.Us.m0, InField.Mesh, InField.Parameters.β, OutField.Us.Mesh; tf_extrapolate = tf_extrapolate, tf_harmonic = tf_harmonic)
        # Convert back to slowness
        OutField.Us.m0 .= inv.(OutField.Us.m0)
        if OutField.coupling_option == 1
            # Convert S-Slownes to S-to-P slowness ratio (i.e. Vp/Vs ratio)
            # This conversion could be done without allocation but I'm being lazy here...
            vp_field = zeros(eltype(OutField.Us), size(OutField.Us.Mesh))
            # Interpolate P-velocity field to the S-slowness inversion mesh
            interpolate_field!(vp_field, InField.Mesh, InField.Parameters.vp, OutField.Us.Mesh; tf_extrapolate = tf_extrapolate, tf_harmonic = tf_harmonic)
            # Compute ratio
            OutField.Us.m0 .*= vp_field
        end
    end

    # Revert isotropic velocities back to reference velocities
    # revert_parameters!(OutField, InField.Parameters) # <---- NOT NEEDED IF STARTING MODEL IS ASSUMED ISOTROPIC!

    return nothing
end
# USED IN BUILDING INPUTS

# Interpolate between meshes
function interpolate_field!(outfield, inMesh::RegularGrid{G}, infield, outMesh::RegularGrid{G, 3}; tf_extrapolate = false, tf_harmonic = false) where {G}
    # Check that coordinate systems are equivalent
    if (inMesh.Geometry.λ₀ != outMesh.Geometry.λ₀) || (inMesh.Geometry.ϕ₀ != outMesh.Geometry.ϕ₀) ||
        (inMesh.Geometry.R₀ != outMesh.Geometry.R₀) || (inMesh.Geometry.β != outMesh.Geometry.β)
        error("Need to update interpolate_field! for meshes with different origins!")
    end

    for (k, qz) in enumerate(outMesh.x[3])
        for (j, qy) in enumerate(outMesh.x[2])
            for (i, qx) in enumerate(outMesh.x[1])
                outfield[i,j,k] = linearly_interpolate(inMesh.x, infield, (qx,qy,qz); tf_extrapolate = tf_extrapolate, tf_harmonic = tf_harmonic)
            end
        end
    end

    return nothing
end
