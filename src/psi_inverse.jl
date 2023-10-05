# Main inversion function
function psi_inverse!(Model::ModelTauP, ΔM::SeismicPerturbationModel, Obs::Vector{<:Observable}, S::SolverLSQR)
    # Assign Jacobian indices to parameters. Returns total number of parameters (i.e. Jacobian columns).
    npar = assign_jacobian_indices!(ΔM)

    # Allocate solution vectors
    x = zeros(npar)

    # Compute kernels and relative residuals for all observations
    # For TauP models, this only needs to be done once as the rays are 1D predictions
    K, b, = psi_forward(Obs, Model)
    nobs = length(b)

    # Initial sum of squared (relative) residuals
    wssrₖ = sum(x -> x^2, b; init = 0.0)
    # Initial F-statistic
    fcrit = quantile(FDist(nobs - 1, nobs - 1), 0.95)
    fstat = 10.0*fcrit

    # Inner inversion iterations (i.e. no ray tracing)
    kiter = 0
    while (fstat > fcrit) && (kiter < S.nonlinit)
        println("Starting iteration ", string(kiter + 1), ".")
        # Initialise the Jacobian row (i.e. observation) counter
        nobs = 0

        # Build Jacobian Matrix
        Aᵢ, Aⱼ, Aᵥ = psi_build_jacobian(ΔM, Obs, K; Δi = nobs)

        # Update row counter when finished building Jacobian for all observations in a set
        nobs = maximum(Aᵢ)
        println("Number of parameters: ", string(npar), ".")
        println("Number of observations: ", string(nobs), ".")

        # Compute row-sum of the Jacobian squared (RSJS)
        accumarray!(x, Aⱼ, Aᵥ, x -> x^2);
        # Assign to parameters
        fill_rsjs!(ΔM, x)
        # Reset solution vector
        fill!(x, zero(eltype(x)))

        # Build Constraint Equations
        build_constraints!(Aᵢ, Aⱼ, Aᵥ, b, ΔM; Δi = nobs, tf_jac_scale = S.tf_jac_scale)

        # Update row counter when finished building Jacobian for all observations in a set
        nrows = maximum(Aᵢ)
        println("Number of constraints: ", string(nrows - nobs), ".")

        # Build and solve sparse linearised system
        A = sparse(Aᵢ, Aⱼ, Aᵥ, nrows, npar, +)
        lsqr!(x, A, b; damp = S.damp, atol = S.atol, conlim = S.conlim, maxiter = S.maxiter, verbose = S.verbose, log = false)

        # Update Parameters
        update_parameters!(ΔM, x)
        # Reset solution vector
        fill!(x, zero(eltype(x)))

        # Update kernels
        update_kernel!(K, ΔM, Obs)
        
        # Re-evalute weighted sum of the squared resdiuals
        b = zeros(nobs) # Reset b-vector
        wssrₖ₊₁ = 0.0
        for i in eachindex(K)
            _, b[i] = evaluate_kernel(K[i], Obs[i])
            wssrₖ₊₁ += b[i]^2
        end

        # Compute new fstat
        fstat = wssrₖ/wssrₖ₊₁
        # Display fit summary for this iteration
        println("F-stat = ", string(fstat), " | F-crit = ", string(fcrit))
        println("wssrₖ = ", string(wssrₖ), " | wssrₖ₊₁ = ", string(wssrₖ₊₁))
        println("χ²ₖ = ", string(wssrₖ/nobs), " | ", "χ²ₖ₊₁ = ", string(wssrₖ₊₁/nobs), "\n")
        # Update prior fit
        wssrₖ = wssrₖ₊₁

        # Update iteration counter
        kiter += 1
    end
    # Update Model
    update_model!(Model, ΔM)

    return nothing
end

# Jacobian building functions
function psi_build_jacobian(ΔM::SeismicPerturbationModel, Obs::Vector{<:Observable}, K::Vector{<:ObservableKernel}; Δi = 0)
    # Initialise storage arrays for linear system
    Aᵢ = Vector{Int}()
    Aⱼ = Vector{Int}()
    Aᵥ = Vector{Float64}()

    if ~isnothing(ΔM.Velocity)
        psi_jacobian!(Aᵢ, Aⱼ, Aᵥ, ΔM.Velocity, Obs, K; Δi = Δi)
    end
    if ~isnothing(ΔM.Interface)
        error("Interface inversion not yet implemented.")
    end
    if ~isnothing(ΔM.Hypocenter)
        error("Hypocenter inversion not yet implemented.")
    end
    if ~isnothing(ΔM.SourceStatics)
        psi_jacobian!(Aᵢ, Aⱼ, Aᵥ, ΔM.SourceStatics, Obs; Δi = Δi)
    end
    if ~isnothing(ΔM.ReceiverStatics)
        error("Receiver static inversion not yet implemented.")
    end

    return Aᵢ, Aⱼ, Aᵥ
end

# Build Jacobian Block for Isotropic Slowness Parameters
function psi_jacobian!(Aᵢ, Aⱼ, Aᵥ, ΔM::IsotropicSlowness, Obs::Vector{<:Observable}, K::Vector{<:ObservableKernel}; Δi = 0)
    # Loop over kernels
    for i in eachindex(K)
        # Differentiate observable kernel with respect to parameter field
        j, ∂bᵢ∂mⱼ = psi_differentiate_kernel(ΔM, K[i], K[i].b)
        # Weight partial by observation uncertainty
        ∂bᵢ∂mⱼ ./= Obs[i].error
        # Append to Jacobian
        append!(Aᵢ, fill((i + Δi), length(j)))
        append!(Aⱼ, j)
        append!(Aᵥ, ∂bᵢ∂mⱼ)
    end

    return nothing
end

# Observable Sensitivity: Isotropic slowness + Isotropic Velocity Model + Shear Travel Time
function psi_differentiate_kernel(ΔM::IsotropicSlowness, K::ObservableKernel{<:IsotropicVelocity}, b::Type{<:TravelTime})

    # Get Jacobian index and mesh
    if K.b <: TravelTime{<:CompressionalWave}
        jcol = ΔM.up.jcol[1] - 1
        Mesh = ΔM.up.Mesh
    elseif K.b <: TravelTime{<:ShearWave}
        jcol = ΔM.us.jcol[1] - 1
        Mesh = ΔM.us.Mesh
    else
        error("Unrecognized kernel observable "*string(K.b)*".")
    end

    # Allocate storage array for the partials
    T = eltype(K.q₀) # Assume same element type for partials as for static
    n = length(K.w) # Number of partials
    Wj = Array{Int}(undef, n, 8) # Parameter jacobian index
    Wv = Array{T}(undef, n, 8) # Parameter sensitivity weights, 8 per node
    # Loop over elements in kernel
    for i in eachindex(K.x)
        # Convert to inversion field coordinates
        kx₁, kx₂, kx₃ = global_to_local(K.x[i][1], K.x[i][2], K.x[i][3], Mesh.CS)
        # Map partial derivative to inversion parameter. Travel-time slowness partial is simply the kernel weight.
        j, v = get_weights(Mesh, kx₁, kx₂, kx₃, K.w[i]; Δi = jcol)
        # Store jacobian indices and values
        Wj[i,:] .= j
        Wv[i,:] .= v
    end
    # Accumulate the weights
    jm, ∂t∂m = accumlate_weights(Wj, Wv)

    return jm, ∂t∂m
end

# Map sensitivities to Jacobian
function get_weights(Mesh::RegularGrid, qx₁, qx₂, qx₃, w; Δi = 0)
    # Return trilinear interpolation weights
    wi, wv = trilinear_weights(Mesh.x[1], Mesh.x[2], Mesh.x[3], qx₁, qx₂, qx₃; tf_extrapolate = false, scale = w)
    # Apply shift to index
    wj = (wi[1] + Δi, wi[2] + Δi, wi[3] + Δi, wi[4] + Δi, wi[5] + Δi, wi[6] + Δi, wi[7] + Δi, wi[8] + Δi)
    return wj, wv
end

# Accumulates weights
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

# Build Jacobian Block for Seismic Source Statics
function psi_jacobian!(Aᵢ, Aⱼ, Aᵥ, ΔM::SeismicSourceStatics, Obs::Vector{<:Observable}; Δi = 0)
    # Loop over observations
    for (i, b) in enumerate(Obs)
        # Dictionary key for static
        k = (b.source_id, nameof(typeof(b)), nameof(typeof(b.phase)))
        if haskey(ΔM.jcol, k)
            # Get Jacobian index
            j = ΔM.jcol[k]
            # Append to Jacobian
            append!(Aᵢ, i + Δi)
            append!(Aⱼ, j)
            append!(Aᵥ, 1.0/Obs[i].error) # Static sensitivity is always 1
        else
            println("No source static for key, ", k)
        end
    end

    return nothing
end



# Assigns Jacobian index to a ParameterField
function assign_jacobian_indices!(ΔM::SeismicPerturbationModel; jcol = 0)
    # Assign Jacobian indices to all fields in a Seismic Perturbation Model
    # Each call to assign_jacobian_indices! updates and returns the Jacobian row counter
    if ~isnothing(ΔM.Velocity)
        jcol = assign_jacobian_indices!(ΔM.Velocity; jcol = jcol)
    end
    if ~isnothing(ΔM.Interface)
        jcol = assign_jacobian_indices!(ΔM.Interface; jcol = jcol)
    end
    if ~isnothing(ΔM.Hypocenter)
        jcol = assign_jacobian_indices!(ΔM.Hypocenter; jcol = jcol)
    end
    if ~isnothing(ΔM.SourceStatics)
        jcol = assign_jacobian_indices!(ΔM.SourceStatics; jcol = jcol)
    end
    if ~isnothing(ΔM.ReceiverStatics)
        jcol = assign_jacobian_indices!(ΔM.ReceiverStatics; jcol = jcol)
    end

    return jcol
end

function assign_jacobian_indices!(ΔM::IsotropicSlowness; jcol = 0)

    # Indices of P-slowness parameters
    if ~isnothing(ΔM.up)
        m = length(ΔM.up.Mesh)
        ΔM.up.jcol[1] = 1 + jcol
        ΔM.up.jcol[2] = m + jcol
        jcol += m
    end
    # Indices of S-slowness parameters
    if ~isnothing(ΔM.us)
        m = length(ΔM.us.Mesh)
        ΔM.us.jcol[1] = 1 + jcol
        ΔM.us.jcol[2] = m + jcol
        jcol += m
    end

    return jcol
end

function assign_jacobian_indices!(ΔM::SeismicStatics; jcol = 0)
    # Loop over static keys
    for k in eachindex(ΔM.jcol)
        # Increment jacobian index
        jcol += 1
        # Assign index to key
        ΔM.jcol[k] = jcol
    end

    return jcol
end



function fill_rsjs!(ΔM::SeismicPerturbationModel, x)

    if ~isnothing(ΔM.Velocity)
        fill_rsjs!(ΔM.Velocity, x)
    end
    if ~isnothing(ΔM.Interface)
        error("Interface RSJS not yet defined.")
    end
    if ~isnothing(ΔM.Hypocenter)
        error("Hypocenter RSJS not yet defined.")
    end
    if ~isnothing(ΔM.SourceStatics)
        fill_rsjs!(ΔM.SourceStatics, x)
    end
    if ~isnothing(ΔM.ReceiverStatics)
        fill_rsjs!(ΔM.ReceiverStatics, x)
    end

    return nothing
end

function fill_rsjs!(ΔM::IsotropicSlowness, x)
    # Update P-wave Slowness Sensitivities (if not empty)
    if ~isnothing(ΔM.up)
        fill_inversion_field!(ΔM.up.RSJS, x, ΔM.up.jcol)
    end

    # Update S-wave Slowness Sensitivities (if not empty)
    if ~isnothing(ΔM.us)
        fill_inversion_field!(ΔM.us.RSJS, x, ΔM.us.jcol)
    end

    return nothing
end

function fill_rsjs!(ΔM::SeismicStatics, x)
    # Loop over keys and extract Jacobian element
    for k in eachindex(ΔM.jcol)
        ΔM.RSJS[k] = x[ΔM.jcol[k]]
    end

    return nothing
end

function root_mean_rsjs(J; δJ = 0.0)
    g = zero(eltype(J))
    n = 0
    for i in eachindex(J)
        if J[i] > δJ
            g += J[i]
            n += 1
        end
    end
    # Check for non-null sensitivity
    if n > 0
        g /= n
    else
        g = 1.0
    end

    return sqrt(g)
end

# Convenient function to loop over Jacobian indices
function fill_inversion_field!(f, x, jx)
    k = 0
    for i in jx[1]:jx[2]
        k += 1
        f[k] = x[i]
    end

    return nothing
end



function update_parameters!(ΔM::SeismicPerturbationModel, x)
    if ~isnothing(ΔM.Velocity)
        update_parameters!(ΔM.Velocity, x)
    end
    if ~isnothing(ΔM.Interface)
        error("Interface parameter update not yet defined.")
    end
    if ~isnothing(ΔM.Hypocenter)
        error("Hypocenter parameter update not yet defined.")
    end
    if ~isnothing(ΔM.SourceStatics)
        update_parameters!(ΔM.SourceStatics, x)
    end
    if ~isnothing(ΔM.ReceiverStatics)
        update_parameters!(ΔM.ReceiverStatics, x)
    end

    return nothing
end

function update_parameters!(ΔM::IsotropicSlowness, x)
    if ~isnothing(ΔM.up)
        fill_inversion_field!(ΔM.up.δm, x, ΔM.up.jcol)
        # Cummulative Perturbations
        ΔM.up.Δm .+= ΔM.up.δm
    end

    if ~isnothing(ΔM.us)
        fill_inversion_field!(ΔM.us.δm, x, ΔM.us.jcol)
        ΔM.us.Δm .+= ΔM.us.δm
    end

    return nothing
end

function update_parameters!(ΔM::SeismicStatics, x)
    for k in eachindex(ΔM.jcol)
        ΔM.δm[k] = x[ΔM.jcol[k]]
        ΔM.Δm[k] += x[ΔM.jcol[k]]
    end

    return nothing
end


function update_kernel!(K::Vector{<:ObservableKernel}, ΔM::SeismicPerturbationModel, Obs::Vector{<:Observable})
    if ~isnothing(ΔM.Velocity)
        update_kernel!(K, ΔM.Velocity)
    end
    if ~isnothing(ΔM.Interface)
        error("Interface kernel update not yet defined.")
    end
    if ~isnothing(ΔM.Hypocenter)
        error("Hypocenter kernel update not yet defined.")
    end
    if ~isnothing(ΔM.ReceiverStatics)
        update_kernel!(K, ΔM.ReceiverStatics, Obs)
    end
    if ~isnothing(ΔM.SourceStatics)
        update_kernel!(K, ΔM.SourceStatics, Obs)
    end

    return nothing
end

function update_kernel!(K::Vector{<:ObservableKernel}, ΔM::InversionParameter)
    for i in eachindex(K)
        update_kernel!(K[i], ΔM)
    end

    return nothing
end

function update_kernel!(K::ObservableKernel{<:IsotropicVelocity,}, ΔM::IsotropicSlowness)
    if ~isnothing(K.m.vp) && ~isnothing(ΔM.up)
        for (i, xq) in enumerate(K.x)
            # Map global kernel coordinates to local
            kx₁, kx₂, kx₃ = global_to_local(xq[1], xq[2], xq[3], ΔM.up.Mesh.CS)
            # Interpolate the slowness perturbations to the kernel
            δu = linearly_interpolate(ΔM.up.Mesh.x[1], ΔM.up.Mesh.x[2], ΔM.up.Mesh.x[3], ΔM.up.δm, kx₁, kx₂, kx₃;
            tf_extrapolate = false, tf_harmonic = false)
            # Prior kernel slowness
            up = 1.0/K.m.vp[i]
            # Updated kernel slowness
            K.m.vp[i] = 1.0/(up + δu)
        end
    end

    if ~isnothing(K.m.vs) && ~isnothing(ΔM.us)
        for (i, xq) in enumerate(K.x)
            # Map global kernel coordinates to local
            kx₁, kx₂, kx₃ = global_to_local(xq[1], xq[2], xq[3], ΔM.us.Mesh.CS)
            # Interpolate the slowness perturbations to the kernel
            δu = linearly_interpolate(ΔM.us.Mesh.x[1], ΔM.us.Mesh.x[2], ΔM.us.Mesh.x[3], ΔM.us.δm, kx₁, kx₂, kx₃;
            tf_extrapolate = false, tf_harmonic = false)
            # Prior kernel slowness
            us = 1.0/K.m.vs[i]
            # Updated kernel slowness
            K.m.vs[i] = 1.0/(us + δu)
        end
    end

    return nothing
end

function update_kernel!(K::Vector{<:ObservableKernel}, ΔM::InversionParameter, Obs::Vector{<:Observable})
    for i in eachindex(K)
        update_kernel!(K[i], ΔM, Obs[i])
    end

    return nothing
end

function update_kernel!(K::ObservableKernel, ΔM::SeismicSourceStatics, b::Observable)
    kiop = (b.source_id, nameof(typeof(b)), nameof(typeof(b.phase)))
    if haskey(ΔM.δm, kiop)
        K.q₀[1] += ΔM.δm[kiop]
    end

    return nothing
end



function update_model!(Model::AbstractModel, ΔM::SeismicPerturbationModel)
    if ~isnothing(ΔM.Velocity)
        update_model!(Model, ΔM.Velocity)
    end
    if ~isnothing(ΔM.Interface)
        error("Interface model update not yet defined.")
    end
    if ~isnothing(ΔM.Hypocenter)
        error("Hypocenter model update not yet defined.")
    end
    if ~isnothing(ΔM.SourceStatics)
        update_model!(Model, ΔM.SourceStatics)
    end
    if ~isnothing(ΔM.ReceiverStatics)
        update_model!(Model, ΔM.ReceiverStatics)
    end

    return nothing
end

# MODEL UPDATES ARE ADDING TOTAL PERTURBATION!
# The Δm fields are the cumulative inner non-linear perturbations
# The δm fields are the incremental inner non-linear (i.e. non-linear) perturbations
# For multiple ray tracing iterations (i.e. outer non-linear loop), we will need to
# reset Δm at each outer iteration. Otherwise, we will be adding the same perturbations
# mulitple times.
function update_model!(Model::ModelTauP{<:IsotropicVelocity,}, ΔM::IsotropicSlowness)
    # Update P-wave Slowness (if not empty)
    if ~isnothing(ΔM.up)
        # Map perturbations to model
        for (k, qx₃) in enumerate(Model.Mesh.x[3])
            # Radial TauP depth
            rq = Model.Rₜₚ - ΔM.up.Mesh.CS.R₀ - qx₃
            # Reference slowness at this depth
            uq = piecewise_linearly_interpolate(Model.r, Model.nd, Model.vp, rq; tf_extrapolate = true, tf_harmonic = true)
            uq = 1.0/uq
            # Loop over remaining nodes and update P-wave velocity perturbation
            for (i, qx₁) in enumerate(Model.Mesh.x[1]), (j, qx₂) in enumerate(Model.Mesh.x[2])
                Δu = linearly_interpolate(ΔM.up.Mesh.x[1], ΔM.up.Mesh.x[2], ΔM.up.Mesh.x[3], ΔM.up.Δm, qx₁, qx₂, qx₃;
                     tf_extrapolate = false, tf_harmonic = false)
                # Add velocity perturbation to model
                Model.m.vp[i, j, k] += (1.0/(uq + Δu)) - (1.0/uq)
            end
        end
    end

    # Update S-wave Slowness (if not empty)
    if ~isnothing(ΔM.us)
        # Map perturbations to model
        for (k, qx₃) in enumerate(Model.Mesh.x[3])
            # Radial TauP depth
            rq = Model.Rₜₚ - ΔM.us.Mesh.CS.R₀ - qx₃
            # Reference slowness
            uq = piecewise_linearly_interpolate(Model.r, Model.nd, Model.vs, rq; tf_extrapolate = true, tf_harmonic = true)
            uq = 1.0/uq
            # Loop over remaining nodes and update S-wave velocity perturbation
            for (i, qx₁) in enumerate(Model.Mesh.x[1]), (j, qx₂) in enumerate(Model.Mesh.x[2])
                Δu = linearly_interpolate(ΔM.us.Mesh.x[1], ΔM.us.Mesh.x[2], ΔM.us.Mesh.x[3], ΔM.us.Δm, qx₁, qx₂, qx₃;
                     tf_extrapolate = false, tf_harmonic = false)
                # Add velocity perturbation to model
                Model.m.vs[i, j, k] += (1.0/(uq + Δu)) - (1.0/uq)
            end
        end
    end

    return nothing
end

function update_model!(Model::AbstractModel, ΔM::SeismicSourceStatics)
    for k in eachindex(ΔM.Δm)
        if haskey(Model.Sources.static, k)
            # Accumulate static
            Model.Sources.static[k] += ΔM.Δm[k]
        else
            # Add static
            Model.Sources.static[k] = ΔM.Δm[k]
        end
    end

    return nothing
end

function update_model!(Model::AbstractModel, ΔM::SeismicReceiverStatics)
    for k in eachindex(ΔM.Δm)
        if haskey(Model.Receivers.static, k)
            Model.Receivers.static[k] += ΔM.Δm[k]
        else
            Model.Receivers.static[k] = ΔM.Δm[k]
        end
    end

    return nothing
end



# Build Constraint Equations
function build_constraints!(Aᵢ, Aⱼ, Aᵥ, b, ΔM::SeismicPerturbationModel; Δi = 0, tf_jac_scale = false)
    if ~isnothing(ΔM.Velocity)
        Δi = build_constraints!(Aᵢ, Aⱼ, Aᵥ, b, ΔM.Velocity; Δi = Δi, tf_jac_scale = tf_jac_scale)
    end
    if ~isnothing(ΔM.Interface)
        error("Interface regularisation not yet defined.")
    end
    if ~isnothing(ΔM.Hypocenter)
        error("Hypocenter regularisation not yet defined.")
    end
    if ~isnothing(ΔM.SourceStatics)
        Δi = build_constraints!(Aᵢ, Aⱼ, Aᵥ, b, ΔM.SourceStatics; Δi = Δi) # No Jacobian scaling for statics
    end
    if ~isnothing(ΔM.ReceiverStatics)
        Δi = build_constraints!(Aᵢ, Aⱼ, Aᵥ, b, ΔM.ReceiverStatics; Δi = Δi) # No Jacobian scaling for statics
    end

    return Δi
end

function build_constraints!(Aᵢ, Aⱼ, Aᵥ, b, ΔM::SeismicStatics; Δi = 0)
    # Loop over static keys
    for kj in eachindex(ΔM.jcol)
        # Define the damping parameter key
        ki = (kj[2], kj[3])
        # Define the constraint if the damping is > 0
        if ΔM.μ[ki] > 0.0
            Δi += 1 # Increment row counter
            j = ΔM.jcol[kj] # Jacobian index
            w = ΔM.μ[ki]/ΔM.σ[kj] # Define the weight
            # Append constraint
            push!(Aᵢ, Δi)
            push!(Aⱼ, j)
            push!(Aᵥ, w)
            # Constraint value
            if ΔM.tf_jump
                δm = ΔM.μ[ki]*ΔM.Δm[kj]/ΔM.σ[kj]
                push!(b, -δm)
            else
                push!(b, 0.0)
            end
        end
    end

    return Δi
end

function build_constraints!(Aᵢ, Aⱼ, Aᵥ, b, ΔU::IsotropicSlowness; Δi = 0, tf_jac_scale = false)

    # P-slowness
    if ~isnothing(ΔU.up) && isnothing(ΔU.us)
        Δj = (ΔU.up.jcol[1] - 1)
        Δi = build_spatial_contraints!(Aᵢ, Aⱼ, Aᵥ, b, ΔU.up.Mesh, ΔU.up.Δm, ΔU.up.m₀, ΔU.up.σₘ, ΔU.up.μ[1], ΔU.up.μjump[1],
        ΔU.up.λ, ΔU.up.λjump[1], ΔU.up.RSJS; tf_jac_scale = tf_jac_scale, Δi = Δi, Δj = Δj)
    end

    # S-slowness
    if ~isnothing(ΔU.us) && isnothing(ΔU.up)
        Δj = (ΔU.us.jcol[1] - 1)
        Δi = build_spatial_contraints!(Aᵢ, Aⱼ, Aᵥ, b, ΔU.us.Mesh, ΔU.us.Δm, ΔU.us.m₀, ΔU.us.σₘ, ΔU.us.μ[1], ΔU.us.μjump[1],
        ΔU.us.λ, ΔU.us.λjump[1], ΔU.us.RSJS; tf_jac_scale = tf_jac_scale, Δi = Δi, Δj = Δj)
    end

    # Joint Inversion
    if ~isnothing(ΔU.up) && ~isnothing(ΔU.us)
        error("Missing Coupling Constraints.")
    end

    return Δi
end

# Constraints for spatially discretised variables
function build_spatial_contraints!(Aᵢ, Aⱼ, Aᵥ, b, Mesh, Δm, m₀, σₘ, μ, μjump, λ, λjump, rsjs; tf_jac_scale = false, Δi = 0, Δj = 0)
    # Weight adjustment for Jacobian scaling
    if tf_jac_scale
        w = root_mean_rsjs(rsjs)
        m̅ = mean(m₀)
        σ̅ = mean(σₘ)
    else
        w = 1.0
        m̅ = 1.0
        σ̅ = 1.0
    end
    # Scale damping and smoothing multipliers
    μ = w*σ̅*m̅*μ
    λ1 = w*m̅*λ[1]
    λ2 = w*m̅*λ[2]
    λ3 = w*m̅*λ[3]

    # Build Damping Constraints
    Dᵢ, Dⱼ, Dᵥ, δm, Δi = damping_constraint(μ, m₀, σₘ, Δm, μjump; Δi = Δi, Δj = Δj)
    # Append constraints
    append!(Aᵢ, Dᵢ)
    append!(Aⱼ, Dⱼ)
    append!(Aᵥ, Dᵥ)
    append!(b, δm)

    # Smoothing Constraint
    Lᵢ, Lⱼ, Lᵥ, δL, Δi = smoothing_constraint(Mesh, m₀, Δm, λ1, λ2, λ3, λjump; Δi = Δi, Δj = Δj)
    # Append constraints
    append!(Aᵢ, Lᵢ)
    append!(Aⱼ, Lⱼ)
    append!(Aᵥ, Lᵥ)
    append!(b, δL)

    return Δi
end

# Damping System
function damping_constraint(μ, m₀, σₘ, Δm, tf_jump; Δi = 0, Δj = 0)
    # Damping constrain--minimise sum of squared parameter perturbations
    np = length(m₀) # Number of parameters
    Dᵢ = collect((1 + Δi):(np + Δi)) # Row index
    Dⱼ = collect((1 + Δj):(np + Δj)) # Column index
    Dᵥ = μ./(σₘ[:].*m₀[:]) # Coefficient
    # Constraint vector
    if tf_jump
        δm = -Dᵥ.*Δm[Dⱼ]
    else
        δm = zeros(np)
    end
    Δi = Δi + np

    return Dᵢ, Dⱼ, Dᵥ, δm, Δi
end

# Smoothing System
function smoothing_constraint(Mesh, m₀, Δm, λᵢ, λⱼ, λₖ, tf_jump; Δi = 0, Δj = 0)
    # Weighted Laplacian Smoothing Constraint
    Lᵢ, Lⱼ, Lᵥ = sparse_laplacian(Mesh; wᵢ = λᵢ, wⱼ = λⱼ, wₖ = λₖ)
    # Scale Laplacian by starting model (i.e. smoothing fractional perturbations)
    Lᵥ .*= 1.0./m₀[Lⱼ]
    # Constraint vector
    np = length(Mesh)
    δL = zeros(np)
    if tf_jump
        for k in eachindex(Lᵢ)
            i = Lᵢ[k]
            j = Lⱼ[k]
            δL[i] -= Δm[j]*Lᵥ[k]
        end
    end

    # Shift the row and column indices
    Lᵢ .+= Δi
    Lⱼ .+= Δj
    Δi += np

    return Lᵢ, Lⱼ, Lᵥ, δL, Δi
end

function sparse_laplacian(Mesh::RegularGrid{T, 3, R}; wᵢ = 1.0, wⱼ = 1.0, wₖ = 1.0) where {T<:CoordinateSystem, R<:AbstractRange}
    # ASSUMES GRID IS LARGER THAN OR EQUAL TO 4x4x4
    nx₁, nx₂, nx₃ = size(Mesh)

    # Number of central finite-difference coefficients
    ncoeff = 3*((nx₁ - 2)*nx₂*nx₃ + (nx₂ - 2)*nx₁*nx₃ + (nx₃ - 2)*nx₁*nx₂)
    # Add number of boundary finite-difference coefficients
    ncoeff += 8*(nx₂*nx₃ + nx₁*nx₃ + nx₁*nx₂)
    # Allocate storage arrays
    Lᵢ = zeros(Int, ncoeff)
    Lⱼ = zeros(Int, ncoeff)
    Lᵥ = zeros(Float64, ncoeff)
    # Initialize counter
    ind = 1
    # Loop over all nodes and compute finite-difference coefficients
    for i in 1:nx₁, j in 1:nx₂, k in 1:nx₃
        n₀ = subscripts_to_index((nx₁, nx₂, nx₃), (i, j, k))
        # x₁-dimension
        if (i > 1) && (i < nx₁)
            # Central difference (2ⁿᵈ-order)
            n₋ = subscripts_to_index((nx₁, nx₂, nx₃), (i-1, j, k))
            n₊ = subscripts_to_index((nx₁, nx₂, nx₃), (i+1, j, k))
            # i - 1
            Lᵢ[ind] = n₀
            Lⱼ[ind] = n₋
            Lᵥ[ind] = wᵢ*1.0
            ind += 1
            # i
            Lᵢ[ind] = n₀
            Lⱼ[ind] = n₀
            Lᵥ[ind] = -wᵢ*2.0
            ind += 1
            # i + 1
            Lᵢ[ind] = n₀
            Lⱼ[ind] = n₊
            Lᵥ[ind] = wᵢ*1.0
            ind += 1
        else
            # Forward/Backward difference (2ⁿᵈ-order)
            if (i == 1)
                n₁ = subscripts_to_index((nx₁, nx₂, nx₃), (i+1, j, k))
                n₂ = subscripts_to_index((nx₁, nx₂, nx₃), (i+2, j, k))
                n₃ = subscripts_to_index((nx₁, nx₂, nx₃), (i+3, j, k))
            elseif (i == nx₁)
                n₁ = subscripts_to_index((nx₁, nx₂, nx₃), (i-1, j, k))
                n₂ = subscripts_to_index((nx₁, nx₂, nx₃), (i-2, j, k))
                n₃ = subscripts_to_index((nx₁, nx₂, nx₃), (i-3, j, k))
            end
            # i
            Lᵢ[ind] = n₀
            Lⱼ[ind] = n₀
            Lᵥ[ind] = wᵢ*2.0
            ind += 1
            # i ± 1
            Lᵢ[ind] = n₀
            Lⱼ[ind] = n₁
            Lᵥ[ind] = -wᵢ*5.0
            ind += 1
            # i ± 2
            Lᵢ[ind] = n₀
            Lⱼ[ind] = n₂
            Lᵥ[ind] = wᵢ*4.0
            ind += 1
            # i ± 3
            Lᵢ[ind] = n₀
            Lⱼ[ind] = n₃
            Lᵥ[ind] = -wᵢ*1.0
            ind += 1
        end
        # x₂-dimension
        if (j > 1) && (j < nx₂)
            # Central difference (2ⁿᵈ-order)
            n₋ = subscripts_to_index((nx₁, nx₂, nx₃), (i, j-1, k))
            n₊ = subscripts_to_index((nx₁, nx₂, nx₃), (i, j+1, k))
            # i - 1
            Lᵢ[ind] = n₀
            Lⱼ[ind] = n₋
            Lᵥ[ind] = wⱼ*1.0
            ind += 1
            # i
            Lᵢ[ind] = n₀
            Lⱼ[ind] = n₀
            Lᵥ[ind] = -wⱼ*2.0
            ind += 1
            # i + 1
            Lᵢ[ind] = n₀
            Lⱼ[ind] = n₊
            Lᵥ[ind] = wⱼ*1.0
            ind += 1
        else
            # Forward/Backward difference (2ⁿᵈ-order)
            if (j == 1)
                n₁ = subscripts_to_index((nx₁, nx₂, nx₃), (i, j+1, k))
                n₂ = subscripts_to_index((nx₁, nx₂, nx₃), (i, j+2, k))
                n₃ = subscripts_to_index((nx₁, nx₂, nx₃), (i, j+3, k))
            elseif (j == nx₂)
                n₁ = subscripts_to_index((nx₁, nx₂, nx₃), (i, j-1, k))
                n₂ = subscripts_to_index((nx₁, nx₂, nx₃), (i, j-2, k))
                n₃ = subscripts_to_index((nx₁, nx₂, nx₃), (i, j-3, k))
            end
            # i
            Lᵢ[ind] = n₀
            Lⱼ[ind] = n₀
            Lᵥ[ind] = wⱼ*2.0
            ind += 1
            # i ± 1
            Lᵢ[ind] = n₀
            Lⱼ[ind] = n₁
            Lᵥ[ind] = -wⱼ*5.0
            ind += 1
            # i ± 2
            Lᵢ[ind] = n₀
            Lⱼ[ind] = n₂
            Lᵥ[ind] = wⱼ*4.0
            ind += 1
            # i ± 3
            Lᵢ[ind] = n₀
            Lⱼ[ind] = n₃
            Lᵥ[ind] = -wⱼ*1.0
            ind += 1
        end
        # x₃-dimension
        if (k > 1) && (k < nx₃)
            # Central difference (2ⁿᵈ-order)
            n₋ = subscripts_to_index((nx₁, nx₂, nx₃), (i, j, k-1))
            n₊ = subscripts_to_index((nx₁, nx₂, nx₃), (i, j, k+1))
            # i - 1
            Lᵢ[ind] = n₀
            Lⱼ[ind] = n₋
            Lᵥ[ind] = wₖ*1.0
            ind += 1
            # i
            Lᵢ[ind] = n₀
            Lⱼ[ind] = n₀
            Lᵥ[ind] = -wₖ*2.0
            ind += 1
            # i + 1
            Lᵢ[ind] = n₀
            Lⱼ[ind] = n₊
            Lᵥ[ind] = wₖ*1.0
            ind += 1
        else
            # Forward/Backward difference (2ⁿᵈ-order)
            if (k == 1)
                n₁ = subscripts_to_index((nx₁, nx₂, nx₃), (i, j, k+1))
                n₂ = subscripts_to_index((nx₁, nx₂, nx₃), (i, j, k+2))
                n₃ = subscripts_to_index((nx₁, nx₂, nx₃), (i, j, k+3))
            elseif (k == nx₃)
                n₁ = subscripts_to_index((nx₁, nx₂, nx₃), (i, j, k-1))
                n₂ = subscripts_to_index((nx₁, nx₂, nx₃), (i, j, k-2))
                n₃ = subscripts_to_index((nx₁, nx₂, nx₃), (i, j, k-3))
            end
            # i
            Lᵢ[ind] = n₀
            Lⱼ[ind] = n₀
            Lᵥ[ind] = wₖ*2.0
            ind += 1
            # i ± 1
            Lᵢ[ind] = n₀
            Lⱼ[ind] = n₁
            Lᵥ[ind] = -wₖ*5.0
            ind += 1
            # i ± 2
            Lᵢ[ind] = n₀
            Lⱼ[ind] = n₂
            Lᵥ[ind] = wₖ*4.0
            ind += 1
            # i ± 3
            Lᵢ[ind] = n₀
            Lⱼ[ind] = n₃
            Lᵥ[ind] = -wₖ*1.0
            ind += 1
        end
    end

    return Lᵢ, Lⱼ, Lᵥ
end




function compute_dlnV(Model::ModelTauP{<:IsotropicVelocity})
    dims = size(Model.Mesh)
    dlnVp = zeros(dims)
    dlnVs = zeros(dims)

    for k in 1:dims[3]
        rq = Model.Rₜₚ - Model.Mesh.CS.R₀ - Model.Mesh.x[3][k]
        vp = piecewise_linearly_interpolate(Model.r, Model.nd, Model.vp, rq; tf_extrapolate = true, tf_harmonic = true)
        vs = piecewise_linearly_interpolate(Model.r, Model.nd, Model.vs, rq; tf_extrapolate = true, tf_harmonic = true)
        for i in 1:dims[1], j in 1:dims[2]
            dlnVp[i, j, k] = Model.m.vp[i, j, k]/vp
            dlnVs[i, j, k] = Model.m.vs[i, j, k]/vs
        end
    end

    return dlnVp, dlnVs
end