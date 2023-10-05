#################### FORWARD PROBLEM WITH TAUP ####################

# PSI_FORWARD: TravelTime + ModelTauP
function psi_forward(b::SeismicObservable, Model::ModelTauP)
    # Compute ray path (source-to-receiver)
    x₁, x₂, x₃, rₑ, t, L = return_taup_path(Model, b.phase.name, b.source_id, b.receiver_id)
    # Build ObservableKernel
    K = ObservableKernel(b, Model, x₁, x₂, x₃, t)
    # Add entry point time
    if typeof(b) <: TravelTime
        K.q₀ .+= t[1]
    end
    # Add source and receiver statics
    K.q₀ .+= get_source_static(Model.Sources.static, b)
    K.q₀ .+= get_receiver_static(Model.Receivers.static, b)
    # Update prediction and relative residual for static correction
    q = evaluate_kernel(K)
    r = (b.obs - q)/b.error
    # println("Ray integration error: ", q - t[end], " s.")

    return K, r, q
end
# PSI_FORWARD: Vector{TravelTime} + ModelTauP
function psi_forward(B::Vector{<:SeismicObservable}, Model::AbstractModel)
    # Allocate storage arrays
    T = typeof(B[1].obs)
    n = length(B)
    q = Vector{T}(undef, n)
    r = Vector{T}(undef, n)
    K = allocate_kernel_array(Model.m, n; T = T)

    # Compute kernels
    for i in eachindex(B)
        K[i], r[i], q[i] = psi_forward(B[i], Model)
    end

    return K, r, q
end
# Allocates an array to store ObservableKernel for an IsotropicVelocity model
function allocate_kernel_array(::IsotropicVelocity, n::Int; T = Float64)
    # Kernel field types
    mtype = IsotropicVelocity{Vector{T}} # Model parameterisation
    wtype = Vector{T} # Weights
    xtype = Vector{NTuple{3, T}} # Coordinates
    qtype = Vector{T} # Static
    # Allocate Kernel Array
    K = Vector{ObservableKernel{mtype, wtype, xtype, qtype}}(undef, n)
    
    return K
end
# Allocates an array to store ObservableKernel for an HexagonalVelocity model
function allocate_kernel_array(::HexagonalVelocity, n::Int; T = Float64)
    # Kernel field types
    mtype = HexagonalVelocity{Vector{T}} # Model parameterisation
    wtype = Array{T, 2} # Weights
    xtype = Vector{NTuple{3, T}} # Coordinates
    qtype = Vector{T} # Static
    # Allocate Kernel Array
    K = Vector{ObservableKernel{mtype, wtype, xtype, qtype}}(undef, n)
    
    return K
end

# Evaluate Kernel: Integration of kernel values
function evaluate_kernel(K::ObservableKernel)
    q = K.q₀[1]
    nw = size(K.w, 1)
    # for i in eachindex(K.x)
    for i in 1:nw # Loop over weights NOT coordinates (finite-frequency coordinate have different dimensions)
        q += kernel_value(K.b, K.m, K.w, i)
    end

    return q
end
function evaluate_kernel(K::ObservableKernel, b::Observable)
    q = evaluate_kernel(K)
    r = (b.obs - q)/b.error

    return q, r
end

# Kernel value for a P-wave TravelTime in an IsotropicVelocity model
function kernel_value(::Type{<:TravelTime{<:CompressionalWave}}, m::IsotropicVelocity, w, i)
    return w[i]/m.vp[i]
end
# Kernel value for a S-wave TravelTime in an IsotropicVelocity model
function kernel_value(::Type{<:TravelTime{<:ShearWave}}, m::IsotropicVelocity, w, i)
    return w[i]/m.vs[i]
end
# Kernel value for a P-wave TravelTime in an HexagonalVelocity model
function kernel_value(::Type{<:TravelTime{<:CompressionalWave}}, m::HexagonalVelocity, w, i)
    # Compute 2x the angle between propagation direction and symmetry axis
    cosΔϕ = cos(w[i,2] - m.ϕ[i]) 
    sinθᵣ, cosθᵣ = sincos(w[i,3])
    sinθₐ, cosθₐ = sincos(m.θ[i])
    cosα = cosΔϕ*cosθᵣ*cosθₐ + sinθᵣ*sinθₐ
    cos2α = 2.0*(cosα^2) - 1.0
    # Elliptical anisotropic factor
    F = 1.0 + m.fp[i]*cos2α
    # Hexagonal anisotropic factor 4θ-component
    if ~isnothing(m.gs)
        cos4α = 2.0*(cos2α^2) - 1.0
        F += (m.gs[i]*cos4α)
    end

    return w[i,1]/(m.vp[i]*F)
end
# Kernel value for a S-wave TravelTime in a HexagonalVelocity model
function kernel_value(::Type{<:TravelTime{<:ShearWave}}, m::HexagonalVelocity, w, i)
    # Get cosine and sine terms for ray and symmetry axis orientations
    sinΔϕ, cosΔϕ = sincos(w[i,2] - m.ϕ[i]) 
    sinθᵣ, cosθᵣ = sincos(w[i,3])
    sinθₐ, cosθₐ = sincos(m.θ[i])
    # Cosine of 1x, 2x, and 4x the angle between the propagation direction and symmetry axis
    cosα = cosΔϕ*cosθᵣ*cosθₐ + sinθᵣ*sinθₐ
    cos2α = 2.0*(cosα^2) - 1.0
    cos4α = 2.0*(cos2α^2) - 1.0
    # Angle between the polarisation direction and projection of symmetry axis into ray-normal plane
    β = atan(-sinΔϕ*cosθₐ, cosΔϕ*sinθᵣ*cosθₐ - cosθᵣ*sinθₐ) - w[i, 4]
    # Shear slowness polarised parallel to symmetry axis
    u4 = 1.0/(m.vs[i]*((1.0 + m.fs[i])/(1.0 + m.gs[i]))*(1.0 + m.gs[i]*cos4α))
    # Shear slowness polarised normal to symmetry axis
    u2 = 1.0/(m.vs[i]*(1.0 + m.fs[i]*cos2α))
    # Anisotropic shear slowness
    u = u2 - (u2 - u4)*(cos(β)^2)

    return w[i,1]*u
end
# Kernel value for SplittingIntensity in a HexagonalVelocity model
function kernel_value(::Type{<:SplittingIntensity}, m::HexagonalVelocity, w, i)
    # Get cosine and sine terms for ray and symmetry axis orientations
    sinΔϕ, cosΔϕ = sincos(w[i,2] - m.ϕ[i]) 
    sinθᵣ, cosθᵣ = sincos(w[i,3])
    sinθₐ, cosθₐ = sincos(m.θ[i])
    # Cosine of 1x, 2x, and 4x the angle between the propagation direction and symmetry axis
    cosα = cosΔϕ*cosθᵣ*cosθₐ + sinθᵣ*sinθₐ
    cos2α = 2.0*(cosα^2) - 1.0
    cos4α = 2.0*(cos2α^2) - 1.0
    # Angle between the polarisation direction and projection of symmetry axis into ray-normal plane
    β = atan(-sinΔϕ*cosθₐ, cosΔϕ*sinθᵣ*cosθₐ - cosθᵣ*sinθₐ) - w[i, 4]
    # Shear slowness polarised parallel to symmetry axis
    u4 = 1.0/(m.vs[i]*((1.0 + m.fs[i])/(1.0 + m.gs[i]))*(1.0 + m.gs[i]*cos4α))
    # Shear slowness polarised normal to symmetry axis
    u2 = 1.0/(m.vs[i]*(1.0 + m.fs[i]*cos2α))
    # Splitting intensity
    Δs = 0.5*(u2 - u4)*sin(2.0*β)

    return w[i,1]*Δs
end



# Build an ObservableKernel for any SeismicObservable predictable by a IsotropicVelocity ModelTauP
function ObservableKernel(B::SeismicObservable, Model::ModelTauP{<:IsotropicVelocity}, x₁, x₂, x₃, tq)
    # Allocate kernel structure
    T = eltype(tq)
    nq = length(tq)
    K = ObservableKernel(typeof(B), IsotropicVelocity(T, B.phase, nq), similar(tq),
    Vector{NTuple{3, T}}(undef, nq), zeros(T,1))

    # Store ray path nodes as tuple in kernel
    map!((n₁,n₂,n₃) -> (n₁,n₂,n₃), K.x, x₁, x₂, x₃)

    # Compute ray node segment lengths
    fill_raynode_weights!(K.w, x₁, x₂, x₃)

    # Define effective node velocities
    fill_effective_velocities!(K, tq)

    # Interpolate existing velocity perturbations to the kernel model
    interpolate_model!(B.phase, K, Model; tf_extrapolate = true, tf_harmonic = false)

    return K
end
# Build an ObservableKernel for any SeismicObservable predictable by a HexagonalVelocity ModelTauP
function ObservableKernel(B::SeismicObservable, Model::ModelTauP{<:HexagonalVelocity}, x₁, x₂, x₃, tq)
    # Allocate kernel structure
    T = eltype(tq)
    nq = length(tq)
    K = ObservableKernel(typeof(B), HexagonalVelocity(T, B.phase, nq; tf_elliptical = isnothing(Model.m.gs)),
    Array{T}(undef, (nq, 4)), Vector{NTuple{3, T}}(undef, nq), zeros(T,1))

    # Store ray path nodes as tuple in kernel
    map!((n₁,n₂,n₃) -> (n₁,n₂,n₃), K.x, x₁, x₂, x₃)

    # Compute ray node segment lengths and orientations
    fill_raynode_weights!(K.w, x₁, x₂, x₃)

    # Compute ray node polarisations for S-waves
    if typeof(B.phase) <: ShearWave
        fill_raynode_polarisations!(K, Model, B.source_id, B.phase.paz)
    end

    # Define effective node velocities
    fill_effective_velocities!(K, tq)

    # Interpolate existing velocity perturbations to the kernel model
    interpolate_model!(B.phase, K, Model; tf_extrapolate = true)

    return K
end


# Scalar ray node weights
function fill_raynode_weights!(wq::Vector{T}, x₁, x₂, x₃) where {T}
    nq = size(wq, 1)
    # Define initial segment length between nodes (i - 1) and i
    wⱼ = sqrt(((x₁[2] - x₁[1])^2) + ((x₂[2] - x₂[1])^2) + ((x₃[2] - x₃[1])^2))
    # Half-weight for the first ray node
    wq[1] = 0.5*wⱼ
    # Assign inner ray node weights
    for i in 2:(nq - 1)
        # Length of ray segment between nodes i and (i + 1)
        wⱼ₊₁ = sqrt(((x₁[i+1] - x₁[i])^2) + ((x₂[i+1] - x₂[i])^2) + ((x₃[i+1] - x₃[i])^2))
        # Ray node weight is the average of its segment lengths
        wq[i] = 0.5*(wⱼ + wⱼ₊₁)
        # Update segment length between nodes (i - 1) and i for next iteration
        wⱼ = wⱼ₊₁
    end
    # Half-weight for last ray node
    wq[nq] = 0.5*wⱼ

    return nothing
end
# Vector ray node weights
function fill_raynode_weights!(wq::Array{T, 2}, x₁, x₂, x₃) where {T}
    nq = size(wq, 1)
    # Define initial segment length between nodes (i - 1) and i
    wⱼ = sqrt(((x₁[2] - x₁[1])^2) + ((x₂[2] - x₂[1])^2) + ((x₃[2] - x₃[1])^2))
    # Half-weight for the first ray node
    wq[1, 1] = 0.5*wⱼ
    # Segment vector components --> Note component re-ordering!
    r₃ = (x₁[2] - x₁[1])
    r₁ = (x₂[2] - x₂[1])
    r₂ = (x₃[2] - x₃[1])
    # Spherical segment parameters
    wq[1, 2] = atan(r₂, r₁)
    wq[1, 3] = atan(r₃, sqrt((r₁^2) + (r₂^2)))
    # Assign inner ray node weights
    for i in 2:(nq - 1)
        # Length of ray segment between nodes i and (i + 1)
        wⱼ₊₁ = sqrt(((x₁[i+1] - x₁[i])^2) + ((x₂[i+1] - x₂[i])^2) + ((x₃[i+1] - x₃[i])^2))
        # Ray node weight is the average of its segment lengths
        wq[i, 1] = 0.5*(wⱼ + wⱼ₊₁)
        # Segment vector components --> Note component re-ordering!
        r₃ = (x₁[i+1] - x₁[i-1])
        r₁ = (x₂[i+1] - x₂[i-1])
        r₂ = (x₃[i+1] - x₃[i-1])
        # Segment orientation
        wq[i, 2] = atan(r₂, r₁)
        wq[i, 3] = atan(r₃, sqrt((r₁^2) + (r₂^2)))
        # Update segment length between nodes (i - 1) and i for next iteration
        wⱼ = wⱼ₊₁
    end
    # Half-weight for last ray node
    wq[nq, 1] = 0.5*wⱼ
    # Segment vector components --> Note component re-ordering!
    r₃ = (x₁[nq] - x₁[nq-1])
    r₁ = (x₂[nq] - x₂[nq-1])
    r₂ = (x₃[nq] - x₃[nq-1])
    # Spherical segment parameters
    wq[nq, 2] = atan(r₂, r₁)
    wq[nq, 3] = atan(r₃, sqrt((r₁^2) + (r₂^2)))

    return nothing
end
function fill_raynode_polarisations!(K::ObservableKernel, Model::AbstractModel, sid, ζ)
    # Geographic source coordinates
    nₛ = Model.Sources.id[sid]
    λ₀ = Model.Sources.λ[nₛ]
    ϕ₀ = Model.Sources.ϕ[nₛ]
    c = π/180.0 # Degrees-to-radians mulitplier
    for (i, xi) in enumerate(K.x)
        # Geographic ray coordinates
        λᵢ, ϕᵢ, _ = global_to_geographic(xi[1], xi[2], xi[3]; radius = Model.Mesh.CS.R₀)
        # λᵢ, ϕᵢ, _ = global_to_geographic(K.x[end][1], K.x[end][2], K.x[end][3]; radius = Model.Mesh.CS.R₀) # No correction +0.01 ms difference
        # Back-azimuth of source with respect to the ray node
        _, βᵢ = inverse_geodesic(ϕᵢ, λᵢ, ϕ₀, λ₀; tf_degrees = true)
        βᵢ = 270.0 - βᵢ # Convert back-azimuth to azimuth counter-clockwise positive from east
        βᵢ *= c # Convert to radians
        # The polarisation azimuth in the ray-normal plane. This correction is necessary because
        # the polarisation azimuth was measured in the QTL coordinate system which may not correspond
        # exactly to the ray-aligned coordinates. The correction is not a scalar because the
        # polarisation direction twists along the ray.
        # -> Actually, I beleive the correction to the polarisation azimuth should be a scalar
        # -> Just need to correct for difference between ray azimuth and back-azimuth at surface
        # -> The polarisation azimuth should be constant in QTL reference frame
        K.w[i, 4] = ζ + βᵢ - K.w[i, 2]
    end

    return nothing
end
# Compute ray node effective velocities (as opposed to interpolating)
function fill_effective_velocities!(K::ObservableKernel{<:SeismicVelocity}, tq)
    nq = length(tq)
    if ~isnothing(K.m.vp)
        K.m.vp[1] = 2.0*K.w[1, 1]/(tq[2] - tq[1])
        for i in 2:(nq - 1)
            # Average travel-time along adjacent segments
            tᵢ = 0.5*((tq[i] - tq[i-1]) + (tq[i+1] - tq[i]))
            K.m.vp[i] = K.w[i, 1]/tᵢ
        end
        K.m.vp[nq] = 2.0*K.w[nq, 1]/(tq[nq] - tq[nq-1])
    end
    if ~isnothing(K.m.vs)
        K.m.vs[1] = 2.0*K.w[1, 1]/(tq[2] - tq[1])
        for i in 2:(nq - 1)
            # Average travel-time along adjacent segments
            tᵢ = 0.5*((tq[i] - tq[i-1]) + (tq[i+1] - tq[i]))
            K.m.vs[i] = K.w[i, 1]/tᵢ
        end
        K.m.vs[nq] = 2.0*K.w[nq, 1]/(tq[nq] - tq[nq-1])
    end

    return nothing
end



##### STATIC DICTIONARY FUNCTIONS ######

# Use 'nameof(b)' 'nameof(b.phase)' to get just the structure name as a symbol
function get_source_static(δ::Dict, b::Observable)
    # Define key (id + observable + phase)
    k = (b.source_id, nameof(typeof(b)), nameof(typeof(b.phase)))
    if haskey(δ, k)
        δₖ = δ[k]
    else
        δₖ = zero(valtype(δ))
    end

    return δₖ
end
function get_receiver_static(δ::Dict, b::Observable)
    # Define key (id + observable + phase)
    k = (b.receiver_id, nameof(typeof(b)), nameof(typeof(b.phase)))
    if haskey(δ, k)
        δₖ = δ[k]
    else
        # println("No Static for key: ", k)
        δₖ = zero(valtype(δ))
    end

    return δₖ
end





## RAY TRACING ##

# Return a single ray path through a TauP model
function return_taup_path(Model::ModelTauP, aphase, sid, rid)
    # Source coordinates
    iₛ = Model.Sources.id[sid]
    λ₀ = Model.Sources.λ[iₛ]
    ϕ₀ = Model.Sources.ϕ[iₛ]
    r₀ = Model.Sources.r[iₛ]
    # Receiver coordinates
    iᵣ = Model.Receivers.id[rid]
    λ₁ = Model.Receivers.λ[iᵣ]
    ϕ₁ = Model.Receivers.ϕ[iᵣ]
    r₁ = Model.Receivers.r[iᵣ]
    # Compute TauP ray
    λ, ϕ, r, t, L, _ = return_taup_path(aphase, λ₀, ϕ₀, r₀, λ₁, ϕ₁, r₁, Model.PathObj, Model.Mesh; Rₜ = Model.Rₜₚ, dl₀ = Model.dl₀)
    # Convert to global model coordinates
    x₁, x₂, x₃ = geographic_to_global(λ, ϕ, r; radius = Model.Mesh.CS.R₀)

    return x₁, x₂, x₃, r, t, L
end

# Returns the TauP ray path through the local mesh in true geographic coordinates (λ, ϕ, r) along with
# the travel-time along the ray (t). Additionally, the ray path in local model coordinates (x, y, z)
# is also returned along with the acr-distance along the path (d)
function return_taup_path(phase, λ₀, ϕ₀, r₀, λ₁, ϕ₁, r₁, PathObj, Mesh::AbstractMesh; Rₜ = 6371.0, dl₀ = 0.0)
    # Convert elevations to TauP model depths
    ΔR = Rₜ - Mesh.CS.R₀
    h₀ = ΔR - r₀
    h₁ = ΔR - r₁
    # Source-to-receiver arc degrees and bearing
    Δ, α = inverse_geodesic(ϕ₀, λ₀, ϕ₁, λ₁; tf_degrees = true)
    # Compute TauP ray
    d, r, t = taup_path!(PathObj, phase, Δ, h₀, h₁)
    # Total ray path length
    L = taup_ray_length(d, r, Rₜ)
    # Convert TauP depth to elevation (in-place operation of Rₜ - R₀ - r)
    r .= (ΔR .- r)
    # Retrieve geographic coordinates
    ϕ, λ = direct_geodesic(ϕ₀, λ₀, d, α; tf_degrees = true)
    # Extract local ray path (includes first node outside the model domain)
    return_local_path!(d, r, t, λ, ϕ, Mesh; tf_first_out = true)
    # Re-sample ray path
    if dl₀ > 0.0
        # Re-sample ray path
        d, r, t = resample_path(d, r, t, dl₀; R₀ = Rₜ)
        # Re-derive geographic coordinates
        ϕ, λ = direct_geodesic(ϕ₀, λ₀, d, α; tf_degrees = true)
        # Extract local ray path (excludes first node outside model domain)
        x, y, z = return_local_path!(d, r, t, λ, ϕ, Mesh; tf_first_out = false)
    end

    return λ, ϕ, r, t, L, x, y, z, d
end

function taup_ray_length(d, r, Rₜₚ)
    ns = length(d) - 1
    c = π/180.0
    L = 0.0
    for i in 1:ns
        sinΔ, cosΔ = sincos(c*d[i])
        R1 = Rₜₚ - r[i]
        x1 = R1*sinΔ
        z1 = R1*cosΔ

        sinΔ, cosΔ = sincos(c*d[i+1])
        R2 = Rₜₚ - r[i+1]
        x2 = R2*sinΔ
        z2 = R2*cosΔ

        L += sqrt((x2 - x1)^2 + (z2 - z1)^2)
    end

    return L
end

# Trims the ray path provided in polar (d,r) and true geographic (λ, ϕ) coordinates
# to only include ray nodes inside local mesh. Also returns raypath in the local
# coordinate system (x, y, z)
function return_local_path!(d, r, t, λ, ϕ, Mesh::RegularGrid; tf_first_out = true)
    # Total ray nodes
    M = length(d)
    # Local path coordinates
    x, y, z = geographic_to_local(λ, ϕ, r, Mesh.CS)
    # Model limits
    xminmax, yminmax, zminmax = extrema(Mesh)
    # Account for elevation in maximum z-coordinate
    zmax = maximum(z) # Account for elevation in maximum z-coordinate

    if tf_first_out
        # Find first node outside model counting from receiver
        iout = get_first_out(x, y, z, xminmax[1], xminmax[2], yminmax[1], yminmax[2], zminmax[1], zmax; i = M, Δi = -1)
        iout += -1 # Keep last node out
        # Check first node out is not receiver
        if iout >= (M - 1)
            error("Receiver located outside model space!")
        end
    else
        # Find first node inside model counting from source-side
        iout = get_first_in(x, y, z, xminmax[1], xminmax[2], yminmax[1], yminmax[2], zminmax[1], zmax; i = 1, Δi = 1)
        iout += -1 # Last node outside domain
    end

    # Remove source-side nodes from local path
    if iout > 0
        splice!(d, 1:iout)
        splice!(r, 1:iout)
        splice!(t, 1:iout)
        splice!(λ, 1:iout)
        splice!(ϕ, 1:iout)
        splice!(x, 1:iout)
        splice!(y, 1:iout)
        splice!(z, 1:iout)
    end

    return x, y, z
end

# Does this give uniform along-ray spacing? I think no. Spacing between nodes is constant
# if they fall on the same segment but can change when they cross segments? Think about resampling V-shape path.
function resample_path(d, r, t, dl₀; R₀ = 6371.0)
    # Cartesian ccoordinates in plane containing path
    K = π/180.0
    x₁ = similar(d)
    x₂ = similar(d)
    for i in eachindex(d)
        x₁[i], x₂[i] = sincos(K*d[i])
        x₁[i] *= (R₀ + r[i])
        x₂[i] *= (R₀ + r[i])
    end
    # Normalised along-path distance
    l = cumsum(sqrt.(diff(x₁).^2 + diff(x₂).^2))
    pushfirst!(l, 0.0)
    # Define number of samples
    N = 1 + round(Int, (l[end] - l[1])/dl₀)
    # Re-sample the ray in *cartesian* coordinates
    x₁ = resample(l, x₁, N)
    x₂ = resample(l, x₂, N)
    t  = resample(l, t, N)
    # Convert back to polar coordinates
    d = atand.(x₁, x₂)
    r = sqrt.((x₁.^2) + (x₂.^2)) .- R₀

    return d, r, t
end



# RANDOM UTILITIES

function mean_source_delay(r, B::Vector{<:SeismicObservable}, S::SeismicSources)
    # Allocate vector to store delays
    Δt = similar(r)
    # Associate each observable to a source index
    sid = Vector{Int}(undef, length(Δt))
    for i in eachindex(B)
        sid[i] = S.id[B[i].source_id]
        Δt[i] = r[i]*B[i].error
    end
    # Initialise accumulation arrays
    msd = zeros(length(S.id))
    num = zeros(Int, length(S.id))
    # Sum the delays for every source
    accumarray!(msd, sid, Δt)
    # Sum the number of observations for each source
    accumarray!(num, sid, Δt, (x) -> 1)
    # Compute mean
    msd ./= num
    # Demeaned residuals
    Δt .-= msd[sid]

    return Δt, msd, sid
end