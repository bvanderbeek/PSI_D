# Small changes to make
# SeismicPhase
# + Store period here instead of observation structure


##################### MODEL STRUCTURES #######################

# Abstract Mesh: Describes the spatial discretisation of model parameters
# Only implemented regular grids. Consider using Mesh.jl package for more comples geometries.
abstract type AbstractMesh end

# Regular Grid: Grid of dimension 'N' with uniform spacing in each dimension.
struct RegularGrid{G<:CoordinateSystem, N, R<:AbstractRange} <: AbstractMesh
    Geometry::G # Coordinate System Geometry
    x::NTuple{N, R} # Coordinate Tuple
end
# Add functions to base for this mesh
# Dimensions of mesh
function Base.ndims(::RegularGrid{T, N, R}) where {T, N, R}
    return N
end
# Size returns the dimensions of the mesh
function Base.size(Mesh::RegularGrid{T, 3, R}) where {T, R}
    return length(Mesh.x[1]), length(Mesh.x[2]), length(Mesh.x[3])
end
# Length returns the number of elements in the mesh
function Base.length(Mesh::RegularGrid)
    return prod(size(Mesh))
end
# Extrema returns the minimum/maximum values of the mesh in each dimension
function Base.extrema(Mesh::RegularGrid{T, 3, R}) where {T, R}
    return extrema(Mesh.x[1]), extrema(Mesh.x[2]), extrema(Mesh.x[3])
end

# Makes more sense for 'id' to be an array of source identifiers and 'coordinates' to be a dictionary
# of source locations idexed by 'id'. Same for SeismicReceivers.
# Basic Seismic Sources Structure
struct SeismicSources{S, F, K, N}
    id::Dict{S, Int} # Unique identifier
    coordinates::Array{NTuple{N, F}} # Source Coordinate Tuples
    statics::Dict{Tuple{S, K, K}, F} # Static dictionary (id, :observable, :phase)
end
# Build SeismicSources from arrays of identifiers and coordinates with empty static dictionary
function SeismicSources(id, coordinates; ObservableKey = Symbol, PhaseKey = Symbol, StaticType = Float64)
    # Initialise SeismicSources with empty dictionaries
    idKey = eltype(id)
    n = length(id)

    return SeismicSources(Dict(zip(id, 1:n)), coordinates, Dict{Tuple{idKey, ObservableKey, PhaseKey}, StaticType}())
end
# Build SeismicSources with zero-valued statics for every (id, static_observable, static_phase) tuple
function SeismicSources(id, coordinates, static_observable, static_phases; static_type = Float64)
    # Initialise SeismicSources with empty dictionaries
    S = SeismicSources(id, coordinates; ObservableKey = eltype(static_observable), PhaseKey = eltype(static_phases), StaticType = static_type)
    # Fill static dictionary
    initialise_static_dictionaries!(S, id, static_observable, static_phases)

    return S
end

# Basic Seismic Receivers Structure
struct SeismicReceivers{S, F, K, N}
    id::Dict{S, Int} # Unique identifier
    coordinates::Array{NTuple{N, F}} # Source Coordinate Tuples
    statics::Dict{Tuple{S, K, K}, F} # Static dictionary (id, :observable, :phase)
end
function SeismicReceivers(id, coordinates; ObservableKey = Symbol, PhaseKey = Symbol, StaticType = Float64)
    # Initialise SeismicReceivers with empty dictionaries
    idKey = eltype(id)
    n = length(id)

    return SeismicReceivers(Dict(zip(id, 1:n)), coordinates, Dict{Tuple{idKey, ObservableKey, PhaseKey}, StaticType}())
end
function SeismicReceivers(id, coordinates, static_observable, static_phases; static_type = Float64)
    # Initialise SeismicReceivers with empty dictionaries
    R = SeismicReceivers(id, coordinates; ObservableKey = eltype(static_observable), PhaseKey = eltype(static_phases), StaticType = static_type)
    # Fill static dictionary
    initialise_static_dictionaries!(R, id, static_observable, static_phases)

    return R
end

# Initialise static dictionaries for SeismicSources and SeismicReceivers
function initialise_static_dictionaries!(S, Ki, Ko, Kp)
    for k in eachindex(Kp)
        for j in eachindex(Ko)
            for i in eachindex(Ki)
                S.static[(Ki[i], Ko[j], Kp[k])] = 0.0
            end
        end
    end

    return nothing
end

abstract type ModelParameterization end
abstract type SeismicVelocity <: ModelParameterization end

# Isotropic Velocity
struct IsotropicVelocity{T} <: SeismicVelocity
    vp::Union{T, Nothing}
    vs::Union{T, Nothing}
end
function IsotropicVelocity(T::DataType, dim)
    return IsotropicVelocity(zeros(dim), zeros(dim))
end
function return_isotropic_velocities(Parameters::IsotropicVelocity)
    return deepcopy(Parameters.vp), deepcopy(Parameters.vs)
end
function return_velocity_fields(Parameters::IsotropicVelocity)
    return getfield(Parameters, :vp), getfield(Parameters, :vs)
end


# Hexagonal Vectoral Velocity
struct HexagonalVectoralVelocity{T, R} <: SeismicVelocity
    α::Union{T, Nothing} # Symmetry axis P-velocity defined such that α² = c₃₃/ρ
    β::Union{T, Nothing} # Symmetry axis S-velocity defined such that β² = c₄₄/ρ
    f::Union{T, Nothing} # Anisotropic magnitude (f ≥ 0)
    azimuth::Union{T, Nothing} # Anisotropic symmetry axis azimuth
    elevation::Union{T, Nothing} # Anisotropic symmetry axis elevation
    ratio_ϵ::Union{R, Nothing} # Thomsen parameter ratio ϵ/f where ϵ = (c₁₁ - c₃₃)/(2c₃₃)
    ratio_η::Union{R, Nothing} # Thomsen parameter ratio η/f = (ϵ - δ)/f; δ is defined in note below
    ratio_γ::Union{R, Nothing} # Thomsen parameter ratio γ/f where γ = (c₆₆ - c₄₄)/(2c₄₄)
    tf_exact::Bool # Use exact (true) or weak (false) form of Thomsen's phase velocity equations
    # Exact form of Thomsen's velocity implies the following definition of δ,
    #   δ = ( 2(c₁₃ + c₄₄)² - (c₃₃ - c₄₄)(c₁₁ + c₃₃ - 2c₄₄) ) / (2c₃₃²); Thomsen, 1986; Eq. 8c.
    # Alternatively, the weak form of the δ parameter may be defined as
    #   δ = ( (c₁₃² + c₄₄²) - (c₃₃² - c₄₄²) ) / ( 2c₃₃(c₃₃ - c₄₄) ); Thomsen, 1986; Eq. 17.
    # A simplier and no-less-accurate definition of the δ parameter is,
    #   δ = (c₁₃ - c₃₃ + 2c₄₄)/c₃₃; Mensch and Rasolofosaon, 1997; Eq. 20.
end
function return_thomsen_parameters(Parameters::HexagonalVectoralVelocity{T, R}, index) where {T, R}
    # Reference velocities and anisotropic strength
    α = isnothing(Parameters.α) ? 0.0 : Parameters.α[index]
    β = isnothing(Parameters.β) ? 0.0 : Parameters.β[index]
    f = isnothing(Parameters.f) ? 0.0 : Parameters.f[index]
    # Anisotropic fractions
    if R <: Number
        ϵ = isnothing(Parameters.ratio_ϵ) ? 0.0 : f*Parameters.ratio_ϵ
        η = isnothing(Parameters.ratio_η) ? 0.0 : f*Parameters.ratio_η
        γ = isnothing(Parameters.ratio_γ) ? 0.0 : f*Parameters.ratio_γ
    else
        # Ratio parameters may be stored as single element vectors such that they can be updated.
        # This re-definition of index handles this situtation.
        index = min(index, length(Parameters.ratio_η))
        ϵ = isnothing(Parameters.ratio_ϵ) ? 0.0 : f*Parameters.ratio_ϵ[index]
        η = isnothing(Parameters.ratio_η) ? 0.0 : f*Parameters.ratio_η[index]
        γ = isnothing(Parameters.ratio_γ) ? 0.0 : f*Parameters.ratio_γ[index]
    end
    # Note, returns the Thomsen parameter δ = ϵ - η
    return α, β, ϵ, ϵ - η, γ
end
function return_anisotropic_ratios(Parameters::HexagonalVectoralVelocity{T, R}, index) where {T, R}
    # Anisotropic fractions
    if R <: Number
        ratio_ϵ = isnothing(Parameters.ratio_ϵ) ? 0.0 : Parameters.ratio_ϵ
        ratio_η = isnothing(Parameters.ratio_η) ? 0.0 : Parameters.ratio_η
        ratio_γ = isnothing(Parameters.ratio_γ) ? 0.0 : Parameters.ratio_γ
    else
        # Ratio parameters may be stored as single element vectors such that they can be updated.
        # This re-definition of index handles this situtation.
        index = min(index, length(Parameters.ratio_η))
        ratio_ϵ = isnothing(Parameters.ratio_ϵ) ? 0.0 : Parameters.ratio_ϵ[index]
        ratio_η = isnothing(Parameters.ratio_η) ? 0.0 : Parameters.ratio_η[index]
        ratio_γ = isnothing(Parameters.ratio_γ) ? 0.0 : Parameters.ratio_γ[index]
    end

    return ratio_ϵ, ratio_η, ratio_γ
end
function return_isotropic_velocities(Parameters::HexagonalVectoralVelocity{T, R}, index) where {T, R}
    α, β, ϵ, δ, γ = return_thomsen_parameters(Parameters, index)
    vip = α*sqrt( 1.0 + (16/15)*ϵ + (4/15)*δ )
    vis = β*sqrt( 1.0 + (2/3)*γ + (2/15)*((α^2)/(β^2))*(ϵ - δ) )
    return vip, vis
end
function return_isotropic_velocities(Parameters::HexagonalVectoralVelocity{T, R}) where {T, R}
    vp, vs = (similar(Parameters.f), similar(Parameters.f))
    for i in eachindex(Parameters.f)
        vp[i], vs[i] = return_isotropic_velocities(Parameters, i)
    end

    return vp, vs
end
function return_reference_velocities(Parameters::HexagonalVectoralVelocity{T,R}, vip, vis, index) where {T, R}
    _, _, ϵ, δ, γ = return_thomsen_parameters(Parameters, index)
    α = vip/sqrt( 1.0 + (16/15)*ϵ + (4/15)*δ )
    β = sqrt( (vis^2)*( 1.0 - 2.0*((vip^2)/(vis^2))*((ϵ - δ)/(15.0 + 16.0*ϵ + 4.0*δ)) )/( 1.0 + (2/3)*γ ) )
    return α, β
end
function return_velocity_fields(Parameters::HexagonalVectoralVelocity)
    return getfield(Parameters, :α), getfield(Parameters, :β)
end


# Forward Modelling Methods and Parameter Containers
abstract type ForwardMethod end
abstract type ForwardMethodParameters <: ForwardMethod end
# TauP
struct ForwardTauP <: ForwardMethod end
struct ParametersTauP{J} <: ForwardMethodParameters
    PathObj::TauP.JavaCall.JavaObject{J} # TauP Java Path Object
    reference_model::String
    radius::Float64
    dL::Float64
end
function ParametersTauP(; reference_model = "ak135", dL = 10.0)
    PathObj = buildPathObj(reference_model)
    ModelTauP = load_taup_model(reference_model)

    return ParametersTauP(PathObj, reference_model, ModelTauP.r[end], dL)
end
# Shortest Path
struct ForwardShortestPath <: ForwardMethod end
struct ParametersShortestPath <: ForwardMethodParameters end
# Finite Frequency
struct ForwardFiniteFrequency <: ForwardMethod end
struct ParametersFiniteFrequency <: ForwardMethodParameters end
# Container for parameters
struct ForwardMethodsContainer
    TauP::Union{ParametersTauP, Nothing}
    ShortestPath::Union{ParametersShortestPath, Nothing}
    FiniteFrequency::Union{ParametersFiniteFrequency, Nothing}
end


# Basic PSI Model Structure
struct PsiModel{P<:ModelParameterization, G, S, R}
    Mesh::G # Mesh
    Sources::S # Seismic Sources
    Receivers::R # Seismic Receivers
    Parameters::P # Model Parameters
    Methods::ForwardMethodsContainer
end
function PsiModel(::Type{<:IsotropicVelocity}, Mesh, Sources, Receivers, Methods)
    vp, vs = return_taup_velocities(Methods.TauP.reference_model, Mesh)
    # Build seismic velocity parameters from isotropic velocities
    Parameters = IsotropicVelocity(vp, vs)

    return PsiModel(Mesh, Sources, Receivers, Parameters, Methods)
end
function PsiModel(::Type{<:HexagonalVectoralVelocity}, ratio_ϵ, ratio_η, ratio_γ, Mesh, Sources, Receivers, Methods)
    vp, vs = return_taup_velocities(Methods.TauP.reference_model, Mesh)
    # Build seismic velocity parameters from isotropic velocities
    T = eltype(vp)
    Parameters = HexagonalVectoralVelocity(vp, vs, zeros(T, size(Mesh)), zeros(T, size(Mesh)), zeros(T, size(Mesh)),
    ratio_ϵ, ratio_η, ratio_γ, false)

    return PsiModel(Mesh, Sources, Receivers, Parameters, Methods)
end
function return_taup_velocities(reference_model, Mesh)
    # Load the reference TauP model
    ModelTauP = load_taup_model(reference_model)
    # Difference between TauP and Mesh radii
    ΔR = ModelTauP.r[end] - Mesh.Geometry.R₀
    # Create null isotropic seismic velocity arrays
    T = eltype(ModelTauP.vp)
    vp = zeros(T, size(Mesh))
    vs = zeros(T, size(Mesh))
    # Fill velocities
    for (k, zk) in enumerate(Mesh.x[3])
        # Convert Mesh elevation to radial depth in TauP model
        qr = ΔR - zk
        vp_k = piecewise_linearly_interpolate(ModelTauP.r, ModelTauP.dindex, ModelTauP.vp, qr; tf_extrapolate = true, tf_harmonic = false)
        vs_k = piecewise_linearly_interpolate(ModelTauP.r, ModelTauP.dindex, ModelTauP.vs, qr; tf_extrapolate = true, tf_harmonic = false)
        vp[:,:,k] .= vp_k
        vs[:,:,k] .= vs_k
    end

    return vp, vs
end

############## OBSERVABLE STRUCTURES ##############

# Observable: Any measurable quantity that can be predicted. Given a Model
# and an Observable, there should be no ambiguity in how to make a prediction.
abstract type Observable end

# Seismic Observable: Any quantity measured from a seismic wave
abstract type SeismicObservable <: Observable end
# These should all have the same fields?

# Seismic observables are always associated with a phase. Define phase types.
abstract type SeismicPhase end
# Body Wave Seismic Phases
struct CompressionalWave <: SeismicPhase
    name::String
    period::Float64
end
struct ShearWave <: SeismicPhase
    name::String
    period::Float64
    paz::Float64 # Polarization azimuth in QTL-coordinates (e.g., Radial = 0, Transverse = π/2)
end
# If no polarisation provided, default to radially polarised
function ShearWave(name, period)
    return ShearWave(name, period, 0.0)
end 

# TravelTime Observation
struct TravelTime{P, F, S, R, T} <: SeismicObservable
    Phase::P
    Forward::F
    source_id::S
    receiver_id::R
    observation::T
    error::T
end

# Splitting Intensity Observation
struct SplittingIntensity{P, F, S, R, T} <: SeismicObservable
    Phase::P
    Forward::F
    source_id::S
    receiver_id::R
    observation::T
    error::T
end

# Splitting Parameters
struct SplittingParameters{P, F, S, R, T} <: SeismicObservable
    Phase::P
    Forward::F
    source_id::S
    receiver_id::R
    observation::T
    error::T
end

struct ObservableKernel{B<:Observable, P<:ModelParameterization, C, W, T}
    Observation::B
    Parameters::P
    coordinates::C
    weights::W
    static::T
end








# PSI FORWARD: Parameter File
function psi_forward(parameter_file::String)
    # Load inputs
    PsiParameters, Observations, ForwardModel, _ = build_inputs(parameter_file)
    # Call forward problem
    predictions, relative_residuals, Kernels = psi_forward(Observations, ForwardModel)
    # Save results
    if !isempty(PsiParameters["Output"]["output_directory"])
        # Create time-stamped directory?
        if PsiParameters["Output"]["tf_time_stamp"]
            date_now, time_now = split(string(now()), "T")
            date_now, time_now = (split(date_now, "-"), split(time_now, ":"))
            PsiParameters["Output"]["output_directory"] *= "/"*date_now[1][3:4]*prod(date_now[2:end])*"_"*prod(time_now[1:2])*time_now[3][1:2]
        end
        mkpath(PsiParameters["Output"]["output_directory"])
        # Save predictions and copy of parameter file
        write_observations(PsiParameters["Output"]["output_directory"], Observations; alt_data = predictions, prepend = "SYN")
        path_filename = splitdir(parameter_file)
        cp(parameter_file, PsiParameters["Output"]["output_directory"]*"/"*path_filename[2])
    end

    return PsiParameters, Observations, ForwardModel, relative_residuals, Kernels
end
# PSI FORWARD: Abstract Observable + PSI Model
function psi_forward(Observation::Observable, Model::PsiModel)
    # Build the kernel structure for the particular observable
    Kernel = return_kernel(Observation, Model)
    # Evaluate the kernel
    prediction, relative_residual = evaluate_kernel(Kernel)

    return prediction, relative_residual, Kernel
end
# PSI FORWARD: Vector of Observables + PSI Model
# Assumption: Single observation type
# Assumption: Kernel weight tuple has fixed names
function psi_forward(Observations::Vector{B}, Model::PsiModel{P}) where {B <: Observable, P}
    # Allocate storage arrays
    n = length(Observations)
    val_type = eltype(Model.Mesh.x[1]) # Type assumed for model parameter and coordinate values
    obs_type = typeof(Observations[1].observation) # Type that stores an observation.......COULD FAIL FOR MULTIOBSERVABLES
    predictions = Vector{obs_type}(undef, n)
    relative_residuals = Vector{obs_type}(undef, n)
    # Vector of kernels
    param_type = return_kernel_parameter_type(P, val_type)
    coord_type = Vector{NTuple{ndims(Model.Mesh), val_type}}
    weight_type = Vector{NamedTuple{(:dr, :azimuth, :elevation), NTuple{3, val_type}}} # Assuming fixed names for weights!
    KernelTypes = return_kernel_types(B, param_type, coord_type, weight_type, obs_type)
    Kernels = Vector{KernelTypes}(undef, n); # Vector{ObservableKernel{B, param_type, coord_type, weight_type, Vector{obs_type}}}(undef, n)

    # Compute kernels
    for i in eachindex(Observations)
        predictions[i], relative_residuals[i], Kernels[i] = psi_forward(Observations[i], Model)
    end

    return predictions, relative_residuals, Kernels
end
function return_kernel_parameter_type(::Type{IsotropicVelocity{T}}, V) where {T <: Array}
    return IsotropicVelocity{Vector{V}}
end
function return_kernel_parameter_type(::Type{HexagonalVectoralVelocity{T, R}}, V) where {T <: Array, R <: Number}
    return HexagonalVectoralVelocity{Vector{V}, R}
end
function return_kernel_parameter_type(::Type{HexagonalVectoralVelocity{T, R}}, V) where {T <: Array, R <: Array}
    return HexagonalVectoralVelocity{Vector{V}, Vector{V}}
end
function return_kernel_types(ObsTypes::Union, param_type, coord_type, weight_type, value_type)
    KernelTypes = Union{}
    SubType = ObsTypes
    while isa(SubType, Union)
        # Identify concrete and Union types (not clear in which field they will be stored)
        aType, SubType = isa(SubType.b, Union) ? (SubType.a, SubType.b) : (SubType.b, SubType.a)
        KernelTypes = Union{ObservableKernel{aType, param_type, coord_type, weight_type, Vector{value_type}}, KernelTypes}
    end
    KernelTypes = Union{ObservableKernel{SubType, param_type, coord_type, weight_type, Vector{value_type}}, KernelTypes}

    return KernelTypes
end
function return_kernel_types(ObsType::DataType, param_type, coord_type, weight_type, value_type)
    return ObservableKernel{ObsType, param_type, coord_type, weight_type, Vector{value_type}}
end


# RETURN KERNEL: TauP + Seismic Observable
function return_kernel(Observation::T, Model::PsiModel) where {T <: SeismicObservable}
    # Compute the ray path
    ray_lon, ray_lat, ray_elv, ray_time = call_taup_path(Observation, Model)
    # Convert geographic coordinates to global coordinate Tuple
    kernel_coordinates = map((n1,n2,n3) -> (n1,n2,n3), ray_lon, ray_lat, ray_elv)
    for (i, x) in enumerate(kernel_coordinates)
        kernel_coordinates[i] = geographic_to_global(x[1], x[2], x[3]; radius = Model.Mesh.Geometry.R₀)
    end
    # Kernel weights defined using local ray orientations
    kernel_weights = return_ray_kernel_weights(kernel_coordinates)
    for (i, w) in enumerate(kernel_weights)
        azm, elv = ecef_to_local_orientation(ray_lon[i], ray_lat[i], w.azimuth, w.elevation)
        kernel_weights[i] = (dr = w.dr, azimuth = azm, elevation = elv)
    end
    # Kernel parameters interpolate_model_to_kernel()
    KernelParameters = return_kernel_parameters(Observation, Model, kernel_coordinates)
    # Kernel Static
    kernel_static = return_kernel_static(Observation, Model)
    T <: TravelTime ? kernel_static[1] += ray_time[1] : nothing

    return ObservableKernel(Observation, KernelParameters, kernel_coordinates, kernel_weights, kernel_static)
end
function return_kernel_static(Observation, Model)
    kernel_static = get_source_static(Model.Sources.statics, Observation)
    kernel_static = kernel_static .+ get_receiver_static(Model.Receivers.statics, Observation)
    return [kernel_static]
end

# RETURN RAY KERNEL WEIGHTS: Ray segment lengths and orientations for any source-to-receiver path
# First index in path is assumed to be the source position while the last index corresponds to the receiver
function return_ray_kernel_weights(ray_coordinates::Vector{NTuple{3, T}}) where {T}
    # Allocate kernel weights
    nw = length(ray_coordinates)
    # Three weights for each ray path node (segment length, azimuth, and elevation)
    kernel_weights = Vector{NamedTuple{(:dr, :azimuth, :elevation), NTuple{3, T}}}(undef, nw)
    # Define initial segment length between nodes (i - 1) and i
    dx = ray_coordinates[2] .- ray_coordinates[1]
    wj = sqrt(sum(dx.^2))
    # Half-weight for the first ray node
    dr_i = 0.5*wj
    # Segement orientation
    azm_i = atan(dx[2], dx[1])
    elv_i = atan(dx[3], sqrt((dx[1]^2) + (dx[2]^2)))
    kernel_weights[1] = (dr = dr_i, azimuth = azm_i, elevation = elv_i)
    # Assign inner ray node weights
    for i in 2:(nw - 1)
        dx = ray_coordinates[i+1] .- ray_coordinates[i]
        # Length of ray segment between nodes i and (i + 1)
        wk = sqrt(sum(dx.^2))
        # Ray node weight is the average of its segment lengths
        dr_i = 0.5*(wj + wk)
        # Segement orientation
        dx = ray_coordinates[i+1] .- ray_coordinates[i-1] # Use orientation for segment (i -1) to (i + 1)
        azm_i = atan(dx[2], dx[1])
        elv_i = atan(dx[3], sqrt((dx[1]^2) + (dx[2]^2)))
        kernel_weights[i] = (dr = dr_i, azimuth = azm_i, elevation = elv_i)
        # Update segment length between nodes (i - 1) and i for next iteration
        wj = wk
    end
    dx = ray_coordinates[nw] .- ray_coordinates[nw-1]
    # Half-weight for last ray node
    dr_i = 0.5*sqrt(sum(dx.^2))
    # Segement orientation
    azm_i = atan(dx[2], dx[1])
    elv_i = atan(dx[3], sqrt((dx[1]^2) + (dx[2]^2)))
    kernel_weights[nw] = (dr = dr_i, azimuth = azm_i, elevation = elv_i)

    return kernel_weights
end


function return_kernel_parameters(Observation::SeismicObservable, Model::PsiModel, kernel_coordinates)
    return return_kernel_parameters(Observation.Phase, Model, kernel_coordinates)
end
# RETURN KERNEL PARAMETERS: P-wave + Isotropic Velocity Parameters
function return_kernel_parameters(::CompressionalWave, Model::PsiModel{<:IsotropicVelocity}, kernel_coordinates)
    # Initialise new parameter structure
    n = length(kernel_coordinates)
    T = eltype(Model.Parameters.vp)
    KernelParameters = IsotropicVelocity(zeros(T, n), nothing)
    # Linearly interpolate model to the kernel
    interpolate_kernel_parameters!(KernelParameters, kernel_coordinates, Model)

    return KernelParameters
end
# RETURN KERNEL PARAMETERS: S-wave + Isotropic Velocity Parameters
function return_kernel_parameters(::ShearWave, Model::PsiModel{<:IsotropicVelocity}, kernel_coordinates)
    # Initialise new parameter structure
    n = length(kernel_coordinates)
    T = eltype(Model.Parameters.vs)
    KernelParameters = IsotropicVelocity(nothing, zeros(T, n))
    # Linearly interpolate model to the kernel
    interpolate_kernel_parameters!(KernelParameters, kernel_coordinates, Model)

    return KernelParameters
end
# RETURN KERNEL PARAMETERS: P-wave + Hexagonal Vectoral Parameters
function return_kernel_parameters(::CompressionalWave, Model::PsiModel{<:HexagonalVectoralVelocity}, kernel_coordinates)
    # Define length and element type for kernel parameters
    n = length(kernel_coordinates)
    T = eltype(Model.Parameters.α)
    # Define uniform or spatially variable anisotropic ratios
    if length(Model.Parameters.ratio_ϵ) == 1
        ratio_ϵ = Model.Parameters.ratio_ϵ
        ratio_η = Model.Parameters.ratio_η
    else
        ratio_ϵ = zeros(T, n)
        ratio_η = zeros(T, n)
    end
    # Initialise new parameter structure
    if Model.Parameters.tf_exact
        # Exact qP-phase velocity requires α, β, ϵ, η
        KernelParameters = HexagonalVectoralVelocity(zeros(T, n), zeros(T, n), zeros(T, n), zeros(T, n), zeros(T, n),
        ratio_ϵ, ratio_η, nothing, Model.Parameters.tf_exact)
    else
        # Weak qP-phase velocity approximation only requires α, ϵ, η
        KernelParameters = HexagonalVectoralVelocity(zeros(T, n), nothing, zeros(T, n), zeros(T, n), zeros(T, n),
        ratio_ϵ, ratio_η, nothing, Model.Parameters.tf_exact)
    end
    # Linearly interpolate model to the kernel
    interpolate_kernel_parameters!(KernelParameters, kernel_coordinates, Model)
    
    return KernelParameters
end
# RETURN KERNEL PARAMETERS: S-wave + Hexagonal Vectoral Parameters
function return_kernel_parameters(::ShearWave, Model::PsiModel{<:HexagonalVectoralVelocity}, kernel_coordinates)
    # Define length and element type for kernel parameters
    n = length(kernel_coordinates)
    T = eltype(Model.Parameters.β)
    # Define uniform or spatially variable anisotropic ratios
    if length(Model.Parameters.ratio_γ) == 1
        ratio_ϵ = Model.Parameters.ratio_ϵ
        ratio_η = Model.Parameters.ratio_η
        ratio_γ = Model.Parameters.ratio_γ
    else
        ratio_ϵ = zeros(T, n)
        ratio_η = zeros(T, n)
        ratio_γ = zeros(T, n)
    end
    # Initialise new parameter structure
    if Model.Parameters.tf_exact
        # Exact qS-phase velocities requires α, β, ϵ, η, γ
        KernelParameters = HexagonalVectoralVelocity(zeros(T, n), zeros(T, n), zeros(T, n), zeros(T, n), zeros(T, n),
        ratio_ϵ, ratio_η, ratio_γ, Model.Parameters.tf_exact)
    else
        # Weak qS-phase velocity requires α, β, η, γ
        KernelParameters = HexagonalVectoralVelocity(zeros(T, n), zeros(T, n), zeros(T, n), zeros(T, n), zeros(T, n),
        nothing, ratio_η, ratio_γ, Model.Parameters.tf_exact)
    end
    # Linearly interpolate model to the kernel
    interpolate_kernel_parameters!(KernelParameters, kernel_coordinates, Model)
    
    return KernelParameters
end



# INTERPOLATE KERNEL PARAMETERS: Isotropic Velocity Parameters
function interpolate_kernel_parameters!(KernelParameters::IsotropicVelocity, kernel_coordinates::Vector{NTuple{3, T}}, Model::PsiModel{<:IsotropicVelocity}) where {T}
    for (i, qx_global) in enumerate(kernel_coordinates)
        # Convert the global kernel coordinates into the local coordinate system
        qx = global_to_local(qx_global[1], qx_global[2], qx_global[3], Model.Mesh.Geometry)
        # Get trilinear interpolation weights
        wind, wval = trilinear_weights(Model.Mesh.x, qx; tf_extrapolate = false, scale = 1.0)
        # Interpolate field flag
        tf_interpolate = (vp = ~isnothing(KernelParameters.vp), vs = ~isnothing(KernelParameters.vs))
        # Interpolate fields
        for (j, wj) in enumerate(wval)
            k = wind[j]
            if tf_interpolate.vp
                KernelParameters.vp[i] += wj*Model.Parameters.vp[k]
            end
            if tf_interpolate.vs
                KernelParameters.vs[i] += wj*Model.Parameters.vs[k]
            end
        end
    end

    return nothing
end
# INTERPOLATE KERNEL PARAMETERS: Hexagonal Thomsen Parameters
function interpolate_kernel_parameters!(KernelParameters::HexagonalVectoralVelocity, kernel_coordinates::Vector{NTuple{3, T}}, Model::PsiModel{<:HexagonalVectoralVelocity}) where {T}
    # Interpolation flags
    tf_interp_α = ~isnothing(KernelParameters.α)
    tf_interp_β = ~isnothing(KernelParameters.β)
    tf_interp_f = ~isnothing(KernelParameters.f)
    tf_interp_angles = ~isnothing(KernelParameters.azimuth)
    tf_interp_ratio_ϵ = ~isnothing(KernelParameters.ratio_ϵ) && (length(KernelParameters.ratio_ϵ) > 1)
    tf_interp_ratio_η = ~isnothing(KernelParameters.ratio_η) && (length(KernelParameters.ratio_η) > 1)
    tf_interp_ratio_γ = ~isnothing(KernelParameters.ratio_γ) && (length(KernelParameters.ratio_γ) > 1)
    # Loop to interpolate parameters to Kernel
    for (i, qx_global) in enumerate(kernel_coordinates)
        # Convert the global kernel coordinates into the local coordinate system
        qx = global_to_local(qx_global[1], qx_global[2], qx_global[3], Model.Mesh.Geometry)
        # Get trilinear interpolation weights
        wind, wval = trilinear_weights(Model.Mesh.x, qx; tf_extrapolate = false, scale = 1.0)
        # Interpolate fields
        (s1, s2, s3) = (0.0, 0.0, 0.0)
        for (j, wj) in enumerate(wval)
            k = wind[j]
            tf_interp_α ? KernelParameters.α[i] += wj*Model.Parameters.α[k] : nothing
            tf_interp_β ? KernelParameters.β[i] += wj*Model.Parameters.β[k] : nothing
            tf_interp_f ? KernelParameters.f[i] += wj*Model.Parameters.f[k] : nothing
            tf_interp_ratio_ϵ ? KernelParameters.ratio_ϵ[i] += wj*Model.Parameters.ratio_ϵ[k] : nothing
            tf_interp_ratio_η ? KernelParameters.ratio_η[i] += wj*Model.Parameters.ratio_η[k] : nothing
            tf_interp_ratio_γ ? KernelParameters.ratio_γ[i] += wj*Model.Parameters.ratio_γ[k] : nothing
            # Angle interpolation must be done without anisotropic magnitude weighting to preserve orientations where f = 0
            if tf_interp_angles
                # Trigonometric factors
                sinϕ, cosϕ = sincos(Model.Parameters.azimuth[k])
                sinθ, cosθ = sincos(Model.Parameters.elevation[k])
                # Weighted sum of anisotropic vector components
                s1 += wj*cosθ*cosϕ
                s2 += wj*cosθ*sinϕ
                s3 += wj*sinθ
            end
        end
        # Recover orientation
        if tf_interp_angles
            KernelParameters.azimuth[i] = atan(s2, s1)
            KernelParameters.elevation[i] = atan(s3, sqrt((s1^2) + (s2^2)))
        end
    end

    return nothing
end


# EVALUATE KERNEL: Vector of ObservableKernel
function evaluate_kernel(Kernels::Vector{<:ObservableKernel})
    nobs = length(Kernels)
    predictions = similar(Kernels[1].static, nobs)
    relative_residuals = similar(Kernels[1].static, nobs)
    for (i, iKernel) in enumerate(Kernels)
        predictions[i], relative_residuals[i] = evaluate_kernel(iKernel)
    end

    return predictions, relative_residuals
end
# EVALUATE KERNEL: Travel Time
function evaluate_kernel(Kernel::ObservableKernel{<:TravelTime})
    # Initialise prediction with static
    prediction = Kernel.static[1]
    # Integrate kernel
    for (i, w) in enumerate(Kernel.weights)
        vi = kernel_phase_velocity(Kernel.Observation.Phase, Kernel, i)
        prediction += w.dr/vi
    end
    relative_residual = (Kernel.Observation.observation - prediction)/Kernel.Observation.error

    return prediction, relative_residual
end
# EVALUATE KERNEL: Splitting Intensity
function evaluate_kernel(Kernel::ObservableKernel{<:SplittingIntensity})
    # Initialise prediction with static
    prediction = Kernel.static[1]
    # Integrate kernel
    for (i, w) in enumerate(Kernel.weights)
        si = kernel_splitting_intensity(Kernel.Observation.Phase, Kernel, i)
        prediction += w.dr*si
    end
    relative_residual = (Kernel.Observation.observation - prediction)/Kernel.Observation.error

    return prediction, relative_residual
end


function kernel_phase_velocity(::CompressionalWave, Kernel::ObservableKernel{T1, T2}, index) where {T1, T2 <: IsotropicVelocity}
    return Kernel.Parameters.vp[index]
end
function kernel_phase_velocity(::ShearWave, Kernel::ObservableKernel{T1, T2}, index) where {T1, T2 <: IsotropicVelocity}
    return Kernel.Parameters.vs[index]
end
function kernel_phase_velocity(::CompressionalWave, Kernel::ObservableKernel{T1, T2}, index) where {T1, T2 <: HexagonalVectoralVelocity}
    # Extract Thomsen Parameters
    α, β, ϵ, δ, _ = return_thomsen_parameters(Kernel.Parameters, index)
    # Compute qP phase velocity
    if Kernel.Parameters.tf_exact
        vqp, _ = qp_phase_velocity_thomsen(Kernel.weights[index].azimuth, Kernel.weights[index].elevation,
        Kernel.Parameters.azimuth[index], Kernel.Parameters.elevation[index], α, ϵ, δ, β)
    else
        vqp, _ = qp_phase_velocity_thomsen(Kernel.weights[index].azimuth, Kernel.weights[index].elevation,
        Kernel.Parameters.azimuth[index], Kernel.Parameters.elevation[index], α, ϵ, δ)
    end

    return vqp
end
function kernel_phase_velocity(Phase::ShearWave, Kernel::ObservableKernel{T1, T2}, index) where {T1, T2 <: HexagonalVectoralVelocity}
    # Extract Thomsen Parameters
    α, β, ϵ, δ, γ = return_thomsen_parameters(Kernel.Parameters, index)
    # Compute qS phase velocities
    if Kernel.Parameters.tf_exact
        vqs1, vqs2, _, ζ = qs_phase_velocities_thomsen(Kernel.weights[index].azimuth, Kernel.weights[index].elevation,
        Phase.paz, Kernel.Parameters.azimuth[index], Kernel.Parameters.elevation[index], α, β, ϵ, δ, γ)
    else
        vqs1, vqs2, _, ζ = qs_phase_velocities_thomsen(Kernel.weights[index].azimuth, Kernel.weights[index].elevation,
        Phase.paz, Kernel.Parameters.azimuth[index], Kernel.Parameters.elevation[index], α, β, ϵ - δ, γ)
    end
    # Effective anisotropic shear slowness
    uq = (1.0/vqs2) - ((1.0/vqs2) - (1.0/vqs1))*(cos(ζ)^2)

    return 1.0/uq
end
function kernel_splitting_intensity(Phase::ShearWave, Kernel::ObservableKernel{T1, T2}, index) where {T1, T2 <: HexagonalVectoralVelocity}
    # Extract Thomsen Parameters
    α, β, ϵ, δ, γ = return_thomsen_parameters(Kernel.Parameters, index)
    # Compute qS phase velocities
    if Kernel.Parameters.tf_exact
        vqs1, vqs2, _, ζ = qs_phase_velocities_thomsen(Kernel.weights[index].azimuth, Kernel.weights[index].elevation,
        Phase.paz, Kernel.Parameters.azimuth[index], Kernel.Parameters.elevation[index], α, β, ϵ, δ, γ)
    else
        vqs1, vqs2, _, ζ = qs_phase_velocities_thomsen(Kernel.weights[index].azimuth, Kernel.weights[index].elevation,
        Phase.paz, Kernel.Parameters.azimuth[index], Kernel.Parameters.elevation[index], α, β, ϵ - δ, γ)
    end
    # Splitting Intensity
    si = 0.5*((1.0/vqs2) - (1.0/vqs1))*sin(2.0*ζ)

    return si
end

function qp_phase_velocity(propagation_azimuth, propagation_elevation, Parameters::HexagonalVectoralVelocity, index)
    # Get Thomsen parameters
    α, β, ϵ, δ, _ = return_thomsen_parameters(Parameters, index)
    # Compute trigonometric terms
    cosθ = symmetry_axis_cosine(Parameters.azimuth[index], Parameters.elevation[index], propagation_azimuth, propagation_elevation)
    cosθ_2 = cosθ^2
    sinθ_2 = 1.0 - cosθ_2
    # Compute qP phase velocity
    if Parameters.tf_exact
        # Compute D-parameter (Thomsen, 1986; Eq. 10d)
        g = 1.0 - ((β/α)^2)
        f = g^(-2)
        D = 0.5*g*( sqrt( 1.0 + 4.0*δ*f*sinθ_2*cosθ_2 + 4.0*ϵ*(g + ϵ)*f*(sinθ_2^2) ) - 1.0 )
        # Exact qP-phase velocity (Thomsen, 1986; Eq. 10a)
        vqp = α*sqrt( 1.0 + ϵ*sinθ_2 + D )
    else
        # Approximate qP-phase velocity
        vqp = α*sqrt( 1.0 + 2.0*ϵ*(sinθ_2^2) + 2.0*δ*sinθ_2*cosθ_2 )
        # vqp = α*( 1.0 + ϵ*(sinθ_2^2) + δ*sinθ_2*cosθ_2 ) # Weak Approximation (Thomsen, 1986; Eq. 16a)
    end

    return vqp, cosθ
end
function qs_phase_velocities(propagation_azimuth, propagation_elevation, qt_polarization, Parameters::HexagonalVectoralVelocity, index)
    # Get Thomsen parameters
    α, β, ϵ, δ, γ = return_thomsen_parameters(Parameters, index)
    # Compute angle between propagation direction and symmetry axis
    cosθ, ζ = symmetry_axis_cosine(Parameters.azimuth[index], Parameters.elevation[index], propagation_azimuth, propagation_elevation, qt_polarization)
    cosθ_2 = cosθ^2
    sinθ_2 = 1.0 - cosθ_2
    # Compute qS phase velocities
    if Parameters.tf_exact
        # Compute D-parameter (Thomsen, 1986; Eq. 10d)
        g = 1.0 - ((β/α)^2)
        f = g^(-2)
        D = 0.5*g*( sqrt( 1.0 + 4.0*δ*f*sinθ_2*cosθ_2 + 4.0*ϵ*(g + ϵ)*f*(sinθ_2^2) ) - 1.0 )
        # Exact qS-phase velocities (Thomsen, 1986; Eq. 10b,c)
        vqs1 = β*sqrt( 1.0 + ((α/β)^2)*(ϵ*sinθ_2 - D) )
        vqs2 = β*sqrt( 1.0 + 2.0*γ*sinθ_2 )
    else
        # Approximate qS-phase velocities
        vqs1 = β*sqrt( 1.0 + 2.0*((α/β)^2)*(ϵ - δ)*sinθ_2*cosθ_2 )
        vqs2 = β*sqrt( 1.0 + 2.0*γ*sinθ_2 )
        # vqs1 = β*( 1.0 + ((α/β)^2)*(ϵ - δ)*sinθ_2*cosθ_2 ) # Weak approximation (Thomsen, 1986; Eq. 16b)
        # vqs2 = β*( 1.0 + γ*sinθ_2 ) # Weak approximation (Thomsen, 1986; Eq. 16c)
    end

    return vqs1, vqs2, cosθ, ζ
end

# DEPRECATE THESE FUNCTIONS USING ABOVE
function qp_phase_velocity_thomsen(propagation_azimuth, propagation_elevation, symmetry_azimuth, symmetry_elevation, α, ϵ, δ)

    # Compute trigonometric terms
    cosθ = symmetry_axis_cosine(symmetry_azimuth, symmetry_elevation, propagation_azimuth, propagation_elevation)
    cosθ_2 = cosθ^2
    sinθ_2 = 1.0 - cosθ_2
    # Approximate qP-phase velocity
    vqp = α*sqrt( 1.0 + 2.0*ϵ*(sinθ_2^2) + 2.0*δ*sinθ_2*cosθ_2 )
    # vqp = α*( 1.0 + ϵ*(sinθ_2^2) + δ*sinθ_2*cosθ_2 ) # Weak Approximation

    return vqp, cosθ
end
function qp_phase_velocity_thomsen(propagation_azimuth, propagation_elevation, symmetry_azimuth, symmetry_elevation, α, ϵ, δ, β)
    
    # Compute trigonometric terms
    cosθ = symmetry_axis_cosine(symmetry_azimuth, symmetry_elevation, propagation_azimuth, propagation_elevation)
    cosθ_2 = cosθ^2
    sinθ_2 = 1.0 - cosθ_2
    # Compute D-parameter (Thomsen, 1986; Eq. 10d)
    g = 1.0 - ((β/α)^2)
    f = g^(-2)
    D = 0.5*g*( sqrt( 1.0 + 4.0*δ*f*sinθ_2*cosθ_2 + 4.0*ϵ*(g + ϵ)*f*(sinθ_2^2) ) - 1.0 )
    # Exact qP-phase velocity (Thomsen, 1986; Eq. 10a)
    vqp = α*sqrt( 1.0 + ϵ*sinθ_2 + D )

    return vqp, cosθ
end
function qs_phase_velocities_thomsen(propagation_azimuth, propagation_elevation, qt_polarization,
    symmetry_azimuth, symmetry_elevation, α, β, η, γ)

    # Compute angle between propagation direction and symmetry axis
    cosθ, ζ = symmetry_axis_cosine(symmetry_azimuth, symmetry_elevation, propagation_azimuth, propagation_elevation, qt_polarization)
    cosθ_2 = cosθ^2
    sinθ_2 = 1.0 - cosθ_2

    # Approximate qS-phase velocities
    vqs1 = β*sqrt( 1.0 + 2.0*((α/β)^2)*η*sinθ_2*cosθ_2 )
    vqs2 = β*sqrt( 1.0 + 2.0*γ*sinθ_2 )
    # vqs1 = β*( 1.0 + ((α/β)^2)*η*sinθ_2*cosθ_2 ) # Weak approximation
    # vqs2 = β*( 1.0 + γ*sinθ_2 ) # Weak approximation

    return vqs1, vqs2, cosθ, ζ
end
function qs_phase_velocities_thomsen(propagation_azimuth, propagation_elevation, qt_polarization,
    symmetry_azimuth, symmetry_elevation, α, β, ϵ, δ, γ)

    # Compute angle between propagation direction and symmetry axis
    cosθ, ζ = symmetry_axis_cosine(symmetry_azimuth, symmetry_elevation, propagation_azimuth, propagation_elevation, qt_polarization)
    cosθ_2 = cosθ^2
    sinθ_2 = 1.0 - cosθ_2

    # Compute D-parameter (Thomsen, 1986; Eq. 10d)
    g = 1.0 - ((β/α)^2)
    f = g^(-2)
    D = 0.5*g*( sqrt( 1.0 + 4.0*δ*f*sinθ_2*cosθ_2 + 4.0*ϵ*(g + ϵ)*f*(sinθ_2^2) ) - 1.0 )
    # Exact qS-phase velocities (Thomsen, 1986; Eq. 10b,c)
    vqs1 = β*sqrt( 1.0 + ((α/β)^2)*(ϵ*sinθ_2 - D) )
    vqs2 = β*sqrt( 1.0 + 2.0*γ*sinθ_2 )

    return vqs1, vqs2, cosθ, ζ
end


function symmetry_axis_cosine(symmetry_azimuth, symmetry_elevation, propagation_azimuth, propagation_elevation)
    cosΔλ = cos(propagation_azimuth - symmetry_azimuth) 
    sinϕp, cosϕp = sincos(propagation_elevation)
    sinϕs, cosϕs = sincos(symmetry_elevation)
    # Cosine of angle between propagation direction and symmetry axis
    cosθ = cosΔλ*cosϕp*cosϕs + sinϕp*sinϕs
    
    return cosθ
end
function symmetry_axis_cosine(symmetry_azimuth, symmetry_elevation, propagation_azimuth, propagation_elevation, qt_polarization)
    sinΔλ, cosΔλ = sincos(propagation_azimuth - symmetry_azimuth) 
    sinϕp, cosϕp = sincos(propagation_elevation)
    sinϕs, cosϕs = sincos(symmetry_elevation)
    # Cosine of angle between propagation direction and symmetry axis
    cosθ = cosΔλ*cosϕp*cosϕs + sinϕp*sinϕs
    # Angle between polarization vector and projection of symmetry axis in QT-plane (i.e. ray-normal plane)
    # Do not return cos(ζ). The sign of this angle is important for splitting intensity.
    ζ = atan(-sinΔλ*cosϕs, cosΔλ*sinϕp*cosϕs - cosϕp*sinϕs) - qt_polarization
    
    return cosθ, ζ
end


# Use 'nameof(b)' 'nameof(b.phase)' to get just the structure name as a symbol
function get_source_static(δ::Dict, b::Observable)
    # Define key (id + observable + phase)
    k = (b.source_id, nameof(typeof(b)), nameof(typeof(b.Phase)))
    if haskey(δ, k)
        δₖ = δ[k]
    else
        δₖ = return_null_static(b)
    end

    return δₖ
end
function get_receiver_static(δ::Dict, b::Observable)
    # Define key (id + observable + phase)
    k = (b.receiver_id, nameof(typeof(b)), nameof(typeof(b.Phase)))
    if haskey(δ, k)
        δₖ = δ[k]
    else
        # println("No Static for key: ", k)
        δₖ = return_null_static(b)
    end

    return δₖ
end
function return_null_static(b::SeismicObservable)
    return zero(eltype(b.observation))
end
function return_null_static(b::SplittingParameters)
    return (zero(eltype(b.observation)), zero(eltype(b.observation)))
end



function call_taup_path(Observation, Model)
    # Source coordinates
    src_ind = Model.Sources.id[Observation.source_id]
    src_lon = Model.Sources.coordinates[src_ind][1]
    src_lat = Model.Sources.coordinates[src_ind][2]
    src_elv = Model.Sources.coordinates[src_ind][3]
    # Receiver coordinates
    rcv_ind = Model.Receivers.id[Observation.receiver_id]
    rcv_lon = Model.Receivers.coordinates[rcv_ind][1]
    rcv_lat = Model.Receivers.coordinates[rcv_ind][2]
    rcv_elv = Model.Receivers.coordinates[rcv_ind][3]
    # Convert elevations to TauP model depths
    ΔR = Model.Methods.TauP.radius - Model.Mesh.Geometry.R₀
    src_dpt = ΔR - src_elv
    rcv_dpt = ΔR - rcv_elv
    # Source-to-receiver arc degrees and bearing
    dist_deg, azim = inverse_geodesic(src_lat, src_lon, rcv_lat, rcv_lon; tf_degrees = true)
    # Get ray path
    #PathObj = buildPathObj(Model.Methods.TauP.reference_model)
    #ray_deg, ray_elv, ray_time = taup_path!(PathObj, Observation.Phase.name, dist_deg, src_dpt, rcv_dpt)
    ray_deg, ray_elv, ray_time = taup_path!(Model.Methods.TauP.PathObj, Observation.Phase.name, dist_deg, src_dpt, rcv_dpt)
    # Convert ray depth in TauP model to ray elevation
    ray_elv .= (ΔR .- ray_elv)
    # Retrieve geographic coordinates
    ray_lat, ray_lon = direct_geodesic(src_lat, src_lon, ray_deg, azim; tf_degrees = true)
    # Extract local ray path (includes first node outside the model domain)
    return_local_path!(ray_deg, ray_elv, ray_time, ray_lon, ray_lat, Model.Mesh; tf_first_out = true)
    # Re-sample ray path
    if Model.Methods.TauP.dL > 0.0
        # Re-sample ray path
        ray_deg, ray_elv, ray_time = resample_path(ray_deg, ray_elv, ray_time, Model.Methods.TauP.dL; R₀ = Model.Mesh.Geometry.R₀)
        # Re-derive geographic coordinates
        ray_lat, ray_lon = direct_geodesic(src_lat, src_lon, ray_deg, azim; tf_degrees = true)
        # Extract local ray path (excludes first node outside model domain)
        return_local_path!(ray_deg, ray_elv, ray_time, ray_lon, ray_lat, Model.Mesh; tf_first_out = false)
    end

    return ray_lon, ray_lat, ray_elv, ray_time
end

# Trims the ray path provided in polar (d,r) and true geographic (λ, ϕ) coordinates
# to only include ray nodes inside local mesh. Also returns raypath in the local
# coordinate system (x, y, z)
function return_local_path!(d, r, t, λ, ϕ, Mesh::RegularGrid; tf_first_out = true)
    # Total ray nodes
    M = length(d)
    # Local path coordinates
    x, y, z = geographic_to_local(λ, ϕ, r, Mesh.Geometry)
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















######### INTERPOLATIONS ##########

function trilinear_weights(x::NTuple{3, R}, qx::NTuple{3, T}; tf_extrapolate = false, scale = 1.0) where {R, T}
    return trilinear_weights(x[1], x[2], x[3], qx[1], qx[2], qx[3]; tf_extrapolate = tf_extrapolate, scale = scale)
end
function trilinear_weights(x, y, z, qx, qy, qz; tf_extrapolate = false, scale = 1.0, out_of_bounds_value = 0.0)
    # Grid dimensions and spacing
    nx = length(x)
    ny = length(y)
    nz = length(z)
    Δx = step(x)
    Δy = step(y)
    Δz = step(z)
    # For extrapolation, truncate query points such that they are inside the model domain
    if tf_extrapolate
        qx = min(max(qx, minimum(x)), maximum(x))
        qy = min(max(qy, minimum(y)), maximum(y))
        qz = min(max(qz, minimum(z)), maximum(z))
    end
    # Nearest lower index
    ix = 1 + floor(Int, (qx - x[1])/Δx)
    jy = 1 + floor(Int, (qy - y[1])/Δy)
    kz = 1 + floor(Int, (qz - z[1])/Δz)
    # Bounds check
    if (ix < 0) || (ix > nx) || (jy < 0) || (jy > ny) || (kz < 0) || (kz > nz)
        # Uses 'extrap_weight' weight if outside model domain
        scale = out_of_bounds_value
    end
    # Truncate to lower index inside domain
    ix = min(max(ix, 1), nx - 1)
    jy = min(max(jy, 1), ny - 1)
    kz = min(max(kz, 1), nz - 1)
    # Normalised distances from nearest lower index (c000)
    fx₁ = (qx - x[ix])/Δx
    fx₂ = (qy - y[jy])/Δy
    fx₃ = (qz - z[kz])/Δz
    # Weight indices
    i₁ = subscripts_to_index((nx, ny, nz), (  ix,   jy, kz)) # c000
    i₂ = subscripts_to_index((nx, ny, nz), (ix+1,   jy, kz)) # c100
    i₃ = subscripts_to_index((nx, ny, nz), (  ix, jy+1, kz)) # c010
    i₄ = subscripts_to_index((nx, ny, nz), (ix+1, jy+1, kz)) # c110
    i₅ = subscripts_to_index((nx, ny, nz), (  ix,   jy, kz+1)) # c001
    i₆ = subscripts_to_index((nx, ny, nz), (ix+1,   jy, kz+1)) # c101
    i₇ = subscripts_to_index((nx, ny, nz), (  ix, jy+1, kz+1)) # c011
    i₈ = subscripts_to_index((nx, ny, nz), (ix+1, jy+1, kz+1)) # c111
    # Weight values
    w₁ = scale*(1.0 - fx₁)*(1.0 - fx₂)*(1.0 - fx₃) # c000
    w₂ = scale*fx₁*(1.0 - fx₂)*(1.0 - fx₃) # c100
    w₃ = scale*(1.0 - fx₁)*fx₂*(1.0 - fx₃) # c010
    w₄ = scale*fx₁*fx₂*(1.0 - fx₃) # c110
    w₅ = scale*(1.0 - fx₁)*(1.0 - fx₂)*fx₃ # c001
    w₆ = scale*fx₁*(1.0 - fx₂)*fx₃ # c101
    w₇ = scale*(1.0 - fx₁)*fx₂*fx₃ # c011
    w₈ = scale*fx₁*fx₂*fx₃ # c111
    # # Build static vectors for weights
    # wind = @SVector [i₁, i₂, i₃, i₄, i₅, i₆, i₇, i₈]
    # wval = @SVector [w₁, w₂, w₃, w₄, w₅, w₆, w₇, w₈]
    # return wind, wval
    
    return (i₁, i₂, i₃, i₄, i₅, i₆, i₇, i₈), (w₁, w₂, w₃, w₄, w₅, w₆, w₇, w₈)
end
# Single query point interpolation
function linearly_interpolate(x::NTuple{3, R}, v::Array{T, 3}, qx::NTuple{3, T}; tf_extrapolate = false, tf_harmonic = false) where {R, T}
    # Get linear interpolation weights for query point
    wind, wval = trilinear_weights(x, qx; tf_extrapolate = tf_extrapolate, scale = 1.0)
    # Return interpolated value as a weighted arithmetic (harmonic) average
    qv = zero(eltype(v))
    if tf_harmonic
        for i in eachindex(wind)
            qv += wval[i]/v[wind[i]]
        end
        qv = 1.0/qv
    else
        for i in eachindex(wind)
            qv += wval[i]*v[wind[i]]
        end
    end

    return qv
end
# Array of query points interpolation
function linearly_interpolate!(qv, x, v, qx; tf_extrapolate = false, tf_harmonic = false)
    # Point-wise linear interpolation
    for i in eachindex(qx)
        qv[i] = linearly_interpolate(x, v, qx[i]; tf_extrapolate = tf_extrapolate, tf_harmonic = tf_harmonic)
    end

    return nothing
end
# function linearly_interpolate(x, v, q; tf_extrapolate = false, tf_harmonic = false)
#     qv = similar(v, eltype(v), size(q))
#     linearly_interpolate!(qv, x, v, qx; tf_extrapolate = tf_extrapolate, tf_harmonic = tf_harmonic)

#     return qv
# end

## 1D LINEAR INTERPOLATION -- REGULAR GRIDS ##
function linear_weights(x::AbstractRange, qx::Number; tf_extrapolate = false, scale = 1.0)
    nx = length(x)
    Δx = step(x)
    # For extrapolation, truncate query points such that they are inside the model domain
    if tf_extrapolate
        qx = min(max(qx, minimum(x)), maximum(x))
    end
    # Nearest lower-index
    ix = 1 + floor(Int, (qx - x[1])/Δx)
    # Bounds check
    if (ix < 0) || (ix > nx)
        # Force NaN weights if outside model domain
        scale = NaN
    end
    # Truncate to lower index inside domain
    ix = min(max(ix, 1), nx - 1)
    # Normalised distances from nearest lower index
    fx₁ = (qx - x[ix])/Δx
    # Weight indices
    i₁ = subscripts_to_index((nx,), (ix,))
    i₂ = subscripts_to_index((nx,), (ix + 1,))
    # Weight values
    w₁ = scale*(1.0 - fx₁)
    w₂ = scale*fx₁
    # # Build static vectors for weights
    # wind = @SVector [i₁, i₂]
    # wval = @SVector [w₁, w₂]
    # return wind, wval

    return (i₁, i₂), (w₁, w₂)
end
# Single query point interpolation
function linearly_interpolate(x::AbstractRange, v, qx::Number; tf_extrapolate = false, tf_harmonic = false)
    # Get linear interpolation weights for query point
    wind, wval = linear_weights(x, qx; tf_extrapolate = tf_extrapolate, scale = 1.0)
    # Return interpolated value as a weighted arithmetic (harmonic) average
    if tf_harmonic
        qv = 1.0/((wval[1]/v[wind[1]]) + (wval[2]/v[wind[2]]))
    else
        qv = (wval[1]*v[wind[1]]) + (wval[2]*v[wind[2]])
    end

    return qv
end
## 1D LINEAR INTERPOLATION -- IRREGULAR GRIDS -- MONOTONIC ##
function linearly_interpolate(x, v, qx::Number; tf_extrapolate = false, tf_harmonic = false)
    # Identify minimum and maximum indices
    n = length(x)
    if x[1] < x[n]
        imin = 1
        imax = n
    else
        imin = n
        imax = 1
    end
    # Interpolate
    if (qx < x[imin])
        # Minimum bounds
        if tf_extrapolate
            qv = v[imin]
        else
            qv = NaN
        end
    elseif (qx > x[imax])
        # Maximum bounds
        if tf_extrapolate
            qv = v[imax]
        else
            qv = NaN
        end
    else
        # Inside
        j = searchsortedlast(x, qx)  # Find the index of the largest sample point less than or equal to qx[i]
        w = (qx - x[j]) / (x[j+1] - x[j])  # Compute the interpolation factor
        # Return interpolated value as a weighted arithmetic (harmonic) average
        if tf_harmonic
            qv = 1.0/(((1.0 - w)/v[j]) + (w/v[j+1]))
        else
            qv = (1.0 - w)*v[j] + w*v[j+1] 
        end
    end

    return qv
end

function piecewise_linearly_interpolate(x, d, v, qx::Number; tf_extrapolate = false, tf_harmonic = false)
    # Initialisation
    m = length(x) # Number of samples
    n = length(d) # Number of discontinuities
    j = 1 # Discontinuity d-index (upper side)
    ib = d[j] # x-index of first discontinuity below query point (upper side)
    # Locate first discontinuity below query point
    while (qx > x[ib]) && (ib < m)
        j += 1
        if j > n
            ib = m
        else
            ib = d[j]
        end
    end
    # x-index above query point
    if j == 1
        # No boundary above, use first sample index
        ia = 1
    else
        # Use under-side of boundary above (hence +1)
        ia = 1 + d[j - 1]
    end
    # Linear interpolation on contiuous interval
    if qx == x[ib]
        # When at boundary, average velocities on either side
        if tf_harmonic
            qv = 2.0/((1.0/v[ib]) + (1.0/v[ib + 1]))
        else
            qv = 0.5*(v[ib] + v[ib + 1])
        end
    else
        qv = linearly_interpolate(x[ia:ib], v[ia:ib], qx; tf_extrapolate = tf_extrapolate, tf_harmonic = tf_harmonic)
    end

    return qv
end

## 1D LINEAR RESAMPLING ##
function resample(x, v, N::Int)
    # Initialization
    M  = length(x)
    vq = similar(v, N)
    # Sampling interval
    x₀ = x[1]
    x₁ = x[end]
    # Initialise index into reference points
    i  = 1
    # Loop over query points
    for (j, xq) in enumerate(range(start = x₀, stop = x₁, length = N))
        # Update counter
        while ((i + 1) < M) && (xq > x[i + 1])
            i += 1
        end
        # Gradient
        g = (v[i+1] - v[i])/(x[i+1] - x[i])
        # Linearly interpolate to define query point
        vq[j] = g*(xq - x[i]) + v[i]
    end
    # # For comparison, using the interpolations package is less efficient
    # # for such a simple resampling.
    # function resample_interp(x, v, N)
    #     F = LinearInterpolation(x, v, extrapolation_bc=Flat())
    #     return F(range(x[1], x[end], N))
    # end

    return vq
end


