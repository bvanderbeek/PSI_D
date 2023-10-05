#################### SOLVERS ####################

# Solver: Structures for storing parameters relevant to the inverse problem solver
abstract type Solver end

# Linearised solver methods
abstract type LinearisedSolver <: Solver end

# Solver LSQR
struct SolverLSQR <: LinearisedSolver
    damp::Number # Solver internal damping parameter; should generally be zero
    atol::Number # Stopping tolerance; see lsqr documentation (1e-6)
    conlim::Number # Stopping tolerance; see lsqr documentation (1e8)
    maxiter::Int # Maximum number of solver iterations (maximum(size(A)))
    verbose::Bool # Output solver iteration info
    # Custom options
    tf_jac_scale::Bool # Apply Jacobian scaling to system
    nonlinit::Int # Maximum number of non-linear iterations for iterative linearised approach
end




#################### OBSERVABLES ####################

# Observable: Any measurable quantity that can be predicted. Given a model
# and an Observable, there should be no ambiguity in how to make a prediction.
abstract type Observable end
# Seismic Observable: Any quantity measured from a seismic wave
abstract type SeismicObservable <: Observable end
# These should all have the same fields?

# Seismic observables are always associated with a phase. Define phase types.
abstract type SeismicPhase end
# Fundamental Seismic Phases
struct CompressionalWave <: SeismicPhase
    name::String
end
struct ShearWave <: SeismicPhase
    name::String
    paz::Float64
end
function ShearWave(aphase::String)
    return ShearWave(aphase, 0.0)
end
# Additional phases that may become relevant are (1) Rayleigh, (2) Love,
# (3) Reflections, and (4) Conversions.

# Seismic observables are also associated with a forward (i.e. prediction) method.
# Define forward method types.
abstract type ForwardMethod end
struct RayTheory <: ForwardMethod end
struct FresnelVolume <: ForwardMethod end
struct FiniteFrequency <: ForwardMethod end

# Make SeismicObservable a structure and add field 'obs_type'?
# + All SeismicObservable have same fields
# + Can still dispatch on type parameters

# TravelTime Observation
struct TravelTime{P, F, S, R, T} <: SeismicObservable
    phase::P
    forward::F
    source_id::S
    receiver_id::R
    period::T
    obs::T
    error::T
end

# Splitting Intensity Observation
struct SplittingIntensity{P, F, S, R, T} <: SeismicObservable
    phase::P
    forward::F
    source_id::S
    receiver_id::R
    period::T
    obs::T
    error::T
end




#################### MODEL PARAMETERISATIONS ####################

# Model Parameterisation: Describes the combination of parameters that can be used
# to define a model. Unused parameters can be assigned empty arrays/values of the same
# type as the used parameters. These parameters can also be used to define the inversion unknowns.
#
# Parameter sets are always complete (e.g. IsotropicVelocity has vp and vs fields). However, not all
# fields need to be defined or inverted for.
abstract type InversionParameter end
abstract type ModelParameterisation <: InversionParameter end

# Elastic Models
abstract type ElasticModel <: ModelParameterisation end

# Seismic Velocity Model
abstract type SeismicVelocity <: ElasticModel end

# Isotropic Velocity
struct IsotropicVelocity{T} <: SeismicVelocity
    vp::Union{T, Nothing}
    vs::Union{T, Nothing}
end
# Create IsotropicVelocity structure with zero-valued arrays
function IsotropicVelocity(T::DataType, dims)
    return IsotropicVelocity(zeros(T, dims), zeros(T, dims))
end
function IsotropicVelocity(T::DataType, ::CompressionalWave, dims)
    return IsotropicVelocity(zeros(T, dims), nothing)
end
function IsotropicVelocity(T::DataType, ::ShearWave, dims)
    return IsotropicVelocity(nothing, zeros(T, dims))
end

# Hexagonal Anisotropy
struct HexagonalVelocity{T} <: SeismicVelocity
    vp::Union{T, Nothing}
    vs::Union{T, Nothing}
    fp::Union{T, Nothing}
    fs::Union{T, Nothing}
    gs::Union{T, Nothing}
    ϕ::Union{T, Nothing}
    θ::Union{T, Nothing}
end
# Create IsotropicVelocity structure with zero-valued arrays
function HexagonalVelocity(T::DataType, dims)
    return HexagonalVelocity(zeros(T, dims), zeros(T, dims), zeros(T, dims), zeros(T, dims),
    zeros(T, dims), zeros(T, dims), zeros(T, dims))
end
function HexagonalVelocity(T::DataType, ::CompressionalWave, dims; tf_elliptical = false)
    if tf_elliptical
        S = HexagonalVelocity(zeros(T, dims), nothing, zeros(T, dims), nothing,
        nothing, zeros(T, dims), zeros(T, dims))
    else
        S = HexagonalVelocity(zeros(T, dims), nothing, zeros(T, dims), nothing,
        zeros(T, dims), zeros(T, dims), zeros(T, dims))
    end

    return S
end
function HexagonalVelocity(T::DataType, ::ShearWave, dims; tf_elliptical = false)
    if tf_elliptical
        S = HexagonalVelocity(nothing, zeros(T, dims), nothing, zeros(T, dims),
        nothing, zeros(T, dims), zeros(T, dims))
    else
        S = HexagonalVelocity(nothing, zeros(T, dims), nothing, zeros(T, dims),
        zeros(T, dims), zeros(T, dims), zeros(T, dims))
    end

    return S
end

# Chevrot-Thomsen Parameters
struct ChevrotThosmen{T}
    α::Union{T, Nothing} # √(c11/ρ)
    β::Union{T, Nothing} # √(c66/ρ)
    ϵ::Union{T, Nothing} # 0.5*(c11 - c33)/c11
    γ::Union{T, Nothing} # 0.5*(c66 - c44)/c66
    δ::Union{T, Nothing} # (c13 - c33 + 2*c44)/c11
    ϕ::Union{T, Nothing} # Symmetry axis azimuth
    θ::Union{T, Nothing} # Symmetry axis elevation
end

# Isotropic Slowness
abstract type SeismicSlowness <: ElasticModel end
struct IsotropicSlowness{T} <: SeismicSlowness
    up::Union{T, Nothing} # Compressional Slowness (s/km)
    us::Union{T, Nothing} # Shear Slowness (s/km)
end

# ABC Parameterisation for hexagonal anisotropy
struct ABCSlowness{T} <: SeismicSlowness
    up::Union{T, Nothing}
    us::Union{T, Nothing}
    A::Union{T, Nothing}
    B::Union{T, Nothing}
    C::Union{T, Nothing}
    x2::Union{T, Nothing} # Ratio between S2 and P anisotropic fractions, fs/fp
    x1::Union{T, Nothing} # Ratio between S1 and P anisotropic fractions, gs/fp
    s::Union{T, Nothing} # Defines fast (+) or slow (-) symmetry axis
end


#################### OBSERVABLE KERNEL ####################

# Observable Kernel: The kernel structure holds all the information required to make a prediction.
# Given an Observable and a ObservableKernel, there should be no ambiguity in how to make a prediction.
# Predictions from an ObservableKernel can all be written in the following form:
#   q = q₀ + ∑ f(mᵢ, wᵢ)
# where 'f' is some function of the model parameters 'mᵢ' and the weight 'wᵢ'

# struct ObservableKernel{O<:Observable, T1, T2, T3, T4}
#     b::O # Observation
#     m::T1 # Model parameters
#     w::T2 # Weights (e.g., the length of a ray segment influenced by a particular velocity node)
#     x::T3 # Coordinates of model parameters
#     q₀::T4 # Static
# end

struct ObservableKernel{T1, T2, T3, T4} # <-- Changed for SI sensitivity kernels. See above
    b::DataType # Observable type
    m::T1 # Model parameters
    w::T2 # Weights (e.g., the length of a ray segment influenced by a particular velocity node)
    x::T3 # Coordinates of model parameters
    q₀::T4 # Static
end
# Builds an ObservableKernel given kernel model values, weights, and coordinates
function ObservableKernel(b::Type{<:Observable}, m::ModelParameterisation, w, x₁, x₂, x₃)
    # Initialize kernel structure with correct types
    T = eltype(x₁) # Element type
    dims = size(x₁) # Dimensions of kernel (assumes x₁, x₂, and x₃ are all equal dimensions)
    K = ObservableKernel(b, m, w, Array{NTuple{3, T}}(undef, dims), zeros(T, 1))
    # Map coordinates to the kernel coordinate tuple
    map!((n₁,n₂,n₃) -> (n₁,n₂,n₃), K.x, x₁, x₂, x₃)

    return K
end



#################### MESHES ####################

# Abstract Mesh: Describes the spatial discretisation of model parameters
# Only implemented regular grids. Consider using Mesh.jl package for more comples geometries
abstract type AbstractMesh end

# Regular Grid: Grid of dimension 'N' with uniform spacing in each dimension.
struct RegularGrid{T<:CoordinateSystem, N, R<:AbstractRange} <: AbstractMesh
    CS::T # Coordinate System
    x::NTuple{N, R} # Coordinate Tuple
end
# Add functions to base for this mesh
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




#################### SEISMIC SOURCES ####################

# Seismic Sources: Simple structure for storing seismic source data in geographic coordinates
struct SeismicSources{S, F, K}
    id::Dict{S,Int}
    ϕ::Vector{F}
    λ::Vector{F}
    r::Vector{F}
    static::Dict{Tuple{S, K, K}, F}
end
function SeismicSources(id, ϕ, λ, r; Ko = Symbol, Kp = Symbol)
    # Initialise SeismicSources with empty dictionaries
    Ki = eltype(id)
    T = eltype(ϕ)
    n = length(id)
    S = SeismicSources(Dict(zip(id, 1:n)), ϕ, λ, r, Dict{Tuple{Ki, Ko, Kp}, T}())

    return S
end
function SeismicSources(id, ϕ, λ, r, static_types, static_phases)
    # Initialise SeismicSources with empty dictionaries
    S = SeismicSources(id, ϕ, λ, r)
    # Fill static dictionary
    initialise_static_dictionaries!(S, id, static_types, static_phases)

    return S
end




#################### SEISMIC RECEIVERS ####################

# Seismic Receivers: Simple structure for storing seismic receiver data in geographic coordinates
struct SeismicReceivers{S, F, K}
    id::Dict{S,Int}
    ϕ::Vector{F}
    λ::Vector{F}
    r::Vector{F}
    static::Dict{Tuple{S, K, K}, F}
end
function SeismicReceivers(id, ϕ, λ, r; Ko = Symbol, Kp = Symbol)
    # Initialise SeismicReceivers with empty dictionaries
    Ki = eltype(id)
    T = eltype(ϕ)
    n = length(id)
    R = SeismicReceivers(Dict(zip(id, 1:n)), ϕ, λ, r, Dict{Tuple{Ki, Ko, Kp}, T}())

    return R
end
function SeismicReceivers(id, ϕ, λ, r, static_types, static_phases)
    # Initialise SeismicReceivers with empty dictionaries
    R = SeismicReceivers(id, ϕ, λ, r)
    # Fill static dictionary
    initialise_static_dictionaries!(R, id, static_types, static_phases)

    return R
end



#################### MODELS ####################

# Abstract Model: Structures containing all the parameters required to make a prediction
# for some observable
abstract type AbstractModel end

# TauP Model Structure: A 1D spherical radial earth velocity model
# While rays are traced in a 1D model, a 3D model is stored so that these 1D rays can
# be re-integrated to give travel-time predictions in hetergeneous earth.
struct ModelTauP{P, G, S, R, F, J} <: AbstractModel
    refmodel::String # Name of TauP reference model
    PathObj::TauP.JavaCall.JavaObject{J} # TauP Java Path Object
    Rₜₚ::F # TauP model radius (km)
    r::Vector{F} # TauP depth -- piece-wise linear with discontinuities (km)
    nd::Vector{Int} # Discontinuity index
    vp::Vector{F} # P velocity (km/s)
    vs::Vector{F} # S velocity (km/s)
    Sources::S # SeismicSources Structure
    Receivers::R # SeismicReceivers Structure
    Mesh::G # Mesh Structure for 3D perturbations to reference TauP model
    m::P # Model perturbation array
    dl₀::F # Ray resampling interval (km)
end
# Outer constructor function to build from a TauP model file
function ModelTauP(Vtype::Type{<:SeismicVelocity}, refmodel, Sources, Receivers, Mesh::RegularGrid{T, 3, R}, dl₀; Δm = nothing) where
    {T<:GeographicCoordinates, R}

    # Load the TauP Velocity model structure
    TP = load_taup_model(refmodel)
    # Initialise the TauP Path Object
    PathObj = buildPathObj(refmodel)
    # Initialise ModelTauP structure
    dims = size(Mesh)
    if isnothing(Δm)
        # Empty perturbation fields
        Model = ModelTauP(refmodel, PathObj, TP.r[end], TP.r, TP.dindex, TP.vp, TP.vs, Sources, Receivers, Mesh, 
            Vtype(eltype(TP.vp), dims), dl₀)
    else
        # Pre-defined perturbations
        Model = ModelTauP(refmodel, PathObj, TP.r[end], TP.r, TP.dindex, TP.vp, TP.vs, Sources, Receivers, Mesh, Δm, dl₀)
    end

    return Model
end



#################### INVERSION PARAMETERS ####################

# Parameter Field: Structure to hold parameters discretised in space
struct ParameterField{T,N} <: InversionParameter
    Mesh::AbstractMesh # Mesh defining discretisation of parameters
    Δm::Array{T,N} # Cummulative perturbations to parameters
    δm::Array{T,N} # Incremental perturbations to parameters
    m₀::Array{T,N} # Starting model values
    σₘ::Array{T,N} # A priori fractional uncertainty
    # Regularisation parameters
    μ::Vector{T} # Damping Weight (square-root of Lagrangian Multiplier)
    μjump::Vector{Bool} # Penalize cummulative (true) or incremental (false) perturbations
    λ::Vector{T} # Smoothing Weight (square-root of Lagrangian Multiplier)
    λjump::Vector{Bool} # Penalize cummulative (true) or incremental (false) roughness
    # Jacobian info
    jcol::Vector{Int} # Jacobian column index for parameter 
    RSJS::Array{T,N} # Row-sum of the Jacobian-squared (i.e. a proxy for parameter sensitivity)
end
# Initialises a ParameterField
function ParameterField(Mesh, T)
    dims = size(Mesh) # Mesh dimensions
    N = length(dims) # Number of dimensions
    P = length(Mesh) # Number of parameters

    return ParameterField(Mesh, zeros(T, dims), zeros(T, dims), Array{T}(undef, dims), Array{T}(undef, dims),
    zeros(T, 1), [false], zeros(T, N), [false], [1, P], zeros(T, dims))
end
# Custom display of structure
function Base.show(io::IO, obj::ParameterField)
    S = split(string(typeof(obj.Mesh)),"{")
    println(io, "ParameterField")
    println(io, "--> Mesh:             ", S[1])
    println(io, "--> Size:             ", size(obj.Mesh))
    println(io, "--> Coordinates:      ", S[2])
    println(io, "--> Parameterisation: ", typeof(obj.m₀))

    return nothing
end

# Statics. Even though these structures are the same, need to break them up for dispatch.
abstract type SeismicStatics <: InversionParameter end

# Seismic Sources
struct SeismicSourceStatics{K, T, L} <: SeismicStatics
    Δm::Dict{K,T} # Dictionary of (ID, Static Type, Static Phase) *keys* and cummulative static perturbations
    δm::Dict{K,T} # Dictionary of (ID, Static Type, Static Phase) *keys* and incremental static perturbations
    σ::Dict{K,T} # Dictionary of (ID, Static Type, Static Phase) *keys* and static uncertainty
    μ::Dict{L,T} # Dictionary of (Static Type, Static Phase) *keys* and their damping factor
    tf_jump::Bool
    jcol::Dict{K,Int} # Dictionary of (ID, Static Type, Static Phase) *keys* and Jacobian column index
    RSJS::Dict{K,T} # Dictionary of (ID, Static Type, Static Phase) *keys* and row-sum of Jacobian squared value
end
function SeismicSourceStatics(id, static_types, static_phases, tf_jump, μ, σ)

    # Initialise dictionaries and statics structure
    Ki = eltype(id)
    Ko = eltype(static_types)
    Kp = eltype(static_phases)
    T = eltype(μ)
    S = SeismicSourceStatics(Dict{Tuple{Ki, Ko, Kp}, T}(), Dict{Tuple{Ki, Ko, Kp}, T}(), Dict{Tuple{Ki, Ko, Kp}, T}(),
    Dict{Tuple{Ko, Kp}, T}(), tf_jump, Dict{Tuple{Ki, Ko, Kp}, Int}(), Dict{Tuple{Ki, Ko, Kp}, T}())
    # Fill the dictionaries
    initialise_static_dictionaries!(S, id, static_types, static_phases, μ, σ)

    return S
end

# Seismic Receivers
struct SeismicReceiverStatics{K, T, L} <: SeismicStatics
    Δm::Dict{K,T}
    δm::Dict{K,T} # Dictionary of (ID, Static Type, Static Phase) *keys* and static value
    σ::Dict{K,T} # Dictionary of (ID, Static Type, Static Phase) *keys* and static uncertainty
    μ::Dict{L,T} # Dictionary of (Static Type, Static Phase) *keys* and their damping factor
    tf_jump::Bool
    jcol::Dict{K,Int} # Dictionary of (ID, Static Type, Static Phase) *keys* and Jacobian column index
    RSJS::Dict{K,T} # Dictionary of (ID, Static Type, Static Phase) *keys* and row-sum of Jacobian squared value
end
function SeismicReceiverStatics(id, static_types, static_phases, tf_jump, μ, σ)

    # Initialise dictionaries and statics structure
    Ki = eltype(id)
    Ko = eltype(static_types)
    Kp = eltype(static_phases)
    T = eltype(μ)
    S = SeismicReceiverStatics(Dict{Tuple{Ki, Ko, Kp}, T}(), Dict{Tuple{Ki, Ko, Kp}, T}(), Dict{Tuple{Ki, Ko, Kp}, T}(),
    Dict{Tuple{Ko, Kp}, T}(), tf_jump, Dict{Tuple{Ki, Ko, Kp}, Int}(), Dict{Tuple{Ki, Ko, Kp}, T}())
    # Fill the dictionaries
    initialise_static_dictionaries!(S, id, static_types, static_phases, μ, σ)

    return S
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




####### UTILITY FUNCTIONS ######

# Simplifies extraction of source or receiver coordinates
function get_seismic_coordinates(S, id)
    n = S.id[id]
    λ = S.λ[n]
    ϕ = S.ϕ[n]
    r = S.r[n]

    return λ, ϕ, r, n
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
# Initialise static dictionaries for inversion
function initialise_static_dictionaries!(S::SeismicStatics, Ki, Ko, Kp, μ, σ)
    # Fill dictionaries
    n = 0
    for k in eachindex(Kp) # Phase Type
        for j in eachindex(Ko) # Observation Type
            μₖⱼ = μ[k][j] # Damping for (phase, observable) pair
            σₖⱼ = σ[k][j] # Uncertainty for (phase, observable) pair
            S.μ[(Ko[j], Kp[k])] = μₖⱼ # Damping parameter--can be unique for each (Observable, SeismicPhase) pair
            for i in eachindex(Ki) # Unique Identifyer (i.e. receiver or source ID)
                n += 1
                S.Δm[(Ki[i], Ko[j], Kp[k])] = 0.0  # Assign initial cummulative static perturbation
                S.δm[(Ki[i], Ko[j], Kp[k])] = 0.0 # Assign initial incremental static perturbation
                S.σ[(Ki[i], Ko[j], Kp[k])] = σₖⱼ # Assign static uncertainty
                S.jcol[(Ki[i], Ko[j], Kp[k])] = n # Assign unique index number for Jacobian
                S.RSJS[(Ki[i], Ko[j], Kp[k])] = 0.0 # Initial row-sum of Jacobian squared
            end
        end
    end

    return nothing
end