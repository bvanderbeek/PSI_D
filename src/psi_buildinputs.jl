# Build input structures required to run code
function build_inputs(param_file)
    # Parse input toml-file
    D = TOML.parsefile(param_file)

    # Build the observation structure 
    Obs = build_observations(D["Observations"])

    # Model
    if isempty(D["Model"]["theModel"])
        # model_type = eval(Symbol(D["Model"]["type"]))
        Model = build_model(PsiModel, D["Model"])
    else
        # model_type = eval(Symbol(D["Model"]["type"]))
        Model = load_model(PsiModel, D["Model"])
    end

    # Inversion Parameters
    if haskey(D, "Invert")
        if haskey(D["Invert"], "ParameterFile")
            error("Loading an existing inverse model file not yet defined!")
        else
            InvParam = build_inversion_parameters(D["Invert"], Model)
        end
    else
        InvParam = nothing
    end

    # Solver
    if haskey(D, "Solver")
        solver_type = eval(Symbol(D["Solver"]["type"]))
        Solv = build_solver(solver_type, D["Solver"])
    else
        Solv = nothing
    end

    return D, Obs, Model, InvParam, Solv
end

# Type unstable but loading data is not currently a performance concern
function build_observations(D::Dict; FwdType = ForwardTauP,  dlm = ",", ValueType = Float64)
    SIDType, RIDType = (Int, String) # Assumed types for source and receiver IDs!!!
    # Allocate observation vector
    ObsUnion = Union{}
    nobs = 0
    for bkey in eachindex(D) # Loop over SeismicObservable types
        ObsType = eval(Symbol(bkey))
        ObsValType = ObsType <: SplittingParameters ? NTuple{2, ValueType} : ValueType
        for pkey in eachindex(D[bkey]) # Loop over SeismicPhase types
            PhsType = eval(Symbol(pkey))
            ObsUnion = Union{ObsType{PhsType, FwdType, SIDType, RIDType, ObsValType}, ObsUnion}
            nobs += countlines(D[bkey][pkey]["filename"])
        end
    end
    bvec = Vector{ObsUnion}(undef, nobs)
    # Fill observation vector
    iobs = 0
    for bkey in eachindex(D) # Loop over SeismicObservable types
        ObsType = eval(Symbol(bkey))
        for pkey in eachindex(D[bkey]) # Loop over SeismicPhase types
            PhsType = eval(Symbol(pkey))
            for line in readlines(D[bkey][pkey]["filename"]) # Loop over observations
                iobs += 1
                line = split(line, dlm)
                aPhase, sid, rid, b, σ = parse_data_row(ObsType, PhsType, line; ValueType = ValueType)
                bvec[iobs] = ObsType(aPhase, FwdType(), sid, rid, b, σ)
            end
        end
    end

    return bvec
end
function parse_data_row(::Type{<:SeismicObservable}, PhsType, line; ValueType = Float64)
    b, σ = ( parse(ValueType, line[1]), parse(ValueType, line[2]) )
    T, phs, sid, rid = (parse(ValueType, line[3]), string(strip(line[4])), parse(Int, line[5]), string(strip(line[6])))
    # Channel name assumed to be stored in column 7
    aPhase = length(line) > 7 ? PhsType(phs, T, parse(ValueType, line[8])) : PhsType(phs, T)
    return aPhase, sid, rid, b, σ
end
function parse_data_row(::Type{SplittingParameters}, PhsType, line; ValueType = Float64)
    b = ( parse(ValueType, line[1]), parse(ValueType, line[2]) )
    σ = ( parse(ValueType, line[3]), parse(ValueType, line[4]) )
    T, phs, sid, rid = ( parse(ValueType, line[5]), string(strip(line[6])), parse(Int, line[7]), string(strip(line[8])) )
    # Channel name assumed to be stored in column 9
    aPhase = length(line) > 9 ? PhsType(phs, T, parse(ValueType, line[10])) : PhsType(phs, T)
    return aPhase, sid, rid, b, σ
end

# Determines the formatting for source and receiver IDs...DEPRECATED! FIXED TYPES FOR SOURCE AND RECEIVER ID
function get_id_type(f; index = 1, dlm = ",")
    # Determine if source IDs are integers or strings
    line = readline(f)
    line = split(line, dlm, keepempty = false)
    if isnothing(tryparse(Int, strip(line[index])))
        id_type = String
        convert_id = (x) -> string(x)
    else
        id_type = Int
        convert_id = (x) -> parse(Int, x)
    end

    return id_type, convert_id
end



function build_model(::Type{PsiModel}, D::Dict)
    # Define coordinate system
    coords = eval(Symbol(D["CoordinateSystem"]["type"]))
    Geometry = build_coordinate_system(coords, D["CoordinateSystem"])

    # Define mesh
    mesh_type = eval(Symbol(D["Mesh"]["type"]))
    Mesh = build_mesh(mesh_type, D["Mesh"], Geometry)

    # Define sources and receivers
    Sources = build_sources(D["Aquisition"]["source_data"])
    Receivers = build_receivers(D["Aquisition"]["receiver_data"])

    # Define forward method parameters
    Methods = build_forward_methods(D["Methods"])

    # Define Parameterisation
    parameterisation = eval(Symbol(D["parameterisation"]))
    
    return build_model(parameterisation, D, Mesh, Sources, Receivers, Methods)
end
function build_model(::Type{<:IsotropicVelocity}, D::Dict, Mesh, Sources, Receivers, Methods)
    return PsiModel(IsotropicVelocity, Mesh, Sources, Receivers, Methods)
end
function build_model(::Type{<:HexagonalVectoralVelocity}, D::Dict, Mesh, Sources, Receivers, Methods)
    ratio_ϵ = D["Parameters"]["ratio_epsilon"]
    ratio_η = D["Parameters"]["ratio_eta"]
    ratio_γ = D["Parameters"]["ratio_gamma"]
    return PsiModel(HexagonalVectoralVelocity, ratio_ϵ, ratio_η, ratio_γ, Mesh, Sources, Receivers, Methods)
end

function build_coordinate_system(::Type{LocalGeographic}, D::Dict)
    # Extract parameters
    R₀ = D["R_0"]
    λ₀ = D["Lon_0"]
    ϕ₀ = D["Lat_0"]
    β  = D["Rot"]

    return LocalGeographic(λ₀, ϕ₀, R₀, β)
end

function build_mesh(::Type{RegularGrid}, D::Dict, Geometry::LocalGeographic)
    # Extract Parameters
    DX1 = D["DX_1"]
    DX2 = D["DX_2"]
    DX3 = D["DX_3"]
    NX1 = D["NX_1"]
    NX2 = D["NX_2"]
    NX3 = D["NX_3"]
    X3_start = haskey(D, "X3_start") ? D["X3_start"] : 0.0 # Option to shift model surface to include elevation
    # Grid coordinate vectors
    x1 = range(start = -DX1, stop = DX1, length = NX1)
    x2 = range(start = -DX2, stop = DX2, length = NX2)
    x3 = range(start = X3_start, stop = -DX3 + X3_start, length = NX3)

    return RegularGrid(Geometry, (x1, x2, x3));
end

# Read seismic source data from file
function build_sources(f; dlm = ",", data_type = Float64)
    # Determine if IDs are integers or strings
    # id_type, convert_id = get_id_type(f; index = 1, dlm = dlm)
    id_type = Int # Force Integer ID

    # Number of sources
    n = countlines(f)
    # Allocate storage arrays
    id = Vector{id_type}(undef, n)
    coordinates = Vector{NTuple{3, data_type}}(undef, n)
    # Loop over data lines
    k = 0
    for line in readlines(f)
        # Update counter
        k += 1
        # Split the line
        line = split(line, dlm, keepempty = false)
        # id[k] = convert_id(strip(line[1]))
        id[k] = parse(Int, line[1])
        lon = parse(data_type, line[2])
        lat = parse(data_type, line[3])
        elv = parse(data_type, line[4])
        coordinates[k] = (lon, lat, elv)
    end

    return SeismicSources(id, coordinates)
end

# Read seismic source data from file
function build_receivers(f; dlm = ",", data_type = Float64)
    # Determine if IDs are integers or strings
    # id_type, convert_id = get_id_type(f; index = 1, dlm = dlm)
    id_type = String # Force String ID

    # Number of sources
    n = countlines(f)
    # Allocate storage arrays
    id = Vector{id_type}(undef, n)
    coordinates = Vector{NTuple{3, data_type}}(undef, n)
    # Loop over data lines
    k = 0
    for line in readlines(f)
        # Update counter
        k += 1
        # Split the line
        line = split(line, dlm, keepempty = false)
        # id[k] = convert_id(strip(line[1]))
        id[k] = strip(line[1])
        lon = parse(data_type, line[2])
        lat = parse(data_type, line[3])
        elv = parse(data_type, line[4])
        coordinates[k] = (lon, lat, elv)
    end
    
    return SeismicReceivers(id, coordinates)
end

# Define calculation parameters for forward methods
function build_forward_methods(D::Dict)
    if haskey(D, "TauP")
        reference_model = D["TauP"]["reference_model"]
        dL = D["TauP"]["DL"]
        MethodTauP = ParametersTauP(; reference_model = reference_model, dL = dL)
    else
        MethodTauP = ParametersTauP()
    end
    MethodShortestPath = ParametersShortestPath()
    MethodFiniteFrequency = ParametersFiniteFrequency()

    return ForwardMethodsContainer(MethodTauP, MethodShortestPath, MethodFiniteFrequency)
end



# Create a PsiModel from an existing file
function load_model(::Type{PsiModel}, D::Dict)
    # Model file and type
    theModel = D["theModel"]
    ftype = split(theModel, ".")
    if ftype[end] == "dat"
        # Model coordinate system
        coords = eval(Symbol(D["CoordinateSystem"]["type"]))
        # Model mesh
        mesh_type = eval(Symbol(D["Mesh"]["type"]))
        # Model parameterisation
        parameterisation = eval(Symbol(D["parameterisation"]))
        # Read the model and mesh
        Parameters, Mesh = read_model(coords, mesh_type, parameterisation, theModel)
    else
        error("Loading model files of type "*ftype[end]*" is not supported.")
    end

    # Define sources and receivers
    Sources = build_sources(D["Aquisition"]["source_data"])
    Receivers = build_receivers(D["Aquisition"]["receiver_data"])

    # Define forward method parameters
    Methods = build_forward_methods(D["Methods"])

    return PsiModel(Mesh, Sources, Receivers, Parameters, Methods)
end

# Loads a model dat-file defined in a LocalGeographic coordinate system, on a regular grid, with HexagonalVelocity parameters
function read_model(::Type{LocalGeographic}, ::Type{RegularGrid}, parameterisation, f; dlm = ",", T = Float64)
    io = open(f)
    # Read header line that defines coordinate system
    line = readline(io)
    line = split(line, dlm)
    R₀ = parse(T, line[1])
    λ₀ = parse(T, line[2])
    ϕ₀ = parse(T, line[3])
    β = parse(T, line[4])
    # Read header line that specifies model dimensions
    line = readline(io)
    line = split(line, dlm)
    nx = (parse(Int, line[1]), parse(Int, line[2]), parse(Int, line[3]))
    # Read next header line that defines extent of model (arc-wdith, arc_height, max_depth)
    line = readline(io)
    line = split(line, dlm)
    Δx = (parse(T, line[1]), parse(T, line[2]), parse(T, line[3]))
    dz_ext = length(line) > 3 ? parse(T, line[4]) : 0.0 # Read starting depth for vertically extended models
    # Build the coordinate system for this model
    Geometry = LocalGeographic(λ₀, ϕ₀, R₀, β)
    # Build the mesh for this model
    x₁ = range(start = -Δx[1], stop = Δx[1], length = nx[1]) # Local spherical coordinates!
    x₂ = range(start = -Δx[2], stop = Δx[2], length = nx[2]) # Local spherical coordinates!
    x₃ = range(start = dz_ext, stop = -Δx[3] + dz_ext, length = nx[3]) # Reversed! Depth vector include vertical extension
    Mesh = RegularGrid(Geometry, (x₁, x₂, x₃))
    # Read in model parameters. This will close the file.
    Parameters = read_model_parameters(io, parameterisation, Mesh; dlm = dlm, T = T)

    return Parameters, Mesh
end

function read_model_parameters(io, parameterisation::Type{IsotropicVelocity}, Mesh; dlm = ",", T = Float64)
    # Allocate model structure
    Parameters = IsotropicVelocity(T, size(Mesh))
    # Read and populate model
    k = 0
    for line in readlines(io)
        # Update counter
        k += 1
        # Split the line
        line = split(line, dlm)
        # Store hexagonal velocity parameters
        Parameters.vp[k] = parse(T, line[4])
        Parameters.vs[k] = parse(T, line[5])
    end
    close(io)

    return Parameters
end

function read_model_parameters(io, parameterisation::Type{HexagonalVectoralVelocity}, Mesh; dlm = ",", T = Float64)
    # Allocate model structure
    nxyz = size(Mesh)
    line = readline(io)
    line = strip.(split(line, dlm))
    if length(line) == 1
        tf_read_ratios = true
        tf_exact = parse(Bool, line[1])
        Parameters = HexagonalVectoralVelocity(zeros(nxyz), zeros(nxyz), zeros(nxyz), zeros(nxyz), zeros(nxyz),
        zeros(nxyz), zeros(nxyz), zeros(nxyz), tf_exact)
    elseif length(line) == 4
        tf_read_ratios = false
        tf_exact, ratio_ϵ, ratio_η, ratio_γ = (parse(Bool, line[1]), parse(T, line[2]), parse(T, line[3]), parse(T, line[4]))
        Parameters = HexagonalVectoralVelocity(zeros(nxyz), zeros(nxyz), zeros(nxyz), zeros(nxyz), zeros(nxyz),
        ratio_ϵ, ratio_η, ratio_γ, tf_exact)
    else
        error("Unrecognized header format!")
    end
    # Read and populate model
    k = 0
    for line in readlines(io)
        # Update counter
        k += 1
        # Split the line
        line = split(line, dlm)
        # Store hexagonal velocity parameters
        Parameters.α[k] = parse(T, line[4])
        Parameters.β[k] = parse(T, line[5])
        Parameters.f[k] = parse(T, line[6])
        Parameters.azimuth[k] = parse(T, line[7])
        Parameters.elevation[k] = parse(T, line[8])
        if tf_read_ratios
            Parameters.ratio_ϵ[k], Parameters.ratio_η[k], Parameters.ratio_γ[k] = (parse(T, line[9]), parse(T, line[10]), parse(T, line[11]))
        end
    end
    close(io)

    return Parameters
end

# INVERSION PARAMETERS

# Builds a SeismicPerturbationModel
function build_inversion_parameters(D::Dict, Model)

    # Seismic Source Statics
    if haskey(D, "SourceStatics")
        SourceStatics = build_source_statics(D["SourceStatics"], collect(keys(Model.Sources.id)))
    else
        SourceStatics = nothing
    end
    # Seismic Receiver Statics
    if haskey(D, "ReceiverStatics")
        ReceiverStatics = build_receiver_statics(D["ReceiverStatics"], collect(keys(Model.Receivers.id)))
    else
        ReceiverStatics = nothing
    end
    # Seismic Velocity Parameters
    if haskey(D, "Velocity")
        Velocity = build_velocity_parameters(D["Velocity"], Model)
    else
        Velocity = nothing
    end

    # The following parameter fields have yet to be implemented
    Hypocenter = haskey(D, "Hypocenter") ? error("Hypocenter inversion not implemented.") : nothing
    Interface = haskey(D, "Interface") ? error("Interface inversion not implemented.") : nothing

    return SeismicPerturbationModel(Velocity = Velocity, Interface = Interface, Hypocenter = Hypocenter, 
    SourceStatics = SourceStatics, ReceiverStatics = ReceiverStatics)
end

# Build InverseSeismicVelocity
function build_velocity_parameters(D::Dict, Model::PsiModel)
    if haskey(D, "Isotropic")
        Isotropic = build_isotropic_parameters(D["Isotropic"], Model)
    else
        Isotropic = nothing
    end

    if haskey(D, "Anisotropic")
        Anisotropic = build_anisotropic_parameters(D["Anisotropic"], Model)
    else
        Anisotropic = nothing
    end

    return InverseSeismicVelocity(Isotropic, Anisotropic)
end

# Build Isotropic Parameter Fields
function build_isotropic_parameters(D::Dict, Model::PsiModel)
    # Build the isotropic parameter mesh
    mesh_type = eval(Symbol(D["Mesh"]["type"]))
    Mesh = build_mesh(mesh_type, D["Mesh"], Model.Mesh.Geometry)

    # P-wave parameter field
    if haskey(D, "P")
        P = ParameterField(Mesh)
        P.wdamp .= D["P"]["damping_weight"]
        P.tf_damp_cumulative .= D["P"]["tf_min_cumulative"]
        P.wsmooth .= D["P"]["smoothing_weights"]
        P.tf_smooth_cumulative .= D["P"]["tf_smooth_cumulative"]
    else
        P = nothing
    end
    # S-wave parameter field
    if haskey(D, "S")
        S = ParameterField(Mesh)
        S.wdamp .= D["S"]["damping_weight"]
        S.tf_damp_cumulative .= D["S"]["tf_min_cumulative"]
        S.wsmooth .= D["S"]["smoothing_weights"]
        S.tf_smooth_cumulative .= D["S"]["tf_smooth_cumulative"]
    else
        S = nothing
    end

    return build_isotropic_parameters(eval(Symbol(D["parameterisation"])), Model, P, S, D["coupling_option"])
end
# Build InverseIsotropicSlowness
function build_isotropic_parameters(::Type{InverseIsotropicSlowness}, Model::PsiModel, P, S, coupling_option)
    # Build structure
    IsotropicParameters = InverseIsotropicSlowness(P, S, coupling_option)
    # Interpolate starting model parameters
    interpolate_parameters!(IsotropicParameters, Model; tf_extrapolate = true, tf_harmonic = true)

    return IsotropicParameters
end

# Build Anisotropic Parameter Fields
# -> Replaces build_parameter_field(::Type{InverseAzRadVector}, D::Dict, Model::PsiModel{<:HexagonalVectoralVelocity})
function build_anisotropic_parameters(D::Dict, Model::PsiModel)
    # Build the anisotropic parameter mesh
    mesh_type = eval(Symbol(D["Mesh"]["type"]))
    Mesh = build_mesh(mesh_type, D["Mesh"], Model.Mesh.Geometry)

    if haskey(D, "Orientations")
        parameterisation = eval(Symbol(D["Orientations"]["parameterisation"]))
        Orientations = build_anisotropic_parameters(eval(Symbol(D["Orientations"]["parameterisation"])), Mesh, Model,
        D["Orientations"]["damping_weight"], D["Orientations"]["tf_min_cumulative"],
        D["Orientations"]["smoothing_weights"], D["Orientations"]["tf_smooth_cumulative"])
    else
        Orientations = nothing
    end
    if haskey(D, "Fractions")
        error("Anisotropic fraction inversion not yet implemented!")
    else
        Fractions = nothing
    end
    
    return InverseAnisotropicVector(Orientations, Fractions)
end
function build_anisotropic_parameters(::Type{InverseAzRadVector}, Mesh, Model::PsiModel{<:HexagonalVectoralVelocity},
    wdamp::Number, tf_damp_cumulative::Bool, wsmooth::Vector, tf_smooth_cumulative::Bool)
    # Build parameter fields for the AzRad vector components
    A, B, C = (
        ParameterField(Mesh; wdamp = [wdamp], tf_damp_cumulative = [tf_damp_cumulative], wsmooth = wsmooth, tf_smooth_cumulative = [tf_smooth_cumulative]),
        ParameterField(Mesh; wdamp = [wdamp], tf_damp_cumulative = [tf_damp_cumulative], wsmooth = wsmooth, tf_smooth_cumulative = [tf_smooth_cumulative]),
        ParameterField(Mesh; wdamp = [wdamp], tf_damp_cumulative = [tf_damp_cumulative], wsmooth = wsmooth, tf_smooth_cumulative = [tf_smooth_cumulative])
    )
    # Check that the starting model is isotropic. If not, we need to interpolate the starting model parameters
    any(Model.Parameters.f .> 0.0) ? error("Missing anisotropic starting model interpolation!") : nothing

    return InverseAzRadVector(A, B, C)
end

# Build SeismicSourceStatics
function build_source_statics(D::Dict, id; value_type = Float64)
    # Initialise dictionaries in statics structure
    Ki, Ko, Kp = (eltype(id), Symbol, Symbol) # ID-Observable-Phase Tuple for keys
    Statics = SeismicSourceStatics(Dict{Tuple{Ki, Ko, Kp}, value_type}(), Dict{Tuple{Ki, Ko, Kp}, value_type}(), Dict{Tuple{Ki, Ko, Kp}, value_type}(),
    Dict{Tuple{Ko, Kp}, value_type}(), Dict{Tuple{Ko, Kp}, Bool}(), Dict{Tuple{Ki, Ko, Kp}, Int}(), Dict{Tuple{Ki, Ko, Kp}, value_type}())
    # Fill the static dictionaries
    initialize_static_parameters!(Statics, D, id)

    return Statics
end
# Build SeismicReceiverStatics
function build_receiver_statics(D::Dict, id; value_type = Float64)
    # Initialise dictionaries in statics structure
    Ki, Ko, Kp = (eltype(id), Symbol, Symbol) # ID-Observable-Phase Tuple for keys
    Statics = SeismicReceiverStatics(Dict{Tuple{Ki, Ko, Kp}, value_type}(), Dict{Tuple{Ki, Ko, Kp}, value_type}(), Dict{Tuple{Ki, Ko, Kp}, value_type}(),
    Dict{Tuple{Ko, Kp}, value_type}(), Dict{Tuple{Ko, Kp}, Bool}(), Dict{Tuple{Ki, Ko, Kp}, Int}(), Dict{Tuple{Ki, Ko, Kp}, value_type}())
    # Fill the static dictionaries
    initialize_static_parameters!(Statics, D, id)
    
    return Statics
end
function initialize_static_parameters!(S::SeismicStatics, D::Dict, id)
    n = 0
    for obs in eachindex(D) # Observable
        key_obs = Symbol(obs)
        for (p, phs) in enumerate(D[obs]["phases"]) # Phase
            key_phs = Symbol(phs)
            S.tf_damp_cumulative[(key_obs, key_phs)] = D[obs]["tf_jump"][p] # Damping parameter--can be unique for each (Observable, SeismicPhase) pair
            S.wdamp[(key_obs, key_phs)] = D[obs]["damping"][p]
            for key_id in id # Identifier
                n += 1
                S.dm[(key_id, key_obs, key_phs)] = 0.0  # Assign initial cummulative static perturbation
                S.ddm[(key_id, key_obs, key_phs)] = 0.0 # Assign initial incremental static perturbation
                S.uncertainty[(key_id, key_obs, key_phs)] = D[obs]["uncertainty"][p] # Assign static uncertainty
                S.jcol[(key_id, key_obs, key_phs)] = n # Assign unique index number for Jacobian
                S.RSJS[(key_id, key_obs, key_phs)] = 0.0 # Initial row-sum of Jacobian squared
            end
        end
    end

    return nothing
end


function build_solver(::Type{SolverLSQR}, D::Dict)
    # Hard-coded parameters
    damp = 0.0 # Keep zero-valued to prevent additional damping within solver
    verbose = false # If true, prints summary of every iteration of the solver
    # Extract parameters
    atol = D["atol"]
    conlim = D["conlim"]
    maxiter = convert(Int, D["maxiter"])
    nonliniter = convert(Int, D["nonliniter"])
    tf_jac_scale = D["tf_jac_scale"]

    return SolverLSQR(damp, atol, conlim, maxiter, verbose, tf_jac_scale, nonliniter)
end