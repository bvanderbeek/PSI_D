# Build input structures required to run code
function build_inputs(param_file)
    # Parse input toml-file
    D = TOML.parsefile(param_file)

    # Observations
    if isempty(D["Observations"]["theObservations"])
        # Build the observation structure 
        Obs = build_observations(D["Observations"])
    else
        error("Loading an existing observation structure is not yet implemented.")
    end

    # Model
    if isempty(D["Model"]["theModel"])
        model_type = eval(Symbol(D["Model"]["type"]))
        Model = build_model(model_type, D["Model"])
    else
        model_type = eval(Symbol(D["Model"]["type"]))
        Model = load_model(model_type, D["Model"])
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

# Returns vector of observations
function build_observations(D::Dict)
    # Parse inputs
    f = D["data_file"] # Data file with observations
    obs = eval(Symbol(D["type"])) # Type of observations in data file
    phase = eval(Symbol(D["phase"])) # Phase type in data file
    forward = eval(Symbol(D["forward"])) # Forward modelling method for these data

    return fill_observation_vector(f, obs, phase, forward; dlm = ",", data_type = Float64)
end

# Loads SeismicObservable observations of a specific kind into an array
function fill_observation_vector(f, Obs::Type{<:SeismicObservable}, phase, forward; dlm = ",", data_type = Float64)
    # Determing source/receiver ID format
    sid_type, convert_sid = get_id_type(f; index = 5, dlm = dlm)
    rid_type, convert_rid = get_id_type(f; index = 6, dlm = dlm)

    # Read polarisation angle from file? Polarisation is assumed to be stored in 8th column.
    line = readline(f)
    line = split(line, dlm)
    tf_read_paz = (length(line) > 7)

    # Allocate storage vector
    n = countlines(f)
    B = Vector{Obs{phase, forward, sid_type, rid_type, data_type}}(undef, n)
    # Read in observations
    k = 0
    for line in readlines(f)
        # Update counter 
        k += 1
        # Parse data row
        line = split(line, dlm)
        b = parse(data_type, line[1])
        σ = parse(data_type, line[2])
        T = parse(data_type, line[3])
        p = string(strip(line[4]))
        sid = convert_sid(strip(line[5]))
        rid = convert_rid(strip(line[6]))
        chn = strip(line[7])
        # Build observable
        if tf_read_paz
            paz = parse(data_type, line[8])
            B[k] = Obs(phase(p, T, paz), forward(), sid, rid, b, σ)
        else
            B[k] = Obs(phase(p, T), forward(), sid, rid, b, σ)
        end
    end

    return B
end

# Determines the formatting for source and receiver IDs
function get_id_type(f; index = 1, dlm = ",")
    # Determine if source IDs are integers or strings
    line = readline(f)
    line = split(line, dlm)
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
    # Grid coordinate vectors
    x1 = range(start = -DX1, stop = DX1, length = NX1)
    x2 = range(start = -DX2, stop = DX2, length = NX2)
    x3 = range(start = 0.0, stop = -DX3, length = NX3)

    return RegularGrid(Geometry, (x1, x2, x3));
end

# Read seismic source data from file
function build_sources(f; dlm = ",", data_type = Float64)
    # Determine if IDs are integers or strings
    id_type, convert_id = get_id_type(f; index = 1, dlm = dlm)

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
        line = split(line, dlm)
        id[k] = convert_id(strip(line[1]))
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
    id_type, convert_id = get_id_type(f; index = 1, dlm = dlm)

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
        line = split(line, dlm)
        id[k] = convert_id(strip(line[1]))
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
    # Read next header line that defines extent of model (min-x₁, min-x₂, min-x₃, max-x₁, max-x₂, max-x₃)
    line = readline(io)
    line = split(line, dlm)
    xlim = (parse(T, line[1]), parse(T, line[2]), parse(T, line[3]), parse(T, line[4]), parse(T, line[5]), parse(T, line[6]))
    # Build the coordinate system for this model
    Geometry = LocalGeographic(λ₀, ϕ₀, R₀, β)
    # Build the mesh for this model
    x₁ = range(start = xlim[1], stop = xlim[4], length = nx[1])
    x₂ = range(start = xlim[2], stop = xlim[5], length = nx[2])
    x₃ = range(start = xlim[6], stop = xlim[3], length = nx[3]) # Reversed!
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

# Build SeismicReceiverStatics
function build_source_statics(D::Dict, id)
    # Extract parameters
    static_types = Symbol.(D["types"])
    static_phases = Symbol.(D["phases"])
    tf_jump = D["tf_jump"]
    damping = D["damping"]
    unc = D["uncertainty"]

    return SeismicSourceStatics(id, static_types, static_phases, tf_jump, damping, unc)
end
# Build SeismicReceiverStatics
function build_receiver_statics(D::Dict, id)
    # Extract parameters
    static_types = Symbol.(D["types"])
    static_phases = Symbol.(D["phases"])
    tf_jump = D["tf_jump"]
    damping = D["damping"]
    unc = D["uncertainty"]

    return SeismicReceiverStatics(id, static_types, static_phases, tf_jump, damping, unc)
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