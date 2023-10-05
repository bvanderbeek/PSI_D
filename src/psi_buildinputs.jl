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
    if isempty(D["InversionParameters"]["theParameters"])
        InvParam = build_inversion_parameters(D["InversionParameters"], Model)
    else
        error("Loading an existing parameter structure not yet defined")
    end

    # Solver
    solver_type = eval(Symbol(D["Solver"]["type"]))
    Solv = build_solver(solver_type, D["Solver"])

    return Obs, Model, InvParam, Solv, D
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
            B[k] = Obs(phase(p, paz), forward(), sid, rid, T, b, σ)
        else
            B[k] = Obs(phase(p), forward(), sid, rid, T, b, σ)
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

function build_model(::Type{ModelTauP}, D::Dict)
    # Define coordinate system
    coords = eval(Symbol(D["CoordinateSystem"]["type"]))
    CS = build_coordinate_system(coords, D["CoordinateSystem"])

    # Define mesh
    mesh_type = eval(Symbol(D["Mesh"]["type"]))
    Mesh = build_mesh(mesh_type, D["Mesh"], CS)

    # Define sources and receivers
    Sources = build_sources(D["Aquisition"]["source_data"])
    Receivers = build_receivers(D["Aquisition"]["receiver_data"])

    # Build TauP model
    parameterisation = eval(Symbol(D["parameterisation"]))
    refmodel = D["refModel"]
    dl₀ = D["DL"]
    
    return ModelTauP(parameterisation, refmodel, Sources, Receivers, Mesh, dl₀)
end

# Create a ModelTauP from an existing file
function load_model(::Type{ModelTauP}, D::Dict)
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
        # Load the model and mesh
        Δm, Mesh = load_model(coords, mesh_type, parameterisation, theModel)
    else
        error("Loading model files of type "*ftype[end]*" is not supported.")
    end
    # Ray discretisation increment for TauP paths
    dl₀ = D["DL"]
    # Load the TauP Velocity model structure
    refmodel = D["refModel"]
    TP = load_taup_model(refmodel)
    # Initialise the TauP Path Object
    PathObj = buildPathObj(refmodel)

    # Compute velocity perturbations relative to reference model
    Rₜₚ = TP.r[end]
    for (k, rq) in enumerate(Mesh.x[3])
        # TauP depth
        rq = Rₜₚ - Mesh.CS.R₀ - rq
        # Interpolate Taup velocities
        vp = piecewise_linearly_interpolate(TP.r, TP.dindex, TP.vp, rq; tf_extrapolate = true, tf_harmonic = true)
        vs = piecewise_linearly_interpolate(TP.r, TP.dindex, TP.vs, rq; tf_extrapolate = true, tf_harmonic = true)
        # Subtract reference velocities from model velocities
        for i in eachindex(Mesh.x[1]), j in eachindex(Mesh.x[2])
            Δm.vp[i,j,k] -= vp
            Δm.vs[i,j,k] -= vs
        end
    end

    # Define sources and receivers
    Sources = build_sources(D["Aquisition"]["source_data"])
    Receivers = build_receivers(D["Aquisition"]["receiver_data"])

    return ModelTauP(refmodel, PathObj, Rₜₚ, TP.r, TP.dindex, TP.vp, TP.vs, Sources, Receivers, Mesh, Δm, dl₀)
end

# Loads a model dat-file defined in a LocalGeographic coordinate system, on a regular grid, with HexagonalVelocity parameters
function load_model(::Type{LocalGeographic}, ::Type{RegularGrid}, ::Type{HexagonalVelocity}, f; dlm = ",", T = Float64)
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
    CS = LocalGeographic(λ₀, ϕ₀, R₀, β)
    # Build the mesh for this model
    x₁ = range(start = xlim[1], stop = xlim[4], length = nx[1])
    x₂ = range(start = xlim[2], stop = xlim[5], length = nx[2])
    x₃ = range(start = xlim[6], stop = xlim[3], length = nx[3]) # Reversed!
    Mesh = RegularGrid(CS, (x₁, x₂, x₃))
    # Allocate model structure
    A = HexagonalVelocity(T, nx)
    # Read and populate model
    k = 0
    for line in readlines(io)
        # Update counter
        k += 1
        # Split the line
        line = split(line, dlm)
        # Store hexagonal velocity parameters
        A.vp[k] = parse(T, line[4])
        A.vs[k] = parse(T, line[5])
        A.fp[k] = parse(T, line[6])
        A.fs[k] = parse(T, line[7])
        A.gs[k] = parse(T, line[8])
        A.ϕ[k] = parse(T, line[9])
        A.θ[k] = parse(T, line[10])
    end
    close(io)

    return A, Mesh
end

# Loads a model dat-file defined in a LocalGeographic coordinate system, on a regular grid, with HexagonalVelocity parameters
function load_model(::Type{LocalGeographic}, ::Type{RegularGrid}, ::Type{IsotropicVelocity}, f; dlm = ",", T = Float64)
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
    CS = LocalGeographic(λ₀, ϕ₀, R₀, β)
    # Build the mesh for this model
    x₁ = range(start = xlim[1], stop = xlim[4], length = nx[1])
    x₂ = range(start = xlim[2], stop = xlim[5], length = nx[2])
    x₃ = range(start = xlim[6], stop = xlim[3], length = nx[3]) # Reversed!
    Mesh = RegularGrid(CS, (x₁, x₂, x₃))
    # Allocate model structure
    A = IsotropicVelocity(T, nx)
    # Read and populate model
    k = 0
    for line in readlines(io)
        # Update counter
        k += 1
        # Split the line
        line = split(line, dlm)
        # Store hexagonal velocity parameters
        A.vp[k] = parse(T, line[4])
        A.vs[k] = parse(T, line[5])
    end
    close(io)

    return A, Mesh
end


function build_coordinate_system(::Type{LocalGeographic}, D::Dict)
    # Extract parameters
    R₀ = D["R_0"]
    λ₀ = D["Lon_0"]
    ϕ₀ = D["Lat_0"]
    β  = D["Rot"]

    return LocalGeographic(λ₀, ϕ₀, R₀, β)
end

function build_mesh(::Type{RegularGrid}, D::Dict, CS::LocalGeographic)
    # Extract Parameters
    Δλ = D["DX_1"]
    Δϕ = D["DX_2"]
    Δr = D["DX_3"]
    nx₁ = D["NX_1"]
    nx₂ = D["NX_2"]
    nx₃ = D["NX_3"]
    # Grid coordinate vectors
    x₁ = range(start = -Δλ, stop = Δλ, length = nx₁)
    x₂ = range(start = -Δϕ, stop = Δϕ, length = nx₂)
    x₃ = range(start = 0.0, stop = -Δr, length = nx₃)

    return RegularGrid(CS, (x₁, x₂, x₃));
end

# Read seismic source data from file
function build_sources(f; dlm = ",", data_type = Float64)
    # Determine if IDs are integers or strings
    id_type, convert_id = get_id_type(f; index = 1, dlm = dlm)

    # Number of sources
    n = countlines(f)
    # Allocate storage arrays
    id = Vector{id_type}(undef, n)
    lon = Vector{data_type}(undef, n)
    lat = Vector{data_type}(undef, n)
    elv = Vector{data_type}(undef, n)
    # Loop over data lines
    k = 0
    for line in readlines(f)
        # Update counter
        k += 1
        # Split the line
        line = split(line, dlm)
        id[k] = convert_id(strip(line[1]))
        lon[k] = parse(data_type, line[2])
        lat[k] = parse(data_type, line[3])
        elv[k] = parse(data_type, line[4])
    end

    return SeismicSources(id, lat, lon, elv)
end

# Read seismic source data from file
function build_receivers(f; dlm = ",", data_type = Float64)
    # Determine if IDs are integers or strings
    id_type, convert_id = get_id_type(f; index = 1, dlm = dlm)

    # Number of sources
    n = countlines(f)
    # Allocate storage arrays
    id = Vector{id_type}(undef, n)
    lon = Vector{data_type}(undef, n)
    lat = Vector{data_type}(undef, n)
    elv = Vector{data_type}(undef, n)
    # Loop over data lines
    k = 0
    for line in readlines(f)
        # Update counter
        k += 1
        # Split the line
        line = split(line, dlm)
        id[k] = convert_id(strip(line[1]))
        lon[k] = parse(data_type, line[2])
        lat[k] = parse(data_type, line[3])
        elv[k] = parse(data_type, line[4])
    end

    return SeismicReceivers(id, lat, lon, elv)
end

function build_inversion_parameters(D::Dict, Model)
    if haskey(D, "Velocity")
        parameterisation = eval(Symbol(D["Velocity"]["parameterisation"]))
        Velocity = build_parameter_field(parameterisation, D["Velocity"], Model)
    else
        Velocity = nothing
    end

    if haskey(D, "Interface")
        error("Interface not implemented.")
    else
        Interface = nothing
    end

    if haskey(D, "Hypocenter")
        error("Hypocenter not implemented.")
    else
        Hypocenter = nothing
    end

    if haskey(D, "SourceStatics")
        SourceStatics = build_source_statics(D["SourceStatics"], collect(keys(Model.Sources.id)))
    else
        SourceStatics = nothing
    end

    if haskey(D, "ReceiverStatics")
        error("Receiver Statics not implemented.")
    else
        ReceiverStatics = nothing
    end

    return SeismicPerturbationModel(Velocity = Velocity, Interface = Interface, Hypocenter = Hypocenter, 
    SourceStatics = SourceStatics, ReceiverStatics = ReceiverStatics)
end

function build_parameter_field(::Type{IsotropicSlowness}, D::Dict, Model::ModelTauP)
    # Define mesh
    mesh_type = eval(Symbol(D["Mesh"]["type"]))
    Mesh = build_mesh(mesh_type, D["Mesh"], Model.Mesh.CS)

    # Extract damping parameters
    tf_Djump = D["tf_Djump"]
    damping = D["damping"]
    unc = D["uncertainty"]

    # Extract smoothing parameters
    tf_Sjump = D["tf_Sjump"]
    smoothing = D["smoothing"]

    # Extract boolean indentifying for which parameters we are inverting
    tf_up_us = D["tf_invert"]

    # Build ParameterField for P-slowness
    if tf_up_us[1]
        # Initialise parameter field
        up = ParameterField(Mesh, eltype(damping))
        # Interpolate model to parameter grid
        interpolate_model!(up, Model, :up; tf_extrapolate = true, tf_harmonic = true)
        # Assign uncertainty
        up.σₘ .= unc[1]
        # Assign damping parameters
        up.μ[1] = damping[1]
        up.μjump[1] = tf_Djump
        # Assign smoothing parameters
        up.λ .= smoothing[1]
        up.λjump[1] = tf_Sjump
    else
        up = nothing
    end

    # Build ParameterField for S-slowness
    if tf_up_us[2]
        # Initialise parameter field
        us = ParameterField(Mesh, eltype(damping))
        # Interpolate model to parameter grid
        interpolate_model!(us, Model, :us; tf_extrapolate = true, tf_harmonic = true)
        # Assign uncertainty
        us.σₘ .= unc[2]
        # Assign damping parameters
        us.μ[1] = damping[2]
        us.μjump[1] = tf_Djump
        # Assign smoothing parameters
        us.λ .= smoothing[2]
        us.λjump[1] = tf_Sjump
    else
        us = nothing
    end

    return IsotropicSlowness(up, us)
end

function build_source_statics(D::Dict, id)
    # Extract parameters
    static_types = Symbol.(D["types"])
    static_phases = Symbol.(D["phases"])
    tf_jump = D["tf_jump"]
    damping = D["damping"]
    unc = D["uncertainty"]

    return SeismicSourceStatics(id, static_types, static_phases, tf_jump, damping, unc)
end

function build_solver(::Type{SolverLSQR}, D::Dict)
    # Hard-coded parameters
    damp = 0 # Keep zero-valued to prevent additional damping within solver
    verbose = false # If true, prints summary of every iteration of the solver
    # Extract parameters
    atol = D["atol"]
    conlim = D["conlim"]
    maxiter = D["maxiter"]
    nonliniter = D["nonliniter"]
    tf_jac_scale = D["tf_jac_scale"]

    return SolverLSQR(damp, atol, conlim, maxiter, verbose, tf_jac_scale, nonliniter)
end