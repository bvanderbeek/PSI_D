
function psi_forward_seismic_dijkstra(Observations, Model; length_0 = 0.0, pred = 0)
    # Make the graph. Needs to be in global cartesian
    G = make_graph(Model)

    # Dictionary of origin indices (keys) and the observation and destination indices (values)
    itinerary, tf_reverse = build_itinerary(Observations, Model.Sources, Model.Receivers)
    if tf_reverse
        start_points, end_points = Model.Receivers.coordinates, Model.Sources.coordinates
    else
        start_points, end_points = Model.Sources.coordinates, Model.Receivers.coordinates
    end

    # Loop over initialisation points
    num_obs, obs_type = length(Observations), typeof(Observations[1].observation) # Observation value type (assumed uniform...would fail for SplittingParameters + TravelTimes, for example)
    predictions, rel_res, tt_sp = Vector{obs_type}(undef, num_obs), Vector{obs_type}(undef, num_obs), zeros(num_obs)
    Kernels = allocate_kernels_vector(Observations, Model)
    n, npt = 0, length(itinerary)
    for (departure, destinations) in itinerary
        # Define phase for SeismicDijkstra
        if departure[2] == "P"
            phase_i = SeismicDijkstra.BodyP()
        elseif departure[2] == "S"
            phase_i = SeismicDijkstra.BodySmean()
        else
            error("Foreign phase!")
        end
        lon_start, lat_start, elv_start = start_points[departure[1]]
        a_i = geographic_to_local(lon_start, lat_start, elv_start, Model.Mesh.Geometry)
        x_start = geographic_to_global(a_i[1], a_i[2], a_i[3]; radius = Model.Mesh.Geometry.R₀)
        D = SeismicDijkstra.initialize_dijkstra(G, SVector(x_start); phase = phase_i, length_0 = length_0, pred = pred)
        @time SeismicDijkstra.dijkstra!(D, G; phase = phase_i)

        # Loop over end points to extract times and paths
        for (j_obs, i_end) in destinations
            lon_end, lat_end, elv_end = end_points[i_end]
            b_i = geographic_to_local(lon_end, lat_end, elv_end, Model.Mesh.Geometry)
            x_end = geographic_to_global(b_i[1], b_i[2], b_i[3]; radius = Model.Mesh.Geometry.R₀)
            t_min, xyz_path, vert_path = SeismicDijkstra.get_path(D, G, SVector(x_end); phase = phase_i, length_0 = length_0)
            !tf_reverse && reverse!(xyz_path; dims = 2) # Reverse the path for source-initialisation

            t_min, xyz_path = SeismicDijkstra.refine_path(G, xyz_path, Model.Methods.ShortestPath.dl;
                phase = phase_i, dist = length_0, min_vert = Model.Methods.ShortestPath.min_vert)

            # Rotate to true global cartesian! Needed for kernel creation!
            xyz_path .= Model.Mesh.Geometry.Rₘ*xyz_path

            # Convert path to vector of tuples
            # fine_path = refine_path(xyz_path, dr; min_vert = 2) # Finely sample path
            kernel_coordinates = Vector{NTuple{3, Float64}}(undef, size(xyz_path,2))
            [kernel_coordinates[k] = (xyz_path[1,k], xyz_path[2,k], xyz_path[3,k]) for k in eachindex(kernel_coordinates)]

            # Kernel weights defined using local ray orientations
            kernel_weights = return_ray_kernel_weights(kernel_coordinates)
            for (k, w) in enumerate(kernel_weights)
                x_k, y_k, z_k = kernel_coordinates[k]
                lon_k, lat_k, _ = global_to_geographic(x_k, y_k, z_k; radius = Model.Mesh.Geometry.R₀)
                azm, elv = ecef_to_local_orientation(lon_k, lat_k, w.azimuth, w.elevation)
                kernel_weights[k] = (dr = w.dr, azimuth = azm, elevation = elv)
            end

            kernel_parameters = return_kernel_parameters(Observations[j_obs], Model, kernel_coordinates)
            kernel_static = return_kernel_static(Observations[j_obs], Model)
            Kernels[j_obs] = ObservableKernel(Observations[j_obs], kernel_parameters, kernel_coordinates, kernel_weights, kernel_static)

            # Predictions and residuals
            predictions[j_obs], rel_res[j_obs] = evaluate_kernel(Kernels[j_obs])
            tt_sp[j_obs] = t_min
        end

        n += 1
        println("Finished initialisation point " * string(n) * " of " * string(npt) * " (",departure[2],").")
    end

    return predictions, rel_res, Kernels, tt_sp
end

function allocate_kernels_vector(Observations::Vector{B}, Model::PsiModel{P}) where {B <: Observable, P}
    # Allocate storage arrays
    n = length(Observations)
    val_type = eltype(Model.Mesh.x[1]) # Type assumed for model parameter and coordinate values
    obs_type = typeof(Observations[1].observation) # Type that stores an observation.......COULD FAIL FOR MULTIOBSERVABLES
    # Vector of kernels
    param_type = return_kernel_parameter_type(P, val_type)
    coord_type = Vector{NTuple{ndims(Model.Mesh), val_type}}
    weight_type = Vector{NamedTuple{(:dr, :azimuth, :elevation), NTuple{3, val_type}}} # Assuming fixed names for weights!
    KernelTypes = return_kernel_types(B, param_type, coord_type, weight_type, obs_type)
    Kernels = Vector{KernelTypes}(undef, n) # Vector{ObservableKernel{B, param_type, coord_type, weight_type, Vector{obs_type}}}(undef, n)

    return Kernels
end

function make_graph(Model::PsiModel)
    fs = Model.Methods.ShortestPath.forward_star
    leaf_size = Model.Methods.ShortestPath.leaf_size
    grid_noise = Model.Methods.ShortestPath.grid_noise
    iso_trace = Model.Methods.ShortestPath.iso_trace
    lon_inc, lat_inc, elv_inc = step(Model.Mesh.x[1]), step(Model.Mesh.x[2]), step(Model.Mesh.x[3])
    dlon, dlat, delv = grid_noise*lon_inc, grid_noise*lat_inc, grid_noise*elv_inc

    num_vertices = length(Model.Mesh)
    xg, yg, zg = zeros(num_vertices), zeros(num_vertices), zeros(num_vertices)
    n = 0
    for elv_k in Model.Mesh.x[3]
        for lat_j in Model.Mesh.x[2]
            for lon_i in Model.Mesh.x[1]
                n += 1
                # Define grid noise
                ddlon, ddlat, ddelv = (2.0 * rand() - 1.0) * dlon, (2.0 * rand() - 1.0) * dlat, (2.0 * rand() - 1.0) * delv
                # Compute noisey cartesian coordinates
                lon_ijk, lat_ijk, elv_ijk = lon_i + ddlon, lat_j + ddlat, elv_k + ddelv
                xg[n], yg[n], zg[n] = geographic_to_global(lon_ijk, lat_ijk, elv_ijk; radius = Model.Mesh.Geometry.R₀)
            end
        end
    end

    vert_weights = get_vertex_weights(xg, yg, zg, Model)
    if iso_trace
        vert_weights = isotropic_weights(vert_weights)
    end

    return SeismicDijkstra.StructuredGraph3D(xg, yg, zg, size(Model.Mesh), vert_weights; r_neighbours=fs, leafsize=leaf_size)
end
function get_vertex_weights(xq, yq, zq, Model::PsiModel{<:IsotropicVelocity})
    lon_1, dlon = Model.Mesh.x[1][1], step(Model.Mesh.x[1])
    lat_1, dlat = Model.Mesh.x[2][1], step(Model.Mesh.x[2])
    elv_1, delv = Model.Mesh.x[3][1], step(Model.Mesh.x[3])

    # Nearest neighbour interpolation
    # Grids are the same size but graph node positions can be perturbed
    nxyz, num_vertices = size(Model.Mesh), length(Model.Mesh)
    vp, vs = zeros(num_vertices), zeros(num_vertices)
    for ind in eachindex(xq) # Grid array NOT grid vector
        xq_i, yq_j, zq_k = xq[ind], yq[ind], zq[ind]
        lon_q, lat_q, elv_q = global_to_geographic(xq_i, yq_j, zq_k; radius = Model.Mesh.Geometry.R₀) # LOCAL GEO
        inn = min(max(1 + round(Int, (lon_q - lon_1)/dlon), 1), nxyz[1])
        jnn = min(max(1 + round(Int, (lat_q - lat_1)/dlat), 1), nxyz[2])
        knn = min(max(1 + round(Int, (elv_q - elv_1)/delv), 1), nxyz[3])

        vp[ind] = Model.Parameters.vp[inn, jnn, knn]
        vs[ind] = Model.Parameters.vs[inn, jnn, knn]
    end

    return SeismicDijkstra.IsotropicVelocity(vp, vs)
end
function get_vertex_weights(xq, yq, zq, Model::PsiModel{<:HexagonalVectoralVelocity})
    lon_1, dlon = Model.Mesh.x[1][1], step(Model.Mesh.x[1])
    lat_1, dlat = Model.Mesh.x[2][1], step(Model.Mesh.x[2])
    elv_1, delv = Model.Mesh.x[3][1], step(Model.Mesh.x[3])

    # Nearest neighbour interpolation
    # Grids are the same size but graph node positions can be perturbed
    nxyz, num_vertices = size(Model.Mesh), length(Model.Mesh)
    alpha, beta, azm, elv = zeros(num_vertices), zeros(num_vertices), zeros(num_vertices), zeros(num_vertices)
    epsilon, delta, gamma = zeros(num_vertices), zeros(num_vertices), zeros(num_vertices)
    for ind in eachindex(xq) # Grid array NOT grid vector
        xq_i, yq_j, zq_k = xq[ind], yq[ind], zq[ind]
        lon_q, lat_q, elv_q = global_to_geographic(xq_i, yq_j, zq_k; radius = Model.Mesh.Geometry.R₀)
        inn = min(max(1 + round(Int, (lon_q - lon_1)/dlon), 1), nxyz[1])
        jnn = min(max(1 + round(Int, (lat_q - lat_1)/dlat), 1), nxyz[2])
        knn = min(max(1 + round(Int, (elv_q - elv_1)/delv), 1), nxyz[3])

        rϵ = typeof(Model.Parameters.ratio_ϵ) <: Array ? Model.Parameters.ratio_ϵ[inn,jnn,knn] : Model.Parameters.ratio_ϵ
        rη = typeof(Model.Parameters.ratio_η) <: Array ? Model.Parameters.ratio_η[inn,jnn,knn] : Model.Parameters.ratio_η
        rγ = typeof(Model.Parameters.ratio_γ) <: Array ? Model.Parameters.ratio_γ[inn,jnn,knn] : Model.Parameters.ratio_γ

        f = Model.Parameters.f[inn,jnn,knn]
        epsilon[ind], delta[ind], gamma[ind] = f*rϵ, f*(rϵ - rη), f*rγ
        alpha[ind], beta[ind] = Model.Parameters.α[inn,jnn,knn], Model.Parameters.β[inn,jnn,knn]

        # Anisotropy orientation in ecef cartesian coordinates
        azm_local, elv_local = Model.Parameters.azimuth[inn,jnn,knn], Model.Parameters.elevation[inn,jnn,knn]
        sinλ, cosλ = sincos(azm_local)
        sinϕ, cosϕ = sincos(elv_local)
        s_local = cosϕ*cosλ, cosϕ*sinλ, sinϕ
        sx, sy, sz = local_to_global_vector(s_local, (lon_q, lat_q, elv_q), Model.Mesh.Geometry) # CHECK ME! LOCAL OR TRUE GEO?
        azm[ind], elv[ind] = atan(sy, sx), atan(sz, sqrt((sx^2) + (sy^2)))
    end
    
    return SeismicDijkstra.ThomsenVelocity(alpha, beta, epsilon, delta, gamma, azm, elv)
end
function isotropic_weights(vert_weights::SeismicDijkstra.IsotropicVelocity)
    return vert_weights
end
function isotropic_weights(vert_weights::SeismicDijkstra.ThomsenVelocity)
    vp, vs = zeros(length(vert_weights.alpha)), zeros(length(vert_weights.alpha))
    for i in eachindex(vert_weights.alpha)
        α_i, β_i = vert_weights.alpha[i], vert_weights.beta[i]
        ϵ_i, δ_i, γ_i = vert_weights.epsilon[i], vert_weights.delta[i], vert_weights.gamma[i]
        vp[i] = α_i*sqrt( 1.0 + (16/15)*ϵ_i + (4/15)*δ_i )
        vs[i] = β_i*sqrt( 1.0 + (2/3)*γ_i + (2/15)*((α_i^2)/(β_i^2))*(ϵ_i - δ_i) )
    end
    return SeismicDijkstra.IsotropicVelocity(vp, vs)
end

# Itinerary is a list of start points and destinations for ray tracing.
# The start points are defined by a unique (a, Phase) tuple where the "a" is
# either a source or receiver index and the Phase is either "P" or "S". The
# destinations for each start point is a vector of unique (k_obs, b) tuples where
# "b" is either a source or receiver index and "k_obs" is the corresponding index
# in the observation vector. The itinerary is a dictionary with the start point as
# the keys and the destinations as the values.
function build_itinerary(Observations, Sources, Receivers)
    # Phase identifying functions
    get_phase(Phase::CompressionalWave) = "P"
    get_phase(Phase::ShearWave) = "S"

    # Check Assumptions
    for (k, B_k) in enumerate(Observations)
        !isa(B_k.Forward, ForwardShortestPath) && error("Mixed forward methods not supported!")
        phs_k = uppercase(B_k.Phase.name)
        phs_k != "P" && phs_k != "S" && error("Only direct phases are supported!")
    end

    # Source initialization itinerary
    src_init = Dict{Tuple{Int, String}, Vector{NTuple{2, Int}}}()
    [src_init[(Sources.id[B_k.source_id], get_phase(B_k.Phase))] = Vector{NTuple{2, Int}}() for B_k in Observations]
    for (k, B_k) in enumerate(Observations)
        push!(src_init[(Sources.id[B_k.source_id], get_phase(B_k.Phase))], (k, Receivers.id[B_k.receiver_id]))
    end
    # Receiver initialization itinerary
    rcv_init = Dict{Tuple{Int, String}, Vector{NTuple{2, Int}}}()
    [rcv_init[(Receivers.id[B_k.receiver_id], get_phase(B_k.Phase))] = Vector{NTuple{2, Int}}() for B_k in Observations]
    for (k, B_k) in enumerate(Observations)
        push!(rcv_init[(Receivers.id[B_k.receiver_id], get_phase(B_k.Phase))], (k, Sources.id[B_k.source_id]))
    end
    # Return itinerary with fewest initialzation points
    itinerary, tf_reverse = length(rcv_init) < length(src_init) ? (rcv_init, true) : (src_init, false)

    return itinerary, tf_reverse
end