
function write_coordinates_to_vtk(output_file::String, Data)
    n = length(Data.id)
    xg, yg, zg, v = Vector{Float64}(undef, n), Vector{Float64}(undef, n), Vector{Float64}(undef, n), Vector{Float64}(undef, n)
    for k in eachindex(Data.coordinates)
        lon, lat, elv = Data.coordinates[k]
        xg[k], yg[k], zg[k] = geographic_to_global(lon, lat, elv; radius = 6371.0)
        v[k] = elv
    end
    cells = [MeshCell(VTKCellTypes.VTK_VERTEX, (i, )) for i = 1:n]
    vtk_grid(output_file, xg, yg, zg, cells) do vtk
        vtk["elevation"] = v
    end

    return nothing
end
# Write PsiModel{<:IsotropicVelocity} to VTK-file 
function write_model_to_vtk(output_file::String, Model::PsiModel{<:IsotropicVelocity}; vp_ref = nothing, vs_ref = nothing)
    # Define global coordinates
    xg, yg, zg = global_coordinate_arrays(Model.Mesh)
    # Write VTK file
    vtk_grid(output_file, xg, yg, zg) do vtk
        !isnothing(vp_ref) ? vtk["Vp_ref"] = vp_ref : nothing
        !isnothing(vs_ref) ? vtk["Vs_ref"] = vs_ref : nothing
        vtk["Vp"] = Model.Parameters.vp
        vtk["Vs"] = Model.Parameters.vs
    end

    return nothing
end
# Write PsiModel{<:HexagonalVectoralVelocity} to VTK-file 
function write_model_to_vtk(output_file::String, Model::PsiModel{<:HexagonalVectoralVelocity};
    vp_ref = nothing, vs_ref = nothing, tf_isotropic_velocities = true)
    # Define global 3D coordinate arrays
    xg, yg, zg = global_coordinate_arrays(Model.Mesh)
    # Define velocities to write
    vp, vs = tf_isotropic_velocities ? return_isotropic_velocities(Model.Parameters) : return_velocity_fields(Model.Parameters)
    # Define the global anisotropic vector
    sx, sy, sz = global_anisotropic_vector(Model)
    # Write VTK file
    vtk_grid(output_file, xg, yg, zg) do vtk
        !isnothing(vp_ref) ? vtk["Vp_ref"] = vp_ref : nothing
        !isnothing(vs_ref) ? vtk["Vs_ref"] = vs_ref : nothing
        vtk["Vp"] = vp
        vtk["Vs"] = vs
        vtk["AniVec"] = (sx, sy, sz)
        vtk["epsilon_ratio"] = Model.Parameters.ratio_ϵ
        vtk["eta_ratio"] = Model.Parameters.ratio_η
        vtk["gamma_ratio"] = Model.Parameters.ratio_γ
    end

    return nothing
end
function write_sampling_to_vtk(output_directory::String, PerturbationModel::SeismicPerturbationModel)
    if !isnothing(PerturbationModel.Interface)
        write_sampling_to_vtk(output_directory, PerturbationModel.Interface)
    end
    if !isnothing(PerturbationModel.Velocity)
        write_sampling_to_vtk(output_directory, PerturbationModel.Velocity)
    end

    return nothing
end
function write_sampling_to_vtk(output_directory::String, Velocity::InverseSeismicVelocity)
    if !isnothing(Velocity.Isotropic)
        write_sampling_to_vtk(output_directory, Velocity.Isotropic)
    end
    if !isnothing(Velocity.Anisotropic)
        write_sampling_to_vtk(output_directory, Velocity.Anisotropic)
    end

    return nothing
end
function write_sampling_to_vtk(output_directory::String, Isotropic::InverseIsotropicSlowness)
    if !isnothing(Isotropic.Up)
        xg, yg, zg = global_coordinate_arrays(Isotropic.Up.Mesh)
        vtk_grid(output_directory*"/RSJS_InverseIsotropicSlowness_Up", xg, yg, zg) do vtk
            vtk["RSJS"] = Isotropic.Up.RSJS
        end
    end
    if !isnothing(Isotropic.Us)
        xg, yg, zg = global_coordinate_arrays(Isotropic.Us.Mesh)
        vtk_grid(output_directory*"/RSJS_InverseIsotropicSlowness_Us", xg, yg, zg) do vtk
            vtk["RSJS"] = Isotropic.Us.RSJS
        end
    end

    return nothing
end
function write_sampling_to_vtk(output_directory::String, Anisotropic::InverseAnisotropicVector)
    if !isnothing(Anisotropic.Fractions)
        write_sampling_to_vtk(output_directory, Anisotropic.Fractions)
    end
    if !isnothing(Anisotropic.Orientations)
        write_sampling_to_vtk(output_directory, Anisotropic.Orientations)
    end

    return nothing
end
function write_sampling_to_vtk(output_directory::String, Orientations::InverseAzRadVector)
    xg, yg, zg = global_coordinate_arrays(Orientations.A.Mesh)
    vtk_grid(output_directory*"/RSJS_InverseAzRadVector_A", xg, yg, zg) do vtk
        vtk["RSJS"] = Orientations.A.RSJS
    end

    xg, yg, zg = global_coordinate_arrays(Orientations.B.Mesh)
    vtk_grid(output_directory*"/RSJS_InverseAzRadVector_B", xg, yg, zg) do vtk
        vtk["RSJS"] = Orientations.B.RSJS
    end

    xg, yg, zg = global_coordinate_arrays(Orientations.C.Mesh)
    vtk_grid(output_directory*"/RSJS_InverseAzRadVector_C", xg, yg, zg) do vtk
        vtk["RSJS"] = Orientations.C.RSJS
    end

    return nothing
end
# Return global cartesian coordinate arrays
function global_coordinate_arrays(Mesh)
    xg = zeros(size(Mesh))
    yg = zeros(size(Mesh))
    zg = zeros(size(Mesh))
    for (k, x3) in enumerate(Mesh.x[3])
        for (j, x2) in enumerate(Mesh.x[2])
            for (i, x1) in enumerate(Mesh.x[1])
                xg[i,j,k], yg[i,j,k], zg[i,j,k] = local_to_global(x1, x2, x3, Mesh.Geometry)
            end
        end
    end

    return xg, yg, zg
end
# Return anisotropic vector in the global cartesian coordinate system
function global_anisotropic_vector(Model::PsiModel{<:HexagonalVectoralVelocity})
    sx = zeros(size(Model.Mesh))
    sy = zeros(size(Model.Mesh))
    sz = zeros(size(Model.Mesh))
    for (k, x3) in enumerate(Model.Mesh.x[3])
        for (j, x2) in enumerate(Model.Mesh.x[2])
            for (i, x1) in enumerate(Model.Mesh.x[1])
                lon_lat_elv = local_to_geographic(x1, x2, x3, Model.Mesh.Geometry)
                sinϕ, cosϕ = sincos(Model.Parameters.azimuth[i,j,k])
                sinθ, cosθ = sincos(Model.Parameters.elevation[i,j,k])
                v = (Model.Parameters.f[i,j,k]*cosθ*cosϕ, Model.Parameters.f[i,j,k]*cosθ*sinϕ, Model.Parameters.f[i,j,k]*sinθ)
                sx[i,j,k], sy[i,j,k], sz[i,j,k] = local_to_global_vector(v, lon_lat_elv, Model.Mesh.Geometry)
            end
        end
    end

    return sx, sy, sz
end



# Write PsiModel structures to ascii-files
function write_psi_model(output_directory::String, Model::PsiModel;
    tf_write_sources = true, tf_write_receivers = true, tf_write_parameters = true)

    tf_write_sources ? write_psi_structure(output_directory*"/Sources.dat", Model.Sources) : nothing
    tf_write_receivers ? write_psi_structure(output_directory*"/Receivers.dat", Model.Receivers) : nothing
    tf_write_parameters ? write_psi_structure(output_directory*"/Parameters.dat", Model.Parameters, Model.Mesh) : nothing
    return nothing
end

# Write SeismicSources or SeismicReceivers to ascii data file
function write_psi_structure(out_file, S::Union{<:SeismicSources, <:SeismicReceivers})
    fid = open(out_file, "w")
    for id in eachindex(S.id)
        k = S.id[id]
        println(fid, id,", ",S.coordinates[k][1],", ",S.coordinates[k][2],", ",S.coordinates[k][3])
    end
    close(fid)

    # Write Statics
    if !isempty(S.statics)
        static_file = splitdir(out_file)
        static_file = static_file[1]*"/Statics_"*static_file[2]
        sid = open(static_file, "w")
        for (k, v) in S.statics
            println(sid, k[1],", ",k[2],", ",k[3],", ", v)
        end
        close(sid)
    end

    return nothing
end
# Write model parameters to ascii data file
function write_psi_structure(out_file, Parameters::ModelParameterization, Mesh::RegularGrid)
    fid = open(out_file, "w")
    # Write header information
    print_geometry(fid, Mesh.Geometry)
    print_mesh(fid, Mesh)
    print_header(fid, Parameters)
    n = 0
    for (k, zk) in enumerate(Mesh.x[3])
        for (j, yj) in enumerate(Mesh.x[2])
            for (i, xi) in enumerate(Mesh.x[1])
                n += 1
                print_parameters(fid, n, (xi, yj, zk), Parameters)
            end
        end
    end

    close(fid)
    return nothing
end

# Print LocalGeographic geometry data
function print_geometry(fid, Geometry::LocalGeographic)
    println(fid, Geometry.R₀,", ",Geometry.λ₀,", ",Geometry.ϕ₀,", ",Geometry.β)
    return nothing
end
# Print RegularGrid data
function print_mesh(fid, Mesh::RegularGrid)
    n1, n2, n3 = size(Mesh)
    xminmax, yminmax, zminmax = (extrema(Mesh.x[1]), extrema(Mesh.x[2]), extrema(Mesh.x[3]))
    Δx = (xminmax[2] - xminmax[1], yminmax[2] - yminmax[1], zminmax[2] - zminmax[1])
    println(fid, n1,", ",n2,", ",n3)
    println(fid, 0.5*Δx[1],", ",0.5*Δx[2],", ",Δx[3],", ",zminmax[2]) # Also print start depth for vertically extended models
    return nothing
end
# Print any necessary parameter headers
function print_header(fid, Parameters)
    return nothing
end
# Print header information for HexagonalVectoralVelocity parameter data file
function print_header(fid, Parameters::HexagonalVectoralVelocity)
    if length(Parameters.ratio_ϵ) == 1
        println(fid, Parameters.tf_exact,", ",Parameters.ratio_ϵ[1],", ",Parameters.ratio_η[1],", ",Parameters.ratio_γ[1])
    else
        println(fid, Parameters.tf_exact)
    end
    return nothing
end
# Print header information for ElasticVoigt parameter data file
function print_header(fid, Parameters::ElasticVoigt)
    if size(Parameters.ρ) == size(Parameters.c11)
        println(fid, false)
    else
        println(fid, true)
    end
    return nothing
end
# Print IsotropicVelocity parameters
function print_parameters(fid, n, coords, Parameters::IsotropicVelocity)
    println(fid, coords[1],", ",coords[2],", ",coords[3],", ",Parameters.vp[n],", ",Parameters.vs[n])
    return nothing
end
# Print HexagonalVectoralVelocity parameters
function print_parameters(fid, n, coords, Parameters::HexagonalVectoralVelocity)
    α, β, _ = return_thomsen_parameters(Parameters, n)
    if length(Parameters.ratio_ϵ) == 1
        println(fid, coords[1],", ",coords[2],", ",coords[3],", ",α,", ",β,", ",
        Parameters.f[n],", ",Parameters.azimuth[n],", ",Parameters.elevation[n])
    else
        println(fid, coords[1],", ",coords[2],", ",coords[3],", ",α,", ",β,", ",
        Parameters.f[n],", ",Parameters.azimuth[n],", ",Parameters.elevation[n],", ",
        Parameters.ratio_ϵ[n],", ",Parameters.ratio_η[n],", ",Parameters.ratio_γ[n])
    end

    return nothing
end
# Print ElasticVoigt parameters
function print_parameters(fid, n, coords, Parameters::ElasticVoigt)
    n == 1 ? (@warn "Incomplete function: ElasticVoigt parameters need to be rotated back to global cartesian!") : nothing
    a = size(Parameters.ρ) == size(Parameters.c11) ? 1.0e-9 : 1.0
    println(fid, coords[1],", ",coords[2],", ",coords[3],", ",
    a*Parameters.c11[n],", ",a*Parameters.c12[n],", ",a*Parameters.c13[n],", ",a*Parameters.c14[n],", ",a*Parameters.c15[n],", ",
    a*Parameters.c16[n],", ",a*Parameters.c22[n],", ",a*Parameters.c23[n],", ",a*Parameters.c24[n],", ",a*Parameters.c25[n],", ",
    a*Parameters.c26[n],", ",a*Parameters.c33[n],", ",a*Parameters.c34[n],", ",a*Parameters.c35[n],", ",a*Parameters.c36[n],", ",
    a*Parameters.c44[n],", ",a*Parameters.c45[n],", ",a*Parameters.c46[n],", ",a*Parameters.c55[n],", ",a*Parameters.c56[n],", ",
    a*Parameters.c66[n],", ",Parameters.ρ[n])
    return nothing
end



# Write Observations
function write_observations(out_dir, Observations::Vector{<:Observable}; alt_data = nothing, prepend = "")
    # Generate a dictionary of files to which to write the observations
    OutFiles = Dict{NTuple{2, String}, IOStream}()
    SubType = eltype(Observations)
    while isa(SubType, Union)
        # Identify concrete and Union types (not clear in which field they will be stored)
        aType, SubType = isa(SubType.b, Union) ? (SubType.a, SubType.b) : (SubType.b, SubType.a)
        # Extract Observable and Phase types
        s = split(string(aType), "{")
        a_phs = split(s[2], ",")
        a_obs, a_phs = (s[1], a_phs[1])
        # Store dictionary entry for observation file
        OutFiles[(a_obs, a_phs)] = open(out_dir*"/"*prepend*"_"*a_obs*"_"*a_phs*".dat", "w")
    end
    # Last entry
    s = split(string(SubType), "{")
    a_phs = split(s[2], ",")
    a_obs, a_phs = (s[1], a_phs[1])
    OutFiles[(a_obs, a_phs)] = open(out_dir*"/"*prepend*"_"*a_obs*"_"*a_phs*".dat", "w")

    # Write observations to their respective files
    for (i, B) in enumerate(Observations)
        # Extract Observable and Phase types
        s = split(string(B), "{")
        a_phs = split(s[2], ",")
        a_obs, a_phs = (s[1], a_phs[1])
        # Print observation to file
        isnothing(alt_data) ? print_observation(OutFiles[(a_obs, a_phs)], B) : print_observation(OutFiles[(a_obs, a_phs)], B; observation = alt_data[i])
    end

    # Close all the Observable files
    for k in eachindex(OutFiles)
        close(OutFiles[k])
    end

    return nothing
end
function print_observation(fid, B::SeismicObservable; observation = B.observation)
    if typeof(B.Phase) <: ShearWave
        println(fid, observation,", ",B.error,", ",B.Phase.period,", ",B.Phase.name,", ",
        B.source_id,", ",B.receiver_id,", ","???",", ",B.Phase.paz)
    else
        println(fid, observation,", ",B.error,", ",B.Phase.period,", ",B.Phase.name,", ",
        B.source_id,", ",B.receiver_id,", ","???")
    end

    return nothing
end
function print_observation(fid, SP::SplittingParameters; observation = SP.observation)
    println(fid, observation[1],", ",observation[2],", ",SP.error[1],", ",SP.error[2],", ", 
    SP.Phase.period,", ",SP.Phase.name,", ",SP.source_id,", ",SP.receiver_id,", ","Q",", ",SP.Phase.paz)

    return nothing
end