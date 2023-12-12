
function write_model(output_file::String, Model::PsiModel{<:IsotropicVelocity}; vp_ref = nothing, vs_ref = nothing)
    # Define global coordinates
    xg, yg, zg = global_coordinate_arrays(Model.Mesh)
    # Get velocity fields
    vp, vs = return_velocity_fields(Model.Parameters)
    # Write fractional velocity anomalies instead of absolute velocities
    if ~isnothing(vp_ref)
        vp = vp .- vp_ref
        vp ./= vp_ref
    end
    if ~isnothing(vs_ref)
        vs = vs .- vs_ref
        vs ./= vs_ref
    end

    return write_model(output_file, xg, yg, zg, vp, vs, zeros(size(vp)), zeros(size(vp)), zeros(size(vp)), 0.0, 0.0, 0.0)
end

function write_model(output_file::String, Model::PsiModel{<:HexagonalVectoralVelocity}; tf_isotropic = true, vp_ref = nothing, vs_ref = nothing)
    # Define global coordinates
    xg, yg, zg = global_coordinate_arrays(Model.Mesh)
    # Define the global anisotropic vector
    sx, sy, sz = global_anisotropic_vector(Model)
    # Return isotropic or Thomsen velocities
    if tf_isotropic
        vp, vs = return_isotropic_velocities(Model.Parameters)
    else
        vp, vs = return_velocity_fields(Model.Parameters)
    end
    # Write fractional velocity anomalies instead of absolute velocities
    if ~isnothing(vp_ref)
        vp = vp .- vp_ref
        vp ./= vp_ref
    end
    if ~isnothing(vs_ref)
        vs = vs .- vs_ref
        vs ./= vs_ref
    end

    return write_model(output_file, xg, yg, zg, vp, vs, sx, sy, sz,
    Model.Parameters.ratio_ϵ, Model.Parameters.ratio_η, Model.Parameters.ratio_γ)
end
function write_model(output_file, xg, yg, zg, vp, vs, sx, sy, sz, ratio_ϵ, ratio_η, ratio_γ)
    # Write the VTK file
    vtk_grid(output_file, xg, yg, zg) do vtk
        vtk["Vp"] = vp
        vtk["Vs"] = vs
        vtk["AnisotropicVector"] = (sx, sy, sz)
        vtk["epsilon_ratio"] = ratio_ϵ
        vtk["eta_ratio"] = ratio_η
        vtk["gamma_ratio"] = ratio_γ
    end

    return nothing
end

function return_isotropic_velocities(Parameters::HexagonalVectoralVelocity)
    vp = zeros(size(Parameters.f))
    vs = zeros(size(Parameters.f))
    for i in eachindex(Parameters.f)
        vp[i], vs[i] = return_isotropic_velocities(Parameters, i)
    end

    return vp, vs
end

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


# NEED TO BE CHECKED

# Write Model
function write_psi_structure(out_file, Parameters, Mesh::RegularGrid)
    fid = open(out_file, "w")
    print_header(fid, Mesh)
    for (k, zk) in enumerate(Mesh.x[3])
        for (j, yj) in enumerate(Mesh.x[2])
            for (i, xi) in enumerate(Mesh.x[1])
                print_parameters(fid, (xi, yj, zk), Parameters)
            end
        end
    end
    close(fid)

    return nothing
end
function print_header(fid, Mesh)
    print_geometry(fid, Mesh.Geometry)
    print_mesh(fid, Mesh)
    return nothing
end
function print_geometry(fid, Geometry::LocalGeographic)
    println(fid, Geometry.R₀,", ",Geometry.λ₀,", ",Geometry.ϕ₀,", ",Geometry.β)
    return nothing
end
function print_mesh(fid, Mesh::RegularGrid)
    n1, n2, n3 = size(Mesh)
    println(fid, n1,", ",n2,", ",n3)
    println(fid, Mesh.x[1][1],", ",Mesh.x[2][1],", ",Mesh.x[3][1],", ",Mesh.x[1][end],", ",Mesh.x[2][end],", ",Mesh.x[3][end])
    return nothing
end
function print_parameters(fid, n, coords, Parameters::HexagonalVectoralVelocity)
    α, β, ϵ, δ, γ = return_thomsen_parameters(Parameters, n)
    println(fid, coords[1],", ",coords[2],", ",coords[3],", ",α,", ",β,", ",ϵ,", ",δ,", ",γ,", ",
    Parameters.azimuth[n],", ",Parameters.elevation[n])
    return nothing
end
# Write Observations
function write_psi_structure(out_file, Observations::Vector{<:Observable})
    fid = open(out_file, "w")
    for obs in Observations
        print_observation(fid, obs)
    end
    close(fid)

    return nothing
end
function write_psi_structure(out_file, Observations::Vector{<:Observable}, predictions)
    fid = open(out_file, "w")
    for (i, obs) in enumerate(Observations)
        print_observation(fid, obs; observation = predictions[i])
    end
    close(fid)

    return nothing
end
function print_observation(fid, SP::SplittingParameters; observation = SP.observation)
    println(fid, observation[1],", ",observation[2],", ",SP.error[1],", ",SP.error[2],", ", 
    SP.Phase.period,", ",SP.Phase.name,", ",SP.source_id,", ",SP.receiver_id,", ","Q",", ",SP.Phase.paz)

    return nothing
end