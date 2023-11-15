function write_model(Model::PsiModel{<:HexagonalVectoralVelocity}, output_file::String)
    # Define global coordinates
    xg, yg, zg = global_coordinate_arrays(Model.Mesh)
    # Compute isotropic velocities
    vp, vs = return_isotropic_velocities(Model.Parameters)
    # Define the global anisotropic vector
    sx, sy, sz = global_anisotropic_vector(Model)
    # Write the VTK file
    vtk_grid(output_file, xg, yg, zg) do vtk
        vtk["Vp"] = vp
        vtk["Vs"] = vs
        vtk["AnisotropicVector"] = (sx, sy, sz)
        vtk["epsilon_ratio"] = Model.Parameters.ratio_ϵ
        vtk["eta_ratio"] = Model.Parameters.ratio_η
        vtk["gamma_ratio"] = Model.Parameters.ratio_γ
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