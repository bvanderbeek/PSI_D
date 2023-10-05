# INTERPOLATION ROUTINES

# # Update this
# interpolate_model!(rₑ, K.m, M.r, M.v; vp_interp = true, vs_interp = false)
# # Then we can delete this
# linear_interpolation_1d


## MODEL INTERPOLATIONS ##
# Interpolates an IsotropicVelocity ModelTauP to a ParameterField
function interpolate_model!(ΔM::ParameterField, Model::ModelTauP{<:SeismicVelocity}, field::Symbol;
    tf_extrapolate = true, tf_harmonic = true)
    # Create views into the correct field
    if field == :up
        @views v = Model.vp
        @views dv = Model.m.vp
    elseif field == :us
        @views v = Model.vs
        @views dv = Model.m.vs
    else
        error("IsotropicSlowness has no field "*string(field)*".")
    end

    # Interpolate
    for (k, zq) in enumerate(ΔM.Mesh.x[3])
        # Radial TauP depth
        rq = Model.Rₜₚ - ΔM.Mesh.CS.R₀ - zq
        # Get reference 1D velocity
        vq = piecewise_linearly_interpolate(Model.r, Model.nd, v, rq;
        tf_extrapolate = tf_extrapolate, tf_harmonic = tf_harmonic)
        # Loop over remaining grid dimensions
        for (j, yq) in enumerate(ΔM.Mesh.x[2]), (i, xq) in enumerate(ΔM.Mesh.x[1])
            # Interpolate any velocity anomalies to the query point. Cannot use harmonic interpolation
            # as these velocity anomalies can be zero-valued.
            dvq = linearly_interpolate(Model.Mesh.x[1], Model.Mesh.x[2], Model.Mesh.x[3], dv, xq, yq, zq;
            tf_extrapolate = tf_extrapolate, tf_harmonic = false)
            # Store P-wave slowness
            ΔM.m₀[i,j,k] = 1.0/(vq + dvq)
        end
    end

    return nothing
end

# Interpolates an IsotropicVelocity ModelTauP to an ObservableKernel for P-waves
function interpolate_model!(::CompressionalWave, K::ObservableKernel{P}, Model::ModelTauP{M}; tf_extrapolate = false, tf_harmonic = false) where
    {P<:IsotropicVelocity, M<:IsotropicVelocity}
    nq = size(K.w,1) # <-- NEW Loop over number of weights NOT coordinates (ff kernels have extra coordinate)
    for i in 1:nq # <-- NEW
        xq = K.x[i] # <-- NEW
    # for (i, xq) in enumerate(K.x) # <-- OLD
        # Map kernel coordinates to local coordinates
        qx, qy, qz = global_to_local(xq[1], xq[2], xq[3], Model.Mesh.CS)
        # Interpolate velocity perturbations to kernel position
        dm = linearly_interpolate(Model.Mesh.x[1], Model.Mesh.x[2], Model.Mesh.x[3], Model.m.vp, qx, qy, qz;
        tf_extrapolate = tf_extrapolate, tf_harmonic = tf_harmonic)
        # Add perturbation to kernel
        K.m.vp[i] += dm
    end

    return nothing
end

# Interpolates an IsotropicVelocity ModelTauP to an ObservableKernel for S-waves
function interpolate_model!(::ShearWave, K::ObservableKernel{P}, Model::ModelTauP{M}; tf_extrapolate = false, tf_harmonic = false) where
    {P<:IsotropicVelocity, M<:IsotropicVelocity}
    nq = size(K.w,1) # <-- NEW Loop over number of weights NOT coordinates (ff kernels have extra coordinate)
    for i in 1:nq # <-- NEW
        xq = K.x[i] # <-- NEW
    # for (i, xq) in enumerate(K.x) # <-- OLD
        # Map kernel coordinates to local coordinates
        qx, qy, qz = global_to_local(xq[1], xq[2], xq[3], Model.Mesh.CS)
        # Interpolate velocity perturbations to kernel position
        dm = linearly_interpolate(Model.Mesh.x[1], Model.Mesh.x[2], Model.Mesh.x[3], Model.m.vs, qx, qy, qz;
        tf_extrapolate = tf_extrapolate, tf_harmonic = tf_harmonic)
        # Add perturbation to kernel
        K.m.vs[i] += dm
    end

    return nothing
end

# Interpolates an HexagonalVelocity ModelTauP to an ObservableKernel for P- or S-waves
function interpolate_model!(::W, K::ObservableKernel{Pk}, Model::ModelTauP{Pm}; tf_extrapolate = false) where
    {W<:SeismicPhase, Pk<:HexagonalVelocity, Pm<:HexagonalVelocity}

    # Create views such that the interpolation is general for both P- and S-waves
    if W <: CompressionalWave
        @views v = Model.m.vp
        @views f = Model.m.fp
        @views vq = K.m.vp
        @views fq = K.m.fp
    elseif W <: ShearWave
        @views v = Model.m.vs
        @views f = Model.m.fs
        @views vq = K.m.vs
        @views fq = K.m.fs
    else
        error("Unrecognized SeismicPhase, "*string(W)*".")
    end
    # If 4-theta terms are present, model is hexagonal. Otherwise, model is elliptical.
    tf_hexagonal = ~isnothing(Model.m.gs)

    nq = size(K.w,1) # <-- NEW Loop over number of weights NOT coordinates (ff kernels have extra coordinate)
    for i in 1:nq # <-- NEW
        xq = K.x[i] # <-- NEW
    # for (i, xq) in enumerate(K.x) # <-- OLD
        # Map kernel coordinates to local coordinates
        qx, qy, qz = global_to_local(xq[1], xq[2], xq[3], Model.Mesh.CS)
        # Get interpolation weights
        wind, wval = trilinear_weights(Model.Mesh.x[1], Model.Mesh.x[2], Model.Mesh.x[3], qx, qy, qz;
        tf_extrapolate = tf_extrapolate, scale = 1.0)
        # Interpolate the unit-magnitude a, b, c vectoral anisotropy parameters
        aⱼ = 0.0
        bⱼ = 0.0
        cⱼ = 0.0
        for (n, j) in enumerate(wind)
            # Get symmetry axis signs and cosines
            sin2ϕ, cos2ϕ = sincos(2.0*Model.m.ϕ[j])
            sinθ, cosθ = sincos(Model.m.θ[j])
            # Weighted sum of anisotropic components
            aⱼ += wval[n]*(cosθ^2)*cos2ϕ
            bⱼ += wval[n]*(cosθ^2)*sin2ϕ
            cⱼ += wval[n]*sinθ
            # Weighted sum of velocity and anisotorpic strength
            vq[i] += wval[n]*v[j]
            fq[i] += wval[n]*f[j]
            # Interpolate the 4-theta component and store in gs-field
            if tf_hexagonal
                g = return_anisotropic_4mag(W, Model, j)
                K.m.gs[i] += wval[n]*g
            end
        end
        # Recover orientation
        K.m.ϕ[i] = 0.5*atan(bⱼ, aⱼ)
        K.m.θ[i] = atan(cⱼ, sqrt(sqrt((aⱼ^2) + (bⱼ^2))))
    end

    return nothing
end
# # Interpolates an HexagonalVelocity ModelTauP to an ObservableKernel for P- or S-waves
# function interpolate_model!(::W, K::ObservableKernel{Pk}, Model::ModelTauP{Pm}; tf_extrapolate = false) where
#     {W<:SeismicPhase, Pk<:HexagonalVelocity, Pm<:HexagonalVelocity}

#     # Create views such that the interpolation is general for both P- and S-waves
#     if W <: CompressionalWave
#         @views v = Model.m.vp
#         @views f = Model.m.fp
#         @views vq = K.m.vp
#         @views fq = K.m.fp
#     elseif W <: ShearWave
#         @views v = Model.m.vs
#         @views f = Model.m.fs
#         @views vq = K.m.vs
#         @views fq = K.m.fs
#     else
#         error("Unrecognized SeismicPhase, "*string(W)*".")
#     end
#     # If 4-theta terms are present, model is hexagonal. Otherwise, model is elliptical.
#     tf_hexagonal = ~isnothing(Model.m.gs)

#     for (i, xq) in enumerate(K.x)
#         # Map kernel coordinates to local coordinates
#         qx, qy, qz = global_to_local(xq[1], xq[2], xq[3], Model.Mesh.CS)
#         # Get nearest neighbor
#         iw, jw, kw = nearest_neighbor(Model.Mesh.x[1], Model.Mesh.x[2], Model.Mesh.x[3], qx, qy, qz)
#         vq[i] += v[iw,jw,kw] # Adding because ModelTauP stores velocity perturbations NOT absolute values
#         fq[i] = f[iw,jw,kw]
#         if tf_hexagonal
#             ind = subscripts_to_index(size(Model.Mesh), (iw, jw, kw))
#             g = return_anisotropic_4mag(W, Model, ind)
#             K.m.gs[i] = g
#         end
#         K.m.ϕ[i] = Model.m.ϕ[iw,jw,kw]
#         K.m.θ[i] = Model.m.θ[iw,jw,kw]
#     end

#     return nothing
# end
function return_anisotropic_4mag(::Type{<:CompressionalWave}, M::ModelTauP{<:HexagonalVelocity}, i)
    # Convert linear index into 3rd-dimension subscript
    dims = size(Model.Mesh)
    k = 1 + floor(Int, (i - 1)/(dims[1]*dims[2]))
    # The TauP radial depth
    rq = Model.Rₜₚ - Model.Mesh.CS.R₀ - Model.Mesh.x[3][k]
    # Interpolate P- and S-velocities from TauP model
    vp = piecewise_linearly_interpolate(Model.r, Model.nd, Model.vp, rq; tf_extrapolate = true, tf_harmonic = true)
    vs = piecewise_linearly_interpolate(Model.r, Model.nd, Model.vs, rq; tf_extrapolate = true, tf_harmonic = true)
    # Add perturbations to the reference model velocities
    vp += Model.m.vp[i]
    vs += Model.m.vs[i]
    # Get anisotropic parameters
    fp = Model.m.fp[i]
    fs = Model.m.fs[i]
    gs = Model.m.gs[i]
    # Compute the P-wave 4α-component
    gp = 0.5*(sqrt((1.0 + 2.0*fp + (fp^2)) - 4.0*((vs/vp)^2)*((1.0 + fs)^2)*(1.0 - (1.0/(1.0 + gs)))) - (1.0 + fp))

    return gp
end
function return_anisotropic_4mag(::Type{<:ShearWave}, M::ModelTauP{<:HexagonalVelocity}, i)
    return Model.m.gs[i]
end


function interpolate_model!(Fq::SeismicSlowness, G::RegularGrid{T, 3, R}, F::SeismicVelocity, Q::RegularGrid{T, 3, R}; tf_extrapolate = false, tf_harmonic = false) where {T, R}
    if length(Fq.up) > 0
        for (i, qx₁) in enumerate(Q.x[1]), (j, qx₂) in enumerate(Q.x[2]), (k, qx₃) in enumerate(Q.x[3])
            v = linearly_interpolate(G.x[1], G.x[2], G.x[3], F.vp, qx₁, qx₂, qx₃; tf_extrapolate = tf_extrapolate, tf_harmonic = tf_harmonic)
            Fq.up[i, j, k] = 1.0/v
        end
    end
    if length(Fq.us) > 0
        for (i, qx₁) in enumerate(Q.x[1]), (j, qx₂) in enumerate(Q.x[2]), (k, qx₃) in enumerate(Q.x[3])
            v = linearly_interpolate(G.x[1], G.x[2], G.x[3], F.vs, qx₁, qx₂, qx₃; tf_extrapolate = tf_extrapolate, tf_harmonic = tf_harmonic)
            Fq.us[i, j, k] = 1.0/v
        end
    end

    return nothing
end

function interpolate_model!(Fq::SeismicVelocity, G::RegularGrid{T, 3, R}, F::SeismicVelocity, qx₁, qx₂, qx₃; tf_extrapolate = false, tf_harmonic = false) where {T, R}
    if length(Fq.vp) > 0
        linearly_interpolate!(Fq.vp, G.x[1], G.x[2], G.x[3], F.vp, qx₁, qx₂, qx₃; tf_extrapolate = tf_extrapolate, tf_harmonic = tf_harmonic)
    end
    if length(Fq.vs) > 0
        linearly_interpolate!(Fq.vs, G.x[1], G.x[2], G.x[3], F.vs, qx₁, qx₂, qx₃; tf_extrapolate = tf_extrapolate, tf_harmonic = tf_harmonic)
    end

    return nothing
end



## 3D TRILINEAR INTERPOLATION -- REGULAR GRIDS ##
function trilinear_weights(x::AbstractRange, y::AbstractRange, z::AbstractRange, qx::Number, qy::Number, qz::Number; tf_extrapolate = false, scale = 1.0)
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
        # Forces NaN weights if outside model domain
        scale = NaN
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
function linearly_interpolate(x::AbstractRange, y::AbstractRange, z::AbstractRange, v, qx::Number, qy::Number, qz::Number; tf_extrapolate = false, tf_harmonic = false)
    # Get linear interpolation weights for query point
    wind, wval = trilinear_weights(x, y, z, qx, qy, qz; tf_extrapolate = tf_extrapolate, scale = 1.0)
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
function linearly_interpolate!(qv, x::AbstractRange, y::AbstractRange, z::AbstractRange, v, qx, qy, qz; tf_extrapolate = false, tf_harmonic = false)
    # Point-wise linear interpolation
    for i in eachindex(qx)
        qv[i] = linearly_interpolate(x, y, z, v, qx[i], qy[i], qz[i]; tf_extrapolate = tf_extrapolate, tf_harmonic = tf_harmonic)
    end

    return nothing
end
function linearly_interpolate(x::AbstractRange, y::AbstractRange, z::AbstractRange, v, qx, qy, qz; tf_extrapolate = false, tf_harmonic = false)
    qv = similar(qx, eltype(v), size(qx))
    linearly_interpolate!(qv, x, y, z, v, qx, qy, qz; tf_extrapolate = tf_extrapolate, tf_harmonic = tf_harmonic)

    return qv
end

## 3D NEAREST NEIGHBOR -- REGULAR GRIDS ##
function nearest_neighbor(x::AbstractRange, y::AbstractRange, z::AbstractRange, qx, qy, qz)
    i = 1 + round(Int, (qx - x[1])/step(x))
    j = 1 + round(Int, (qy - y[1])/step(y))
    k = 1 + round(Int, (qz - z[1])/step(z))
    i = min(max(i, 1), length(x))
    j = min(max(j, 1), length(y))
    k = min(max(k, 1), length(z))

    return i, j, k
end



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
function linearly_interpolate(x::AbstractRange, v, qx; tf_extrapolate = false, tf_harmonic = false)
    # Get linear interpolation weights for query point
    wind, wval = linear_weights(x, qx[i]; tf_extrapolate = tf_extrapolate, scale = 1.0)
    # Return interpolated value as a weighted arithmetic (harmonic) average
    if tf_harmonic
        qv = 1.0/((wval[1]/v[wind[1]]) + (wval[2]/v[wind[2]]))
    else
        qv = (wval[1]*v[wind[1]]) + (wval[2]*v[wind[2]])
    end

    return qv
end
# Array of query points interpolation
function linearly_interpolate!(qv, x::AbstractRange, v, qx; tf_extrapolate = false, tf_harmonic = false)
    for i in eachindex(qx)
        qv[i] = linearly_interpolate(x, v, qx[i]; tf_extrapolate = tf_extrapolate, tf_harmonic = tf_harmonic)
    end

    return nothing
end
function linearly_interpolate(x::AbstractRange, v, qx; tf_extrapolate = false)
    qv = similar(v, eltype(v), size(qx))
    linearly_interpolate!(qv, x, v, qx; tf_extrapolate = tf_extrapolate, tf_harmonic = tf_harmonic)

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
function linearly_interpolate!(qv, x, v, qx; tf_extrapolate = false, tf_harmonic = false)
    for i in eachindex(qx)
        qv[i] = linearly_interpolate(x, v, qx; tf_extrapolate = tf_extrapolate, tf_harmonic = tf_harmonic)
    end

    return nothing
end
function linearly_interpolate(x, v, qx; tf_extrapolate = false, tf_harmonic = false)
    qv = similar(v, eltype(v), size(qx))
    linearly_interpolate!(qv, x, v, qx; tf_extrapolate = tf_extrapolate, tf_harmonic = tf_harmonic)

    return qv
end



## 1D PIECEWISE LINEAR INTERPOLATION -- DISCONTINUOUS GRID -- MONOTONIC INCREASING ##
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
function piecewise_linearly_interpolate!(qv, x, d, v, qx; tf_extrapolate = false, tf_harmonic = false)
    for i in eachindex(qx)
        qv[i] = piecewise_linearly_interpolate(x, d, v, qx[i]; tf_extrapolate = tf_extrapolate, tf_harmonic = tf_harmonic)
    end

    return nothing
end
function piecewise_linearly_interpolate(x, d, v, qx; tf_extrapolate = false, tf_harmonic = false)
    qv = similar(v, eltype(v), size(qx))
    piecewise_linearly_interpolate!(qv, x, d, v, qx; tf_extrapolate = tf_extrapolate, tf_harmonic = tf_harmonic)
    
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
