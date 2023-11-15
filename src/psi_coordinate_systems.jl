# COORDINATE SYSTEMS
abstract type CoordinateSystem end
abstract type GeographicCoordinates <: CoordinateSystem end

# 1.0 LOCAL GEOGRAPHIC
# Coordinate system in which the origin is rotated to the equator
struct LocalGeographic{T, R} <: GeographicCoordinates
    λ₀::T # Origin longitude (degrees)
    ϕ₀::T # Origin latitude (degrees)
    R₀::T # Reference radius (degrees)
    β::T # Rotation from east (degrees)
    Rₘ::R # Rotation matrix that maps from local to global coordinates
    # Inner-constructor method to define rotation matrix
    function LocalGeographic(λ₀::T, ϕ₀::T, R₀::T, β::T) where {T}
        C = π/180.0
        Rₘ = rotation_matrix((C*β, -C*ϕ₀, C*λ₀), (1,2,3))
        return new{T, typeof(Rₘ)}(λ₀, ϕ₀, R₀, β, Rₘ)
    end
end
# Local to Global Coordinate Conversion
function local_to_global(λ, ϕ, r, G::LocalGeographic)
    # Convert spherical to cartesian coordinates
    xl, yl, zl = geographic_to_global(λ, ϕ, r; radius = G.R₀)
    # Rotate to global
    xyz = @SVector [xl, yl, zl]
    xyz = G.Rₘ*xyz

    return xyz[1], xyz[2], xyz[3]
end
function local_to_global(λ::AbstractArray, ϕ::AbstractArray, r::AbstractArray, G::LocalGeographic)
    xg = similar(λ)
    yg = similar(ϕ)
    zg = similar(r)
    for i in eachindex(λ)
        xg[i], yg[i], zg[i] = local_to_global(λ[i], ϕ[i], r[i], G)
    end

    return xg, yg, zg
end
# Global to Local Coordinate Conversion
function global_to_local(xg, yg, zg, G::LocalGeographic)
    # Rotate global coordinates to (0°, 0°)
    xyz = @SVector [xg, yg, zg]
    xyz = transpose(G.Rₘ)*xyz
    # Convert to local geographic
    λ, ϕ, r = global_to_geographic(xyz[1], xyz[2], xyz[3]; radius = G.R₀)

    return λ, ϕ, r 
end
function global_to_local(xg::AbstractArray, yg::AbstractArray, zg::AbstractArray, G::LocalGeographic)
    λ = similar(xg)
    ϕ = similar(yg)
    r = similar(zg)
    for i in eachindex(xg)
        λ[i], ϕ[i], r[i] = global_to_local(xg[i], yg[i], zg[i], G)
    end

    return λ, ϕ, r
end

function local_to_global_vector(v, lon_lat_elv, Geometry::LocalGeographic)
    # Rotate vector to true east, north
    R = rotation_matrix(π*Geometry.β/180.0, 3)
    east  = R[1,1]*v[1] + R[1,2]*v[2] + R[1,3]*v[3]
    north = R[2,1]*v[1] + R[2,2]*v[2] + R[2,3]*v[3]
    elv   = R[3,1]*v[1] + R[3,2]*v[2] + R[3,3]*v[3]
    # Return Earth-centered Earth-fixed vector
    return ecef_vector((east, north, elv), lon_lat_elv[2], lon_lat_elv[1])
end

function global_to_local_vector(w, lon_lat_elv, Geometry::LocalGeographic)
    # Rotate to geographic position
    c = π/180.0
    R = rotation_matrix((-c*lon_lat_elv[1], c*lon_lat_elv[2], -c*Geometry.β), (3, 2, 1))
    vx = R[1,1]*w[1] + R[1,2]*w[2] + R[1,3]*w[3]
    vy = R[2,1]*w[1] + R[2,2]*w[2] + R[2,3]*w[3]
    vz = R[3,1]*w[1] + R[3,2]*w[2] + R[3,3]*w[3]
    # Local coordinate system is centered at (0°N, 0°E) such that global (x,y,z) corresponds to local (radial, east, north)
    return (vy, vz, vx)
end

function global_to_local_angles(azimuth, elevation, lon_lat_elv, Geometry::LocalGeographic)
    # Compute the global vector components
    sinϕ, cosϕ = sincos(azimuth)
    sinλ, cosλ = sincos(elevation)
    # Note the order!
    w = (sinλ, cosϕ*cosλ, sinϕ*cosλ)
    # Return the local vector
    v = global_to_local_vector(w, lon_lat_elv, Geometry)
    # Compute the local orientations
    local_azimuth = atan( v[2], v[1] )
    local_elevation = atan( v[3], sqrt((v[1]^2) + (v[2]^2)) )

    return local_azimuth, local_elevation
end
function local_to_global_angles(azimuth, elevation, lon_lat_elv, Geometry::LocalGeographic)
    sinϕ, cosϕ = sincos(azimuth)
    sinθ, cosθ = sincos(elevation)
    v = (cosϕ*cosθ, sinϕ*cosθ, sinθ)
    w = local_to_global_vector(v, lon_lat_elv, Geometry)
    global_azimuth = atan(w[3], w[2])
    global_elevation = atan(w[1], sqrt((w[2]^2) + (w[3]^2)))

    return global_azimuth, global_elevation
end



# GENERAL GEOGRAPHIC CONVERSIONS

# Geographic to Local Coordinate Conversion
function geographic_to_local(λ, ϕ, elevation, G::GeographicCoordinates)
    # Pass through global coordinates
    xg, yg, zg = geographic_to_global(λ, ϕ, elevation; radius = G.R₀)

    return global_to_local(xg, yg, zg, G)
end



# Local to Geographic Coordinate Conversion
function local_to_geographic(λ, ϕ, r, G::GeographicCoordinates)
    # Pass through global coordinates
    xg, yg, zg = local_to_global(λ, ϕ, r, G)

    return global_to_geographic(xg, yg, zg; radius = G.R₀)
end



# Geographic to Global Coordinate Conversion
function geographic_to_global(λ, ϕ, elevation; radius = 6371.0)
    K = π/180.0
    sinλ, cosλ = sincos(K*λ)
    sinϕ, cosϕ = sincos(K*ϕ)
    R = radius + elevation
    x = R*cosϕ*cosλ
    y = R*cosϕ*sinλ
    z = R*sinϕ

    return x, y, z
end
function geographic_to_global(λ::AbstractArray, ϕ::AbstractArray, elevation::AbstractArray; radius = 6371.0)
    x = similar(λ)
    y = similar(ϕ)
    z = similar(elevation)
    for i in eachindex(λ)
        x[i], y[i], z[i] = geographic_to_global(λ[i], ϕ[i], elevation[i], radius = radius)
    end

    return x, y, z
end



# Global to Geographic Coordinate Conversions
function global_to_geographic(x, y, z; radius = 6371.0)
    C = 180.0/π
    λ = C*atan(y, x)
    ϕ = C*atan(z, sqrt(x^2 + y^2))
    elevation = sqrt(x^2 + y^2 + z^2) - radius

    return λ, ϕ, elevation
end
function global_to_geographic(x::AbstractArray, y::AbstractArray, z::AbstractArray; radius = 6371.0)
    λ = similar(x)
    ϕ = similar(y)
    elevation = similar(z)
    for i in eachindex(x)
        λ[i], ϕ[i], elevation[i] = global_to_geographic(x[i], y[i], z[i], radius = radius)
    end

    return λ, ϕ, elevation
end

function ecef_vector(east_north_elv::NTuple{3, T}, latitude, longitude) where {T}
    # ECEF components for vector at (0°N, 0°E)
    w = (east_north_elv[3], east_north_elv[1], east_north_elv[2])
    # Rotate to geographic position
    c = π/180.0
    R = rotation_matrix((-c*latitude, c*longitude), (2, 3))
    sx = R[1,1]*w[1] + R[1,2]*w[2] + R[1,3]*w[3]
    sy = R[2,1]*w[1] + R[2,2]*w[2] + R[2,3]*w[3]
    sz = R[3,1]*w[1] + R[3,2]*w[2] + R[3,3]*w[3]

    return sx, sy, sz
end



# GEODESIC FUNCTIONS
# 1. Do everything in radians! Note that 'sind(x)' is > 2x  slower than 'sin(deg2rad(x))'...
#    Optional argument to convert degrees/radians
# 2. Use 'isapprox(a, b; atol = 0, rtol = Δ)' to compare angles to 90.0! Reasonable choice
#    for Δ may be 10*eps()? Default is sqrt(eps()).

function direct_geodesic(ϕ₁, λ₁, Δ, α; tf_degrees::Bool = true)
    # Convert input from degrees to radians
    if tf_degrees
        K = π/180.0
        ϕ₁ = K*ϕ₁
        λ₁ = K*λ₁
        Δ = K*Δ
        α = K*α
    end
    # Trig computations
    sinϕ₁, cosϕ₁ = sincos(ϕ₁)
    sinΔ, cosΔ = sincos(Δ)
    sinα, cosα = sincos(α)
    # Destination
    sinϕ₂ = sinϕ₁*cosΔ + cosϕ₁*sinΔ*cosα
    λ₂ = λ₁ + atan(sinα*sinΔ*cosϕ₁, cosΔ-sinϕ₁*sinϕ₂)
    ϕ₂ = asin(sinϕ₂)
    # Solution for reference point at poles
    # At ϕ₁ = ±90° λ₂ is non-unique but can be deduced from α
    # by considering the direction α is pointing when you rotate
    # the sphere-centered cartesian x-coordinate which points
    # towards 0°N, 0°S to the poles.
    #
    # Assumes that λ₁ = 0° when α is calculated via inverse_geodesic,
    # however, this to could also be corrected for.
    if isapprox(ϕ₁, -0.5*π) # ϕ₁ == -0.5*π
        λ₂ = α
    elseif isapprox(ϕ₁, 0.5*π) # ϕ₁ == 0.5*π
        λ₂ = π - α
        if λ₂ > π
            λ₂ = λ₂ - 2.0*π
        end
    end
    # Convert output from radians to degrees
    if tf_degrees
        ϕ₂ /= K
        λ₂ /= K
    end

    return ϕ₂, λ₂ 
end
function direct_geodesic(ϕ₁, λ₁, Δ::AbstractArray, α; tf_degrees::Bool = true)
    # Pre-allocate output
    ϕ = similar(Δ)
    λ = similar(Δ)
    # Loop to retrieve geographic coordinates
    for i in eachindex(Δ)
        ϕ[i], λ[i] = direct_geodesic(ϕ₁, λ₁, Δ[i], α; tf_degrees = tf_degrees)
    end

    return ϕ, λ
end

function inverse_geodesic(ϕ₁, λ₁, ϕ₂, λ₂; tf_degrees::Bool = true)
    # Convert input from degrees to radians
    if tf_degrees
        K = π/180.0
        ϕ₁ = K*ϕ₁
        λ₁ = K*λ₁
        ϕ₂ = K*ϕ₂
        λ₂ = K*λ₂
    end
    # Coordinate differences
    Δϕ = ϕ₂ - ϕ₁
    Δλ = λ₂ - λ₁
    # Assign null longitude if at polls for consistent results
    if isapprox(abs(ϕ₁), 0.5*π) # abs(ϕ₁) == 0.5*π
        λ₁ = 0.0
        Δλ = λ₂
    end
    # Trig computations
    sinϕ₁, cosϕ₁ = sincos(ϕ₁)
    sinϕ₂, cosϕ₂ = sincos(ϕ₂)
    sinΔλ, cosΔλ = sincos(Δλ)
    # Haversine formula for distance between two points on a sphere
    h = sqrt( (sin(0.5*Δϕ)^2) + cosϕ₁*cosϕ₂*(sin(0.5*Δλ)^2) )
    h = min(h, 1.0)
    Δ = 2*asin(h)
    # Bearing formula
    α = atan( sinΔλ*cosϕ₂, cosϕ₁*sinϕ₂ - sinϕ₁*cosϕ₂*cosΔλ )
    # Convert output from radians to degrees
    if tf_degrees
        Δ /= K
        α /= K
    end

    return Δ, α
end

# # Intended to be a general distance function for various coordinates systems
# function distance(a₁, a₂, b₁, b₂, CS::GeographicCoordinates)
#     return inverse_geodesic(a₁, a₂, b₁, b₂)
# end