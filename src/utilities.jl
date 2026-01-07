
function subscripts_to_index(dimsize::NTuple{N, Int}, subscripts::NTuple{N, Int}) where {N}
    # Loop over array dimensions
    nd = length(dimsize)
    ind = min(max(subscripts[1], 1), dimsize[1]) - 1
    stride = 1
    for i in 2:nd
        sub = min(max(subscripts[i], 1), dimsize[i]) - 1
        # Count indices
        stride *= dimsize[i - 1]
        ind += stride*sub
    end
    ind += 1

    return ind
end

function index_to_subscripts(dimsize::NTuple{3, Int}, index::Int)
    index -= 1
    k, index = divrem(index, dimsize[1]*dimsize[2])
    j, i = divrem(index, dimsize[1])
    k += 1
    j += 1
    i += 1

    return i, j, k
end
# General but needs to allocate vector for subscripts
function index_to_subscripts(dimsize::NTuple{N, Int}, index::Int) where {N}
    nd = length(dimsize)
    subs = Vector{Int}(undef, nd)
    index -= 1
    for i = nd:-1:2
        stride = prod(dimsize[1:(i-1)])
        subs[i], index = divrem(index, stride)
        subs[i] += 1
    end
    subs[1] = index + 1

    return subs
end


function accumarray!(w, i, v)
    # Loop and sum
    for j in eachindex(v)
        w[i[j]] += v[j]
    end

    return nothing
end
function accumarray!(w, i, v, f::Function)
    # Loop and sum
    for j in eachindex(v)
        w[i[j]] += f(v[j])
    end

    return nothing
end


# Direct search for first node inside boundaries
function get_first_in(x, y, z, xmin, xmax, ymin, ymax, zmin, zmax; i = 1, Δi = 1)
    N = length(x)
    while ( (i > 0) && (i <= N)
        && ~((x[i] >= xmin) && (x[i] <= xmax)
        && (y[i] >= ymin) && (y[i] <= ymax)
        && (z[i] >= zmin) && (z[i] <= zmax)) )
        # Update index
        i += Δi
    end

    return i
end

# Direct search for first node outside boundaries
function get_first_out(x, y, z, xmin, xmax, ymin, ymax, zmin, zmax; i = 1, Δi = 1)
    N = length(x)
    while ( (i > 0) && (i <= N)
        && (x[i] >= xmin) && (x[i] <= xmax)
        && (y[i] >= ymin) && (y[i] <= ymax)
        && (z[i] >= zmin) && (z[i] <= zmax) )
        # Update index
        i += Δi
    end

    return i
end

"""
    rotation_matrix(α, n) -> R::StaticArray

    Returns the 3x3 rotation matrix `R` corresponding to a rotation of
    `α` about axis `n` ∈ [1,2,3]. Rotation angles are measured positive
    counter-clockwise looking at origin from positive end of the 
    rotation axis.

    Returns a static array.

    Angles in radians.
"""
function rotation_matrix(α, n)
    sinα, cosα = sincos(α)
    if n == 1
        return @SMatrix [1.0 0.0 0.0; 0.0 cosα -sinα; 0.0 sinα cosα]
    elseif n == 2
        return @SMatrix [cosα 0.0 sinα; 0.0 1.0 0.0; -sinα 0.0 cosα]
    elseif n == 3
        return @SMatrix [cosα -sinα 0.0; sinα cosα 0.0; 0.0 0.0 1.0]
    else
        error("Requested rotation axis out-of-bounds!")
    end
end
"""
    rotation_matrix(α::Tuple, n::Tuple) -> R::StaticArray

    Returns the 3x3 rotation matrix `R` corresponding to the composition
    of multiple principle rotation matrices, `R` = R(αᵢ)R(αᵢ₋₁)...R(α₁).

    Returns a static array.

    Angles in radians.
"""
function rotation_matrix(α::Tuple, n::Tuple)
    R = @SMatrix [1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0]
    for i in eachindex(n)
        R = rotation_matrix(α[i],n[i])*R
    end
    return R
end



"""
    spherical_to_cartesian(ϕ, θ; r = 1.0) -> x, y, z

    Return cartesian coordinates,`x`, `y`, and `z` from spherical coordinates, 
    `ϕ`, `θ`, and, optionally, `r`. The azimuth (`ϕ`)is measured postive
    counter-clockwise from the +x-axis; the elevation (`θ`) is measured from the
    x,y-plane; and the optional radius (`r`)  defaults to 1.0 if not defined.

    Angles in radians.
"""
function spherical_to_cartesian(ϕ, θ, r = 1.0)
    sinϕ, cosϕ = sincos(ϕ)
    sinθ, cosθ = sincos(θ)
    x = r*cosθ*cosϕ
    y = r*cosθ*sinϕ
    z = r*sinθ
    return x,y,z
end
"""
    cartesian_to_spherical(x, y, z) -> ϕ, θ, r

    Return spherical coordinates,`ϕ`, `θ`, and `r` from cartesian coordinates, 
    `x`, `y`, and, `z`. The azimuth (`ϕ`)is measured postive counter-clockwise
    from the +x-axis; the elevation (`θ`) is measured from the x,y-plane.

    Angles in radians.
"""
function cartesian_to_spherical(x, y, z)
    ϕ = atan(y,x)
    θ = atan(z,sqrt(x^2 + y^2))
    r = sqrt(x^2 + y^2 + z^2)
    return ϕ, θ, r
end

function check_taup_up_down(Events, Stations, Data, ref_model; dR = 0.0, Rref = 6371.0)
    tf_extended = dR > 0.0
    nobs = length(Data)
    kbad, sld = falses(nobs), zeros(nobs)
    tt, dist, baz, phase_out = zeros(nobs), zeros(nobs), zeros(nobs), Vector{String}(undef, nobs)

    TimeObj = buildTimeObj(ref_model)
    for (k, B) in enumerate(Data)
        # Indexing
        evt_k, sta_k = B.source_id, B.receiver_id # Event and station ID
        j_evt, j_sta = Events.id[evt_k], Stations.id[sta_k] # Event and station linear index
        # Event-station distance
        evt_lon, evt_lat, evt_elv = Events.coordinates[j_evt]
        sta_lon, sta_lat, sta_elv = Stations.coordinates[j_sta]
        dist[k], baz[k] = inverse_geodesic(sta_lat, sta_lon, evt_lat, evt_lon; tf_degrees = true)
        # Adjusted source and receiver depths
        evt_dpt, sta_dpt = tf_extended ? (dR - evt_elv, dR - sta_elv) : (-evt_elv, 0.0)
        # Phase names
        if B.Phase.name == "p" || B.Phase.name == "P"
            phase_up, phase_down = "p", "P"
        elseif B.Phase.name == "s" || B.Phase.name == "S"
             phase_up, phase_down = "s", "S"
        else
            error("Only direct phases!")
        end

        # Compare phases
        tt_u, _ = taup_time!(TimeObj, phase_up, dist[k], evt_dpt, sta_dpt)
        tt_u = isnan(tt_u) ? Inf : tt_u
        tt_d, _ = taup_time!(TimeObj, phase_down, dist[k], evt_dpt, sta_dpt)
        tt_d = isnan(tt_d) ? Inf : tt_d
        # Select first arriving phase
        tt[k], phase_out[k] = tt_d < tt_u ? (tt_d, phase_down) : (tt_u, phase_up)
        kbad[k] = isinf(tt[k]) || isnan(tt[k]) ? true : false

        # Compute straight-line-distance
        r_1, r_2 = Rref + evt_elv, Rref + sta_elv
        sinϕ_1, cosϕ_1 = sincosd(evt_lat)
        sinλ_1, cosλ_1 = sincosd(evt_lon)
        sinϕ_2, cosϕ_2 = sincosd(sta_lat)
        sinλ_2, cosλ_2 = sincosd(sta_lon)
        x_1, y_1, z_1 = r_1*cosϕ_1*cosλ_1, r_1*cosϕ_1*sinλ_1, r_1*sinϕ_1
        x_2, y_2, z_2 = r_2*cosϕ_2*cosλ_2, r_2*cosϕ_2*sinλ_2, r_2*sinϕ_2
        sld[k] = sqrt((x_2 - x_1)^2 + (y_2 - y_1)^2 + (z_2 - z_1)^2)
    end
    nbad = sum(kbad)
    nbad > 0 && (@warn "There are "*string(nbad)*" bad events!")

    return kbad, tt, dist, baz, phase_out, sld
end