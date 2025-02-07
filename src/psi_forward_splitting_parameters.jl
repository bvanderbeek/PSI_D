# WARNING! Some coordinate system modification for splitting parameters have been implemented
# 1. Kernel orientations are re-defined to be with respect to the local tangent plane
#
# NOTE THESE CONVENTIONS!
# Always return fast-axis splitting parameters
# Negative split time = fast-axis
# Postitive split time = slow-axis
# Polarization azimuth is measure in ray-aligned QT-plane
# + This means that this angle is constant when computing the phase velocities

# using FFTW
# using LinearAlgebra
# using StaticArrays
# using Plots

# SplittingParameters Structure defined in psi_forward.jl

# Loads SplittingParameters observations
function fill_observation_vector(f, Obs::Type{<:SplittingParameters}, phase, forward; dlm = ",", data_type = Float64)
    # Determing source/receiver ID format
    sid_type, convert_sid = get_id_type(f; index = 7, dlm = dlm)
    rid_type, convert_rid = get_id_type(f; index = 8, dlm = dlm)

    # Read polarisation angle from file? Polarisation is assumed to be stored in 8th column.
    line = readline(f)
    line = split(line, dlm)

    # Allocate storage vector
    n = countlines(f)
    B = Vector{Obs{phase, forward, sid_type, rid_type, NTuple{2,data_type}}}(undef, n)
    # Read in observations
    k = 0
    for line in readlines(f)
        # Update counter 
        k += 1
        # Parse data row
        line = split(line, dlm)
        dt = parse(data_type, line[1])
        faz = parse(data_type, line[2])
        σ_dt = parse(data_type, line[3])
        σ_faz = parse(data_type, line[4])
        T = parse(data_type, line[5])
        p = string(strip(line[6]))
        sid = convert_sid(strip(line[7]))
        rid = convert_rid(strip(line[8]))
        chn = strip(line[9])
        paz = parse(data_type, line[10])
        # Build observable
        B[k] = Obs(phase(p, T, paz), forward(), sid, rid, (dt, faz), (σ_dt, σ_faz))
    end

    return B
end
# EVALUATE KERNEL: Splitting Parameters
function evaluate_kernel(Kernel::ObservableKernel{<:SplittingParameters})
    # Compute split waveform
    S, Ts, Δt_weak, ζ_weak, _ = kernel_split_wavelet(Kernel)
    s1, s2 = (@views S[1,:], @views S[2,:])
    # Search for optimal splitting parameters. Search begins at the estimated position (Δt_weak, ζ_weak)
    split_time, fast_azimuth, _ = splitting_parameters_search(s1, s2, Ts, Δt_weak, ζ_weak)
    # Remove sign of the split time. By my convention, fast-azimuths have negative split times
    split_time = abs(split_time)
    # Update fast-azimuth for the polarization direction (previous calculations were all carried out in the
    # principle shear wave coordinate system)
    fast_azimuth += Kernel.Observation.Phase.paz
    # Compute surface projection of fast-axis...is this step necessary? Depends on data processing.
    R_rg = rotation_matrix(( 0.5*π - Kernel.weights[end].elevation, Kernel.weights[end].azimuth), (2, 3))
    sinζ, cosζ = sincos(fast_azimuth)
    fv = @SVector [cosζ, sinζ, 0.0]
    fv = R_rg*fv
    fast_azimuth = atan(fv[2], fv[1])
    # Add statics to predictions -- 1 element vector with a tuple
    split_time += Kernel.static[1][1]
    fast_azimuth += Kernel.static[1][2]
    # Compute relative residuals
    res_time = (Kernel.Observation.observation[1] - split_time)/Kernel.Observation.error[1]
    res_azim = atan(tan(Kernel.Observation.observation[2] - fast_azimuth))/Kernel.Observation.error[2]

    return (split_time, fast_azimuth), (res_time, res_azim)
end
function kernel_split_wavelet(Kernel::ObservableKernel{<:SplittingParameters};
    order = 1, sampling_period = 0.01*Kernel.Observation.Phase.period, relative_length = 2.0)

    # Construct initial wavelet (linearly polarized on channel 1)
    fbins, U, τ, Ts = fd_gaussian_wavelet(Kernel.Observation.Phase.period, [1.0; 0.0];
    order = order, sampling_period = sampling_period, relative_length = relative_length)
    # Initalize apparent splitting coefficients for the dominant frequency
    γ11, γ12, dominant_frequency = (1.0 + 0.0im, 0.0 + 0.0im, 1.0/Kernel.Observation.Phase.period)
    # Propagate wavelet through anisotropic intervals
    for (index, w) in enumerate(Kernel.weights)
        # qS-phase velocities
        vqs1, vqs2, _, ζ = qs_phase_velocities(Kernel.weights[index].azimuth, Kernel.weights[index].elevation,
        Kernel.Observation.Phase.paz, Kernel.Parameters, index)
        # Delay of qS'-wave (symmetry axis polarization) with respect to qS''-wave
        Δt = w.dr*((1.0/vqs1) - (1.0/vqs2))
        # Split the waveform
        fd_split_wavelet!(fbins, U, Δt, ζ)
        # Update apparent splitting coefficients
        γ11, γ12 = reduced_splitting_operator(dominant_frequency, Δt, ζ, γ11, γ12)
    end
    # Time-domain split waveform
    S = ifft!(ifftshift(U, 2), 2)
    # Analytic estimate of the splitting parameters
    Δt_weak, ζ_weak = weak_splitting_parameters(dominant_frequency, γ11, γ12)

    return S, Ts, Δt_weak, ζ_weak, τ
end


# Wavelet Splitting Functions
# Single interval
function fd_split_wavelet!(fbins, u::Array{<:Complex, 2}, dt::Number, ζ::Number)
    # Polarization angle of the qS'-wave with respect to channel 1 (i.e. u₁)
    sin2ζ, cos2ζ = sincos(2.0*ζ)
    # Rotate and delay all frequency components of wavelet
    for (i, f) in enumerate(fbins)
        # Compute splitting operator
        sinθ, cosθ = sincos(π*f*dt)
        γ₁₁ = cosθ - 1.0im*sinθ*cos2ζ # γ₂₂ = complex conjugate of γ₁₁
        γ₁₂ = -1.0im*sinθ*sin2ζ # γ₂₁ = γ₁₂
        # Compute new components
        u₁ᵢ = γ₁₁*u[1,i] + γ₁₂*u[2,i]
        u₂ᵢ = γ₁₂*u[1,i] + conj(γ₁₁)*u[2,i]
        # Update
        u[1,i] = u₁ᵢ
        u[2,i] = u₂ᵢ
    end
    # Can I speed up this loop by neglecting negative frequencies?
    # The negative frequencies should be the complex conjugates

    return nothing
end
# Mulitple intervals
function fd_split_wavelet!(fbins, u::Array{<:Complex, 2}, dt, ζ)
    for i in eachindex(dt)
        fd_split_wavelet!(fbins, u, dt[i], ζ[i])
    end

    return nothing
end
# Returns the two independent componetns of the single-layer splitting operator
function reduced_splitting_operator(frequency, split_time, split_azimuth)
    # Single-layer splitting operator (see Equation 8 of Rumpker and Silver, 1998)
    sin2ζ, cos2ζ = sincos(2.0*split_azimuth)
    sinθ, cosθ = sincos(π*frequency*split_time)
    γ₁₁ = cosθ - 1.0im*sinθ*cos2ζ # γ₂₂ = complex conjugate of γ₁₁
    γ₁₂ = -1.0im*sinθ*sin2ζ # γ₂₁ = γ₁₂

    return γ₁₁, γ₁₂
end
function reduced_splitting_operator(frequency, split_time, split_azimuth, g11, g12)
    # Single-layer splitting operator
    γ11_i, γ12_i = reduced_splitting_operator(frequency, split_time, split_azimuth)
    # Update splitting operator elements
    γ11 = g11*γ11_i + g12*γ12_i
    γ12 = g11*γ12_i + g12*conj(γ11_i)

    return γ11, γ12
end
function reduced_splitting_operator(frequency, split_times::AbstractArray, split_azimuths::AbstractArray)
    # Build multi-layer splitting operator
    γ11, γ12 = (1.0 + 0.0im, 0.0 + 0.0im)
    for i in eachindex(split_times)
        # Apparent Splitting Operator after passing through the i'th-layer
        γ11, γ12 = reduced_splitting_operator(frequency, split_times[i], split_azimuths[i], γ11, γ12)
    end

    return γ11, γ12
end
function weak_splitting_parameters(frequency, γ11::Complex, γ12::Complex; null_val = 1.0e-3)
    # Extract real and imaginary components of splitting operator
    g11, g12, h11, h12 = (real(γ11), real(γ12), imag(γ11), imag(γ12))
    # Apparent fast-axis and split time assuming Δt << T
    # Because the numerator in the atan2-function is always positive and we are solving for 2ζ,
    # the split direction (ζ) is constrained to the interval [0°, 90°]. Consequently, there is
    # ambiguity as to whether this orientation is a fast- or slow-direction. The sign of the
    # split-time is required to resolve the ambiguity. Consequently, we CANNOT use the
    # atan2-function to evaluate the split time.
    n = (g12^2) + (h12^2)
    ζ = 0.5*atan(n, (g11*g12 + h11*h12))
    Δt = (1.0/(π*frequency))*atan(n / ((g12*h11 - g11*h12)*sin(2.0*ζ)))
    # Catch NaN-valued split
    Δt = isnan(Δt) ? 0.0 : Δt
    # Catch positive split time (i.e. returned the slow-direction)
    Δt, ζ = Δt > 0.0 ? (-Δt, ζ - 0.5*π) : (Δt, ζ)
    # Catch null splitting direction (occurs as ζ -> ± 90°)
    Δt, ζ = abs(h12/h11) < null_val ? (0.0, 0.0) : (Δt, ζ)

    return Δt, ζ
end
function weak_splitting_parameters(frequency, split_times, split_azimuths; null_val = 1.0e-3)
    # Apparent splitting operator
    γ11, γ12 = reduced_splitting_operator(frequency, split_times, split_azimuths)
    # Compute weak splitting parameters
    Δt, ζ = weak_splitting_parameters(frequency, γ11, γ12; null_val = null_val)
    return Δt, ζ, γ11, γ12
end

# Wavelet Functions
function gaussian_wavelet(dominant_period, polarization; order = 1, sampling_period = 0.1*dominant_period, relative_length = 2.0)
    t, s, τ, Ts = gaussian_wavelet(dominant_period; order = order, sampling_period = sampling_period, relative_length = relative_length)
    Sp = vec(polarization)*transpose(s)

    return t, Sp, τ, Ts
end
function fd_gaussian_wavelet(dominant_period, polarization; order = 1, sampling_period = 0.1*dominant_period, relative_length = 2.0)
    f, u, τ, Ts = fd_gaussian_wavelet(dominant_period; order = order, sampling_period = sampling_period, relative_length = relative_length)
    Up = vec(polarization)*transpose(u)

    return f, Up, τ, Ts
end
function gaussian_wavelet(dominant_period; order = 1, sampling_period = 0.1*dominant_period, relative_length = 2.0)
    # Define time sampling
    N = 1 + round(Int, 2*relative_length*dominant_period/sampling_period) # Number of samples
    t = range(start = -dominant_period*relative_length, stop = dominant_period*relative_length, length = N) # Time vector
    # Characteristic period of the Gaussian
    characteristic_period = order == 0 ? T : dominant_period*sqrt(2.0*order)
    # Multiplication of reference Gaussian with n'th Hermite polynomial yields n'th-order Gaussian wavelet
    Hn = hermite_polynomial(order)
    x = (sqrt(8)*π/characteristic_period)*t
    s = -Hn.(x).*exp.(-0.5*(x.^2)) # Sign flip is just a preference such that 2nd-order wavelet is the typical Ricker wavelet shape

    return t, s, characteristic_period, step(t)
end
function fd_gaussian_wavelet(dominant_period; order = 1, sampling_period = 0.1*dominant_period, relative_length = 2.0)
    # Time-domain Gaussain wavelet
    t, s, τ, Ts = gaussian_wavelet(dominant_period; order = order, sampling_period = sampling_period, relative_length = relative_length)
    # Symmetric frequency bins
    f = range(start = -0.5/Ts, stop = 0.5/Ts, length = length(t))
    # Symmetric Fourier transform of time-domain signal
    u = fftshift(fft(s))

    # # Compute Gaussian wavelet in frequency domain
    # τ = order == 0 ? T : T*sqrt(2*order) # Characteristic period
    # a = 4.0*π^2/(τ^2) # Coefficient in Gaussian exponential, exp(-ax²)
    # n = 1 + round(Int, 2*nT*T/sampling_period) # Number of samples in signal
    # Ts = (2*nT*T)/(n-1) # Sampling period
    # # Symmetric frequency bins
    # f = range(start = -0.5/Ts, stop = 0.5/Ts, length = n)
    # # Amplitude spectrum of Gaussian. Couple notes:
    # # 1. Sign flip is just a preference such that 2nd-order wavelet is the typical Ricker wavelet shape
    # # 2. Lost sign of amplitude components which becomes important if we want to do more operations in the frequency domain
    # u = complex.(-sqrt(π/a)*exp.(-((π*f).^2)./a))
    # # Compute the n'th derivative of the Gaussian wavelet.
    # # There are some issues here with the absolute amplitudes I have not yet resolved
    # order == 0 ? u .*= (1.0/Ts) : u .*= (1.0/Ts)*(1.0*im*2.0*π*f).^order
    # # An attempt to introduce correct signs in amplitude spectrum. Needs to be verified.
    # u[1:2:end] .*= -1.0
    # # The time-domian signal is recovered via
    # # s = real(ifft(ifftshift(u)))

    # Return shifted FFT
    return f, u, τ, Ts
end
function hermite_polynomial(order)
    # Hₙ(x) = (-1)ⁿ*exp(x²/2)*dⁿ/dxⁿ[exp(-x²/2)]
    if order == 0
        Hn = x -> 1.0
    elseif order == 1
        Hn = x -> x
    elseif order == 2
        Hn = x -> (x^2) - 1.0
    elseif order == 3
        Hn = x -> (x^3) - 3.0*x
    elseif order == 4
        Hn = x -> (x^4) - 6.0*(x^2) + 3.0
    else
        error("Only Hermite polynomials of orders 0-4 are defined!")
    end

    return Hn
end


# Directed Search for Optimal Splitting Parameters
function splitting_parameters_search(s1, s2, sampling_period, split_time, split_azimuth; objective_function = trace_covariance_minimization,
    δt = 10.0*sampling_period, δζ = 10.0*π/180.0, δt_min = sampling_period, δζ_min = π/180.0, α = 0.5, maxit = 100)

    # Define objective function
    fobj = (x; u1 = s1, u2 = s2, Ts = sampling_period) -> objective_function(u1, u2, Ts, x[1], x[2])
    # Check for a bad initial guess
    fj = objective_function(s1, s2, sampling_period, 0.0, 0.0)
    fk = objective_function(s1, s2, sampling_period, split_time, split_azimuth)
    split_time, split_azimuth = fj <= fk ? (0.0, 0.0) : (split_time, split_azimuth)
    # Start simple search
    x, fmin, numit = pattern_search_optimization(fobj, (split_time, split_azimuth), (δt, δζ); δx_min = (δt_min, δζ_min), α = α, maxit = maxit)
    # Always return the fast-azimuth and corresponding negative split time
    pm = sign(x[2])
    pm = pm == 0.0 ? 1.0 : pm
    split_time, split_azimuth = x[1] > 0.0 ? (-x[1], x[2] - 0.5*pm*π) : x

    return split_time, split_azimuth, fmin, numit
end
# Basic Grid Search for Optimal Splitting Parameters
function splitting_parameters_grid_search(s1, s2, sampling_period, split_times, split_azimuths;
    objective_function = transverse_energy_minimization, tf_plot = false)
    # Define objective function
    fobj = (τ, ζ; u1 = s1, u2 = s2, Ts = sampling_period) -> objective_function(u1, u2, Ts, τ, ζ)
    # Start simple search
    min_time, min_azimuth, fmin, imin, jmin, Fsamp = grid_search_optimization(fobj, split_times, split_azimuths)
    # Always return the fast-azimuth and corresponding negative split time
    pm = sign(min_azimuth)
    pm = pm == 0.0 ? 1.0 : pm
    min_time, min_azimuth = min_time > 0.0 ? (-min_time, min_azimuth - 0.5*pm*π) : (min_time, min_azimuth)
    # Option to plot objective function
    if tf_plot
        x_limits, y_limits = (extrema(split_times), (180.0/π).*extrema(split_azimuths))
        ratio_xy = abs(y_limits[2] - y_limits[1])/abs(x_limits[2] - x_limits[1])
        hf = heatmap((180.0/π)*split_azimuths, split_times, Fsamp, color=:inferno, aspect_ratio = ratio_xy,
        xlim = y_limits, ylim = x_limits, xlabel = "fast-azimuth (°)", ylabel = "split time (s)", title = "Transverse Energy")
        scatter!([180.0*split_azimuths[jmin]/π], [split_times[imin]], label = "")
        display(hf)
    end

    return min_time, min_azimuth, fmin, imin, jmin, Fsamp
end


# Splitting Parameter Objective Functions
function transverse_energy_minimization(q, t, sampling_period, split_time, split_azimuth)
    # Reverse the split for each component
    sinζ, cosζ = sincos(-split_azimuth)
    # Indices of overlapping samples after applying a delay to s₂
    ref_indices, delay_indices = time_delay_discrete(split_time, sampling_period, length(q))
    
    # Loop over overlapping samples
    nsamp = 1 + (ref_indices[2] - ref_indices[1])
    i1, i2 = (ref_indices[1], delay_indices[1])
    E = 0.0 # Initialize transverse energy
    for i in 1:nsamp
        qi1, ti1 = (real(q[i1]), real(t[i1])) # Allows for complex time-domain signal (consequence of ifft)
        qi2, ti2 = (real(q[i2]), real(t[i2]))
        # Rotate the split_azimuth to channel 1
        u1i = cosζ*qi1 - sinζ*ti1
        u2i = sinζ*qi2 + cosζ*ti2
        # Rotate back to reference coordinate system and accumulate s₂-energy
        E += (u1i^2)*(sinζ^2) + (u2i^2)*(cosζ^2) - 2.0*(u1i*u2i)*cosζ*sinζ
        # Increment the sample counters
        i1 += 1
        i2 += 1
    end

    return E
end
function trace_covariance_minimization(s1, s2, sampling_period, split_time, split_azimuth)
    # Reverse the split for each component
    sinζ, cosζ = sincos(-split_azimuth)
    # Indices of overlapping samples after applying a delay to s₂
    ref_indices, delay_indices = time_delay_discrete(split_time, sampling_period, length(s1))
    
    # Loop over overlapping samples
    nsamp = 1 + (ref_indices[2] - ref_indices[1])
    i1, i2 = (ref_indices[1], delay_indices[1])
    c11, c22, c12 = (0.0, 0.0, 0.0) # Initialize trace covariance matrix components
    for i in 1:nsamp
        s1i1, s2i1 = (real(s1[i1]), real(s2[i1])) # Allows for complex time-domain signal (consequence of ifft)
        s1i2, s2i2 = (real(s1[i2]), real(s2[i2]))
        # Rotate the split_azimuth to channel 1
        u1i = cosζ*s1i1 - sinζ*s2i1
        u2i = sinζ*s1i2 + cosζ*s2i2
        # Accumulate trace covariance matrix components
        c11 += u1i^2
        c22 += u2i^2
        c12 += u1i*u2i
        # Increment the sample counters
        i1 += 1
        i2 += 1
    end
    # Determinant of trace covariance matrix (measure of particle motion linearity)
    return c11*c22 - (c12^2)
end
function time_delay_discrete(delay_time, sampling_period, nsamp)
    Δn = round(Int, delay_time/sampling_period)
    if Δn < 0
        # Advance (early) channel 2 with respect to 1
        # Channel 1: ___/\___..
        # Channel 2: ..___/\___
        ref_indices = (1, nsamp + Δn)
        delay_indices = (1 - Δn, nsamp)
    else
        # Delay (late) channel 2 with respect to 1
        # Channel 1: .._/\_____
        # Channel 2: _____/\_..
        ref_indices = (1 + Δn, nsamp)
        delay_indices = (1, nsamp - Δn)
    end

    return ref_indices, delay_indices
end
# This frequency domain approach must be tested
function fd_transverse_energy_minimization(fbins, u::Array{<:Complex, 2}, split_time, split_azimuth, sampling_period)
    # Reverse the split of each frequency component
    E = 0.0 # Accumulates energy on channel 2
    sinζ, cosζ = sincos(split_azimuth) # Rotation components
    for (i, f) in enumerate(fbins)
        # Compute splitting operator for this frequency bin
        sinθ, cosθ = sincos(-π*f*split_time)
        γ₁₁, γ₁₂ = (cosθ - 1.0im*sinθ*cos2ζ, -1.0im*sinθ*sin2ζ)
        # Reverse split
        w₂ᵢ = -γ₁₂*u[1,i] + γ₁₁*u[2,i]
        # Compute amplitude
        E += real(w₂ᵢ)^2 + imag(w₂ᵢ)^2
    end

    return E
end


# Basic Pattern Search Optimization
function pattern_search_optimization(fobj, x::NTuple{2,T}, δx::NTuple{2,T}; δx_min = 0.1.*δx, α = 0.5, maxit = 100) where {T}
    # Search stencil
    stencil = ((1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)) # ((-1, 0), (0, -1), (1, 0), (0, 1))
    # Objective function at initial guess
    fj = fobj(x)
    # Iterate pattern search
    tf_iterate = true
    numit = 0
    while tf_iterate
        # Check stencil points for new minimum
        nmin = 0
        for (n, pn) in enumerate(stencil)
            x_n = x .+ δx.*pn # Proposed update
            fk = fobj(x_n) # Objective function at proposal
            nmin, fj = fk < fj ? (n, fk) : (nmin, fj) # Keep proposal?
        end
        if nmin > 0
            x = x .+ (δx.*stencil[nmin]) # Update guess when new minimum found
        else
            δx = α.*δx # Otherwise, reduce search lengths
            δx = (max(δx[1], δx_min[1]), max(δx[2], δx_min[2]))
        end
        # Continue iterating until minimum step size or maximum number of iterations are reached
        numit += 1
        tf_iterate = any(δx .> δx_min) && (numit < maxit)
    end

    return x, fj, numit
end
# Basic Grid Search Optimization
function grid_search_optimization(fobj, x1, x2)
    Fsamp = zeros(length(x1), length(x2))
    fmin, imin, jmin = (fobj(x1[1], x2[1]), 1, 1)
    for (j, wj) in enumerate(x2)
        for (i, vi) in enumerate(x1)
            fij = fobj(vi, wj)
            (fmin, imin, jmin)  = fij < fmin ? (fij, i, j) : (fmin, imin, jmin)
            Fsamp[i,j] = fij
        end
    end

    return x1[imin], x2[jmin], fmin, imin, jmin, Fsamp
end