"""
Optional deterministic MT19937 random number generator.

A self-contained MT19937 generator with a fixed, reproducible floating-point
stream, selectable via the `USE_NUMPY_RNG` configuration flag as an alternative
to Julia's native `MersenneTwister` (which produces a different stream from the
same seed). Production runs use the Julia native RNG (`USE_NUMPY_RNG = false`);
this module exists only as an optional, partially NumPy-compatible stream.

PARITY SCOPE (verified empirically against `np.random.RandomState`, 2026-06-09).
This module provides PARTIAL sampler-level parity with NumPy's `RandomState` —
a verified subset of samplers matches bit-for-bit; others do not. NO
stream-level parity claim is made: a full simulation run does NOT reproduce a
Python/NumPy run draw-for-draw (agent trait initialization alone hits two
non-matching branches, so the streams diverge at the first agent; generic
draws such as shuffles and integer ranges also consume the stream
differently).

Verified bit-for-bit matches with `np.random.RandomState`:
- seeding (MT19937 state initialization)
- `numpy_rand` (uniform [0,1), genrand_res53: two 32-bit draws, 53-bit precision)
- `numpy_randn` (Marsaglia polar method with pair caching)
- `numpy_gamma` with shape > 1 (Marsaglia-Tsang)

Known NON-matches (do not rely on these for cross-language reproduction):
- `numpy_randint`: uses modulo reduction, not NumPy's masked rejection
  sampling — consumes the stream differently AND is statistically biased for
  ranges that do not divide 2^32.
- `numpy_gamma` with shape < 1 (different algorithm/draw order) and
  shape == 1 (NumPy dispatches to standard_exponential; this uses
  Marsaglia-Tsang).
- `numpy_beta` with a <= 1 and b <= 1 (NumPy uses Johnk's algorithm in that
  region; this always uses the Gamma-ratio method).
- `numpy_exponential`: uses -log(u) where NumPy uses -log(1-u), so individual
  draws differ.
"""

using Random
using RandomNumbers
using RandomNumbers.MersenneTwisters

export NumpyRNG, numpy_rand, numpy_randn, numpy_randint, numpy_seed!
export numpy_gamma, numpy_beta, numpy_uniform, numpy_exponential

"""
A deterministic MT19937 generator with partial sampler-level parity to the
NumPy `RandomState` stream (verified subset only — see the module docstring
for the parity scope; no stream-level parity claim).
Caches one normal draw for the polar method (which generates pairs of normals).
"""
mutable struct NumpyRNG <: Random.AbstractRNG
    mt::MT19937
    normal_cached::Union{Float64, Nothing}
end

"""
Create a NumpyRNG with the given seed (matching np.random.seed(seed)).
"""
function NumpyRNG(seed::Integer)
    return NumpyRNG(MT19937(UInt32(seed)), nothing)
end

"""
Seed the RNG (matching np.random.seed(seed)).
"""
function numpy_seed!(rng::NumpyRNG, seed::Integer)
    rng.mt = MT19937(UInt32(seed))
    rng.normal_cached = nothing
    return rng
end

"""
Generate a random Float64 in [0, 1) matching NumPy's np.random.random().
Uses NumPy's genrand_res53 algorithm: combines two 32-bit values for 53-bit precision.
"""
function numpy_rand(rng::NumpyRNG)::Float64
    # Get two consecutive uint32 values
    a = rand(rng.mt, UInt32)
    b = rand(rng.mt, UInt32)

    # NumPy's genrand_res53 formula
    # (a >> 5) gives 27 bits, (b >> 6) gives 26 bits, total 53 bits
    return ((a >> 5) * 67108864.0 + (b >> 6)) * (1.0 / 9007199254740992.0)
end

"""
Generate n random Float64 values matching NumPy.
"""
function numpy_rand(rng::NumpyRNG, n::Integer)::Vector{Float64}
    return [numpy_rand(rng) for _ in 1:n]
end

"""
Generate random Float64 values matching NumPy for a given shape.
"""
function numpy_rand(rng::NumpyRNG, dims::Tuple)::Array{Float64}
    result = Array{Float64}(undef, dims)
    for i in eachindex(result)
        result[i] = numpy_rand(rng)
    end
    return result
end

"""
Generate a random integer in [0, high).

NOT NumPy-parity: uses modulo reduction on a single uint32, whereas NumPy's
np.random.randint uses masked rejection sampling. Draws differ from NumPy and
the modulo is statistically biased for ranges that do not divide 2^32.
"""
function numpy_randint(rng::NumpyRNG, high::Integer)::Int
    return Int(rand(rng.mt, UInt32) % high)
end

"""
Generate a random integer in [low, high). NOT NumPy-parity (see one-argument
method above).
"""
function numpy_randint(rng::NumpyRNG, low::Integer, high::Integer)::Int
    return low + numpy_randint(rng, high - low)
end

"""
Generate a standard normal random variable using Marsaglia polar method.
Matches NumPy's np.random.randn() exactly by using the same algorithm and caching.
"""
function numpy_randn(rng::NumpyRNG)::Float64
    # Check if we have a cached normal from the previous call
    if rng.normal_cached !== nothing
        result = rng.normal_cached
        rng.normal_cached = nothing
        return result
    end

    # Marsaglia polar method - generates two normals, cache one
    while true
        u1 = numpy_rand(rng)
        u2 = numpy_rand(rng)

        v1 = 2.0 * u1 - 1.0
        v2 = 2.0 * u2 - 1.0
        s = v1 * v1 + v2 * v2

        if s < 1.0 && s != 0.0
            mul = sqrt(-2.0 * log(s) / s)
            # Cache v1 * mul, return v2 * mul (matches NumPy's order)
            rng.normal_cached = v1 * mul
            return v2 * mul
        end
    end
end

"""
Generate n standard normal random variables.
"""
function numpy_randn(rng::NumpyRNG, n::Integer)::Vector{Float64}
    return [numpy_randn(rng) for _ in 1:n]
end

# ============================================================================
# GAMMA DISTRIBUTION - Marsaglia-Tsang method (NumPy-parity only for shape > 1)
# ============================================================================

"""
Generate a Gamma(shape, 1) random variable.

NumPy parity: verified bit-for-bit against np.random.gamma ONLY for shape > 1
(Marsaglia-Tsang). NOT NumPy-parity for shape < 1 (Ahrens-Dieter here; draws
differ from NumPy) or shape == 1 (NumPy dispatches to standard_exponential;
this uses Marsaglia-Tsang).
"""
function numpy_gamma(rng::NumpyRNG, shape::Real)::Float64
    shape = Float64(shape)

    if shape < 1.0
        # Ahrens-Dieter algorithm for shape < 1
        b = (ℯ + shape) / ℯ

        while true
            u1 = numpy_rand(rng)
            u2 = numpy_rand(rng)

            p = b * u1

            if p <= 1.0
                x = p ^ (1.0 / shape)
                if u2 <= exp(-x)
                    return x
                end
            else
                x = -log((b - p) / shape)
                if u2 <= x ^ (shape - 1.0)
                    return x
                end
            end
        end
    end

    # Marsaglia-Tsang method for shape >= 1
    d = shape - 1.0 / 3.0
    c = 1.0 / sqrt(9.0 * d)

    while true
        x = numpy_randn(rng)
        v = 1.0 + c * x

        if v > 0.0
            v = v * v * v
            u = numpy_rand(rng)

            # Squeeze test (fast acceptance)
            if u < 1.0 - 0.0331 * (x * x) * (x * x)
                return d * v
            end

            # Full test
            if log(u) < 0.5 * x * x + d * (1.0 - v + log(v))
                return d * v
            end
        end
    end
end

"""
Generate a Gamma(shape, scale) random variable.
"""
function numpy_gamma(rng::NumpyRNG, shape::Real, scale::Real)::Float64
    return numpy_gamma(rng, shape) * Float64(scale)
end

# ============================================================================
# BETA DISTRIBUTION - Via Gamma (NumPy-parity only when the Gamma draws match)
# ============================================================================

"""
Generate a Beta(a, b) random variable via the Gamma ratio method:
Beta(a,b) = Ga/(Ga+Gb).

NOT NumPy-parity when a <= 1 and b <= 1: NumPy uses Johnk's algorithm in that
region, this implementation always uses the Gamma ratio. Parity also fails
whenever either Gamma draw hits a non-matching gamma branch (shape <= 1; see
numpy_gamma).
"""
function numpy_beta(rng::NumpyRNG, a::Real, b::Real)::Float64
    ga = numpy_gamma(rng, a)
    gb = numpy_gamma(rng, b)
    return ga / (ga + gb)
end

# ============================================================================
# ADDITIONAL DISTRIBUTIONS
# ============================================================================

"""
Generate a Uniform(low, high) random variable (affine transform of
numpy_rand, which is verified NumPy-parity).
"""
function numpy_uniform(rng::NumpyRNG, low::Real=0.0, high::Real=1.0)::Float64
    return low + (high - low) * numpy_rand(rng)
end

"""
Generate an Exponential(scale) random variable.

NOT NumPy-parity: uses -log(u) where NumPy's standard_exponential uses
-log(1-u), so individual draws differ from np.random.exponential.
"""
function numpy_exponential(rng::NumpyRNG, scale::Real=1.0)::Float64
    # Exponential is just -log(U) * scale
    u = numpy_rand(rng)
    while u == 0.0
        u = numpy_rand(rng)
    end
    return -log(u) * Float64(scale)
end

# Implement Random interface for compatibility

# Core Random interface - generate raw UInt32 and UInt64 values
function Random.rand(rng::NumpyRNG, ::Type{UInt32})::UInt32
    return rand(rng.mt, UInt32)
end

function Random.rand(rng::NumpyRNG, ::Type{UInt64})::UInt64
    # Combine two UInt32 values for UInt64
    lo = rand(rng.mt, UInt32)
    hi = rand(rng.mt, UInt32)
    return (UInt64(hi) << 32) | UInt64(lo)
end

function Random.rand(rng::NumpyRNG, ::Type{UInt128})::UInt128
    lo = Random.rand(rng, UInt64)
    hi = Random.rand(rng, UInt64)
    return (UInt128(hi) << 64) | UInt128(lo)
end

# Float64 generation
function Random.rand(rng::NumpyRNG)::Float64
    return numpy_rand(rng)
end

function Random.rand(rng::NumpyRNG, ::Type{Float64})::Float64
    return numpy_rand(rng)
end

# Generate n random Float64 values
function Random.rand(rng::NumpyRNG, n::Integer)::Vector{Float64}
    return numpy_rand(rng, n)
end

# Generate random normal value
function Random.randn(rng::NumpyRNG)::Float64
    return numpy_randn(rng)
end

function Random.randn(rng::NumpyRNG, n::Integer)::Vector{Float64}
    return numpy_randn(rng, n)
end

# Seed function
function Random.seed!(rng::NumpyRNG, seed::Integer)
    numpy_seed!(rng, seed)
    return rng
end

# Required for Julia 1.12+ Random interface
# rng_native_52 returns the Type to indicate what _rand52 should generate
Random.rng_native_52(::NumpyRNG) = Float64

# _rand52 generates a 52-bit random integer using the RNG
# The second argument is the Type returned by rng_native_52
# Following Julia's standard approach for Float64:
# Generate a Float64 in [1.0, 2.0) and extract the 52-bit mantissa
function Random._rand52(rng::NumpyRNG, ::Type{Float64})
    # Generate a random float in [0, 1) and add 1 to get [1, 2)
    f = 1.0 + numpy_rand(rng)
    # Reinterpret as UInt64 and shift left 12 bits to extract mantissa
    return reinterpret(UInt64, f) << 12
end

# ============================================================================
# DISTRIBUTIONS.JL INTEGRATION
# Override rand() for common distributions to use the samplers above.
# Parity caveats from the module docstring apply: only the verified subset
# (uniform, normal, gamma shape > 1) matches NumPy draw-for-draw.
# ============================================================================

using Distributions

# Beta distribution - use our numpy_beta
function Random.rand(rng::NumpyRNG, d::Beta{T}) where T<:Real
    return numpy_beta(rng, d.α, d.β)
end

# Gamma distribution - use our numpy_gamma
function Random.rand(rng::NumpyRNG, d::Gamma{T}) where T<:Real
    return numpy_gamma(rng, shape(d), scale(d))
end

# Normal distribution - use our numpy_randn
function Random.rand(rng::NumpyRNG, d::Normal{T}) where T<:Real
    return d.μ + d.σ * numpy_randn(rng)
end

# Uniform distribution - use our numpy_uniform
function Random.rand(rng::NumpyRNG, d::Uniform{T}) where T<:Real
    return numpy_uniform(rng, d.a, d.b)
end

# Exponential distribution - use our numpy_exponential
function Random.rand(rng::NumpyRNG, d::Exponential{T}) where T<:Real
    return numpy_exponential(rng, d.θ)
end

# LogNormal distribution - exp of normal
function Random.rand(rng::NumpyRNG, d::LogNormal{T}) where T<:Real
    return exp(d.μ + d.σ * numpy_randn(rng))
end
