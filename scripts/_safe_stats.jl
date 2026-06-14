"""
Shared NaN-safe summary helpers for mixed-tier analysis scripts.

v3.5.20: extracted from run_mixed_tier_analysis_full.jl so refutation,
mechanism, placebo, and lambda-sweep drivers all benefit from the same
small-N / empty-batch guards. mean/std on a 0-length vector returns NaN,
and std on a 1-length vector also returns NaN (sample variance needs n>=2).
"""

using Statistics

safe_mean(v) = isempty(v) ? 0.0 : mean(v)
safe_std(v)  = length(v) < 2 ? 0.0 : std(v)
safe_se(std_val::Real, n::Integer) = n > 0 ? std_val / sqrt(n) : 0.0
