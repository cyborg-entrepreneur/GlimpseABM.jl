#!/usr/bin/env julia
"""
Decorrelation exhibit: error-free AI deepens the paradox via correlated commitment.

Two panels:
  D1 - NO_AI_ERRORS vs BASELINE paired contrast (premium tier). Agents are
       individually better-behaved (more confident, better calibrated, explore
       more) yet the market is more correlated/crowded and outcomes collapse.
  D2 - suite-wide generalization: the premium survival trap is predicted by
       frontier-relative crowding (premium-minus-none competition) and by the
       epistemic-horizon recession, across all conditions.

Reads robustness_suite_per_run.csv from a robustness-suite results directory.
Usage:
  julia --project=. scripts/build_decorrelation_exhibit.jl --results results/robustness_<ts>
  julia --project=. scripts/build_decorrelation_exhibit.jl            # auto-picks newest robustness_* dir
"""

using CairoMakie
using CSV
using DataFrames
using Distributions
using Printf
using Statistics

const SECOND_ORDER = ["practical_indeterminism", "agentic_novelty", "competitive_recursion"]

# ---------------------------------------------------------------------------
# CLI / directory discovery
# ---------------------------------------------------------------------------

function find_results_dir(explicit)
    if explicit !== nothing
        if !isdir(explicit)
            println(stderr, "results dir not found: $explicit")
            exit(1)
        end
        return explicit
    end
    candidates = String[]
    if isdir("results")
        for d in readdir("results")
            if startswith(d, "robustness_")
                push!(candidates, joinpath("results", d))
            end
        end
    end
    sort!(candidates, rev=true)
    if isempty(candidates)
        println(stderr, "no results/robustness_* dir found; pass --results")
        exit(1)
    end
    return candidates[1]
end

function parse_args(argv)
    results = nothing
    i = 1
    while i <= length(argv)
        a = argv[i]
        if a == "--results"
            i += 1; results = argv[i]
        elseif startswith(a, "--results=")
            results = split(a, "=", limit=2)[2]
        else
            println(stderr, "unknown argument: $a")
            exit(2)
        end
        i += 1
    end
    return results
end

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

function load(results_dir)
    p = joinpath(results_dir, "robustness_suite_per_run.csv")
    if !isfile(p)
        println(stderr, "missing per-run file: $p")
        exit(1)
    end
    df = CSV.read(p, DataFrame)
    horizon_cols = ["all_agent_emergent_$(d)" for d in SECOND_ORDER]
    horizon_mat = Matrix{Float64}(undef, nrow(df), length(horizon_cols))
    for (j, c) in enumerate(horizon_cols)
        horizon_mat[:, j] = Float64.(df[!, c])
    end
    df[!, :horizon] = vec(mean(horizon_mat; dims=2))
    return df
end

# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------

"""Paired Cohen's d over per-element differences (NaN-tolerant)."""
function cohens_d_paired(a::AbstractVector, b::AbstractVector)
    d = Float64.(a) .- Float64.(b)
    d = filter(!isnan, d)
    if length(d) < 2
        return NaN
    end
    s = std(d; corrected=true)
    if s == 0
        return NaN
    end
    return mean(d) / s
end

"""Pearson correlation (r) and two-sided p-value via t-distribution."""
function pearson_corr_p(x::AbstractVector, y::AbstractVector)
    x = Float64.(x); y = Float64.(y)
    n = length(x)
    mx, my = mean(x), mean(y)
    dx, dy = x .- mx, y .- my
    sxx = sum(dx .^ 2); syy = sum(dy .^ 2)
    if sxx == 0 || syy == 0
        return NaN, NaN
    end
    r = sum(dx .* dy) / sqrt(sxx * syy)
    if abs(r) >= 1.0
        return r, 0.0
    end
    t = r * sqrt((n - 2) / (1 - r^2))
    # two-sided p-value from a t with n-2 df
    df = n - 2
    # use cdf of T via Distributions.TDist if available, but stay header-free:
    # rely on the existing dep.
    p = 2 * (1 - tdist_cdf(abs(t), df))
    return r, p
end

# T-distribution CDF via the project's Distributions dep.
tdist_cdf(t::Real, df::Real) = Float64(cdf(TDist(df), float(t)))

"""Spearman rank correlation rho (ties handled by `tiedrank`-style midrank)."""
function spearman_corr(x::AbstractVector, y::AbstractVector)
    rx = midrank(Float64.(x))
    ry = midrank(Float64.(y))
    r, _ = pearson_corr_p(rx, ry)
    return r
end

"""Average rank for ties; ascending order."""
function midrank(v::AbstractVector{<:Real})
    n = length(v)
    perm = sortperm(v)
    sorted = v[perm]
    ranks = zeros(Float64, n)
    i = 1
    while i <= n
        j = i
        while j <= n && sorted[j] == sorted[i]
            j += 1
        end
        avg = (i + j - 1) / 2.0
        for k in i:(j - 1)
            ranks[perm[k]] = avg
        end
        i = j
    end
    return ranks
end

"""OLS slope/intercept for y = b0 + b1*x."""
function ols_slope_intercept(x::AbstractVector, y::AbstractVector)
    x = Float64.(x); y = Float64.(y)
    mx, my = mean(x), mean(y)
    dx = x .- mx
    sxx = sum(dx .^ 2)
    if sxx == 0
        return 0.0, my
    end
    b1 = sum(dx .* (y .- my)) / sxx
    b0 = my - b1 * mx
    return b1, b0
end

# ---------------------------------------------------------------------------
# D1: paired contrast (focal vs ref) at premium tier
# ---------------------------------------------------------------------------

"""Per-condition premium-tier rows keyed by run_idx."""
function premium_by_run(df::AbstractDataFrame, cond::AbstractString)
    sub = df[(string.(df.condition) .== cond) .& (string.(df.tier) .== "premium"), :]
    # use run_idx as the join key
    return sub
end

function fig_contrast(df::AbstractDataFrame, outdir; focal="NO_AI_ERRORS", ref="BASELINE")
    A = premium_by_run(df, focal)
    B = premium_by_run(df, ref)
    idxA = Set(A.run_idx); idxB = Set(B.run_idx)
    common = sort(collect(intersect(idxA, idxB)))
    A = sort(A[in.(A.run_idx, Ref(common)), :], :run_idx)
    B = sort(B[in.(B.run_idx, Ref(common)), :], :run_idx)

    chans = [
        ("confidence_outcome_decision_confidence_mean", "individual", "decision confidence"),
        ("confidence_outcome_abs_gap_mean",             "individual", "miscalibration |perc−real|"),
        ("explore_share",                               "individual", "exploration share"),
        ("market_ai_action_correlation",                "market",     "AI action-correlation"),
        ("market_crowding_pressure",                    "market",     "crowding pressure"),
        ("mean_competition",                            "market",     "competition faced"),
        ("roi",                                         "outcome",    "realized ROI"),
        ("survival_rate",                               "outcome",    "survival rate"),
    ]
    color = Dict(
        "individual" => colorant"#2e86de",
        "market"     => colorant"#e67e22",
        "outcome"    => colorant"#c0392b",
    )
    ds = Float64[]; labels = String[]; cols_c = []; raws = Tuple{Float64,Float64}[]
    for (c, grp, lab) in chans
        push!(ds, cohens_d_paired(A[!, c], B[!, c]))
        push!(labels, lab); push!(cols_c, color[grp])
        push!(raws, (Float64(mean(Float64.(B[!, c]))), Float64(mean(Float64.(A[!, c])))))
    end
    # y positions: reverse so first channel sits at the top (matches matplotlib y[::-1] convention).
    y = collect(reverse(0:(length(labels) - 1)))

    fig = Figure(size=(950, 580))
    ax = Axis(fig[1, 1];
        xlabel = "paired (within-seed) effect size, Cohen's d   ( " * focal * " − " * ref * ", premium tier )",
        title  = "Error-free AI: individually better-behaved, collectively more crowded, worse outcomes\n" *
                 "blue = individual conduct · orange = market structure · red = outcome   " *
                 "(raw means shown as " * ref * "→" * focal * ")",
        titlesize = 10, xlabelsize = 10,
        yticks = (y, labels),
        yticklabelsize = 9,
    )
    vlines!(ax, [0.0]; color = :black, linewidth = 0.8)
    # Horizontal bars (one per channel) coloured by group.
    for (i, (yi, d, c)) in enumerate(zip(y, ds, cols_c))
        barplot!(ax, [yi], [d]; width = 0.7, color = c, strokecolor = :white,
                 strokewidth = 1.0, direction = :x)
    end
    # Raw-mean annotations next to each bar.
    for (yi, d, (bm, am)) in zip(y, ds, raws)
        sign_str = d >= 0 ? "+" : "-"
        ann = "d=" * (d >= 0 ? "+" : "") * @sprintf("%.1f", d) *
              "   " * @sprintf("%.2f", bm) * "→" * @sprintf("%.2f", am)
        align = d >= 0 ? (:left, :center) : (:right, :center)
        off = d >= 0 ? (6.0, 0.0) : (-6.0, 0.0)
        text!(ax, d, yi; text = ann, offset = off, align = align,
              fontsize = 7.5, color = colorant"#444444")
    end
    # Symmetric x-axis padded by 1.7× the max |d|.
    xmax = max(abs(minimum(filter(!isnan, ds))), abs(maximum(filter(!isnan, ds)))) * 1.7
    xlims!(ax, -xmax, xmax)
    # Legend lower-right.
    elems = [PolyElement(color = color["individual"]),
             PolyElement(color = color["market"]),
             PolyElement(color = color["outcome"])]
    Legend(fig[1, 1], elems,
           ["individual conduct", "market structure", "outcome"];
           labelsize = 8, tellwidth = false, tellheight = false,
           halign = :right, valign = :bottom,
           margin = (10, 10, 10, 10), framevisible = true)

    save(joinpath(outdir, "D1_decorrelation_contrast.png"), fig; px_per_unit = 170 / 72)

    return DataFrame(channel = labels, cohens_d = ds,
                     baseline_mean = [r[1] for r in raws],
                     focal_mean    = [r[2] for r in raws])
end

# ---------------------------------------------------------------------------
# D2: suite-wide premium − none contrasts by condition
# ---------------------------------------------------------------------------

function suite_frame(df::AbstractDataFrame)
    rows = []
    for cond in unique(string.(df.condition))
        g = df[string.(df.condition) .== cond, :]
        none_rows = g[string.(g.tier) .== "none", :]
        prem_rows = g[string.(g.tier) .== "premium", :]
        idx_none = Set(none_rows.run_idx)
        idx_prem = Set(prem_rows.run_idx)
        common = sort(collect(intersect(idx_none, idx_prem)))
        if isempty(common)
            continue
        end
        no = sort(none_rows[in.(none_rows.run_idx, Ref(common)), :], :run_idx)
        pr = sort(prem_rows[in.(prem_rows.run_idx, Ref(common)), :], :run_idx)
        trap = mean(Float64.(pr.survival_rate) .- Float64.(no.survival_rate)) * 100
        comp_excess = mean(Float64.(pr.mean_competition) .- Float64.(no.mean_competition))
        horizon_d = mean(Float64.(pr.horizon) .- Float64.(no.horizon))
        push!(rows, (condition = cond, trap = trap, comp_excess = comp_excess,
                     horizon_d = horizon_d, design = string(g.design[1])))
    end
    return DataFrame(rows)
end

# Category sets for D2 styling (mirrors the Python reference).
const PLACEBO_SET   = Set(["SHAM_AI_LABELS_FREE", "NO_INFORMATION_ADVANTAGE_FREE", "EMERGENT_ADOPTION_BASELINE"])
const FAVORABLE_SET = Set(["ALL_FAVORABLE_TO_FRONTIER"])
const HIGH_DECOR_SET = Set(["NO_AI_ERRORS", "FRONTIER_PUBLIC_OMNISCIENCE", "FRONTIER_REASONING_NO_ERROR"])

# Per-condition (px-offset, draw_leader_line?) spec for D2 labels.
# Conditions not present here are not labelled; only those in label_specs print.
const LABEL_SPECS = Dict(
    "ALL_FAVORABLE_TO_FRONTIER"     => ((10.0, 0.0),    false),
    "NO_AI_ERRORS"                  => ((8.0, 8.0),     false),
    "FRONTIER_PUBLIC_OMNISCIENCE"   => ((-6.0, -16.0),  true),
    "BASELINE"                      => ((22.0, 10.0),   true),
    "AI_COST_2X"                    => ((8.0, -10.0),   false),
    "MARKET_DENSE_LOW_CAPACITY"     => ((10.0, 8.0),    false),
    "MARKET_SLACK_HIGH_CAPACITY"    => ((10.0, 6.0),    false),
    "SHAM_AI_LABELS_FREE"           => ((18.0, -18.0),  true),
    "NO_INFORMATION_ADVANTAGE_FREE" => ((-22.0, 14.0),  true),
    "EMERGENT_ADOPTION_BASELINE"    => ((-18.0, -16.0), true),
)

"""Title-case 'foo_bar baz' style strings (matches Python `.replace("_", " ").title()`)."""
function title_case(s::AbstractString)
    return join(uppercasefirst.(lowercase.(split(replace(s, "_" => " "), " "))), " ")
end

"""Convert (dx_px, dy_px) pixel offsets — interpreted in the matplotlib sense
relative to a 170-DPI render — into data-space deltas for placing labels and
leader-line endpoints. The numerator pairs the matplotlib-side annotation
fontsize (~8pt at 170 DPI) with the larger figure CairoMakie panel; the
denominator scales the offset by the data span so labels float at roughly the
same visual distance regardless of axis units."""
function _px_to_data_offsets(dx_px, dy_px, xspan, yspan, axis_w_px, axis_h_px)
    return (dx_px / axis_w_px) * xspan, (dy_px / axis_h_px) * yspan
end

function fig_suitewide(C::AbstractDataFrame, outdir)
    fig = Figure(size = (1500, 650))
    Label(fig[0, 1:2],
          "Suite-wide: the frontier survival trap scales with correlated-commitment crowding";
          fontsize = 13)

    # Category style helper. Returns the kwargs we pass to scatter!. Placebo
    # points are rendered as open gray circles (transparent fill + gray stroke);
    # the rest are filled colour discs / a green up-triangle.
    function style_for(cond::AbstractString)
        if cond in FAVORABLE_SET
            return (color = colorant"#27ae60", marker = :utriangle, markersize = 13,
                    strokecolor = :white, strokewidth = 0.9)
        elseif cond in HIGH_DECOR_SET
            return (color = colorant"#c0392b", marker = :circle, markersize = 11,
                    strokecolor = :white, strokewidth = 0.9)
        elseif cond in PLACEBO_SET
            return (color = RGBAf(0, 0, 0, 0), marker = :circle, markersize = 10,
                    strokecolor = colorant"#7f8c8d", strokewidth = 1.0)
        else
            return (color = colorant"#2c3e50", marker = :circle, markersize = 8,
                    strokecolor = :white, strokewidth = 0.6)
        end
    end

    for (panel_i, (xcol, xlab)) in enumerate([
            (:comp_excess, "frontier-relative excess competition (premium − none)"),
            (:horizon_d,   "epistemic-horizon recession (premium − none)"),
        ])
        x = Float64.(C[!, xcol]); y = Float64.(C.trap)
        r, p = pearson_corr_p(x, y)
        rho = spearman_corr(x, y)
        title_str = "r=" * (r >= 0 ? "+" : "") * @sprintf("%.2f", r) *
                    " (p=" * @sprintf("%.1e", p) * "),  Spearman ρ=" *
                    (rho >= 0 ? "+" : "") * @sprintf("%.2f", rho)
        ax = Axis(fig[1, panel_i];
            xlabel = xlab, ylabel = "survival trap (pp vs none)",
            xlabelsize = 11, ylabelsize = 11,
            title = title_str, titlesize = 11,
            xticklabelsize = 9, yticklabelsize = 9,
        )
        hlines!(ax, [0.0]; color = :grey, linewidth = 0.6, linestyle = :dot)
        b1, b0 = ols_slope_intercept(x, y)
        xs = collect(range(minimum(x), maximum(x); length = 50))
        lines!(ax, xs, b0 .+ b1 .* xs; color = colorant"#c0392b", linewidth = 1.6,
               linestyle = :dash)
        # Per-point scatter so each condition gets its categorical styling.
        for row in eachrow(C)
            sty = style_for(string(row.condition))
            scatter!(ax, [Float64(row[xcol])], [Float64(row.trap)];
                     color = sty.color, marker = sty.marker, markersize = sty.markersize,
                     strokecolor = sty.strokecolor, strokewidth = sty.strokewidth)
        end
        # Pad axis to make room for the labels around the rim.
        xspan = (length(x) > 1 && maximum(x) != minimum(x)) ? (maximum(x) - minimum(x)) : 1.0
        yspan = (length(y) > 1 && maximum(y) != minimum(y)) ? (maximum(y) - minimum(y)) : 1.0
        xlims!(ax, minimum(x) - 0.10 * xspan, maximum(x) + 0.22 * xspan)
        ylims!(ax, minimum(y) - 0.10 * yspan, maximum(y) + 0.12 * yspan)

        # Labels with optional leader lines (callouts). CairoMakie has no
        # `arrowprops`, so each leader is a thin gray line segment drawn from
        # the data point to a small inset near the label start. Offsets mirror
        # the matplotlib spec proportionally via panel dimensions (~640×540 px).
        axis_w_px, axis_h_px = 640.0, 540.0
        for row in eachrow(C)
            cond_s = string(row.condition)
            spec = get(LABEL_SPECS, cond_s, nothing)
            spec === nothing && continue
            (dx_px, dy_px), with_arrow = spec
            dx, dy = _px_to_data_offsets(dx_px, dy_px, xspan, yspan, axis_w_px, axis_h_px)
            px = Float64(row[xcol]); py = Float64(row.trap)
            lx, ly = px + dx, py + dy
            if with_arrow
                # Shrink the leader away from the point a little so it visually
                # detaches (matches matplotlib's shrinkA=0, shrinkB=2).
                shrink_frac = 0.18
                sx = lx - shrink_frac * (lx - px)
                sy = ly - shrink_frac * (ly - py)
                lines!(ax, [px, sx], [py, sy];
                       color = colorant"#888888", linewidth = 0.6)
            end
            align = dx >= 0 ? (:left, :bottom) : (:right, :bottom)
            text!(ax, lx, ly; text = title_case(cond_s),
                  fontsize = 8, color = colorant"#222222", align = align)
        end
    end

    # 4-entry left-panel legend with category swatches.
    legend_elems = [
        MarkerElement(color = colorant"#2c3e50", marker = :circle, markersize = 9,
                      strokecolor = :white, strokewidth = 0.6),
        MarkerElement(color = colorant"#27ae60", marker = :utriangle, markersize = 12,
                      strokecolor = :white, strokewidth = 0.9),
        MarkerElement(color = colorant"#c0392b", marker = :circle, markersize = 10,
                      strokecolor = :white, strokewidth = 0.9),
        MarkerElement(color = RGBAf(0, 0, 0, 0), marker = :circle, markersize = 10,
                      strokecolor = colorant"#7f8c8d", strokewidth = 1.0),
    ]
    legend_labels = ["paradox supported", "favorable reversal",
                     "high decorrelation", "placebo / emergent"]
    # Place the legend inside the left panel at lower-left, like the Python.
    Legend(fig[1, 1], legend_elems, legend_labels;
           labelsize = 8.5, tellwidth = false, tellheight = false,
           halign = :left, valign = :bottom,
           margin = (10, 10, 10, 10), framevisible = true)

    save(joinpath(outdir, "D2_decorrelation_suitewide.png"), fig; px_per_unit = 170 / 72)
end

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

function main(argv = ARGS)
    results_arg = parse_args(argv)
    results_dir = find_results_dir(results_arg)
    outdir = joinpath(results_dir, "decorrelation_exhibit")
    mkpath(outdir)
    df = load(results_dir)

    d1 = fig_contrast(df, outdir)
    C = suite_frame(df)
    fig_suitewide(C, outdir)

    CSV.write(joinpath(outdir, "D1_contrast_effect_sizes.csv"), d1)
    CSV.write(joinpath(outdir, "D2_suitewide_predictors.csv"), sort(C, :trap))

    println("wrote exhibits to $outdir")
    for f in sort(readdir(outdir))
        println("   ", f)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
