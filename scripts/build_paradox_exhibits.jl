#!/usr/bin/env julia
"""
Paradox-of-future-knowledge exhibit builder.

Consumes the robustness suite outputs (run_robustness_suite.jl) and produces
paradox-centered tables and figures. Survival appears as one supporting metric,
not the analytical focus.

Inputs (from a robustness_* results directory):
  robustness_suite_paradox_scorecard.csv             (wide: condition x tier)
  robustness_suite_paradox_scorecard_delta_vs_baseline.csv
  robustness_suite_summary.csv
  robustness_suite_delta_vs_baseline.csv

Usage:
  julia --project=. scripts/build_paradox_exhibits.jl --results results/robustness_<ts>
  julia --project=. scripts/build_paradox_exhibits.jl            # auto-picks newest robustness_* dir
"""

using CairoMakie
using CSV
using DataFrames
using Printf
using Statistics

# Knightian uncertainty dimensions: AI compresses actor_ignorance (first-order
# benefit) and shifts uncertainty downstream into the other three (second-order).
const FIRST_ORDER_DIM = "actor_ignorance"
const SECOND_ORDER_DIMS = ["practical_indeterminism", "agentic_novelty", "competitive_recursion"]
const ALL_DIMS = vcat(FIRST_ORDER_DIM, SECOND_ORDER_DIMS)
const DIM_LABEL = Dict(
    "actor_ignorance" => "Actor ignorance\n(first-order)",
    "practical_indeterminism" => "Practical\nindeterminism",
    "agentic_novelty" => "Agentic\nnovelty",
    "competitive_recursion" => "Competitive\nrecursion",
)
const MARKET_COLS = [
    "market_opportunity_overlap",
    "market_investment_concentration",
    "market_opportunity_competition",
    "market_ai_herding_intensity",
    "market_ai_action_correlation",
    "market_combo_reuse_pressure",
    "market_crowding_pressure",
]
const TIER_ORDER = ["none", "basic", "advanced", "premium"]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

"""Stable sort of a per-tier frame into the canonical tier order."""
function _tier_sort(df::AbstractDataFrame)
    if "tier_rank" in names(df)
        return sort(df, :tier_rank, alg=Base.Sort.DEFAULT_STABLE)
    end
    df2 = copy(df)
    order_of(t) = begin
        i = findfirst(==(string(t)), TIER_ORDER)
        i === nothing ? 99 : i - 1
    end
    df2[!, :_o] = [order_of(t) for t in df2.tier]
    df2 = sort(df2, :_o, alg=Base.Sort.DEFAULT_STABLE)
    return select(df2, Not(:_o))
end

function find_results_dir(explicit)
    if explicit !== nothing
        if !isdir(explicit)
            println(stderr, "results dir not found: $explicit")
            exit(1)
        end
        return explicit
    end
    # Auto-pick the newest matching entry by name. Like a shell glob this does
    # not filter to directories, so the newest *name* (which may be a file) is
    # chosen; the required-input check in load() then guards correctness.
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

function load(results_dir)
    function rd(name; required=true)
        path = joinpath(results_dir, name)
        if !isfile(path)
            if required
                println(stderr, "missing required input: $path")
                exit(1)
            end
            return nothing
        end
        return CSV.read(path, DataFrame)
    end
    return Dict(
        "scorecard" => rd("robustness_suite_paradox_scorecard.csv"),
        "scorecard_delta" => rd("robustness_suite_paradox_scorecard_delta_vs_baseline.csv"; required=false),
        "summary" => rd("robustness_suite_summary.csv"; required=false),
        "delta" => rd("robustness_suite_delta_vs_baseline.csv"; required=false),
    )
end

"""Highest-rank AI tier present (the frontier comparison vs none)."""
function frontier_tier(df::AbstractDataFrame)
    sc = _tier_sort(df)
    return string(sc.tier[end])
end

# Treat the scorecard's boolean-ish flags consistently: a value counts as true
# when its lowercased, stripped string form is exactly "true".
istrue(x) = lowercase(strip(string(x))) == "true"

# ---------------------------------------------------------------------------
# Markdown rendering (GitHub-flavored "pipe" tables)
# ---------------------------------------------------------------------------

# Render one cell to its display string. Numeric columns honour `floatfmt`
# (a Printf format string); everything else is rendered with `string`, with
# Bool shown as capitalized True/False.
function _cell_str(v, isnumeric::Bool, fmt::Printf.Format)
    if v === missing
        return ""
    elseif isnumeric
        return Printf.format(fmt, float(v))
    elseif v isa Bool
        return v ? "True" : "False"
    else
        return string(v)
    end
end

"""
Hand-format a DataFrame as a GitHub-flavored `pipe` markdown table.

Numeric columns are right-aligned and formatted with `floatfmt`; all other
columns are left-aligned. Each column width is the larger of the widest cell
and the header width plus padding (minimum 2) — the standard `pipe` layout.
"""
function to_markdown(df::AbstractDataFrame; floatfmt::String=".3f")
    fmt = Printf.Format("%" * floatfmt)
    cols = names(df)
    isnum = [eltype(df[!, c]) <: Union{Missing,Real} && !(eltype(df[!, c]) <: Union{Missing,Bool}) for c in cols]
    # Build the string matrix (cells).
    n = nrow(df)
    cellstr = [Vector{String}(undef, n) for _ in cols]
    for (j, c) in enumerate(cols)
        col = df[!, c]
        for i in 1:n
            cellstr[j][i] = _cell_str(col[i], isnum[j], fmt)
        end
    end
    # Column width for the "pipe" format:
    # colwidth = max(widest data cell, header_width + MIN_PADDING) with MIN_PADDING=2.
    # The header gets 2 extra columns of slack when it is the widest element.
    widths = Vector{Int}(undef, length(cols))
    for j in eachindex(cols)
        data_w = 0
        for i in 1:n
            data_w = max(data_w, length(cellstr[j][i]))
        end
        widths[j] = max(data_w, length(cols[j]) + 2)
    end
    pad(s, w, right) = right ? lpad(s, w) : rpad(s, w)
    io = IOBuffer()
    # Header row.
    print(io, "|")
    for j in eachindex(cols)
        print(io, " ", pad(cols[j], widths[j], isnum[j]), " |")
    end
    print(io, "\n|")
    # Separator row.
    for j in eachindex(cols)
        w = widths[j]
        if isnum[j]
            print(io, repeat("-", w + 1), ":", "|")
        else
            print(io, ":", repeat("-", w + 1), "|")
        end
    end
    # Data rows.
    for i in 1:n
        print(io, "\n|")
        for j in eachindex(cols)
            print(io, " ", pad(cellstr[j][i], widths[j], isnum[j]), " |")
        end
    end
    return String(take!(io))
end

# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

"""Per-condition paradox verdict at the frontier tier. Survival is one column."""
function table_paradox_logic(scorecard::AbstractDataFrame, outdir)
    ft = frontier_tier(scorecard)
    rows = scorecard[string.(scorecard.tier) .== ft, :]
    cols = [
        "condition", "category", "theoretical_role", "design",
        "first_order_benefit_observed", "horizon_recession_observed",
        "paradox_logic_supported", "dominant_second_order_channel",
        "knowledge_horizon_tradeoff", "epistemic_horizon_delta_vs_none",
        "first_order_knowledge_gain_vs_none", "survival_effect_pp_vs_none",
    ]
    cols = [c for c in cols if c in names(rows)]
    tbl = select(rows, cols)

    # CSV: write a copy whose Bool columns render as capitalized True/False.
    csv_tbl = copy(tbl)
    for c in names(csv_tbl)
        if eltype(csv_tbl[!, c]) <: Union{Missing,Bool}
            csv_tbl[!, c] = [v === missing ? missing : (v ? "True" : "False") for v in csv_tbl[!, c]]
        end
    end
    CSV.write(joinpath(outdir, "T1_paradox_logic_by_condition.csv"), csv_tbl)

    open(joinpath(outdir, "T1_paradox_logic_by_condition.md"), "w") do f
        write(f, "# Paradox logic by condition (frontier tier = '$ft')\n\n")
        write(f, "`paradox_logic_supported` = first-order benefit AND horizon recession. " *
                 "Survival is a reported consequence, not the criterion.\n\n")
        write(f, to_markdown(tbl; floatfmt="+.3f"))
        write(f, "\n")
    end
    return tbl
end

function table_knightian_dimensions(scorecard::AbstractDataFrame, outdir; condition="BASELINE")
    rows = scorecard[string.(scorecard.condition) .== condition, :]
    if nrow(rows) == 0
        return nothing
    end
    rows = _tier_sort(rows)
    out = DataFrame()
    out.tier = collect(rows.tier)
    for dim in ALL_DIMS
        lvl = "$(dim)_level"
        per = "perceived_$(dim)_level"
        if lvl in names(rows)
            out[!, "$(dim)_realized"] = collect(rows[!, lvl])
        end
        if per in names(rows)
            out[!, "$(dim)_perceived"] = collect(rows[!, per])
        end
    end
    if "epistemic_horizon_pressure" in names(rows)
        out[!, "epistemic_horizon"] = collect(rows.epistemic_horizon_pressure)
    end
    if "survival_effect_pp_vs_none" in names(rows)
        out[!, "survival_pp_vs_none"] = collect(rows.survival_effect_pp_vs_none)
    end
    CSV.write(joinpath(outdir, "T2_knightian_dimensions_baseline.csv"), out)
    open(joinpath(outdir, "T2_knightian_dimensions_baseline.md"), "w") do f
        write(f, "# Knightian dimensions by tier ($condition)\n\n")
        write(f, "AI should compress actor_ignorance (down vs none) while raising the " *
                 "three second-order dimensions (up vs none).\n\n")
        write(f, to_markdown(out; floatfmt=".3f"))
        write(f, "\n")
    end
    return out
end

# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

"""First-order benefit (actor-ignorance reduction) vs second-order horizon pressure, by tier."""
function fig_paradox_mechanism(scorecard::AbstractDataFrame, outdir; condition="BASELINE")
    rows = _tier_sort(scorecard[string.(scorecard.condition) .== condition, :])
    need = ["first_order_knowledge_gain_vs_none", "epistemic_horizon_delta_vs_none"]
    if nrow(rows) == 0 || !all(c -> c in names(rows), need)
        return
    end
    x = Float64.(rows.first_order_knowledge_gain_vs_none)
    y = Float64.(rows.epistemic_horizon_delta_vs_none)
    tiers = string.(rows.tier)
    supported = "paradox_logic_supported" in names(rows) ?
        rows.paradox_logic_supported : fill(false, nrow(rows))

    fig = Figure(size=(650, 500))
    ax = Axis(fig[1, 1];
        xlabel="First-order benefit:\nactor-ignorance reduction vs none →",
        ylabel="Second-order cost:\nepistemic-horizon recession vs none →",
        title="The paradox of future knowledge ($condition)\n" *
              "upper-right = more foresight bought with more downstream uncertainty",
        titlesize=12, xlabelsize=12, ylabelsize=12)
    hlines!(ax, [0.0]; color=:grey, linewidth=0.8, linestyle=:dash)
    vlines!(ax, [0.0]; color=:grey, linewidth=0.8, linestyle=:dash)
    for i in 1:length(x)
        color = istrue(supported[i]) ? colorant"#c0392b" : colorant"#2c3e50"
        scatter!(ax, [x[i]], [y[i]]; markersize=18, color=color,
                 strokecolor=:white, strokewidth=1.5)
        text!(ax, x[i], y[i]; text=tiers[i], offset=(8, 6), fontsize=10, align=(:left, :bottom))
    end
    # Pad the right edge so the outermost tier label (e.g. "premium") doesn't clip.
    xspan = (length(x) > 1 && maximum(x) != minimum(x)) ? (maximum(x) - minimum(x)) : 1.0
    xlims!(ax, minimum(x) - 0.05 * xspan, maximum(x) + 0.22 * xspan)
    text!(ax, 0.02, 0.97; text="red = paradox_logic_supported", space=:relative,
          fontsize=9, color=colorant"#c0392b", align=(:left, :top))
    save(joinpath(outdir, "F1_paradox_mechanism.png"), fig; px_per_unit=160 / 72)
end

function fig_dimensions_by_tier(scorecard::AbstractDataFrame, outdir; condition="BASELINE")
    rows = _tier_sort(scorecard[string.(scorecard.condition) .== condition, :])
    lvl_cols = ["$(d)_level" for d in ALL_DIMS]
    if nrow(rows) == 0 || !all(c -> c in names(rows), lvl_cols)
        return
    end
    tiers = string.(rows.tier)
    ntiers = length(tiers)
    ndims = length(ALL_DIMS)

    fig = Figure(size=(800, 500))
    ax = Axis(fig[1, 1];
        ylabel="Realized emergent level",
        title="Four Knightian dimensions by AI tier ($condition)",
        xticks=(collect(1:ndims), [DIM_LABEL[d] for d in ALL_DIMS]),
        xticklabelsize=9)
    # Grouped bars: dimension on x, tier as the dodge group.
    xs = Int[]; ys = Float64[]; grp = Int[]
    for (ti, _) in enumerate(eachrow(rows))
        r = rows[ti, :]
        for (di, d) in enumerate(ALL_DIMS)
            push!(xs, di); push!(ys, Float64(r["$(d)_level"])); push!(grp, ti)
        end
    end
    barplot!(ax, xs, ys; dodge=grp, color=grp, colormap=:tab10, colorrange=(1, max(10, ntiers)))
    # Legend keyed by tier.
    elems = [PolyElement(color=Makie.wong_colors()[mod1(t, 7)]) for t in 1:ntiers]
    Legend(fig[1, 2], elems, tiers, "tier"; labelsize=9, titlesize=10)
    save(joinpath(outdir, "F2_knightian_dimensions_by_tier.png"), fig; px_per_unit=160 / 72)
end

function fig_perceived_vs_realized(scorecard::AbstractDataFrame, outdir; condition="BASELINE")
    rows = _tier_sort(scorecard[string.(scorecard.condition) .== condition, :])
    if nrow(rows) == 0
        return
    end
    avail = [d for d in ALL_DIMS
             if "$(d)_level" in names(rows) && "perceived_$(d)_level" in names(rows)]
    if isempty(avail)
        return
    end
    tiers = string.(rows.tier)
    ntiers = length(tiers)
    nav = length(avail)

    fig = Figure(size=(round(Int, 340 * nav), 400))
    Label(fig[0, 1:nav],
          "Perceived vs realized uncertainty ($condition) — gap = foresight divergence";
          fontsize=11)
    local first_ax = nothing
    perceived_color = colorant"#5dade2"
    realized_color = colorant"#e67e22"
    for (k, dim) in enumerate(avail)
        ax = Axis(fig[1, k];
            title=replace(DIM_LABEL[dim], "\n" => " "), titlesize=9,
            xticks=(collect(1:ntiers), tiers), xticklabelrotation=deg2rad(30),
            xticklabelsize=8)
        if first_ax === nothing
            ax.ylabel = "uncertainty level"
            first_ax = ax
        else
            linkyaxes!(first_ax, ax)
            hideydecorations!(ax, grid=false)
        end
        xs = collect(1:ntiers)
        per = Float64.(rows[!, "perceived_$(dim)_level"])
        rea = Float64.(rows[!, "$(dim)_level"])
        barplot!(ax, xs .- 0.2, per; width=0.4, color=perceived_color)
        barplot!(ax, xs .+ 0.2, rea; width=0.4, color=realized_color)
    end
    # Single legend on the first panel.
    if first_ax !== nothing
        axislegend(first_ax,
            [PolyElement(color=perceived_color), PolyElement(color=realized_color)],
            ["perceived", "realized"]; labelsize=8)
    end
    save(joinpath(outdir, "F3_perceived_vs_realized.png"), fig; px_per_unit=160 / 72)
end

"""The paradox is the COEXISTENCE of first-order benefit + horizon recession,
which then produces the survival consequence through the crowding channel.
Two panels avoid a misleading signed difference: left = the two epistemic
shifts (both > 0 = paradox); right = survival outcome."""
function fig_paradox_coexistence(scorecard::AbstractDataFrame, outdir)
    ft = frontier_tier(scorecard)
    rows = scorecard[string.(scorecard.tier) .== ft, :]
    need = ["first_order_knowledge_gain_vs_none", "epistemic_horizon_delta_vs_none",
            "survival_effect_pp_vs_none"]
    if nrow(rows) == 0 || !all(c -> c in names(rows), need)
        return
    end
    rows = sort(rows, :survival_effect_pp_vs_none, alg=Base.Sort.DEFAULT_STABLE)  # most-harmed at top
    conds = string.(rows.condition)
    supported = "paradox_logic_supported" in names(rows) ?
        [istrue(v) for v in rows.paradox_logic_supported] : fill(false, nrow(rows))
    n = length(conds)
    y = collect(1:n)

    fig = Figure(size=(1300, round(Int, max(400, 0.45 * n * 72))))
    Label(fig[0, 1:2], "Paradox coexistence across conditions (frontier tier = '$ft')"; fontsize=12)

    fob = colorant"#27ae60"; soc = colorant"#e67e22"
    axL = Axis(fig[1, 1];
        title="Epistemic coexistence at frontier tier\nparadox = BOTH bars > 0", titlesize=10,
        xlabel="Δ vs none (epistemic units)",
        yticks=(y, conds), yticklabelsize=8)
    vlines!(axL, [0.0]; color=:black, linewidth=0.8)
    first_order = Float64.(rows.first_order_knowledge_gain_vs_none)
    second_order = Float64.(rows.epistemic_horizon_delta_vs_none)
    barplot!(axL, y .- 0.2, first_order; width=0.4, color=fob, direction=:x)
    barplot!(axL, y .+ 0.2, second_order; width=0.4, color=soc, direction=:x)
    axislegend(axL,
        [PolyElement(color=fob), PolyElement(color=soc)],
        ["first-order benefit (actor-ignorance reduction)",
         "second-order cost (epistemic-horizon recession)"];
        labelsize=8, position=:rb)

    colors = [s ? colorant"#c0392b" : colorant"#95a5a6" for s in supported]
    axR = Axis(fig[1, 2];
        title="Survival consequence\nred = paradox_logic_supported", titlesize=10,
        xlabel="survival effect (pp vs none)",
        yticks=(y, conds), yticklabelsize=8)
    linkyaxes!(axL, axR)
    hideydecorations!(axR, grid=false)
    vlines!(axR, [0.0]; color=:black, linewidth=0.8)
    barplot!(axR, y, Float64.(rows.survival_effect_pp_vs_none); color=colors, direction=:x)

    save(joinpath(outdir, "F4_paradox_coexistence_by_condition.png"), fig; px_per_unit=160 / 72)
end

"""Crowding in this model is opportunity-level. Left panel: opportunity-level
crowding. Right panel: herding / action-correlation signals. There is
deliberately no sector-level crowding panel."""
function fig_market_commitment(scorecard::AbstractDataFrame, outdir; condition="BASELINE")
    rows = _tier_sort(scorecard[string.(scorecard.condition) .== condition, :])
    if nrow(rows) == 0
        return
    end
    opp_keys = ["market_opportunity_competition", "market_opportunity_overlap",
                "market_investment_concentration"]
    herd_keys = ["market_ai_herding_intensity", "market_ai_action_correlation",
                 "market_combo_reuse_pressure", "market_crowding_pressure"]
    opp_cols = [c for c in opp_keys if c in names(rows)]
    sector_cols = [c for c in herd_keys if c in names(rows)]
    if isempty(opp_cols) && isempty(sector_cols)
        return
    end
    fig = Figure(size=(1150, 450))
    Label(fig[0, 1:2], "Market correlated-commitment by granularity ($condition)"; fontsize=11)
    panels = [
        (opp_cols, "Opportunity-level crowding\n(competition/saturation on funded opportunities)", true),
        (sector_cols, "Herding / action-correlation signals", false),
    ]
    for (pi, (cols, title, is_left)) in enumerate(panels)
        if isempty(cols)
            continue
        end
        labels = [replace(c, "market_" => "") for c in cols]
        vals = [Float64(rows[1, c]) for c in cols]
        ax = Axis(fig[1, pi];
            title=title, titlesize=10,
            xticks=(collect(1:length(cols)), labels),
            xticklabelrotation=deg2rad(25), xticklabelsize=8,
            limits=(nothing, (0, 1.05)))
        if is_left
            ax.ylabel = "level (0–1)"
        end
        barplot!(ax, collect(1:length(cols)), vals; color=colorant"#34495e")
    end
    save(joinpath(outdir, "F5_market_correlated_commitment.png"), fig; px_per_unit=160 / 72)
end

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

function parse_args(argv)
    results = nothing
    output = nothing
    condition = "BASELINE"
    i = 1
    while i <= length(argv)
        a = argv[i]
        if a == "--results"
            i += 1; results = argv[i]
        elseif a == "--output"
            i += 1; output = argv[i]
        elseif a == "--condition"
            i += 1; condition = argv[i]
        elseif startswith(a, "--results=")
            results = split(a, "=", limit=2)[2]
        elseif startswith(a, "--output=")
            output = split(a, "=", limit=2)[2]
        elseif startswith(a, "--condition=")
            condition = split(a, "=", limit=2)[2]
        else
            println(stderr, "unknown argument: $a")
            exit(2)
        end
        i += 1
    end
    return results, output, String(condition)
end

function main(argv=ARGS)
    results_arg, output_arg, condition = parse_args(argv)

    results_dir = find_results_dir(results_arg)
    outdir = output_arg === nothing ? joinpath(results_dir, "paradox_exhibits") : output_arg
    mkpath(outdir)
    data = load(results_dir)
    sc = data["scorecard"]

    println("results: $results_dir")
    println("output:  $outdir")
    println("conditions: ", sort(unique(string.(sc.condition))))

    t1 = table_paradox_logic(sc, outdir)
    table_knightian_dimensions(sc, outdir; condition=condition)
    fig_paradox_mechanism(sc, outdir; condition=condition)
    fig_dimensions_by_tier(sc, outdir; condition=condition)
    fig_perceived_vs_realized(sc, outdir; condition=condition)
    fig_paradox_coexistence(sc, outdir)
    fig_market_commitment(sc, outdir; condition=condition)

    n_supported = "paradox_logic_supported" in names(t1) ?
        count(istrue, t1.paradox_logic_supported) : 0
    println("\nparadox_logic_supported in $n_supported/$(nrow(t1)) conditions (frontier tier)")
    println("exhibits written to $outdir")
    for f in sort(readdir(outdir))
        println("   ", f)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
