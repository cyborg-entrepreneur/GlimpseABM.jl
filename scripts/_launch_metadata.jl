"""
Shared launch metadata helpers for manuscript-facing experiment scripts.

The simulation still uses `premium` as the stable internal tier key. Output
tables expose `frontier_ai` as the manuscript-facing label so we can change
presentation without risking runtime model wiring.
"""

const AI_TIER_DISPLAY_LABELS = Dict(
    "none" => "none",
    "basic" => "basic_ai",
    "advanced" => "advanced_ai",
    "premium" => "frontier_ai",
)
const LAUNCH_METADATA_REPO_ROOT = normpath(joinpath(@__DIR__, ".."))

tier_display_label(tier::AbstractString)::String =
    get(AI_TIER_DISPLAY_LABELS, String(tier), String(tier))

function _safe_cmd_output(cmd::Cmd)::String
    try
        return strip(read(cmd, String))
    catch
        return "unknown"
    end
end

function write_run_provenance!(
    output_dir::String;
    script_name::String,
    parameters::AbstractDict = Dict{String,Any}(),
    notes::AbstractDict = Dict{String,Any}(),
)
    rows = Pair{String,String}[]
    push!(rows, "script" => script_name)
    push!(rows, "generated_at" => Dates.format(now(), dateformat"yyyy-mm-ddTHH:MM:SS"))
    push!(rows, "julia_version" => string(VERSION))
    push!(rows, "threads" => string(Threads.nthreads()))
    push!(rows, "git_commit" => _safe_cmd_output(`git -C $(LAUNCH_METADATA_REPO_ROOT) rev-parse HEAD`))
    push!(rows, "git_branch" => _safe_cmd_output(`git -C $(LAUNCH_METADATA_REPO_ROOT) rev-parse --abbrev-ref HEAD`))
    push!(rows, "git_dirty" => (_safe_cmd_output(`git -C $(LAUNCH_METADATA_REPO_ROOT) status --short`) == "" ? "false" : "true"))
    push!(
        rows,
        "tier_label_semantics" =>
            "Internal key premium is displayed as frontier_ai in manuscript-facing columns.",
   )
    push!(
        rows,
        "emergent_metric_semantics" =>
            "emergent_* aliases summarize all assigned agents; survivor_emergent_* columns report survivor-only values.",
   )
    push!(
        rows,
        "observation_count_semantics" =>
            "*_observations columns report evidence counts behind emergent uncertainty diagnostics.",
   )
    for (key, value) in sort(collect(parameters); by=x -> string(x[1]))
        push!(rows, string(key) => string(value))
    end
    for (key, value) in sort(collect(notes); by=x -> string(x[1]))
        push!(rows, string(key) => string(value))
    end
    df = DataFrame(parameter=first.(rows), value=last.(rows))
    CSV.write(joinpath(output_dir, "run_provenance.csv"), df)
    return df
end
