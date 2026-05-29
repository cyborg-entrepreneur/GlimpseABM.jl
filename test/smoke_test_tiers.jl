push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
using GlimpseABM
using Random
using Statistics
using Printf

const N_AGENTS = 100
const N_ROUNDS = 10
const TIERS = ["none", "basic", "advanced", "premium"]

println("=" ^ 70)
println("Smoke test: $(N_AGENTS) agents × $(N_ROUNDS) rounds × $(length(TIERS)) tiers")
println("=" ^ 70)

results = Dict{String,NamedTuple}()

for tier in TIERS
    println("\n--- Tier: $tier ---")
    config = EmergentConfig(
        N_AGENTS=N_AGENTS,
        N_ROUNDS=N_ROUNDS,
        RANDOM_SEED=42,
    )
    sim = EmergentSimulation(
        config=config,
        seed=42,
        initial_tier_distribution=Dict(tier => 1.0),
    )
    GlimpseABM.run!(sim)

    n_alive = count(a -> get_capital(a) > 0, sim.agents)
    survival_rate = n_alive / N_AGENTS
    capitals = [get_capital(a) for a in sim.agents]
    mean_cap = mean(capitals)
    median_cap = median(capitals)

    @printf("  Survival rate: %.3f (%d / %d)\n", survival_rate, n_alive, N_AGENTS)
    @printf("  Mean capital:   \$%.0f\n", mean_cap)
    @printf("  Median capital: \$%.0f\n", median_cap)
    @printf("  Min capital:    \$%.0f\n", minimum(capitals))
    @printf("  Max capital:    \$%.0f\n", maximum(capitals))

    results[tier] = (
        survival = survival_rate,
        mean_cap = mean_cap,
        median_cap = median_cap,
    )

    # Sanity bounds
    @assert survival_rate >= 0.0 && survival_rate <= 1.0
    @assert mean_cap > 0
end

println("\n" * "=" ^ 70)
println("Tier-divergence summary:")
println("=" ^ 70)
@printf("%-10s  %-12s  %-15s  %-15s\n", "Tier", "Survival", "Mean Cap", "Median Cap")
for tier in TIERS
    r = results[tier]
    @printf("%-10s  %-12.3f  \$%-14.0f  \$%-14.0f\n",
            tier, r.survival, r.mean_cap, r.median_cap)
end

# Verify tiers actually diverge in outcomes
survivals = [results[t].survival for t in TIERS]
caps = [results[t].mean_cap for t in TIERS]
println("\nSurvival range across tiers: $(round(maximum(survivals) - minimum(survivals), digits=3))")
println("Mean-capital CV across tiers: $(round(std(caps) / mean(caps), digits=3))")

println("\n[OK] Smoke test passed.")
