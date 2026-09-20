begin
    using Plots
    using DataFrames
    using Random
    using StatsBase
    using DifferentialEquations
    using Optim 
    using Turing
    using CSV
    using LaTeXStrings
    using StatsPlots
    using KernelDensity
end

# plotly()
gr()

function simulate_MM(n, α, β, κ, init, t, t_final)
    # Arrays to store simulation results
    times = Float64[]
    substrate = Int[]
    enzyme = Int[]
    complex = Int[]
    product = Int[]
    event_types = Int[]

    # Random number generator
    # rng = MersenneTwister(1234)

    t = 0.0

    # Initial conditions
    S = init[:S]
    E = init[:E]
    C = init[:C]
    P = init[:P]
    event_type = 0

    # Store results
    push!(times, t)
    push!(substrate, S)
    push!(enzyme, E)
    push!(complex, C)
    push!(product, P)
    push!(event_types, event_type)

    # Simulation loop
    while t < t_final
        # Calculate propensities
        κ_1 = n^β[:β_1] * κ[:κ_1] * S * E
        κ_m1 = n^β[:β_m1] * κ[:κ_m1] * C
        κ_2 = n^β[:β_2] * κ[:κ_2] * C

        # Calculate total propensity
        κ_0 = κ_1 + κ_m1 + κ_2

        # Calculate time to next reaction
        τ = -log(rand()) / κ_0

        # Update time
        t += τ

        # Update species
        r = rand()
        if (r < κ_1 / κ_0) & (S > 0) & (E > 0)
            S -= 1
            E -= 1
            C += 1
            event_type = 1
        elseif (r < (κ_1 + κ_m1) / κ_0) & (C > 0)
            C -= 1
            S += 1
            E += 1
            event_type = 2
        elseif (C > 0)
            C -= 1
            E += 1
            P += 1
            event_type = 3
        end

        # Store results
        push!(times, t)
        push!(substrate, S)
        push!(enzyme, E)
        push!(complex, C)
        push!(product, P)
        push!(event_types, event_type)
    end

    # Return results
    return DataFrame(times=times, substrate=substrate, enzyme=enzyme, complex=complex, product=product, event_types=event_types, ZS=substrate / (n^α[:α_S]), ZE=enzyme / (n^α[:α_E]), ZC=complex / (n^α[:α_C]), ZP=product / (n^α[:α_P]))
end

function mm_sQSSA!(du, u, p, t)
    κ_1, κ_m1, κ_2, m = p
    κ_m = (κ_m1 + κ_2) / κ_1

    du[1] = -κ_2 * m * u[1] / (κ_m + u[1])
    du[2] = -du[1]
end

function mm_sQSSA_inference!(du, u, p, t)
    κ_m, κ_2, m = p
    du[1] = -κ_2 * m * u[1] / (κ_m + u[1]) # u[1] is h(s)
end

function safe_pos(x; eps = 1e-12)
    return max(real(x), eps)
end

function loglikelihood(θ, data, m, T)
    κ_m, κ_2 = θ
    κ_m = safe_pos(κ_m)
    κ_2 = safe_pos(κ_2)

    n = length(data)
    parms = [κ_m, κ_2, m]
    u0 = [1.0, 1.0]
    tspan = (0.0, T)
    saveat_times = sort(data)
    push!(saveat_times, T)
    sort!(saveat_times)

    ode_prob = ODEProblem(mm_sQSSA_inference!, u0, tspan, parms)
    ode_sol = solve(ode_prob, Tsit5(), saveat=saveat_times)

    ll = 0.0
    for iter in 1:n
        s = safe_pos(ode_sol.u[iter][1])
        # h = safe_pos(ode_sol.u[iter][2])
        ll += log(s) - log(κ_m + s)
    end

    final_term = safe_pos(1.0 - ode_sol.u[n+1][1])
    ll += n * log(κ_2) - n * log(final_term)

    return ll
end

function mle_objective(θ, data, m, T)
    κ_m, κ_2 = θ
    return -loglikelihood([κ_m, κ_2], data, m, T)
end


# Parameters
n = 10^6
S0 = n 
E0 = 10 
α = Dict(:α_S => 1, :α_E => 0, :α_C => 0, :α_P => 1)
β = Dict(:β_1 => 0, :β_m1 => 1, :β_2 => 1)

κ = Dict(:κ_1 => 1.0, :κ_m1 => 0.20, :κ_2 => 0.1)
κ_m = (κ[:κ_m1] + κ[:κ_2]) / κ[:κ_1]
print(κ_m)

# Initial conditions
init = Dict(:S => S0, :E => E0, :C => 0, :P => 0)

# Parameters
κ
m = init[:E] / (n^α[:α_E])
p = [κ[:κ_1], κ[:κ_m1], κ[:κ_2], m]



# Time parameters
t = 0.0     # Initial time
t_final = 5.0   # Final simulation time

# Run simulation
mm_sims = simulate_MM(n, α, β, κ, init, t, t_final)
P_formation_times = mm_sims[!, :times][mm_sims[!, :event_types].==3]

######## MCMC results 
# Random sample of product formation times
T = 4.0
n_sample = 1000
P_formation_times_sample = sample(P_formation_times[P_formation_times.<T], n_sample, replace=false)


@model function mm_Turing_model2(data, m, T)
    κ_2 ~ Uniform(0, 0.5)
    κ_m ~ Uniform(κ_2, 1.0)
    
    θ = [κ_m, κ_2]

    Turing.@addlogprob! loglikelihood(θ, data, m, T)
end

model2 = mm_Turing_model2(P_formation_times_sample, m, T)

chn_nuts2 = sample(model2, NUTS(0.65), 5000)
describe(chn_nuts2)
summarystats(chn_nuts2)

# summary table from the chain
summary_nuts_df = summarystats(chn_nuts2)
CSV.write("chn_nuts2_summary_statistics.csv", summary_nuts_df)

kappa_m = chn_nuts2[:κ_m]
kappa_2 = chn_nuts2[:κ_2]

nuts_parameters_samples = DataFrame(κ_m=kappa_m.data[:, 1], κ_2=kappa_2.data[:, 1])


fname = "nuts_parameters_samples.csv"
CSV.write(fname, nuts_parameters_samples)



gr()

fig = density(kappa_m,
    xlabel=L"\kappa_M", ylabel="Density", linewidth=2, color="grey", label="", fill=(0, 0.5, :grey)  # Fill under the curve
)

scatter!([κ_m], [0],
    shape=:diamond,
    color="black",
    markersize=8,
    label=""
)

fname = "kappa_m_density_nuts"
pdf_fname = fname * ".pdf"
savefig(fig, fname * ".pdf")
savefig(fig, fname * ".svg")
savefig(fig, fname * ".png")


gr()
fig = density(kappa_2,
    xlabel=L"\kappa_P", ylabel="Density", linewidth=2, color="grey", label="", fill=(0, 0.5, :grey)  # Fill under the curve
)

scatter!([κ[:κ_2]], [0],
    shape=:diamond,
    color="black",
    markersize=8,
    label=""
)

fname = "kappa_2_density_nuts"
pdf_fname = fname * ".pdf"
savefig(fig, fname * ".pdf")
savefig(fig, fname * ".svg")
savefig(fig, fname * ".png")



gr()
fig = plot(
    nuts_parameters_samples.κ_m,
    xlabel = "Iteration",
    ylabel = L"\kappa_M",
    label="", 
    lw = 1.0,
    color = "grey"
)

fname = "kappa_m_trace_nuts"
pdf_fname = fname*".pdf"
savefig(fig, fname * ".pdf")
savefig(fig, fname * ".svg")
savefig(fig, fname * ".png")


gr()
fig = plot(
    nuts_parameters_samples.κ_2,
    xlabel = "Iteration",
    ylabel = L"\kappa_P",
    label="", 
    lw = 1.0,
    color = "grey"
)

fname = "kappa_2_trace_nuts"
pdf_fname = fname*".pdf"
savefig(fig, fname * ".pdf")
savefig(fig, fname * ".svg")
savefig(fig, fname * ".png")



########################### MLE results 
T = 3.0
n_sample = 10000
nSim = 5000
mle_values = zeros(nSim, 2)
iter = 1
while iter <= nSim
    try
        P_formation_times_sample = sample(P_formation_times[P_formation_times.<T], n_sample, replace=false)
        x_2 = rand() * 0.5
        x_1 = x_2 + rand() * 0.5
        initial_x = [x_1, x_2]
        mle_results = optimize(θ -> mle_objective(θ, P_formation_times_sample, m, T), initial_x)
        mle_values[iter, :] = Optim.minimizer(mle_results)
        iter += 1
    catch e
        # println("An error occurred: ", e)
    end
end
mle_values

mle_values = DataFrame(κ_m=mle_values[:, 1], κ_2=mle_values[:, 2])
fname = "mle_values"*string(n_sample)*"_samples.csv"
CSV.write(fname, mle_values)


# Estimate the density with a specific bandwidth
kde_result_km = kde(mle_values[!, :κ_m], bandwidth=0.01)

# Plot the density estimate
gr()
fig = plot(kde_result_km.x, kde_result_km.density,
    xlabel=L"\kappa_M",
    ylabel="Density",
    linewidth=2,
    color="grey",
    label="",
    fill=(0, 0.5, :grey)  # Fill under the curve
)
scatter!([κ_m], [0],
    shape=:diamond,
    color="black",
    markersize=8,
    label=""
)

fname = "kappa_m_density_mle_" * string(n_sample) * "_samples"

pdf_fname = fname * ".pdf"
savefig(fig, fname * ".pdf")
savefig(fig, fname * ".svg")
savefig(fig, fname * ".png")

# Estimate the density with a specific bandwidth
kde_result_k2 = kde(mle_values[!, :κ_2], bandwidth=0.01)

# Plot the density estimate
gr()
fig = plot(kde_result_k2.x, kde_result_k2.density,
    xlabel=L"\kappa_P",
    ylabel="Density",
    linewidth=2,
    color="grey",
    label="",
    fill=(0, 0.5, :grey)  # Fill under the curve
)
scatter!([κ[:κ_2]], [0],
    shape=:diamond,
    color="black",
    markersize=8,
    label=""
)

fname = "kappa_2_density_mle_" * string(n_sample) * "_samples"
pdf_fname = fname * ".pdf"
savefig(fig, fname * ".pdf")
savefig(fig, fname * ".svg")
savefig(fig, fname * ".png")


