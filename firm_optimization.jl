using Printf, LinearAlgebra, Random, NLsolve
using Interpolations, Statistics, Roots
using Optim               # for Brent()
using Plots

#───────────────────────────────────────────────────────────────────────────────
# 1.  Parameters
#───────────────────────────────────────────────────────────────────────────────

alpha, gamma = 0.5, 0.3    # Cobb–Douglas exponents
beta,  delta = 0.9, 0.1    # discount factor & depreciation
p_M          = 1.0         # world price of imports
psi          = 0.08        # share of imports that augment capital
eta = 0.01                 # risk aversion constant (mild)

#───────────────────────────────────────────────────────────────────────────────
# 2.  Grids
#───────────────────────────────────────────────────────────────────────────────

K_min, K_max, nK = 0.1, 40.0, 120
K_grid = collect(range(K_min, K_max, length = nK))

τ_L, τ_H = 0.0, 0.4
tau_grid = [τ_L, τ_H]
P_tau    = [0.85 0.15;
            0.10 0.90]
n_tau    = length(tau_grid)

# Bracketing intervals for Brent searches
X_lo, X_hi = 0.0, 10.0
M_lo, M_hi = 0.0, 10.0

#───────────────────────────────────────────────────────────────────────────────
# 3.  Helper functions
#───────────────────────────────────────────────────────────────────────────────

makespline(a, b)        = LinearInterpolation(a, b, extrapolation_bc = Line())
production(K, M)        = K^alpha * ((1 - psi)*M)^gamma
flow_profit(K, M, X, τ) = production(K, M) - (1 + τ)*p_M*M - X
utility(c)              = 1-exp(-eta * c)              # exponential utility
k_prime(K, M, X)        = (1 - delta)*K + X + psi*M

#───────────────────────────────────────────────────────────────────────────────
# 4.  Bellman Iteration
#───────────────────────────────────────────────────────────────────────────────

V        = zeros(nK, n_tau)
Vnext    = similar(V)
policy_M = zeros(nK, n_tau)
policy_X = zeros(nK, n_tau)

tol, max_iter = 1e-6, 1000
iter, bellman_err = 0, Inf

while iter < max_iter && bellman_err > tol
    iter += 1

    # —— reset error for this pass —— 
    bellman_err = 0.0

    # build interpolants of value function
    sV = [makespline(K_grid, V[:, j]) for j in 1:n_tau]

    for (iK, K) in enumerate(K_grid), (jtau, tau) in enumerate(tau_grid)

        # outer maximization over X
        fX = X -> begin
            # inner maximization over M
            fM = M -> begin
                profit  = flow_profit(K, M, X, tau)
                Kp = k_prime(K, M, X)
                EV = P_tau[jtau,1]*sV[1](Kp) + P_tau[jtau,2]*sV[2](Kp)
                return -(utility(profit) + beta*EV)
            end

            resM = optimize(fM, M_lo, M_hi, Brent(); rel_tol=1e-6)
            return Optim.minimum(resM)
        end

        # find optimal X
        resX            = optimize(fX, X_lo, X_hi, Brent(); rel_tol=1e-6)
        val_neg, X_star = Optim.minimum(resX), Optim.minimizer(resX)

        # recover optimal M at X_star
        fM_star = M -> begin
            profit  = flow_profit(K, M, X_star, tau)
            Kp = k_prime(K, M, X_star)
            EV = P_tau[jtau,1]*sV[1](Kp) + P_tau[jtau,2]*sV[2](Kp)
            return -(utility(profit) + beta*EV)
        end
        resM   = optimize(fM_star, M_lo, M_hi, Brent(); rel_tol=1e-6)
        M_star = Optim.minimizer(resM)

        # store results
        best_val         = -val_neg
        Vnext[iK,jtau]   = best_val
        policy_M[iK,jtau] = M_star
        policy_X[iK,jtau] = X_star

        # get error
        bellman_err = max(bellman_err, abs(best_val - V[iK,jtau]))
    end

    # update
    V .= Vnext
    if iter % 20 == 0
        @printf("iter %4d :  err = %.3e\n", iter, bellman_err)
    end
end

println("Converged in $iter iterations. Final error: $bellman_err")

#───────────────────────────────────────────────────────────────────────────────
# 5.  Display average policies
#───────────────────────────────────────────────────────────────────────────────

avg_M = [mean(policy_M[:, s]) for s in 1:n_tau]
avg_X = [mean(policy_X[:, s]) for s in 1:n_tau]

println("\nAverage optimal imports and investment:")
for s in 1:n_tau
    @printf("  τ = %.3f :  M̅ = %.4f   X̅ = %.4f\n",
            tau_grid[s], avg_M[s], avg_X[s])
end

#───────────────────────────────────────────────────────────────────────────────
# 6.  Plot results
#───────────────────────────────────────────────────────────────────────────────
n_points = 15
policy_Kp = [ k_prime(K_grid[i], policy_M[i,s], policy_X[i,s])
              for i in 1:nK, s in 1:n_tau ]

pM = plot(
    K_grid, policy_M[:,1], 
    label="M*(K, τ_L)",
    xlabel="Capital", 
    ylabel="Imports",
    title="Import Policy")
plot!(pM, K_grid, 
      policy_M[:,2], 
      label="M*(K, τ_H)", ls=:dash)
savefig(pM, "plots/imports.png")

pX = plot(
    K_grid[1:n_points], policy_X[1:n_points,1], 
    label="X*(K, τ_L)",
    xlabel="Capital", 
    ylabel="Investment", 
    title="Investment Policy")
plot!(pX, K_grid[1:n_points], 
      policy_X[1:n_points,2], 
      label="X*(K, τ_H)", ls=:dash)
savefig(pX, "plots/investment.png")

pKp = plot(
    K_grid[1:n_points], policy_Kp[1:n_points, 1],
    label   = "K'*(K, τ = $(tau_grid[1]))",
    xlabel  = "Current Capital K",
    ylabel  = "Next-period Capital K'",
    title   = "Capital Policy",
    lw      = 2
)
plot!(
    pKp,
    K_grid[1:n_points], policy_Kp[1:n_points, 2],
    label = "K'*(K, τ = $(tau_grid[2]))",
    ls    = :dash,
    lw    = 2)
savefig(pKp, "plots/capital_policy.png")

pV = plot(K_grid, V[:,1], label="V(K, τ_L)",
          xlabel="Capital", ylabel="Value", title="Value Function")
plot!(pV, K_grid, V[:,2], label="V(K, τ_H)", ls=:dash)
savefig(pV, "plots/value_function.png")

# ───────────────────────────────────────────────────────────────────────────────
# Find convergence to steady states
# ───────────────────────────────────────────────────────────────────────────────

tau_grid = [0.0, 0.4]
τ = 0.4
jtau = findfirst(==(τ), tau_grid)

sM = LinearInterpolation(K_grid, policy_M[:,jtau], extrapolation_bc=Line())
sX = LinearInterpolation(K_grid, policy_X[:,jtau], extrapolation_bc=Line())

# root finding
fK(K) = (1 - delta)*K + sX(K) + psi*sM(K) - K
K_star = find_zero(fK, (minimum(K_grid), maximum(K_grid)))
M_star = sM(K_star)
X_star = sX(K_star)

@printf("Policy steady-state at τ=%.2f → K* = %.4f,  M* = %.4f,  X* = %.4f\n",
        τ, K_star, M_star, X_star)

function simulate(K0; T=50)
    K_path = zeros(T+1); M_path = zeros(T); X_path = zeros(T)
    K_path[1] = K0
    for t in 1:T
        K = K_path[t]
        M = sM(K)
        X = sX(K)
        M_path[t] = M; X_path[t] = X
        K_path[t+1] = (1 - delta)*K + X + psi*M
    end
    return K_path, M_path, X_path
end

# change initial capital
K0 = maximum(K_grid)
T  = 50
K_path, M_path, X_path = simulate(K0, T=T)

# plot
ts = 0:T
p1 = plot(ts, K_path, lw=2, label="Kₜ",
          xlabel="t", ylabel="K", title="Convergence of Kₜ")
hline!([K_star], ls=:dash, lw=2, label="K*")
savefig(p1, "plots/converge_K_exact.png")

p2 = plot(1:T, M_path, lw=2, label="Mₜ",
          xlabel="t", ylabel="M", title="Convergence of Mₜ")
hline!([M_star], ls=:dash, lw=2, label="M*")
savefig(p2, "plots/converge_M_exact.png")

p3 = plot(1:T, X_path, lw=2, label="Xₜ",
          xlabel="t", ylabel="X", title="Convergence of Xₜ")
hline!([X_star], ls=:dash, lw=2, label="X*")
savefig(p3, "plots/converge_X_exact.png")


# ───────────────────────────────────────────────────────────────────────────────
# Optimal Dynamic Paths
# ───────────────────────────────────────────────────────────────────────────────

# redefine spline to avoid errors
makespline(a::AbstractVector, b::AbstractVector) = 
    LinearInterpolation(a, b, extrapolation_bc = Line())

# arrays of callable interpolants
n_tau = length(tau_grid)
m_interp = [ makespline(K_grid, policy_M[:,s]) for s in 1:n_tau ]
x_interp = [ makespline(K_grid, policy_X[:,s]) for s in 1:n_tau ]

K_star = zeros(n_tau)
for s in 1:n_tau
    fK(K) = (1 - delta)*K + x_interp[s](K) + psi*m_interp[s](K) - K
    K_star[s] = find_zero(fK, (minimum(K_grid), maximum(K_grid)))
end

T, T_plot = 1000, 100
rng = MersenneTwister(42) # random number generator

k_path = zeros(T+1)
m_path = zeros(T)
x_path = zeros(T)
y_path = zeros(T)
state  = zeros(Int, T+1)

# initial values
k_path[1] = 2.0
state[1]  = rand(rng) < 0.5 ? 1 : 2

for t in 1:T
    K_cur = k_path[t]
    s_cur = state[t]

    # call to find optimal values
    M_star = m_interp[s_cur](K_cur)
    X_star = x_interp[s_cur](K_cur)

    y_path[t] = K_cur^alpha * ((1-psi)M_star)^gamma
    m_path[t] = M_star
    x_path[t] = X_star

    # law of motion
    k_path[t+1] = (1 - delta)*K_cur + X_star + psi*M_star

    # tariff transition
    u = rand(rng)
    state[t+1] = (u < P_tau[s_cur,1]) ? 1 : 2
end

# plot
ts = 1:T_plot
p = plot(ts, k_path[1 .+ ts], label="Kₜ", lw=2,
         xlabel="t", ylabel="Level", title="Simulated Paths")
plot!(p, ts, m_path[ts], label="Mₜ", lw=2)
plot!(p, ts, x_path[ts], label="Xₜ", lw=2)
plot!(p, ts, y_path[ts], label="Yₜ", lw=2)
for s in 1:n_tau
    hline!(p, [K_star[s]],
           label="K* (τ=$(tau_grid[s]))", ls=:dash, lw=2)
end
plot!(legend=:topright, grid=true)
savefig(p, "plots/simulated_paths.png")

plt_state = plot(ts, state[ts],
    xlabel="t",
    ylabel="Tariff state",
    yticks=( [1,2], ["Low","High"] ),
    markershape=:circle,
    markerstrokewidth=0,
    legend=false,
    title="Tariff State over Time (first $T_plot periods)" )
hline!([1.5], ls=:dash, alpha=0.3)   # optional midpoint guide
savefig(plt_state, "plots/state_transitions.png")