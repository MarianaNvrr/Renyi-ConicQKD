using LinearAlgebra
using JuMP
using ConicQKD # ] add https://github.com/araujoms/ConicQKD.jl.git
using Ket
using Optim
import Hypatia
import Hypatia.Cones
# import JLD2
using Printf
using Parameters

include("Utils_data_bb84.jl")

@with_kw struct epsilon_coeffs{T<:AbstractFloat}
    # Parameters set for variable length computation. Change otherwise.
    ϵEC::T = 1/2*10e-80
    ϵPA::T = 1/2*10e-80
    ϵcompPE::T = 10e-9
end

"Alice state after depolarization"
function alice_depol(v, dAL)
    dA = 2
    ω1 = kron(proj(1, dA), proj(1, dAL)) + kron(proj(2, dA), proj(2, dAL))
    ω2 = kron(proj(1, dA), proj(2, dAL)) + kron(proj(2, dA), proj(1, dAL))
    ω01 = kron(ket(1, dA) * ket(2, dA)', ket(1, dAL) * ket(2, dAL)')
    ω10 = kron(ket(2, dA) * ket(1, dA)', ket(2, dAL) * ket(1, dAL)')
    ω = ((1 + v) / 2 * ω1 + (1 - v) / 2 * ω2 + v * (ω01 + ω10)) / 2
    return ω
end

"Alice state after depolarization and losses"
function alice_depol_loss(v::T, η::T) where {T<:AbstractFloat}
    dA = 2 ; dAL = 3
    ω = η * alice_depol(v, 3) + (1 - η) * kron(I(dA), proj(3, dAL)) / 2
    return ω
end

"Alice's measurements"
function alice_povm()
    PZ = [proj(1, 2), proj(2, 2)]
    PX = 0.5 * [[1 1; 1 1], [1 -1; -1 1]]
    return vcat(PZ, PX)
end

"Bob's measurements"
function bob_povm(pK::T) where {T<:AbstractFloat}
    QZ = pK * [[1 0 0; 0 0 0; 0 0 0], [0 0 0; 0 1 0; 0 0 0]]
    QX = (1 - pK) * 0.5 * [[1 1 0; 1 1 0; 0 0 0], [1 -1 0; -1 1 0; 0 0 0]]
    Q = [proj(3, 3)]
    return vcat(QZ, QX, Q)
end

"Full Alice's and Bob's POVM"
function POVM_AB(pK::T) where {T<:AbstractFloat}
    A = alice_povm()
    B = bob_povm(pK)
    povm = [kron(a, b) for a ∈ A for b ∈ B]

    return povm
end

"Leakage"
function EC_cost_bb84(qber::T, η::T, f::T, pK::T,ϵEC::T,N::T) where {T<:AbstractFloat}
    # H(A|B) 
    leak_EC = binary_entropy(qber)
    leak_EC *= f * η * pK^2                 # EC efficiency and pK
    leak_EC += ceil(log2(inv(ϵEC)))/N   # Correctness cost added in finite corrections
    return leak_EC
end

"QBER for the Z basis"
function qberZ(v::T, η::T, pK::T) where {T<:AbstractFloat}
    A = alice_povm() ; B = bob_povm(pK)
    ω = alice_depol_loss(v, η)
    p_error = sum(real(dot(kron(A[i], B[j]), ω)) for i ∈ 1:2, j ∈ 1:2 if i != j)
    p_click = sum(real(dot(kron(A[i], B[j]), ω)) for i ∈ 1:2, j ∈ 1:2)
    return p_error / p_click
end

"PE correlations with simulated state"
function PE_probabilities_bb84(v::T, η::T, pK::T) where {T<:AbstractFloat}
    ω = alice_depol_loss(v, η)
    n = size(POVM_AB(pK), 1)
    expval = [real(dot(ω, POVM_AB(pK)[i])) for i ∈ (div(n, 2)+1):n]
    return expval
end

"PE correlations with constraint state"
function constraint_probabilities_bb84(ω::AbstractMatrix, pK::T) where {T<:AbstractFloat}
    n = size(POVM_AB(pK), 1)
    return real(dot.(Ref(ω), POVM_AB(pK)[div(n, 2)+1:n]))
end


########  CONIC PROGRAMS

"Fixed-length objective function"
function FixedLenghtOBJ(
    α::T,
    v::T,
    η::T,
    N::T,
    pK::T,
    ϵcompPE::T;
    fast::Bool = true
) where {T<:AbstractFloat}
    dimA = 2 ; dimB = 3
    d = dimA * dimB
    n = size(POVM_AB(pK), 1)

    model = GenericModel{T}()

    # Variables
    @variable(model, ωAB[1:d, 1:d], Hermitian)
    @variable(model, qK)
    @variable(model, q[1:div(n, 2)])
    @variable(model, h_QKD)
    @variable(model, h_KL)

    # Constraints on the state
    @constraint(model, partial_trace(ωAB, 2, [2, 3]) == Hermitian(I(2) / 2))

    # Constraints on probabilities
    @constraint(model, sum(q) + qK == 1)

    # Constraints on exp vals via KL divergence
    p_ωAB = constraint_probabilities_bb84(ωAB, pK) * (1 - pK) # PE probabilities

    # Finite bounds via a Bretagnolle-Huber-Carol estimator 
    C_alphbet = 13 # {perp} U {(0,1) x ((X,Z) x (0,1,perp))}
    δ = sqrt((2 * C_alphbet * log(T(2)) - 2 * log(ϵcompPE)) / N)

    p_sim = PE_probabilities_bb84(v, η, pK) * (1 - pK)
    @constraint(model, [δ; q - p_sim; qK - pK] in Hypatia.EpiNormInfCone{T,T}(1 + 1 + length(q), true))

    # Key map used by both cones
    Ghat_top = [sqrt(pK) * kron(I(2), ket(1, 2) * ket(1, 3)' + ket(2, 2) * ket(2, 3)')]

    vec_dim = Cones.svec_length(Complex, d)
    ωAB_vec = svec(ωAB)

    # Conic program
    if fast
        β = inv(α)
        S = I(6)
        @variable(model, u)
        ZGhat_top = [sqrt(pK) * kron(proj(i, 2), ket(1, 2) * ket(1, 3)' + ket(2, 2) * ket(2, 3)') for i ∈ 1:2]
        @constraint(model, [u; ωAB_vec] in EpiFastRenyiQKDTriCone{T,Complex{T}}(β, Ghat_top, ZGhat_top, 1 + vec_dim; S))
        sβ = β < 1 ? -1 : 1
        @constraint(
            model,
            [h_QKD, qK, pK * (sβ * u + 1 - real(tr(Ghat_top[1] * ωAB * Ghat_top[1]')))] in MOI.ExponentialCone()
        )
        @constraint(model, [h_KL; p_ωAB; q] in Hypatia.EpiRelEntropyCone{T}(1 + 2 * length(q), false))
        @objective(model, Min, α * inv(log(T(2)) * (α - 1)) * (h_KL - h_QKD))
    else
        γ = α * inv(2α - 1)
        @variable(model, uTop)
        @variable(model, uBot)
        @variable(model, ψAB[1:d*3, 1:d*3], Hermitian)

        @constraint(model, tr(ψAB) == 1)
        ψ_vec = svec(ψAB)

        # cone for click events
        Zhat_top = [kron(ket(r, 2) * ket(r, 3)', I(6)) for r ∈ 1:2]
        STop = kron(sum(kron(ket(i, 2), proj(i, 2)) for i ∈ 1:2), sum(ket(i, 3) * ket(i, 2)' for i ∈ 1:2))
        @constraint(
            model,
            [uTop; ωAB_vec; ψ_vec] in
            EpiRenyiQKDTriCone{T,Complex{T}}(γ, Ghat_top, Zhat_top, 1 + length(ωAB_vec) + length(ψ_vec); S = STop)
        )

        # cone for no-click events
        Ghat_bottom = [kron(I(2), sqrt(1 - pK) * (proj(1, 3) + proj(2, 3)) + proj(3, 3))]
        Zhat_bottom = [kron(ket(3, 3)', I(6))]
        SBot = I(6)
        @constraint(
            model,
            [uBot; ωAB_vec; ψ_vec] in EpiRenyiQKDTriCone{T,Complex{T}}(
                γ,
                Ghat_bottom,
                Zhat_bottom,
                1 + length(ωAB_vec) + length(ψ_vec);
                S = SBot
            )
        )

        sγ = γ < 1 ? -1 : 1
        @constraint(model, [h_QKD, 1, sγ * (uTop + uBot)] in MOI.ExponentialCone())
        @constraint(model, [h_KL; p_ωAB; pK; q; qK] in Hypatia.EpiRelEntropyCone{T}(1 + 2 + 2 * length(q), false))
        @objective(model, Min, ((α / (α - 1)) * h_KL + (pK - δ) * h_QKD / (γ - 1)) / log(T(2)))
    end

    # Optimize
    set_optimizer(model, Hypatia.Optimizer{T})
    set_attribute(model, "verbose", true)
    optimize!(model)

    # Extract results
    h_renyi = objective_value(model)

    return h_renyi
end


function TradeoffFunction(
    α::T,
    q_honest::Vector{T}
    ) where {T<:AbstractFloat}

    # Dimensions of registers
    dimA = 2
    dimB = 3
    d = dimA * dimB
    pK = q_honest[1]
    s = div(length(POVM_AB(pK)),2)
    
    # Construct the model
    model = GenericModel{T}()

    # Variables
    @variable(model, λ_QKD[1:1+s])
    @variable(model, ωAB[1:d, 1:d], Hermitian)
    @variable(model, h_QKD)
    @variable(model, h_KL)
    @variable(model, u)
    # Alice's marginal
    @constraint(model, partial_trace(ωAB, 2, [2, 3]) == Hermitian(I(2) / 2))

    # KL divergence
    p_ωAB = constraint_probabilities_bb84(ωAB, pK) * (1 - pK) # PE probabilities
    p = vcat(pK, p_ωAB )
    @constraint(model, [h_KL; p; λ_QKD] ∈ Hypatia.EpiRelEntropyCone{T}(1+2*(1+s),false))

    # Slack constraints
    f_primal = @constraint(model, λ_QKD == q_honest)


    # Objective function
    ####################

    # Key map used by both cones
    Ghat_top = [sqrt(pK) * kron(I(2), ket(1, 2) * ket(1, 3)' + ket(2, 2) * ket(2, 3)')]
    ZGhat_top = [sqrt(pK) * kron(proj(i, 2), ket(1, 2) * ket(1, 3)' + ket(2, 2) * ket(2, 3)') for i ∈ 1:2]
    S = I(6)
    vec_dim = Cones.svec_length(Complex, d)
    ωAB_vec = svec(ωAB)
    β = 1/α
    @constraint(model, [u; ωAB_vec] in EpiFastRenyiQKDTriCone{T,Complex{T}}(β, Ghat_top, ZGhat_top, 1 + vec_dim; S))
    sβ = β < 1 ? -1 : 1


    @constraint(model, [h_QKD, 1, sβ * u + 1 - real(tr(Ghat_top[1]*ωAB*Ghat_top[1]'))] ∈ MOI.ExponentialCone())
    
    @objective(model, Min, α * inv(log(T(2)) * (α - 1)) * (h_KL - pK*h_QKD))


    # Optimize
    ####################
    set_optimizer(model, Hypatia.Optimizer{T})
    set_attribute(model, "verbose", true)
    optimize!(model)

    # Failsafe against numerical failure
    if objective_value(model) < 0
        @error("Failure at finding tradeoff \n")
        return zeros(T, length(λ_QKD))
    else
        f_tradeoff = dual.(f_primal)
    end

    return f_tradeoff
end


function kappa(
        α::T,
        pK::T,
        f_tradeoff::Vector{T}
    ) where {T<:AbstractFloat}

    # Dimensions of registers
    dimA = 2
    dimB = 3
    d = dimA * dimB
    s = size(POVM_AB(pK), 1)
    ΠAB = POVM_AB(pK)[(div(s, 2)+1):s]

    # Coefficients of the weights
    coeff_Cbot   = pK*exp2(inv(α)*(α-1)*f_tradeoff[1])
    coeff_Ctilde = [(1-pK)* exp2(inv(α)*(α-1)*f_tradeoff[i+1])*ΠAB[i] for i ∈ 1:length(f_tradeoff)-1]
    
    # Construct the model
    model = GenericModel{T}()

    # Variables
    @variable(model, ωAB[1:d,1:d], Hermitian)
    @variable(model, u)
    
    # Alice's marginal
    @constraint(model, partial_trace(ωAB, 2, [2, 3]) == Hermitian(I(2) / 2))

    # PE weights
    CtildeWeightSum = sum([real(dot(coeff_c, ωAB)) for coeff_c ∈ coeff_Ctilde])

    # Objective function
    ####################

    # Key map used by both cones
    Ghat_top = [sqrt(pK) * kron(I(2), ket(1, 2) * ket(1, 3)' + ket(2, 2) * ket(2, 3)')]
    ZGhat_top = [sqrt(pK) * kron(proj(i, 2), ket(1, 2) * ket(1, 3)' + ket(2, 2) * ket(2, 3)') for i ∈ 1:2]
    blocks    = [1:size(ZGhat_top[1],1)]
    S = I(6)
    vec_dim = Cones.svec_length(Complex, d)
    ωAB_vec = svec(ωAB)

    β = 1/α
    @constraint(model, [u; ωAB_vec] in EpiFastRenyiQKDTriCone{T,Complex{T}}(β, Ghat_top, ZGhat_top, 1 + vec_dim; S,blocks))
    sβ = β < 1 ? -1 : 1

    # Objective function
    @objective(model, Max, CtildeWeightSum + coeff_Cbot*(sβ*u + 1 - real(tr(Ghat_top[1]*ωAB*Ghat_top[1]'))))

    set_optimizer(model, Hypatia.Optimizer{T})
    set_attribute(model, "verbose", true)
    optimize!(model)

    κ = α*inv(1-α)*log2(objective_value(model))
    return κ
end

#########################

"Finite corrections for the final key rate"
Finite_corrections(α::T, ϵPA::T) where {T<:AbstractFloat} =
    (log2(inv(ϵPA))) * α / (α - 1) - 2 

"Main function to compute key rate"
function Finite_SKR(
    dB::Integer,
    N::T,
    v::T,
    pK::T,
    f::T,
    α::T;
    fast::Bool = true,
    variable::Bool = true
) where {T<:AbstractFloat}

    # Load the epsilons
    @unpack ϵEC, ϵPA, ϵcompPE = epsilon_coeffs{T}()

    η = 10^(-T(dB) / 10)
    qZ = qberZ(v, η, pK)
    leak_EC = EC_cost_bb84(qZ, η, f, pK,ϵEC,N)
    α_low = T(1 + 1e-7);α_high = T(1.5)
    optimal_renyi = α
    if variable 
        println("Initiating variable-length optimization")
        if optimal_renyi == 1.
            println("Optimizing α...")
            obj = αrenyi -> -VariableLengthOBJ(αrenyi, pK, v, η, ϵPA, N)
            sol = Optim.optimize(obj, α_low, α_high, Brent())
            optimal_renyi =sol.minimizer[1]
            h_renyi = -sol.minimum
        else
            h_renyi = VariableLengthOBJ(α, pK,v, η, ϵPA,N)
        end
    else
        if N >=1e10
            println("Initiating fixed-length optimization in the asymptotic regime")
            leak_EC = EC_cost_bb84(qZ, η, f, pK,1.,N)
            obj = αrenyi -> -(FixedLenghtOBJ(αrenyi, v, η, N, pK, T(1.0); fast) - Finite_corrections(αrenyi, ϵPA) / N)
            sol = Optim.optimize(obj, α_low, α_high, Brent())
            optimal_renyi = sol.minimizer[1]
        else
            println("Initiating fixed-length optimization in the finite-size regime")
            if α==1.
            #  Optimization of α parameter 
            obj = αrenyi -> -(FixedLenghtOBJ(αrenyi, v, η, N, pK, ϵcompPE; fast) - Finite_corrections(αrenyi, ϵPA) / N)
            sol = Optim.optimize(obj, α_low, α_high, Brent())
            optimal_renyi = sol.minimizer[1]
            h_renyi = -sol.minimum
            else
                h_renyi = FixedLenghtOBJ(α, v, η, N, pK, ϵcompPE; fast) - Finite_corrections(α, ϵPA) / N
            end
        end
    end
    SKR_Max = h_renyi - leak_EC
    @printf("Optimum found for α-1 = %.5e giving a key rate of SKR = %.2e \n", optimal_renyi - 1, SKR_Max)
    return SKR_Max, optimal_renyi, leak_EC, h_renyi
end

"Variable-length objective function"
function VariableLengthOBJ(α::T, pK::T,v::T, η::T, ϵPA::T,N::T) where {T}
    p_PE = PE_probabilities_bb84(v, η, pK)*(1-pK)
    q_honest = vcat(pK,p_PE)
    f_tradeoff = TradeoffFunction(α,q_honest)
    κ = kappa(α,pK,f_tradeoff)
    h_renyi = dot(q_honest,f_tradeoff) + κ - Finite_corrections(α, ϵPA) / N
    return h_renyi
end

"Function to obtain and save the key rates"
function Instance_bb84(f::Real, N::Real, v::Real, ::Type{T}; fast::Bool = true, variable::Bool=true) where {T}
    # Enforce desired precision
    f = T(f)
    N = T(N)
    v = T(v)
    
    # Create output file
    RATE_BB84 = "ConicQKD2/examples/rebutal/Var3_bb84Rate_bb84_f" * string(f) * "_N1e" * string(count(==('0'), string(Int(N)))) * ".csv"
    file = open(RATE_BB84, "a")
    @printf(file, "f, N, nu \n")
    @printf(file, "%.2f, %.2f, %.2f \n", f, log10(N), v)
    @printf(file, "dB, pK, a-1, leakEC, SKR, dual \n")
    close(file)

    # Start main loop
    for dB ∈ 0:2:50# CHANGE accordingly
        @printf("Transmittance: %d ---------\n", dB)
        pK = T(optimal_pK(N,dB, variable))
        α = optimal_renyi(N,dB,variable)
        SKR, optimal_α, leak_EC, dual = Finite_SKR(dB, N, v, pK, f, α; fast, variable)

        # Record outputs
        file = open(RATE_BB84, "a")
        @printf(file, "%d, %.2f, %.8e, %.12f, %.8e, %.8e \n", dB, pK, optimal_α - T(1), leak_EC, SKR, dual)
        close(file)
    end
end

# Example values
# f=1.16; v = 0.97; N=1e9; T=Float64; fast = true; variable=true

