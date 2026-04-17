import numpy as np

import qics
from qics.quantum import p_tr
from qics.vectorize import lin_to_mat, vec_dim, mat_to_vec


def hbe_dmcv_qics(G_kraus, Z_kraus, bases_AB, alice_amp, p_sim, pK, alpha, eps, Nc, N):
    """Compute the hbe only with all the cones and linear constraints as per Eq. (44)."""

    ϵcompPE = 9e-11
    num_q = 4 * 6
    γ = 1 / (2 - 1 / alpha)
    sγ = -1 if γ < 1 else 1

    dim_rho = G_kraus.shape[1]
    dim_sigma = Z_kraus[1].shape[1]
    dim_out = G_kraus.shape[0]

    v_in_rho  = vec_dim(dim_rho, iscomplex=True, compact=True)
    v_in_sigma  = vec_dim(dim_sigma, iscomplex=True, compact=True)
    v_out = vec_dim(dim_out, iscomplex=True, compact=False)

    # variables (u, ωAB, σSAB)
    c = np.zeros((1 + v_in_rho + v_in_sigma, 1))

    # * Rényi cone constraint

    Id = np.eye(dim_out)
    def Gmap(X):
        return (1 - eps) * G_kraus @ X @ G_kraus.conj().T + eps * Id

    def Zmap(X):
        Z = np.zeros_like(Z_kraus[0], dtype=np.complex128)
        for Zi in Z_kraus:
            Z += Zi @ X @ Zi.conj().T
        return Z

    Gmat = lin_to_mat(Gmap, (dim_rho, dim_out), iscomplex=True, compact=(True, False))
    Zmat = lin_to_mat(Zmap, (dim_sigma, dim_out), iscomplex=True, compact=(True, False))

    G_block = np.block([
        [ -1.0              ,    np.zeros((1, v_in_rho)),   np.zeros((1, v_in_sigma))   ],  # u = 1
        [np.zeros((v_out,1)),    -Gmat,                 np.zeros_like(Zmat)  ],  # X = G(ωAB)
        [np.zeros((v_out,1)),    np.zeros_like(Gmat),   -Zmat                ],  # Y = ZG(σSAB)
    ])  # (18433, 577)

    h = np.block([
        [0.0              ], 
        [np.zeros((v_out, 1))], 
        [np.zeros((v_out, 1))],
    ])

    gamma = 1 / (2 - 1 / alpha) 
    cones = [qics.cones.SandQuasiEntr(dim_out, gamma, iscomplex=True)]


    # * Constraint Tr[σSAB] = 1

    tr_σSAB = lin_to_mat(lambda X: np.trace(X), (dim_sigma, 1), compact=(True, True), iscomplex=True)
    A = np.block([
        [np.zeros((1, 1 + v_in_rho)), tr_σSAB]
        ])
    b = np.array([[1]])

    # * Constraint on the marginal state Tr_B[ωAB] == ωA

    ωA = lin_to_mat(lambda X: p_tr(X,  (4, Nc + 1), 1), (4 * (Nc + 1), 4), iscomplex=True, compact=(True, True))
    A = np.block([
        [A],
        [np.zeros((ωA.shape[0], 1)), ωA, np.zeros((ωA.shape[0], v_in_sigma))]  # p_tr(ωAB) = ωA * ωAB = alice_amp
        ])  
    alice_amp_vec = mat_to_vec(alice_amp, compact=True)
    b = np.vstack([b, alice_amp_vec.reshape((-1, 1))])

    # * Finite bounds via a Bretagnolle-Huber-Carol estimator

    # variables (u, ωAB, σSAB, vec(w), wk, vec(q), qk)
    c = np.vstack([c, np.zeros((2 * num_q + 2, 1))])

    δ = np.sqrt((2 * 25 * np.log(2) - 2 * np.log(ϵcompPE)) / N)

    G_block = np.block([ 
        [G_block, np.zeros((G_block.shape[0], 2 * num_q + 2))],
        [np.zeros((num_q + 1, G_block.shape[1])), -np.eye(num_q + 1),  np.eye(num_q + 1)],  # 0 < w - q
        [np.zeros((num_q + 1, G_block.shape[1])), -np.eye(num_q + 1),  -np.eye(num_q + 1)]  # 0 < w + q
    ]) 

    h = np.vstack([h, (1 - pK) * p_sim, [pK], -(1 - pK) * p_sim, [-pK]])

    A = np.block([
        [A,                            np.zeros((A.shape[0], 2 * num_q + 2)) ],
        [np.zeros((1, A.shape[1])),    np.ones((1, num_q + 1)), np.zeros((1, num_q + 1))]  # sum(w) = δ
    ]) 

    b = np.vstack([b, [δ]])  # sum(w) = δ

    cones.append(qics.cones.NonNegOrthant(num_q + 1))
    cones.append(qics.cones.NonNegOrthant(num_q + 1))

    # * Constraints on probabilities
    A = np.block([
        [A  ],
        [np.zeros((1, A.shape[1] - num_q - 1 )), np.ones((1, num_q + 1))]  # sum(q) = 1
    ]) 

    b = np.vstack([b, [1.0]])  # sum(q) = 1


    # * Constraints on exp vals via KL divergence

    # variables (u, ωAB, σSAB, vec(w), wk, vec(q), qk, h_KL)
    c = np.vstack([c, [alpha / ( (alpha - 1) * np.log(2) )]])

    p_ωAB = []
    for basis_mat in bases_AB:
        constraint_probabilities_dmcv = lin_to_mat(lambda X: np.real(np.trace(X.conj().T @ basis_mat)), (dim_rho, 1), compact=(True, True), iscomplex=True)
        p_ωAB.append((1 - pK) * constraint_probabilities_dmcv)
    p_ωAB = np.vstack(p_ωAB)

    # The cone ordering is different to Hypatia
    G_block = np.block([
        [G_block, np.zeros((G_block.shape[0], 1))],
        [np.zeros((1, G_block.shape[1])), np.array([[-1.0]])],  # constraint h_KL
        [np.zeros((num_q + 1, 1 + v_in_rho + v_in_sigma + num_q + 1)), -np.eye(num_q + 1), np.zeros((num_q + 1, 1))],  # constraint q, qk
        [np.zeros((p_ωAB.shape[0], 1)), -p_ωAB, np.zeros((p_ωAB.shape[0], G_block.shape[1] - p_ωAB.shape[1]))],  # constraint p_ωAB
        [np.zeros((1, G_block.shape[1] + 1))], # constraint p_k
    ]) 
    h = np.vstack([h, np.array([[0.0]]), np.zeros((num_q + 1, 1)), np.zeros((num_q, 1)), np.array([[pK]])])
    A = np.block([
        [A, np.zeros((A.shape[0], 1)) ],   # add new variable h_KL
    ])

    cones.append(qics.cones.ClassRelEntr(num_q + 1))

    # * Exponential cone

    # variables (u, ωAB, σSAB, vec(w), wk, vec(q), qk, h_KL, h_QKD)
    c = np.vstack([c, [(pK - δ) / (np.log(2)*(γ - 1))]])  # add coef of new variable h_QKD

    # The cone ordering is different to Hypatia
    G_block = np.block([
        [G_block, np.zeros((G_block.shape[0], 1))],
        [np.zeros((1, G_block.shape[1])), np.array([[1.0]])],  # -h_QKD in Cone
        [np.array([[-sγ]]), np.zeros((1, G_block.shape[1]))],  # sγ * u in Cone
        [np.zeros((1, G_block.shape[1] + 1))],  # 1 in Cone
    ]) 

    h = np.vstack([h, np.array([[0.0], [0.0], [1.0]])]) # 1 in Cone
    A = np.block([
        [A, np.zeros((A.shape[0], 1)) ],  # add new variable h_QKD
    ]) 

    cones.append(qics.cones.ClassEntr(1))

    # * Solve the cone

    model = qics.Model(A=A, b=b, h=h, c=c, G=G_block, cones=cones)
    solver = qics.Solver(model, max_time=50000, tol_gap=1e-6)
    info = solver.solve()
    p_obj = info["p_obj"]
    # d_obj = info["d_obj"]
    gap = info["opt_gap"]
    num_iter = info["num_iter"] 
    solve_time = info["solve_time"]
    sol_status = info["sol_status"]
    exit_status = info["exit_status"]
    return p_obj, gap, num_iter, solve_time, sol_status, exit_status


if __name__ == "__main__":
    
    import numpy as np
    from qics.vectorize import mat_to_vec

    from julia.api import Julia
    jl = Julia(compiled_modules=False)

    from julia import Main
    Main.include("Instance_dmcv.jl")

    # Parameters
    eps = 1e-8     # depolarization
    N = 1e7
    Nc = 5
    f = 1.0
    Δs = 1.5
    Δ = 4.0
    L = 13

    alpha = 1.0 + Main.optimal_renyi(f, N, L)
    amplitude = Main.optimal_amp(f, L)
    alice_amp = np.array(Main.alice_part(amplitude))
    pK = Main.optimal_pK(f, N, L)
    p_sim = mat_to_vec(Main.simulated_probabilities_dmcv(Δs, Δ, amplitude, L))
    
    R_B = Main.test_basis_dmcv(Nc, Δs, Δ)
    bases_AB = [Main.kron(Main.proj(x + 1, 4), R_B[z]) for x in range(0,4) for z in range(0,6)]
    G_kraus = np.array(Main.gkraus(Main.Float64, Nc)[0])
    Z_kraus = [np.array(Z) for Z in Main.zkraus(Nc)]
    
    h_renyi, gap, num_iter, solve_time, sol_status, exit_status = hbe_dmcv_qics(G_kraus, Z_kraus, bases_AB, alice_amp, p_sim, pK, alpha, eps, Nc, N)
