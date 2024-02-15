using JSON
using CairoMakie
using DataFrames

results = JSON.parsefile("/home/guilherme/2021_NEURALNET/results/results/Processed_results_VF.json")

results_dfs = [
    DataFrame(
        :domain => x["domain"],
        :sequence => i,
        :pos => 1:length(x["fasta_seq"]),
        :aa => map(y -> (isnothing(y) ? missing : string(y)), x["fasta_seq"]),
        :dssp_ss3 => map(y -> (isnothing(y) ? missing : string(y)), x["dssp_ss3"]),
        :dssp_phi => map(y -> (isnothing(y) ? missing : Float64(y)), x["dssp_phi"]),
        :dssp_psi => map(y -> (isnothing(y) ? missing : Float64(y)), x["dssp_psi"]),
        :tinti_ss3_prediction => map(y -> (isnothing(y) ? missing : string(y)), x["tinti_ss3_prediction"]),
        :tinti_phi_prediction => map(y -> (isnothing(y) ? missing : Float64(y)), x["tinti_phi_prediction"]),
        :tinti_psi_prediction => map(y -> (isnothing(y) ? missing : Float64(y)), x["tinti_psi_prediction"]),
    )
    for (i, x) in enumerate(results)
]

merged_results_df = dropmissing(vcat(results_dfs...))
transform!(merged_results_df, :dssp_phi => (x -> map(y -> ((y > 180.0) ? (y - 360.0) : y), x)) => :dssp_phi; renamecols = false)
transform!(merged_results_df, :dssp_psi => (x -> map(y -> ((y > 180.0) ? (y - 360.0) : y), x)) => :dssp_psi; renamecols = false)
transform!(merged_results_df, :tinti_phi_prediction => (x -> map(y -> ((y > 180.0) ? (y - 360.0) : y), x)) => :tinti_phi_prediction; renamecols = false)
transform!(merged_results_df, :tinti_psi_prediction => (x -> map(y -> ((y > 180.0) ? (y - 360.0) : y), x)) => :tinti_psi_prediction; renamecols = false)
subset!(merged_results_df, :dssp_phi => x -> x .!= 0.0)
subset!(merged_results_df, :dssp_psi => x -> x .!= 0.0)

# using KernelDensity
# B = kde((merged_results_df.dssp_phi, merged_results_df.dssp_psi))

fig = Figure(; size = (1200,1000))
ax = Axis(fig[1,1], xlabel = "PHI", ylabel = "PSI", xticks = [-180.0, -90.0, 0.0, 90.0, 180.0], xticklabelsize = 20, yticklabelsize = 20, xlabelsize=30, ylabelsize=30, limits = ((-180.0, +180.0), (-180.0, +180.0)))
sc = scatter!(ax, merged_results_df.dssp_phi, merged_results_df.dssp_psi, color = (:darkgreen, 0.08))
Label(fig[0, :], "Todos os aminoacidos do fold 10 de validacao", fontsize = 30, justification = :center, tellheight=true, tellwidth = false)
# Ramachandran plots

save("FIG_RAMACHANDRAN_REF.png", fig)

# unique(merged_results_df.aa)
# 21-element Vector{String}:
# "N" - Asparagina - OK
# "A" - Alanina - OK
# "I" - Isoleucina - OK
# "D" - Acido Aspartico - OK
# "P" - Prolina - OK
# "R" - Arginina - OK
# "E" - Acido Glutamico - OK
# "L" - Leucina - OK
# "G" - Glicina - OK
# "V" - Valina - OK
# "K" - Lisina - OK
# "S" - Serina - OK
# "T" - Threonina - OK
# "Y" - Tirosina - OK
# "F" - Fenilalanina - 
# "H" - Histidina - OK
# "M" - Methionina - OK
# "W" - Triptofano - OK
# "Q" - Glutamina - OK
# "C" - Cisteina - OK
# "X" - Aminoacido Desconhecido

fig = Figure(; size = (2000,2600))
G_subfig = GridLayout(fig[1,1])
A_subfig = GridLayout(fig[1,2])
F_subfig = GridLayout(fig[1,3])
M_subfig = GridLayout(fig[2,1])
N_subfig = GridLayout(fig[2,2])
D_subfig = GridLayout(fig[2,3])
R_subfig = GridLayout(fig[3,1])
Q_subfig = GridLayout(fig[3,2])
E_subfig = GridLayout(fig[3,3])
K_subfig = GridLayout(fig[4,1])
L_subfig = GridLayout(fig[4,2])
I_subfig = GridLayout(fig[4,3])
H_subfig = GridLayout(fig[5,1])
S_subfig = GridLayout(fig[5,2])
C_subfig = GridLayout(fig[5,3])
P_subfig = GridLayout(fig[6,1])
W_subfig = GridLayout(fig[6,2])
Y_subfig = GridLayout(fig[6,3])
V_subfig = GridLayout(fig[7,1])
T_subfig = GridLayout(fig[7,2])
X_subfig = GridLayout(fig[7,3])

# Glicina (G)
    G_subdf = subset(merged_results_df, :aa => x -> x .== "G")
    axG1 = Axis(G_subfig[1,1], xlabel = "PHI", ylabel = "PSI", xticks = [-180.0, -90.0, 0.0, 90.0, 180.0], xlabelsize=20, ylabelsize=20, limits = ((-180.0, +180.0), (-180.0, +180.0)))
    axG2 = Axis(G_subfig[1,2], xlabel = "PHI", ylabel = "PSI", xticks = [-180.0, -90.0, 0.0, 90.0, 180.0], xlabelsize=20, ylabelsize=20, limits = ((-180.0, +180.0), (-180.0, +180.0)))
    scG1 = scatter!(axG1, G_subdf.dssp_phi, G_subdf.dssp_psi, color = (:red, 0.3))
    scG2 = scatter!(axG2, G_subdf.tinti_phi_prediction, G_subdf.tinti_psi_prediction, color = (:blue, 0.3))
    Label(G_subfig[0, :], "Glicina (G)", fontsize = 30, justification = :center)

# Alanina (A)
    A_subdf = subset(merged_results_df, :aa => x -> x .== "A")
    axA1 = Axis(A_subfig[1,1], xlabel = "PHI", ylabel = "PSI", xticks = [-180.0, -90.0, 0.0, 90.0, 180.0], xlabelsize=20, ylabelsize=20, limits = ((-180.0, +180.0), (-180.0, +180.0)))
    axA2 = Axis(A_subfig[1,2], xlabel = "PHI", ylabel = "PSI", xticks = [-180.0, -90.0, 0.0, 90.0, 180.0], xlabelsize=20, ylabelsize=20, limits = ((-180.0, +180.0), (-180.0, +180.0)))
    scA1 = scatter!(axA1, A_subdf.dssp_phi, A_subdf.dssp_psi, color = (:red, 0.3))
    scA2 = scatter!(axA2, A_subdf.tinti_phi_prediction, A_subdf.tinti_psi_prediction, color = (:blue, 0.3))
    Label(A_subfig[0, :], "Alanina (A)", fontsize = 30, justification = :center)

# Fenilalanina (F)
    F_subdf = subset(merged_results_df, :aa => x -> x .== "F")
    axF1 = Axis(F_subfig[1,1], xlabel = "PHI", ylabel = "PSI", xticks = [-180.0, -90.0, 0.0, 90.0, 180.0], xlabelsize=20, ylabelsize=20, limits = ((-180.0, +180.0), (-180.0, +180.0)))
    axF2 = Axis(F_subfig[1,2], xlabel = "PHI", ylabel = "PSI", xticks = [-180.0, -90.0, 0.0, 90.0, 180.0], xlabelsize=20, ylabelsize=20, limits = ((-180.0, +180.0), (-180.0, +180.0)))
    scF1 = scatter!(axF1, F_subdf.dssp_phi, F_subdf.dssp_psi, color = (:red, 0.3))
    scF2 = scatter!(axF2, F_subdf.tinti_phi_prediction, F_subdf.tinti_psi_prediction, color = (:blue, 0.3))
    Label(F_subfig[0, :], "Fenilalanina (F)", fontsize = 30, justification = :center)

# Methionina (M)
    M_subdf = subset(merged_results_df, :aa => x -> x .== "M")
    axM1 = Axis(M_subfig[1,1], xlabel = "PHI", ylabel = "PSI", xticks = [-180.0, -90.0, 0.0, 90.0, 180.0], xlabelsize=20, ylabelsize=20, limits = ((-180.0, +180.0), (-180.0, +180.0)))
    axM2 = Axis(M_subfig[1,2], xlabel = "PHI", ylabel = "PSI", xticks = [-180.0, -90.0, 0.0, 90.0, 180.0], xlabelsize=20, ylabelsize=20, limits = ((-180.0, +180.0), (-180.0, +180.0)))
    scM1 = scatter!(axM1, M_subdf.dssp_phi, M_subdf.dssp_psi, color = (:red, 0.3))
    scM2 = scatter!(axM2, M_subdf.tinti_phi_prediction, M_subdf.tinti_psi_prediction, color = (:blue, 0.3))
    Label(M_subfig[0, :], "Methionina (M)", fontsize = 30, justification = :center)

# Asparagina (N)
    N_subdf = subset(merged_results_df, :aa => x -> x .== "N")
    axN1 = Axis(N_subfig[1,1], xlabel = "PHI", ylabel = "PSI", xticks = [-180.0, -90.0, 0.0, 90.0, 180.0], xlabelsize=20, ylabelsize=20, limits = ((-180.0, +180.0), (-180.0, +180.0)))
    axN2 = Axis(N_subfig[1,2], xlabel = "PHI", ylabel = "PSI", xticks = [-180.0, -90.0, 0.0, 90.0, 180.0], xlabelsize=20, ylabelsize=20, limits = ((-180.0, +180.0), (-180.0, +180.0)))
    scN1 = scatter!(axN1, N_subdf.dssp_phi, N_subdf.dssp_psi, color = (:red, 0.3))
    scN2 = scatter!(axN2, N_subdf.tinti_phi_prediction, N_subdf.tinti_psi_prediction, color = (:blue, 0.3))
    Label(N_subfig[0, :], "Asparagina (N)", fontsize = 30, justification = :center)

# Acido Aspartico (D)
    D_subdf = subset(merged_results_df, :aa => x -> x .== "D")
    axD1 = Axis(D_subfig[1,1], xlabel = "PHI", ylabel = "PSI", xticks = [-180.0, -90.0, 0.0, 90.0, 180.0], xlabelsize=20, ylabelsize=20, limits = ((-180.0, +180.0), (-180.0, +180.0)))
    axD2 = Axis(D_subfig[1,2], xlabel = "PHI", ylabel = "PSI", xticks = [-180.0, -90.0, 0.0, 90.0, 180.0], xlabelsize=20, ylabelsize=20, limits = ((-180.0, +180.0), (-180.0, +180.0)))
    scD1 = scatter!(axD1, D_subdf.dssp_phi, D_subdf.dssp_psi, color = (:red, 0.3))
    scD2 = scatter!(axD2, D_subdf.tinti_phi_prediction, D_subdf.tinti_psi_prediction, color = (:blue, 0.3))
    Label(D_subfig[0, :], "Acido Aspartico (D)", fontsize = 30, justification = :center)

# Arginina (R)
    R_subdf = subset(merged_results_df, :aa => x -> x .== "R")
    axR1 = Axis(R_subfig[1,1], xlabel = "PHI", ylabel = "PSI", xticks = [-180.0, -90.0, 0.0, 90.0, 180.0], xlabelsize=20, ylabelsize=20, limits = ((-180.0, +180.0), (-180.0, +180.0)))
    axR2 = Axis(R_subfig[1,2], xlabel = "PHI", ylabel = "PSI", xticks = [-180.0, -90.0, 0.0, 90.0, 180.0], xlabelsize=20, ylabelsize=20, limits = ((-180.0, +180.0), (-180.0, +180.0)))
    scR1 = scatter!(axR1, R_subdf.dssp_phi, R_subdf.dssp_psi, color = (:red, 0.3))
    scR2 = scatter!(axR2, R_subdf.tinti_phi_prediction, R_subdf.tinti_psi_prediction, color = (:blue, 0.3))
    Label(R_subfig[0, :], "Arginina (R)", fontsize = 30, justification = :center)

# Glutamina (Q)
    Q_subdf = subset(merged_results_df, :aa => x -> x .== "Q")
    axQ1 = Axis(Q_subfig[1,1], xlabel = "PHI", ylabel = "PSI", xticks = [-180.0, -90.0, 0.0, 90.0, 180.0], xlabelsize=20, ylabelsize=20, limits = ((-180.0, +180.0), (-180.0, +180.0)))
    axQ2 = Axis(Q_subfig[1,2], xlabel = "PHI", ylabel = "PSI", xticks = [-180.0, -90.0, 0.0, 90.0, 180.0], xlabelsize=20, ylabelsize=20, limits = ((-180.0, +180.0), (-180.0, +180.0)))
    scQ1 = scatter!(axQ1, Q_subdf.dssp_phi, Q_subdf.dssp_psi, color = (:red, 0.3))
    scQ2 = scatter!(axQ2, Q_subdf.tinti_phi_prediction, Q_subdf.tinti_psi_prediction, color = (:blue, 0.3))
    Label(Q_subfig[0, :], "Glutamina (Q)", fontsize = 30, justification = :center)

# Acido Glutamico (E)
    E_subdf = subset(merged_results_df, :aa => x -> x .== "E")
    axE1 = Axis(E_subfig[1,1], xlabel = "PHI", ylabel = "PSI", xticks = [-180.0, -90.0, 0.0, 90.0, 180.0], xlabelsize=20, ylabelsize=20, limits = ((-180.0, +180.0), (-180.0, +180.0)))
    axE2 = Axis(E_subfig[1,2], xlabel = "PHI", ylabel = "PSI", xticks = [-180.0, -90.0, 0.0, 90.0, 180.0], xlabelsize=20, ylabelsize=20, limits = ((-180.0, +180.0), (-180.0, +180.0)))
    scE1 = scatter!(axE1, E_subdf.dssp_phi, E_subdf.dssp_psi, color = (:red, 0.3))
    scE2 = scatter!(axE2, E_subdf.tinti_phi_prediction, E_subdf.tinti_psi_prediction, color = (:blue, 0.3))
    Label(E_subfig[0, :], "Acido Glutamico (E)", fontsize = 30, justification = :center)

# Lisina (K)
    K_subdf = subset(merged_results_df, :aa => x -> x .== "K")
    axK1 = Axis(K_subfig[1,1], xlabel = "PHI", ylabel = "PSI", xticks = [-180.0, -90.0, 0.0, 90.0, 180.0], xlabelsize=20, ylabelsize=20, limits = ((-180.0, +180.0), (-180.0, +180.0)))
    axK2 = Axis(K_subfig[1,2], xlabel = "PHI", ylabel = "PSI", xticks = [-180.0, -90.0, 0.0, 90.0, 180.0], xlabelsize=20, ylabelsize=20, limits = ((-180.0, +180.0), (-180.0, +180.0)))
    scK1 = scatter!(axK1, K_subdf.dssp_phi, K_subdf.dssp_psi, color = (:red, 0.3))
    scK2 = scatter!(axK2, K_subdf.tinti_phi_prediction, K_subdf.tinti_psi_prediction, color = (:blue, 0.3))
    Label(K_subfig[0, :], "Lisina (K)", fontsize = 30, justification = :center)

# Leucina (L)
    L_subdf = subset(merged_results_df, :aa => x -> x .== "L")
    axL1 = Axis(L_subfig[1,1], xlabel = "PHI", ylabel = "PSI", xticks = [-180.0, -90.0, 0.0, 90.0, 180.0], xlabelsize=20, ylabelsize=20, limits = ((-180.0, +180.0), (-180.0, +180.0)))
    axL2 = Axis(L_subfig[1,2], xlabel = "PHI", ylabel = "PSI", xticks = [-180.0, -90.0, 0.0, 90.0, 180.0], xlabelsize=20, ylabelsize=20, limits = ((-180.0, +180.0), (-180.0, +180.0)))
    scL1 = scatter!(axL1, L_subdf.dssp_phi, L_subdf.dssp_psi, color = (:red, 0.3))
    scL2 = scatter!(axL2, L_subdf.tinti_phi_prediction, L_subdf.tinti_psi_prediction, color = (:blue, 0.3))
    Label(L_subfig[0, :], "Leucina (L)", fontsize = 30, justification = :center)

# Isoleucina (I)
    I_subdf = subset(merged_results_df, :aa => x -> x .== "I")
    axI1 = Axis(I_subfig[1,1], xlabel = "PHI", ylabel = "PSI", xticks = [-180.0, -90.0, 0.0, 90.0, 180.0], xlabelsize=20, ylabelsize=20, limits = ((-180.0, +180.0), (-180.0, +180.0)))
    axI2 = Axis(I_subfig[1,2], xlabel = "PHI", ylabel = "PSI", xticks = [-180.0, -90.0, 0.0, 90.0, 180.0], xlabelsize=20, ylabelsize=20, limits = ((-180.0, +180.0), (-180.0, +180.0)))
    scI1 = scatter!(axI1, I_subdf.dssp_phi, I_subdf.dssp_psi, color = (:red, 0.3))
    scI2 = scatter!(axI2, I_subdf.tinti_phi_prediction, I_subdf.tinti_psi_prediction, color = (:blue, 0.3))
    Label(I_subfig[0, :], "Isoleucina (I)", fontsize = 30, justification = :center)

# Histidina (H)
    H_subdf = subset(merged_results_df, :aa => x -> x .== "H")
    axH1 = Axis(H_subfig[1,1], xlabel = "PHI", ylabel = "PSI", xticks = [-180.0, -90.0, 0.0, 90.0, 180.0], xlabelsize=20, ylabelsize=20, limits = ((-180.0, +180.0), (-180.0, +180.0)))
    axH2 = Axis(H_subfig[1,2], xlabel = "PHI", ylabel = "PSI", xticks = [-180.0, -90.0, 0.0, 90.0, 180.0], xlabelsize=20, ylabelsize=20, limits = ((-180.0, +180.0), (-180.0, +180.0)))
    scH1 = scatter!(axH1, H_subdf.dssp_phi, H_subdf.dssp_psi, color = (:red, 0.3))
    scH2 = scatter!(axH2, H_subdf.tinti_phi_prediction, H_subdf.tinti_psi_prediction, color = (:blue, 0.3))
    Label(H_subfig[0, :], "Histidina (H)", fontsize = 30, justification = :center)

# Serina (S)
    S_subdf = subset(merged_results_df, :aa => x -> x .== "S")
    axS1 = Axis(S_subfig[1,1], xlabel = "PHI", ylabel = "PSI", xticks = [-180.0, -90.0, 0.0, 90.0, 180.0], xlabelsize=20, ylabelsize=20, limits = ((-180.0, +180.0), (-180.0, +180.0)))
    axS2 = Axis(S_subfig[1,2], xlabel = "PHI", ylabel = "PSI", xticks = [-180.0, -90.0, 0.0, 90.0, 180.0], xlabelsize=20, ylabelsize=20, limits = ((-180.0, +180.0), (-180.0, +180.0)))
    scS1 = scatter!(axS1, S_subdf.dssp_phi, S_subdf.dssp_psi, color = (:red, 0.3))
    scS2 = scatter!(axS2, S_subdf.tinti_phi_prediction, S_subdf.tinti_psi_prediction, color = (:blue, 0.3))
    Label(S_subfig[0, :], "Serina (S)", fontsize = 30, justification = :center)

# Cisteina (C)
    C_subdf = subset(merged_results_df, :aa => x -> x .== "C")
    axC1 = Axis(C_subfig[1,1], xlabel = "PHI", ylabel = "PSI", xticks = [-180.0, -90.0, 0.0, 90.0, 180.0], xlabelsize=20, ylabelsize=20, limits = ((-180.0, +180.0), (-180.0, +180.0)))
    axC2 = Axis(C_subfig[1,2], xlabel = "PHI", ylabel = "PSI", xticks = [-180.0, -90.0, 0.0, 90.0, 180.0], xlabelsize=20, ylabelsize=20, limits = ((-180.0, +180.0), (-180.0, +180.0)))
    scC1 = scatter!(axC1, C_subdf.dssp_phi, C_subdf.dssp_psi, color = (:red, 0.3))
    scC2 = scatter!(axC2, C_subdf.tinti_phi_prediction, C_subdf.tinti_psi_prediction, color = (:blue, 0.3))
    Label(C_subfig[0, :], "Cisteina (C)", fontsize = 30, justification = :center)

# Prolina (P)
    P_subdf = subset(merged_results_df, :aa => x -> x .== "P")
    axP1 = Axis(P_subfig[1,1], xlabel = "PHI", ylabel = "PSI", xticks = [-180.0, -90.0, 0.0, 90.0, 180.0], xlabelsize=20, ylabelsize=20, limits = ((-180.0, +180.0), (-180.0, +180.0)))
    axP2 = Axis(P_subfig[1,2], xlabel = "PHI", ylabel = "PSI", xticks = [-180.0, -90.0, 0.0, 90.0, 180.0], xlabelsize=20, ylabelsize=20, limits = ((-180.0, +180.0), (-180.0, +180.0)))
    scP1 = scatter!(axP1, P_subdf.dssp_phi, P_subdf.dssp_psi, color = (:red, 0.3))
    scP2 = scatter!(axP2, P_subdf.tinti_phi_prediction, P_subdf.tinti_psi_prediction, color = (:blue, 0.3))
    Label(P_subfig[0, :], "Prolina (P)", fontsize = 30, justification = :center)

# Triptofano (W)
    W_subdf = subset(merged_results_df, :aa => x -> x .== "W")
    axW1 = Axis(W_subfig[1,1], xlabel = "PHI", ylabel = "PSI", xticks = [-180.0, -90.0, 0.0, 90.0, 180.0], xlabelsize=20, ylabelsize=20, limits = ((-180.0, +180.0), (-180.0, +180.0)))
    axW2 = Axis(W_subfig[1,2], xlabel = "PHI", ylabel = "PSI", xticks = [-180.0, -90.0, 0.0, 90.0, 180.0], xlabelsize=20, ylabelsize=20, limits = ((-180.0, +180.0), (-180.0, +180.0)))
    scW1 = scatter!(axW1, W_subdf.dssp_phi, W_subdf.dssp_psi, color = (:red, 0.3))
    scW2 = scatter!(axW2, W_subdf.tinti_phi_prediction, W_subdf.tinti_psi_prediction, color = (:blue, 0.3))
    Label(W_subfig[0, :], "Triptofano (W)", fontsize = 30, justification = :center)

# Tirosina (Y)
    Y_subdf = subset(merged_results_df, :aa => x -> x .== "Y")
    axY1 = Axis(Y_subfig[1,1], xlabel = "PHI", ylabel = "PSI", xticks = [-180.0, -90.0, 0.0, 90.0, 180.0], xlabelsize=20, ylabelsize=20, limits = ((-180.0, +180.0), (-180.0, +180.0)))
    axY2 = Axis(Y_subfig[1,2], xlabel = "PHI", ylabel = "PSI", xticks = [-180.0, -90.0, 0.0, 90.0, 180.0], xlabelsize=20, ylabelsize=20, limits = ((-180.0, +180.0), (-180.0, +180.0)))
    scY1 = scatter!(axY1, Y_subdf.dssp_phi, Y_subdf.dssp_psi, color = (:red, 0.3))
    scY2 = scatter!(axY2, Y_subdf.tinti_phi_prediction, Y_subdf.tinti_psi_prediction, color = (:blue, 0.3))
    Label(Y_subfig[0, :], "Tirosina (Y)", fontsize = 30, justification = :center)

# Valiha (V)
    V_subdf = subset(merged_results_df, :aa => x -> x .== "V")
    axV1 = Axis(V_subfig[1,1], xlabel = "PHI", ylabel = "PSI", xticks = [-180.0, -90.0, 0.0, 90.0, 180.0], xlabelsize=20, ylabelsize=20, limits = ((-180.0, +180.0), (-180.0, +180.0)))
    axV2 = Axis(V_subfig[1,2], xlabel = "PHI", ylabel = "PSI", xticks = [-180.0, -90.0, 0.0, 90.0, 180.0], xlabelsize=20, ylabelsize=20, limits = ((-180.0, +180.0), (-180.0, +180.0)))
    scV1 = scatter!(axV1, V_subdf.dssp_phi, V_subdf.dssp_psi, color = (:red, 0.3))
    scV2 = scatter!(axV2, V_subdf.tinti_phi_prediction, V_subdf.tinti_psi_prediction, color = (:blue, 0.3))
    Label(V_subfig[0, :], "Valina (V)", fontsize = 30, justification = :center)

# Threonina (T)
    T_subdf = subset(merged_results_df, :aa => x -> x .== "T")
    axT1 = Axis(T_subfig[1,1], xlabel = "PHI", ylabel = "PSI", xticks = [-180.0, -90.0, 0.0, 90.0, 180.0], xlabelsize=20, ylabelsize=20, limits = ((-180.0, +180.0), (-180.0, +180.0)))
    axT2 = Axis(T_subfig[1,2], xlabel = "PHI", ylabel = "PSI", xticks = [-180.0, -90.0, 0.0, 90.0, 180.0], xlabelsize=20, ylabelsize=20, limits = ((-180.0, +180.0), (-180.0, +180.0)))
    scT1 = scatter!(axT1, T_subdf.dssp_phi, T_subdf.dssp_psi, color = (:red, 0.3))
    scT2 = scatter!(axT2, T_subdf.tinti_phi_prediction, T_subdf.tinti_psi_prediction, color = (:blue, 0.3))
    Label(T_subfig[0, :], "Threonina (T)", fontsize = 30, justification = :center)

# Aminoacido Desconhecido (X)
    X_subdf = subset(merged_results_df, :aa => x -> x .== "X")
    axX1 = Axis(X_subfig[1,1], xlabel = "PHI", ylabel = "PSI", xticks = [-180.0, -90.0, 0.0, 90.0, 180.0], xlabelsize=20, ylabelsize=20, limits = ((-180.0, +180.0), (-180.0, +180.0)))
    axX2 = Axis(X_subfig[1,2], xlabel = "PHI", ylabel = "PSI", xticks = [-180.0, -90.0, 0.0, 90.0, 180.0], xlabelsize=20, ylabelsize=20, limits = ((-180.0, +180.0), (-180.0, +180.0)))
    scX1 = scatter!(axX1, X_subdf.dssp_phi, X_subdf.dssp_psi, color = (:red, 0.3))
    scX2 = scatter!(axX2, X_subdf.tinti_phi_prediction, X_subdf.tinti_psi_prediction, color = (:blue, 0.3))
    Label(X_subfig[0, :], "Aminoacido desconhecido (X)", fontsize = 30, justification = :center)

colsize!(G_subfig, 1, Relative(1/2))
colsize!(G_subfig, 2, Relative(1/2))
colsize!(A_subfig, 1, Relative(1/2))
colsize!(A_subfig, 2, Relative(1/2))
colsize!(F_subfig, 1, Relative(1/2))
colsize!(F_subfig, 2, Relative(1/2))
colsize!(M_subfig, 1, Relative(1/2))
colsize!(M_subfig, 2, Relative(1/2))
colsize!(N_subfig, 1, Relative(1/2))
colsize!(N_subfig, 2, Relative(1/2))
colsize!(D_subfig, 1, Relative(1/2))
colsize!(D_subfig, 2, Relative(1/2))
colsize!(R_subfig, 1, Relative(1/2))
colsize!(R_subfig, 2, Relative(1/2))
colsize!(Q_subfig, 1, Relative(1/2))
colsize!(Q_subfig, 2, Relative(1/2))
colsize!(E_subfig, 1, Relative(1/2))
colsize!(E_subfig, 2, Relative(1/2))
colsize!(K_subfig, 1, Relative(1/2))
colsize!(K_subfig, 2, Relative(1/2))
colsize!(L_subfig, 1, Relative(1/2))
colsize!(L_subfig, 2, Relative(1/2))
colsize!(I_subfig, 1, Relative(1/2))
colsize!(I_subfig, 2, Relative(1/2))
colsize!(H_subfig, 1, Relative(1/2))
colsize!(H_subfig, 2, Relative(1/2))
colsize!(S_subfig, 1, Relative(1/2))
colsize!(S_subfig, 2, Relative(1/2))
colsize!(C_subfig, 1, Relative(1/2))
colsize!(C_subfig, 2, Relative(1/2))
colsize!(P_subfig, 1, Relative(1/2))
colsize!(P_subfig, 2, Relative(1/2))
colsize!(W_subfig, 1, Relative(1/2))
colsize!(W_subfig, 2, Relative(1/2))
colsize!(Y_subfig, 1, Relative(1/2))
colsize!(Y_subfig, 2, Relative(1/2))
colsize!(V_subfig, 1, Relative(1/2))
colsize!(V_subfig, 2, Relative(1/2))
colsize!(T_subfig, 1, Relative(1/2))
colsize!(T_subfig, 2, Relative(1/2))
colsize!(X_subfig, 1, Relative(1/2))
colsize!(X_subfig, 2, Relative(1/2))

Legend(
    fig[end+1, :],
    [
        PolyElement(;color=:red, strokewidth=1),
        PolyElement(;color=color = :blue, strokewidth=1),
    ],
    [
        "Angulo real (DSSP)",
        "Angulo gerado pelo modelo regressor"
    ];
    labelsize=30,
    orientation = :horizontal
)

save("FIG_RAMACHANDRAN_PLOTS.png", fig)

#####
# Section about the Glycine (0,0) exploration
#####

gly_explore_df = subset(merged_results_df, :aa => x -> x .== "G")
gly_explore_df = transform(gly_explore_df, [:tinti_phi_prediction, :tinti_psi_prediction ] => ( (x, y) -> sqrt.(x .^ 2 .+ y .^ 2)) => :absdist; renamecols = false)
gly_explore_df = sort(gly_explore_df, :absdist)