using JSON
using CairoMakie
using DataFrames

results = JSON.parsefile("/home/guilherme/2021_NEURALNET/results/results/Processed_results_VF.json")

results_dfs = [
    DataFrame(
        :sequence => i,
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

# Ramachandran plots
fig = Figure(; size = (2000,2000))
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

# Glicina (G)
    G_subdf = subset(merged_results_df, :aa => x -> x .== "G")
    axG1 = Axis(G_subfig[1,1], xlabel = "PHI", ylabel = "PSI", limits = ((-180.0, +180.0), (-180.0, +180.0)))
    axG2 = Axis(G_subfig[1,2], xlabel = "PHI", ylabel = "PSI", limits = ((-180.0, +180.0), (-180.0, +180.0)))
    scG1 = scatter!(axG1, G_subdf.dssp_phi, G_subdf.dssp_psi, color = (:red, 0.3))
    scG2 = scatter!(axG2, G_subdf.tinti_phi_prediction, G_subdf.tinti_psi_prediction, color = (:blue, 0.3))
    Label(G_subfig[0, :], "Glicina (G)", fontsize = 30, justification = :center)

# Alanina (A)
    A_subdf = subset(merged_results_df, :aa => x -> x .== "A")
    axA1 = Axis(A_subfig[1,1], xlabel = "PHI", ylabel = "PSI", limits = ((-180.0, +180.0), (-180.0, +180.0)))
    axA2 = Axis(A_subfig[1,2], xlabel = "PHI", ylabel = "PSI", limits = ((-180.0, +180.0), (-180.0, +180.0)))
    scA1 = scatter!(axA1, A_subdf.dssp_phi, A_subdf.dssp_psi, color = (:red, 0.3))
    scA2 = scatter!(axA2, A_subdf.tinti_phi_prediction, A_subdf.tinti_psi_prediction, color = (:blue, 0.3))
    Label(A_subfig[0, :], "Alanina (A)", fontsize = 30, justification = :center)

# Fenilalanina (F)
    F_subdf = subset(merged_results_df, :aa => x -> x .== "F")
    axF1 = Axis(F_subfig[1,1], xlabel = "PHI", ylabel = "PSI", limits = ((-180.0, +180.0), (-180.0, +180.0)))
    axF2 = Axis(F_subfig[1,2], xlabel = "PHI", ylabel = "PSI", limits = ((-180.0, +180.0), (-180.0, +180.0)))
    scF1 = scatter!(axF1, F_subdf.dssp_phi, F_subdf.dssp_psi, color = (:red, 0.3))
    scF2 = scatter!(axF2, F_subdf.tinti_phi_prediction, F_subdf.tinti_psi_prediction, color = (:blue, 0.3))
    Label(F_subfig[0, :], "Fenilalanina (F)", fontsize = 30, justification = :center)

# Methionina (M)
    M_subdf = subset(merged_results_df, :aa => x -> x .== "M")
    axM1 = Axis(M_subfig[1,1], xlabel = "PHI", ylabel = "PSI", limits = ((-180.0, +180.0), (-180.0, +180.0)))
    axM2 = Axis(M_subfig[1,2], xlabel = "PHI", ylabel = "PSI", limits = ((-180.0, +180.0), (-180.0, +180.0)))
    scM1 = scatter!(axM1, M_subdf.dssp_phi, M_subdf.dssp_psi, color = (:red, 0.3))
    scM2 = scatter!(axM2, M_subdf.tinti_phi_prediction, M_subdf.tinti_psi_prediction, color = (:blue, 0.3))
    Label(M_subfig[0, :], "Methionina (M)", fontsize = 30, justification = :center)

# Asparagina (N)
    N_subdf = subset(merged_results_df, :aa => x -> x .== "N")
    axN1 = Axis(N_subfig[1,1], xlabel = "PHI", ylabel = "PSI", limits = ((-180.0, +180.0), (-180.0, +180.0)))
    axN2 = Axis(N_subfig[1,2], xlabel = "PHI", ylabel = "PSI", limits = ((-180.0, +180.0), (-180.0, +180.0)))
    scN1 = scatter!(axN1, N_subdf.dssp_phi, N_subdf.dssp_psi, color = (:red, 0.3))
    scN2 = scatter!(axN2, N_subdf.tinti_phi_prediction, N_subdf.tinti_psi_prediction, color = (:blue, 0.3))
    Label(N_subfig[0, :], "Asparagina (N)", fontsize = 30, justification = :center)

# Acido Aspartico (D)
    D_subdf = subset(merged_results_df, :aa => x -> x .== "D")
    axD1 = Axis(D_subfig[1,1], xlabel = "PHI", ylabel = "PSI", limits = ((-180.0, +180.0), (-180.0, +180.0)))
    axD2 = Axis(D_subfig[1,2], xlabel = "PHI", ylabel = "PSI", limits = ((-180.0, +180.0), (-180.0, +180.0)))
    scD1 = scatter!(axD1, D_subdf.dssp_phi, D_subdf.dssp_psi, color = (:red, 0.3))
    scD2 = scatter!(axD2, D_subdf.tinti_phi_prediction, D_subdf.tinti_psi_prediction, color = (:blue, 0.3))
    Label(D_subfig[0, :], "Acido Aspartico (D)", fontsize = 30, justification = :center)

# Arginina (R)
    R_subdf = subset(merged_results_df, :aa => x -> x .== "R")
    axR1 = Axis(R_subfig[1,1], xlabel = "PHI", ylabel = "PSI", limits = ((-180.0, +180.0), (-180.0, +180.0)))
    axR2 = Axis(R_subfig[1,2], xlabel = "PHI", ylabel = "PSI", limits = ((-180.0, +180.0), (-180.0, +180.0)))
    scR1 = scatter!(axR1, R_subdf.dssp_phi, R_subdf.dssp_psi, color = (:red, 0.3))
    scR2 = scatter!(axR2, R_subdf.tinti_phi_prediction, R_subdf.tinti_psi_prediction, color = (:blue, 0.3))
    Label(R_subfig[0, :], "Arginina (R)", fontsize = 30, justification = :center)

# Glutamina (Q)
    Q_subdf = subset(merged_results_df, :aa => x -> x .== "Q")
    axQ1 = Axis(Q_subfig[1,1], xlabel = "PHI", ylabel = "PSI", limits = ((-180.0, +180.0), (-180.0, +180.0)))
    axQ2 = Axis(Q_subfig[1,2], xlabel = "PHI", ylabel = "PSI", limits = ((-180.0, +180.0), (-180.0, +180.0)))
    scQ1 = scatter!(axQ1, Q_subdf.dssp_phi, Q_subdf.dssp_psi, color = (:red, 0.3))
    scQ2 = scatter!(axQ2, Q_subdf.tinti_phi_prediction, Q_subdf.tinti_psi_prediction, color = (:blue, 0.3))
    Label(Q_subfig[0, :], "Glutamina (Q)", fontsize = 30, justification = :center)

# Acido Glutamico (E)
    E_subdf = subset(merged_results_df, :aa => x -> x .== "E")
    axE1 = Axis(E_subfig[1,1], xlabel = "PHI", ylabel = "PSI", limits = ((-180.0, +180.0), (-180.0, +180.0)))
    axE2 = Axis(E_subfig[1,2], xlabel = "PHI", ylabel = "PSI", limits = ((-180.0, +180.0), (-180.0, +180.0)))
    scE1 = scatter!(axE1, E_subdf.dssp_phi, E_subdf.dssp_psi, color = (:red, 0.3))
    scE2 = scatter!(axE2, E_subdf.tinti_phi_prediction, E_subdf.tinti_psi_prediction, color = (:blue, 0.3))
    Label(E_subfig[0, :], "Acido Glutamico (E)", fontsize = 30, justification = :center)

# Lisina (K)
    K_subdf = subset(merged_results_df, :aa => x -> x .== "K")
    axK1 = Axis(K_subfig[1,1], xlabel = "PHI", ylabel = "PSI", limits = ((-180.0, +180.0), (-180.0, +180.0)))
    axK2 = Axis(K_subfig[1,2], xlabel = "PHI", ylabel = "PSI", limits = ((-180.0, +180.0), (-180.0, +180.0)))
    scK1 = scatter!(axK1, K_subdf.dssp_phi, K_subdf.dssp_psi, color = (:red, 0.3))
    scK2 = scatter!(axK2, K_subdf.tinti_phi_prediction, K_subdf.tinti_psi_prediction, color = (:blue, 0.3))
    Label(K_subfig[0, :], "Lisina (K)", fontsize = 30, justification = :center)

# Leucina (L)
    L_subdf = subset(merged_results_df, :aa => x -> x .== "L")
    axL1 = Axis(L_subfig[1,1], xlabel = "PHI", ylabel = "PSI", limits = ((-180.0, +180.0), (-180.0, +180.0)))
    axL2 = Axis(L_subfig[1,2], xlabel = "PHI", ylabel = "PSI", limits = ((-180.0, +180.0), (-180.0, +180.0)))
    scL1 = scatter!(axL1, L_subdf.dssp_phi, L_subdf.dssp_psi, color = (:red, 0.3))
    scL2 = scatter!(axL2, L_subdf.tinti_phi_prediction, L_subdf.tinti_psi_prediction, color = (:blue, 0.3))
    Label(L_subfig[0, :], "Leucina (L)", fontsize = 30, justification = :center)

# Isoleucina (I)
    I_subdf = subset(merged_results_df, :aa => x -> x .== "I")
    axI1 = Axis(I_subfig[1,1], xlabel = "PHI", ylabel = "PSI", limits = ((-180.0, +180.0), (-180.0, +180.0)))
    axI2 = Axis(I_subfig[1,2], xlabel = "PHI", ylabel = "PSI", limits = ((-180.0, +180.0), (-180.0, +180.0)))
    scI1 = scatter!(axI1, I_subdf.dssp_phi, I_subdf.dssp_psi, color = (:red, 0.3))
    scI2 = scatter!(axI2, I_subdf.tinti_phi_prediction, I_subdf.tinti_psi_prediction, color = (:blue, 0.3))
    Label(I_subfig[0, :], "Isoleucina (I)", fontsize = 30, justification = :center)

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


"N"
 "A"
 "I"
 "D"
 "P"
 "R"
 "E"
 "L"
 "G"
 "V"
 "K"
 "S"
 "T"
 "Y"
 "F"
 "H"
 "W"
 "Q"
 "C"
 "X"