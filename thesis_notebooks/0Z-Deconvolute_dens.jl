using JSON
using CairoMakie
using DataFrames
using Chain
using Statistics

inputs = JSON.parsefile("/home/guilherme/2021_NEURALNET/data/data/data_folds_192/fold_01_192/fold_01_192_test_dataset.json"; dicttype=Dict, inttype=Int64, use_mmap=true)

inputs_dfs = [
    @chain DataFrame(
            :domain =>  x["domain"],
            :sequence => i,
            :dssp_ss3 => map(y -> (isnothing(y) ? missing : string(y)), x["dssp_ss3"]),
            :dssp_ss8 => map(y -> (isnothing(y) ? missing : string(y)), x["dssp_ss8"]),
            :dssp_phi => map(y -> (isnothing(y) ? missing : Float64(y)), x["dssp_phi"]),
            :dssp_psi => map(y -> (isnothing(y) ? missing : Float64(y)), x["dssp_psi"]),
        ) begin
            transform!(:sequence => x -> 1:length(x) => :idx)
        end
    for (i, x) in enumerate(inputs)
]

merged_inputs_df = dropmissing(vcat(inputs_dfs...))

using StatsBase

StatsBase.proportionmap(merged_inputs_df.dssp_ss3)
StatsBase.proportionmap(merged_inputs_df.dssp_ss8)

results = JSON.parsefile("/home/guilherme/2021_NEURALNET/results/results/Processed_results_VF.json")

results_dfs = [
    @chain DataFrame(
            :domain =>  x["domain"],
            :sequence => i,
            :aa => map(y -> (isnothing(y) ? missing : string(y)), x["fasta_seq"]),
            :dssp_ss3 => map(y -> (isnothing(y) ? missing : string(y)), x["dssp_ss3"]),
            :dssp_phi => map(y -> (isnothing(y) ? missing : Float64(y)), x["dssp_phi"]),
            :dssp_psi => map(y -> (isnothing(y) ? missing : Float64(y)), x["dssp_psi"]),
            :tinti_ss3_prediction => map(y -> (isnothing(y) ? missing : string(y)), x["tinti_ss3_prediction"]),
            :tinti_phi_prediction => map(y -> (isnothing(y) ? missing : Float64(y)), x["tinti_phi_prediction"]),
            :tinti_psi_prediction => map(y -> (isnothing(y) ? missing : Float64(y)), x["tinti_psi_prediction"]),
            :spot_ss3_prediction => map(y -> (isnothing(y) ? missing : string(y)), x["spot_ss3_prediction"]),
            :spot_phi_prediction => map(y -> (isnothing(y) ? missing : Float64(y)), x["spot_phi_prediction"]),
            :spot_psi_prediction => map(y -> (isnothing(y) ? missing : Float64(y)), x["spot_psi_prediction"]),
            :punet_ss3_prediction => map(y -> (isnothing(y) ? missing : string(y)), x["punet_ss3_prediction"]),
            :punet_phi_prediction => map(y -> (isnothing(y) ? missing : Float64(y)), x["punet_phi_prediction"]),
            :punet_psi_prediction => map(y -> (isnothing(y) ? missing : Float64(y)), x["punet_psi_prediction"]),
            :spider_ss3_prediction => map(y -> (isnothing(y) ? missing : string(y)), x["spider_ss3_prediction"]),
            :spider_phi_prediction => map(y -> (isnothing(y) ? missing : Float64(y)), x["spider_phi_prediction"]),
            :spider_psi_prediction => map(y -> (isnothing(y) ? missing : Float64(y)), x["spider_psi_prediction"]),
            :trimmed =>  x["trimmed"]
        ) begin
            transform!(:sequence => x -> 1:length(x) => :idx)
        end
    for (i, x) in enumerate(results)
]

merged_results_df = dropmissing(vcat(results_dfs...))
transform!(merged_results_df, :dssp_phi => (x -> map(y -> ((y > 180.0) ? (y - 360.0) : y), x)) => :dssp_phi; renamecols = false)
transform!(merged_results_df, :dssp_psi => (x -> map(y -> ((y > 180.0) ? (y - 360.0) : y), x)) => :dssp_psi; renamecols = false)
transform!(merged_results_df, :tinti_phi_prediction => (x -> map(y -> ((y > 180.0) ? (y - 360.0) : y), x)) => :tinti_phi_prediction; renamecols = false)
transform!(merged_results_df, :tinti_psi_prediction => (x -> map(y -> ((y > 180.0) ? (y - 360.0) : y), x)) => :tinti_psi_prediction; renamecols = false)
transform!(merged_results_df, :spider_phi_prediction => (x -> map(y -> ((y > 180.0) ? (y - 360.0) : y), x)) => :spider_phi_prediction; renamecols = false)
transform!(merged_results_df, :spider_psi_prediction => (x -> map(y -> ((y > 180.0) ? (y - 360.0) : y), x)) => :spider_psi_prediction; renamecols = false)
transform!(merged_results_df, :punet_phi_prediction => (x -> map(y -> ((y > 180.0) ? (y - 360.0) : y), x)) => :punet_phi_prediction; renamecols = false)
transform!(merged_results_df, :punet_psi_prediction => (x -> map(y -> ((y > 180.0) ? (y - 360.0) : y), x)) => :punet_psi_prediction; renamecols = false)
transform!(merged_results_df, :spot_phi_prediction => (x -> map(y -> ((y > 180.0) ? (y - 360.0) : y), x)) => :spot_phi_prediction; renamecols = false)
transform!(merged_results_df, :spot_psi_prediction => (x -> map(y -> ((y > 180.0) ? (y - 360.0) : y), x)) => :spot_psi_prediction; renamecols = false)
subset!(merged_results_df, :dssp_phi => x -> x .!= 0.0)

# PHI angles
transform!(merged_results_df, [:dssp_phi, :tinti_phi_prediction] => ( (x, y) -> abs.( x .- y) ) => :tinti_phi_abserror; renamecols = false)
transform!(merged_results_df, :tinti_phi_abserror => (x -> map(y -> (minimum([y, 360-y])), x)) => :tinti_phi_abserror; renamecols = false)
mean(merged_results_df.tinti_phi_abserror)

transform!(merged_results_df, [:dssp_phi, :spider_phi_prediction] => ( (x, y) -> abs.( x .- y) ) => :spider_phi_abserror; renamecols = false)
transform!(merged_results_df, :spider_phi_abserror => (x -> map(y -> (minimum([y, 360-y])), x)) => :spider_phi_abserror; renamecols = false)
mean(merged_results_df.spider_phi_abserror)

transform!(merged_results_df, [:dssp_phi, :punet_phi_prediction] => ( (x, y) -> abs.( x .- y) ) => :punet_phi_abserror; renamecols = false)
transform!(merged_results_df, :punet_phi_abserror => (x -> map(y -> (minimum([y, 360-y])), x)) => :punet_phi_abserror; renamecols = false)
mean(merged_results_df.punet_phi_abserror)

transform!(merged_results_df, [:dssp_phi, :spot_phi_prediction] => ( (x, y) -> abs.( x .- y) ) => :spot_phi_abserror; renamecols = false)
transform!(merged_results_df, :spot_phi_abserror => (x -> map(y -> (minimum([y, 360-y])), x)) => :spot_phi_abserror; renamecols = false)
mean(merged_results_df.spot_phi_abserror)

# PSI angles
transform!(merged_results_df, [:dssp_psi, :tinti_psi_prediction] => ( (x, y) -> abs.( x .- y) ) => :tinti_psi_abserror; renamecols = false)
transform!(merged_results_df, :tinti_psi_abserror => (x -> map(y -> (minimum([y, 360-y])), x)) => :tinti_psi_abserror; renamecols = false)
mean(merged_results_df.tinti_psi_abserror)

transform!(merged_results_df, [:dssp_psi, :spider_psi_prediction] => ( (x, y) -> abs.( x .- y) ) => :spider_psi_abserror; renamecols = false)
transform!(merged_results_df, :spider_psi_abserror => (x -> map(y -> (minimum([y, 360-y])), x)) => :spider_psi_abserror; renamecols = false)
mean(merged_results_df.spider_psi_abserror)

transform!(merged_results_df, [:dssp_psi, :punet_psi_prediction] => ( (x, y) -> abs.( x .- y) ) => :punet_psi_abserror; renamecols = false)
transform!(merged_results_df, :punet_psi_abserror => (x -> map(y -> (minimum([y, 360-y])), x)) => :punet_psi_abserror; renamecols = false)
mean(merged_results_df.punet_psi_abserror)

transform!(merged_results_df, [:dssp_psi, :spot_psi_prediction] => ( (x, y) -> abs.( x .- y) ) => :spot_psi_abserror; renamecols = false)
transform!(merged_results_df, :spot_psi_abserror => (x -> map(y -> (minimum([y, 360-y])), x)) => :spot_psi_abserror; renamecols = false)
mean(merged_results_df.spot_psi_abserror)

# Solvent Accessibility - TODO
# transform!(merged_results_df, [:dssp_acc, :tinti_acc_prediction] => ( (x, y) -> abs.( x .- y) ) => :tinti_acc_abserror; renamecols = false)
# mean(merged_results_df.tinti_acc_abserror)

# transform!(merged_results_df, [:dssp_acc, :spider_acc_prediction] => ( (x, y) -> abs.( x .- y) ) => :spider_acc_abserror; renamecols = false)
# mean(merged_results_df.spider_acc_abserror)

# transform!(merged_results_df, [:dssp_acc, :punet_acc_prediction] => ( (x, y) -> abs.( x .- y) ) => :punet_acc_abserror; renamecols = false)
# mean(merged_results_df.punet_acc_abserror)

# transform!(merged_results_df, [:dssp_acc, :spot_acc_prediction] => ( (x, y) -> abs.( x .- y) ) => :spot_acc_abserror; renamecols = false)
# mean(merged_results_df.spot_acc_abserror)

seqs_acc_df = @chain merged_results_df begin
    groupby(:domain)
    combine(
        [:dssp_ss3, :tinti_ss3_prediction] => ((x, y) -> mean( x .== y)) => :seq_acc,
        :dssp_ss3 => (x -> mean( x .== "E")) => :frac_beta,
        :dssp_ss3 => (x -> sum( x .== "E")) => :num_beta,
        :dssp_ss3 => (x -> mean( x .== "H")) => :frac_alpha,
        :dssp_ss3 => (x -> sum( x .== "H")) => :num_alpha,
        :dssp_ss3 => (x -> mean( x .== "C")) => :frac_coil,
        :dssp_ss3 => (x -> sum( x .== "C")) => :num_coil
    )
end

using FASTX
domains = readlines("/home/guilherme/2021_NEURALNET/data/data/caths40.domlist.dat")
fastaSequences = Vector{String}()

fastaFile = open(FASTA.Reader, "/home/guilherme/2021_NEURALNET/data/data/caths40.fasta.fa")
    for record in fastaFile
        #@show FASTA.identifier(record)
        push!(fastaSequences, FASTA.sequence(String, record))    
    end
close(fastaFile)

len_df = DataFrame(
    :domain => domains,
    :seq_len => map(length,fastaSequences)
)

seqs_acc_df = innerjoin(seqs_acc_df, len_df, on ="domain")

trimmed_seqs_df = subset(seqs_acc_df, :seq_len => x -> x .>= 128)
untrimmed_seqs_df = subset(seqs_acc_df, :seq_len => x -> x .< 128)

using CairoMakie

scatter(trimmed_seqs_df.frac_alpha, trimmed_seqs_df.seq_acc;)
cor(trimmed_seqs_df.frac_alpha, trimmed_seqs_df.seq_acc)
scatter(trimmed_seqs_df.frac_beta, trimmed_seqs_df.seq_acc;)
cor(trimmed_seqs_df.frac_beta, trimmed_seqs_df.seq_acc)
scatter(trimmed_seqs_df.frac_coil, trimmed_seqs_df.seq_acc;)
cor(trimmed_seqs_df.frac_coil, trimmed_seqs_df.seq_acc)

density(trimmed_seqs_df.seq_acc; npoints = 200, offset = 0.0, direction = :x)
density(untrimmed_seqs_df.seq_acc; npoints = 200, offset = 0.0, direction = :x)

sort(seqs_acc_df, :seq_acc)