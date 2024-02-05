#####
# Example evaluation script for TintiNet.jl
# Model IGBT (InceptGOR-Bert): 3-state SS Classifier and PHI, PSI and ACC regressor.
#####

#####
# 0. Loading Packages and defining Hyperparameters
#####

using CUDA, Flux, Transformers, JSON, BSON, Random, Dates, Printf
using IterTools: ncycle;
using Transformers.Basic, Transformers.Pretrain, Transformers.Datasets, Transformers.Stacks, Transformers.BidirectionalEncoder
using BSON: @load, @save
using Flux: @functor, onehot, onecold, mse, shuffle, sigmoid, gradient, unsqueeze
using Flux.Optimise: update!
using Statistics: mean, std

using TintiNet
Random.seed!(102528)

hyperpar = Dict(
    "max_seq_len"       => 192,     # Maximum sequence length on the model (default 192)
    "batch_size"        => 128,     # Batch size for training (default 128)
)

inputfile_path = "/home/guilherme/2021_NEURALNET/data/data/data_folds_192/fold_01_192/fold_01_192_train_sequences.fasta"
# inputfile_path = "/home/guilherme/2021_NEURALNET/data/data/data_folds_192/fold_01_192/fold_01_192_test_sequences.fasta"

# inputfile_path = "/home/guilherme/2021_NEURALNET/TintiNet.jl/examples/example_samples.fasta" ## First argument is the location of the sequence input file

#####
# 1. Obtaining the data
#####

headers, sequences = read_sequences_from_file(inputfile_path, hyperpar["max_seq_len"]; filetype="fasta")

#####
# 2. Loading the saved models
#####

@load "/home/guilherme/2021_NEURALNET/TintiNet.jl/models/TintiNet-IGBT-Classifier-Checkpoint-2024-1-16-03-20-33.bson" checkpoint_model
classifier_model = checkpoint_model |> gpu
Flux.testmode!(classifier_model)

#####
# 3. Model evaluation
#####

findall(headers .== "5cxoB00") # 5653
findall(headers .== "1c75A00") # 22083
findall(headers .== "1d06A00") # 16899
selected_seq_idx = 5653 # SALBIII, 5cxoB00
selected_seq_idx = 22083
selected_seq_idx = 16899

x = sequences[selected_seq_idx:selected_seq_idx]
x = preprocess(x, 192, "-")

x_gpu = fastaVocab(x)
x_gpu = reshape(x_gpu, :, 1) |> gpu
out1 = classifier_model[1](x_gpu)
out2 = classifier_model[2](out1)
out3 = classifier_model[3](out2)
out4 = classifier_model[4](out3)
out5 = classifier_model[5](out4)
out6 = classifier_model[6](out5)
out7 = classifier_model[7](out6)
out8 = classifier_model[8](out7)
out9 = classifier_model[9](out8)
out10 = classifier_model[10](out9)
out11 = classifier_model[11](out10)
out12 = classifier_model[12](out11)
out13 = classifier_model[13](out12)
out14 = classifier_model[14](out13)
out15 = classifier_model[15](out13, out14)
out16 = classifier_model[16](out15; all = true)

n_transformer_layers = 2
attention_matrix = Array{Float64, 3}(undef, 192, 8, n_transformer_layers)
attention_tensor = Array{Float64, 4}(undef, 192, 192, 8, n_transformer_layers)

for this_layer_idx in 1:n_transformer_layers

    this_transformer_layer = classifier_model[16].ts[this_layer_idx]

    query = deepcopy(out15)
    key = deepcopy(out15)
    value = deepcopy(out15)

    qs = size(query)
    ks = size(key)
    vs = size(value)

    #size(ipq) == (h, q_seq_len, batch)
    ipq = @toNd this_transformer_layer.mh.iqproj(query)
    ipk = @toNd this_transformer_layer.mh.ikproj(key)
    ipv = @toNd this_transformer_layer.mh.ivproj(value)

    h = size(ipq, 1)
    hs = div(h,this_transformer_layer.mh.head)

    ipq = reshape(ipq, hs, qs[2], :)
    ipk = reshape(ipk, hs, ks[2], :)
    ipv = reshape(ipv, hs, vs[2], :)

    ipq_cpu = cpu(ipq)
    ipk_cpu = cpu(ipk)

    score = Transformers.batchedmul(ipq_cpu, ipk_cpu; transA = true)
    dk = size(key, 1)

    score = score ./ sqrt(dk)
    score = softmax(score; dims=1)

    for head_idx in 1:size(score, 3)
        attention_tensor[:, :, head_idx, this_layer_idx] = score[:,:,head_idx]
        attention_matrix[:, head_idx, this_layer_idx] = map(sum, eachrow(score[:,:,head_idx])) # this sums to one
    end

end

#####
# Plotting
#####

using CairoMakie
# this_seq_len = 134 #SalBIII, 5cxoB00
# this_seq_len = 71 #CytC553 1c75A00
this_seq_len = 130 #OxySig 1d06A00


## Combined attentions
fig = Figure(; size = (1200,600))
ax1 = Axis(fig[1,1], xticks = (1:10:this_seq_len) , yticks = (1:8), xlabel = "Posicao na sequencia", ylabel = "Cabeca de previsao (\"Head\")", title = "BERT - camada 1")
ax2 = Axis(fig[2,1], xticks = (1:10:this_seq_len) , yticks = (1:8), xlabel = "Posicao na sequencia", ylabel = "Cabeca de previsao (\"Head\")", title = "BERT - camada 2")
hm1 = heatmap!(ax1, attention_matrix[1:this_seq_len,:,1])
hm2 = heatmap!(ax2, attention_matrix[1:this_seq_len,:,2])

save("attentionhm_sequence$(selected_seq_idx)_allheads.png", fig)

## Individual slement-wise attentions
for tf_layer_idx in 1:2
    for head_idx in 1:8
        fig = Figure(; size = (600,600))
        ax = Axis(fig[1,1], yreversed=true)
        hm = heatmap!(ax, attention_tensor[1:this_seq_len,1:this_seq_len,head_idx, tf_layer_idx])

        save("attentionhm_sequence$(selected_seq_idx)_head$(head_idx)_layer$(tf_layer_idx).png", fig)
    end
end

# Clearly, one big figure did not work! - but safekeeping this...
# fig = Figure(; size = (2400,600))
# ax1_1 = Axis(fig[1,1])
# ax2_1 = Axis(fig[1,2])
# ax3_1 = Axis(fig[1,3])
# ax4_1 = Axis(fig[1,4])
# ax5_1 = Axis(fig[1,5])
# ax6_1 = Axis(fig[1,6])
# ax7_1 = Axis(fig[1,7])
# ax8_1 = Axis(fig[1,8])
# ax1_2 = Axis(fig[2,1])
# ax2_2 = Axis(fig[2,2])
# ax3_2 = Axis(fig[2,3])
# ax4_2 = Axis(fig[2,4])
# ax5_2 = Axis(fig[2,5])
# ax6_2 = Axis(fig[2,6])
# ax7_2 = Axis(fig[2,7])
# ax8_2 = Axis(fig[2,8])
# hm1_1 = heatmap!(ax1_1, attention_tensor[:,:,1,1])
# hm2_1 = heatmap!(ax2_1, attention_tensor[:,:,2,1])
# hm3_1 = heatmap!(ax3_1, attention_tensor[:,:,3,1])
# hm4_1 = heatmap!(ax4_1, attention_tensor[:,:,4,1])
# hm5_1 = heatmap!(ax5_1, attention_tensor[:,:,5,1])
# hm6_1 = heatmap!(ax6_1, attention_tensor[:,:,6,1])
# hm7_1 = heatmap!(ax7_1, attention_tensor[:,:,7,1])
# hm8_1 = heatmap!(ax8_1, attention_tensor[:,:,8,1])
# hm1_2 = heatmap!(ax1_2, attention_tensor[:,:,1,2])
# hm2_2 = heatmap!(ax2_2, attention_tensor[:,:,2,2])
# hm3_2 = heatmap!(ax3_2, attention_tensor[:,:,3,2])
# hm4_2 = heatmap!(ax4_2, attention_tensor[:,:,4,2])
# hm5_2 = heatmap!(ax5_2, attention_tensor[:,:,5,2])
# hm6_2 = heatmap!(ax6_2, attention_tensor[:,:,6,2])
# hm7_2 = heatmap!(ax7_2, attention_tensor[:,:,7,2])
# hm8_2 = heatmap!(ax8_2, attention_tensor[:,:,8,2])

training_set = JSON.parsefile("/home/guilherme/2021_NEURALNET/data/data/data_folds_192/fold_01_192/fold_01_192_train_dataset.json"; dicttype=Dict, inttype=Int64, use_mmap=true)
# seqidx = findall( [ values(training_set)[i]["domain"] for i in eachindex(values(training_set)) ] .== "5cxoB00" )[1]
# seqidx = findall( [ values(training_set)[i]["domain"] for i in eachindex(values(training_set)) ] .== "1c75A00" )[1]
seqidx = findall( [ values(training_set)[i]["domain"] for i in eachindex(values(training_set)) ] .== "1d06A00" )[1]
input_seq = values(training_set)[seqidx]["fasta_seq"]
dssp_ss3 = values(training_set)[seqidx]["dssp_ss3"]

ss_predictions = compute_classifier_predictions(
    classifier_model,
    x;
    batched=true,
    eval_batch_size=64,
    use_gpu=true,
    sleep_time_seconds=0.01
)

print("SEQUENCE: $(reduce(*, input_seq))\nDSSP_SS3: $(reduce(*, dssp_ss3))\nMODEL_SS3: $(reduce(*, ss_predictions[1]))")

mean(dssp_ss3[1:134] .== ss_predictions[1][1:134]) # SALBIII, 5cxoB00
mean(dssp_ss3[1:71] .== ss_predictions[1][1:71]) # CytC553, 1c75A00
mean(dssp_ss3[1:130] .== ss_predictions[1][1:130]) # OxySig, 5cxoB00