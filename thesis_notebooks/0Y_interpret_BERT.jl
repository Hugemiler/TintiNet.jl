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
    "max_seq_len"       => 128,     # Maximum sequence length on the model (default 192)
    "batch_size"        => 128,     # Batch size for training (default 128)
)

inputfile_path = "/home/guilherme/2021_NEURALNET/TintiNet.jl/examples/example_samples.fasta" ## First argument is the location of the sequence input file

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
x = sequences[1:1]
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
attention_matrix = Array{Float64, 3}(undef, 128, 8, n_transformer_layers)
attention_tensor = Array{Float64, 4}(undef, 128, 128, 8, n_transformer_layers)

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

fig = Figure(; size = (1000,600))
ax1 = Axis(fig[1,1])
ax2 = Axis(fig[2,1])
hm1 = heatmap!(ax1, attention_matrix[:,:,1])
hm2 = heatmap!(ax2, attention_matrix[:,:,2])


fig = Figure(; size = (2400,600))
ax1_1 = Axis(fig[1,1])
ax2_1 = Axis(fig[1,2])
ax3_1 = Axis(fig[1,3])
ax4_1 = Axis(fig[1,4])
ax5_1 = Axis(fig[1,5])
ax6_1 = Axis(fig[1,6])
ax7_1 = Axis(fig[1,7])
ax8_1 = Axis(fig[1,8])
ax1_2 = Axis(fig[2,1])
ax2_2 = Axis(fig[2,2])
ax3_2 = Axis(fig[2,3])
ax4_2 = Axis(fig[2,4])
ax5_2 = Axis(fig[2,5])
ax6_2 = Axis(fig[2,6])
ax7_2 = Axis(fig[2,7])
ax8_2 = Axis(fig[2,8])

hm1_1 = heatmap!(ax1_1, attention_tensor[:,:,1,1])
hm2_1 = heatmap!(ax2_1, attention_tensor[:,:,2,1])
hm3_1 = heatmap!(ax3_1, attention_tensor[:,:,3,1])
hm4_1 = heatmap!(ax4_1, attention_tensor[:,:,4,1])
hm5_1 = heatmap!(ax5_1, attention_tensor[:,:,5,1])
hm6_1 = heatmap!(ax6_1, attention_tensor[:,:,6,1])
hm7_1 = heatmap!(ax7_1, attention_tensor[:,:,7,1])
hm8_1 = heatmap!(ax8_1, attention_tensor[:,:,8,1])
hm1_2 = heatmap!(ax1_2, attention_tensor[:,:,1,2])
hm2_2 = heatmap!(ax2_2, attention_tensor[:,:,2,2])
hm3_2 = heatmap!(ax3_2, attention_tensor[:,:,3,2])
hm4_2 = heatmap!(ax4_2, attention_tensor[:,:,4,2])
hm5_2 = heatmap!(ax5_2, attention_tensor[:,:,5,2])
hm6_2 = heatmap!(ax6_2, attention_tensor[:,:,6,2])
hm7_2 = heatmap!(ax7_2, attention_tensor[:,:,7,2])
hm8_2 = heatmap!(ax8_2, attention_tensor[:,:,8,2])

fig

ss_predictions = compute_classifier_predictions(
    classifier_model,
    sequences[1:1];
    batched=true,
    eval_batch_size=64,
    use_gpu=true,
    sleep_time_seconds=0.01)

#####
# 4. Saving Prediction Results
#####

write_csv_predictions(output_path, headers, sequences, ss_predictions, phi_predictions, psi_predictions, acc_predictions)
