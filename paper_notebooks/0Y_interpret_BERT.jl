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
using CairoMakie

using TintiNet
Random.seed!(102528)

hyperpar = Dict(
    "max_seq_len"       => 192,     # Maximum sequence length on the model (default 192)
    "batch_size"        => 128,     # Batch size for training (default 128)
)

inputfile_path = "/home/guilherme/2021_NEURALNET/data/data/data_folds_192/fold_01_192/fold_01_192_train_sequences.fasta"
# inputfile_path = "/home/guilherme/2021_NEURALNET/data/data/data_folds_192/fold_01_192/fold_01_192_test_sequences.fasta"
# inputfile_path = "/home/guilherme/2021_NEURALNET/TintiNet.jl/examples/example_samples.fasta" ## First argument is the location of the sequence input file
training_set = JSON.parsefile("/home/guilherme/2021_NEURALNET/data/data/data_folds_192/fold_01_192/fold_01_192_train_dataset.json"; dicttype=Dict, inttype=Int64, use_mmap=true)
# seqidx = findall( [ values(training_set)[i]["domain"] for i in eachindex(values(training_set)) ] .== "5cxoB00" )[1]
seqidx = findall( [ values(training_set)[i]["domain"] for i in eachindex(values(training_set)) ] .== "1c75A00" )[1]
# seqidx = findall( [ values(training_set)[i]["domain"] for i in eachindex(values(training_set)) ] .== "1d06A00" )[1]
input_seq = values(training_set)[seqidx]["fasta_seq"]
dssp_ss3 = values(training_set)[seqidx]["dssp_ss3"]

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

paramCount = sum(length, params(classifier_model))

#####
# 3. Model evaluation
#####

# findall(headers .== "5cxoB00") # 5653
findall(headers .== "1c75A00") # 22083
# findall(headers .== "1d06A00") # 16899
# selected_seq_idx = 5653 # SALBIII, 5cxoB00
selected_seq_idx = 22083
# selected_seq_idx = 16899

x = sequences[selected_seq_idx:selected_seq_idx]
this_seq_len = length(x[1])
# this_seq_len = 134
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

#####
# Interpreting the Conv inceptiGOR Layers
# Save this: heatmap(cpu(out10[1,:,:,1]))
#####

hcplot_dssp_ss3 = vcat(dssp_ss3, repeat(["-"], 192-length(dssp_ss3)))

using Clustering, Distances

dist_res = pairwise(Euclidean(), Matrix(cpu(out10[1,:,:,1])); dims=1)
hcl_res = hclust(dist_res; linkage=:average, branchorder=:optimal)
hclust_res_order = hcl_res.order

fig = Figure(; size = (2000,2000))
ax1 = Axis(
    fig[1,1],
    xlabel = "Posição na sequência primária", ylabel = "Caminho do InceptiGOR8",
    xticks = ((collect(1:10:192) .- 0.5), string.(collect(0:10:192))),
    yticks = (
        collect(1:128),
        vcat(
            repeat([""],7), ["Single"],
            repeat([""],15), ["Conv3"],
            repeat([""],15), ["Conv5"],
            repeat([""],15), ["Conv7"],
            repeat([""],15), ["Conv9"],
            repeat([""],15), ["Conv11"],
            repeat([""],15), ["Conv13"],
            repeat([""],15), ["MaxPool"],
            repeat([""],8)
        )
    ),
    yticklabelrotation = pi/2,
    xticklabelsize = 20, yticklabelsize = 20,
    xlabelsize = 20, ylabelsize = 20, titlesize = 24,
    title = "Output da última camada InceptiGOR, na ordem original da sequência primária", yreversed = false
)
ax2 = Axis(
    fig[2,1],
    ylabel = "Caminho do InceptiGOR8",
    xticks = (collect(1:192), hcplot_dssp_ss3[hcl_res.order]),
    yticks = (
        collect(1:128),
        vcat(
            repeat([""],7), ["Single"],
            repeat([""],15), ["Conv3"],
            repeat([""],15), ["Conv5"],
            repeat([""],15), ["Conv7"],
            repeat([""],15), ["Conv9"],
            repeat([""],15), ["Conv11"],
            repeat([""],15), ["Conv13"],
            repeat([""],15), ["MaxPool"],
            repeat([""],8)
        )
    ),
    yticklabelrotation = pi/2,
    xticklabelsize = 12, yticklabelsize = 20,
    ylabelsize = 20, titlesize = 24,
    title = "Output da última camada InceptiGOR, ordenado por análise hierárquica de agrupamentos", yreversed = false
)
ax2ticks = Axis(
    fig[2,1],
    xlabel = "Estrutura secundária de referência",
    xticks = (collect(1:192), string.(hcl_res.order)),
    xticklabelpad = 20.0,
    xticklabelrotation = pi/2,
    xticklabelsize = 12,
    xlabelsize = 20,
    yreversed = false
)
linkaxes!(ax2ticks, ax2)
hm1 = heatmap!(ax1, cpu(out10[1,:,:,1]), yreversed = true)
hm2 = heatmap!(ax2, cpu(out10[1,:,:,1])[hcl_res.order, :], yreversed = true)
hm2 = heatmap!(ax2ticks, cpu(out10[1,:,:,1])[hcl_res.order, :], yreversed = true)
hideydecorations!(ax2ticks)

Colorbar(fig[3,1], hm1, label = "Nível do sinal na matriz de saída da última camada InceptiGOR8 (unidade arbitrária)", labelsize = 20, vertical = false)

save("inception_sequence$(selected_seq_idx).png", fig)

#####
# Transformer Interpretation
#####

n_transformer_layers = 2
attention_matrix = Array{Float64, 3}(undef, 192, 8, n_transformer_layers)
attention_tensor = Array{Float64, 4}(undef, 192, 192, 8, n_transformer_layers)

for this_layer_idx in 1:n_transformer_layers

    ## 1. Preparing the calculation
    this_transformer_layer = classifier_model[16].ts[this_layer_idx]

    query_mat = deepcopy(out15)
    key_mat = deepcopy(out15)
    value_mat = deepcopy(out15)
    # size(query) == (dims, seq_len)
    ipq = this_transformer_layer.mh.iqproj(query_mat)
    ipk = this_transformer_layer.mh.ikproj(key_mat)
    ipv = this_transformer_layer.mh.ivproj(value_mat)
    h = size(ipq)[1] #h == hs * head
    hs = div(h, this_transformer_layer.mh.head)
    #size(hq) == (hs, seq_len, head)
    hq = permutedims(reshape(ipq, hs, this_transformer_layer.mh.head, :), [1, 3, 2])
    hk = permutedims(reshape(ipk, hs, this_transformer_layer.mh.head, :), [1, 3, 2])
    hv = permutedims(reshape(ipv, hs, this_transformer_layer.mh.head, :), [1, 3, 2])

    # ## Manual approach - for safekeeping
    # cpu_hq = cpu(hq)
    # cpu_hk = cpu(hk)
    # dk = size(cpu_hk, 1)
    # score = Transformers.batchedmul(cpu_hk, cpu_hq; transA = true)
    # score = score ./ sqrt(dk)
    # score = softmax(score; dims=1)

    ## NNlib approach
    #size(hq) == (hs, seq_len, head)
    hq = reshape(ipq, hs, this_transformer_layer.mh.head, :, 1)
    hk = reshape(ipk, hs, this_transformer_layer.mh.head, :, 1)
    hv = reshape(ipv, hs, this_transformer_layer.mh.head, :, 1)

    dpa_nnlib = dot_product_attention_scores(cpu(hq),cpu(hk))
    
    for head_idx in 1:8
        attention_tensor[:, :, head_idx, this_layer_idx] = dpa_nnlib[:,:,head_idx, 1]
        attention_matrix[:, head_idx, this_layer_idx] = map(sum, eachrow(dpa_nnlib[:,:,head_idx, 1])) # this sums to one if col
    end

end

#####
# Plotting
#####

## Combined attentions
# fig = Figure(; size = (1200,600))
fig = Figure(; size = (700,600))
ax1 = Axis(fig[1,1], xticks = (1:10:this_seq_len) , yticks = (1:8), xlabel = "Posicao na sequencia", ylabel = "Cabeca de previsao (\"Head\")", title = "BERT - camada 1", yreversed = false)
ax2 = Axis(fig[2,1], xticks = (1:10:this_seq_len) , yticks = (1:8), xlabel = "Posicao na sequencia", ylabel = "Cabeca de previsao (\"Head\")", title = "BERT - camada 2", yreversed = false)
hm1 = heatmap!(ax1, attention_matrix[1:this_seq_len,:,1])
hm2 = heatmap!(ax2, attention_matrix[1:this_seq_len,:,2])

save("attentionhm_sequence$(selected_seq_idx)_allheads.png", fig)

## Individual slement-wise attentions
for tf_layer_idx in 1:2
    for head_idx in 1:8

        ## Those lines of code confirm the orientation of the Plotting
        # map(sum, eachrow(attention_tensor[1:this_seq_len,1:this_seq_len,head_idx, tf_layer_idx]))
        # map(sum, eachcol(attention_tensor[1:this_seq_len,1:this_seq_len,head_idx, tf_layer_idx]))
        # attention_tensor[:,10, head_idx, tf_layer_idx] .+= 1.0
        # attention_tensor[10, :,head_idx, tf_layer_idx] .+= 1.0

        fig = Figure(; size = (600,600))
        ax = Axis(
            fig[1,1], yreversed=false,
            xlabel = "Residuo KEY (\"recebendo\" atencao)",
            ylabel = "Residuo QUERY (\"prestando\" atencao)",
            xticks = ((collect(1:10:this_seq_len) .- 0.5), string.(collect(0:10:this_seq_len))),
            yticks = ((collect(1:10:this_seq_len) .- 0.5), string.(collect(0:10:this_seq_len))),
            title = "BERT, Camada Transformer $(tf_layer_idx), Cabeca (Head) $(head_idx)",
            aspect = DataAspect(),
            xgridcolor = :white,
            ygridcolor = :white,
            xgridwidth = 0.3,
            ygridwidth = 0.3,
            # xminorgridcolor = :white,
            # yminorgridcolor = :white,
            # xminorgridvisible = true,
            # yminorgridvisible = true,
            # xminorticks = IntervalsBetween(5),
            # yminorticks = IntervalsBetween(5),
        )
        hm = heatmap!(ax, attention_tensor[1:this_seq_len,1:this_seq_len,head_idx, tf_layer_idx])
        translate!(hm, 0, 0, -100)

        save("attentionhm_sequence$(selected_seq_idx)_head$(head_idx)_layer$(tf_layer_idx).png", fig)
    end

    fig = Figure(; size = (600,600))
    ax = Axis(
        fig[1,1], yreversed=false,
        xlabel = "Residuo KEY (\"recebendo\" atencao)",
        ylabel = "Residuo QUERY (\"prestando\" atencao)",
        xticks = ((collect(1:10:this_seq_len) .- 0.5), string.(collect(0:10:this_seq_len))),
        yticks = ((collect(1:10:this_seq_len) .- 0.5), string.(collect(0:10:this_seq_len))),
        title = "BERT, Camada Transformer $(tf_layer_idx), Agregado",
        aspect = DataAspect(),
        xgridcolor = :white,
        ygridcolor = :white,
        xgridwidth = 0.3,
        ygridwidth = 0.3,
        # xminorgridcolor = :white,
        # yminorgridcolor = :white,
        # xminorgridvisible = true,
        # yminorgridvisible = true,
        # xminorticks = IntervalsBetween(5),
        # yminorticks = IntervalsBetween(5),
    )
    hm = heatmap!(ax, sum(attention_tensor[1:this_seq_len,1:this_seq_len,:,tf_layer_idx]; dims = 3)[:,:,1])
    translate!(hm, 0, 0, -100)

    save("attentionhm_sequence$(selected_seq_idx)_aggregate_layer$(tf_layer_idx).png", fig)

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

ss_predictions = compute_classifier_predictions(
    classifier_model,
    x;
    batched=true,
    eval_batch_size=64,
    use_gpu=true,
    sleep_time_seconds=0.01
)

print("SEQUENCE: $(reduce(*, input_seq))\nDSSP_SS3: $(reduce(*, dssp_ss3))\nMODEL_SS3: $(reduce(*, ss_predictions[1]))")

# mean(dssp_ss3[1:134] .== ss_predictions[1][1:134]) # SALBIII, 5cxoB00
mean(dssp_ss3[1:71] .== ss_predictions[1][1:71]) # CytC553, 1c75A00
# mean(dssp_ss3[1:130] .== ss_predictions[1][1:130]) # OxySig, 5cxoB00