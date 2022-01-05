#####
#
# Example script for TintiNet.jl
# Model IGBT (InceptGOR-Bert): 3-state SS Regressor
#
# To run training, revise your code and uncomment the last line.
# On slower GPUs, an epoch may take several minutes with the default configuration.
# Adjust model size using the `hyperpar` and `config` dictionaries.
#
#####

#####
# 0. Loading Packages and defining Hyperparameters
#####

using CUDA, Zygote, Flux, Transformers, JSON, BSON, Random, Dates, Printf
using IterTools: ncycle;
using Transformers.Basic, Transformers.Pretrain, Transformers.Datasets, Transformers.Stacks, Transformers.BidirectionalEncoder
using BSON: @load, @save
using Flux: @functor, onehot, onecold, mse, shuffle, sigmoid, gradient, unsqueeze
using Flux.Optimise: update!
using Statistics: mean, std

using TintiNet
dformat = Dates.DateFormat("yyyy-m-d-HH-MM-SS")
Random.seed!(102528)

##
# Algorithm Hyperparameters

hyperpar = Dict(
    "max_seq_len"       => 128,     # Maximum sequence length on the model (default 192)
    "batch_size"        => 128,     # Batch size for training (default 128)
    "number_of_epochs"  => 200,     # Number of epochs for training (10 for the example file)
    "learning_rate"     => 1e-3,    # Learning rate for optimizer (default 0.001)
    "smooth_epsilon"    => 1e-6     # Small value to smooth onehot result before logcrossentropy (default 1e-6)
)

##
# Model Configuration

const config = Dict(
  "input_vocab_size"                => 22,      # Size of the SEQUENCE/FASTA vocabulary, usually 20 aminoacids + "X" + "-" 
  "output_size"                     => 5,       # Output size. Usually 5 due to sin/cos phi, sin/cos psi and acc.
  "embed_dimension"                 => 128,     # Size of the individual Embed vectors
  "bert_num_hidden_layers"          => 2,       # Number of consecutive Transformer layers on the BERT encoder
  "bert_transformer_layer_heads"    => 8,       # Number of heads in each Transformer layer
  "bert_transformer_layer_hidden"   => 128,     # Hidden size of the positionwise linear layer in each Transformer layer
  "bert_transformer_dropout_prob"   => 0.1,     # Dropout probability for the Transformer layers
  "bert_transformer_activation"     => gelu,    # Activation function for each Transformer layer (default: gelu)
  "detection_num_filters"           => 32       # Number of filters in each convolutional decoder layer
)

#####
# 2. Obtaining and shaping the data
#####

## 2.1. Training sequences, structures and lengths

remapnan(x) = x == nothing ? missing : x

training_set_datapath = "../data/fold01training_dataset.json"

open(training_set_datapath, "r") do file
    
    entries = JSON.parse(file)
    num_entries = length(entries)
    
    parsed_sequences = Vector{Vector{String}}(undef, num_entries)
    parsed_phis = Vector{Vector{Union{Missing, Float64}}}(undef, num_entries)
    parsed_psis = Vector{Vector{Union{Missing, Float64}}}(undef, num_entries)
    parsed_accs = Vector{Vector{Union{Missing, Float64}}}(undef, num_entries)
    parsed_lengths = Vector{Int}(undef, num_entries)

    for (i,j) in enumerate(entries)
        parsed_sequences[i] = string.(j["fasta_seq"])
        parsed_phis[i] = map(remapnan, j["dssp_phi"])
        parsed_psis[i] = map(remapnan, j["dssp_psi"])
        parsed_accs[i] = map(remapnan, j["dssp_acc"])
        parsed_lengths[i] = length(parsed_sequences[i])
    end

    order_idx = sortperm(parsed_lengths)

    global train_sequences_ordered = parsed_sequences[order_idx]
    global train_phis_ordered = parsed_phis[order_idx]
    global train_psis_ordered = parsed_psis[order_idx]
    global train_accs_ordered = parsed_accs[order_idx]
    global train_lengths_ordered = parsed_lengths[order_idx]
    
end

## 2.2. Testing sequences, structures and lengths

testing_set_datapath = "../data/fold01testing_dataset.json"

open(testing_set_datapath, "r") do file

    entries = JSON.parse(file)
    num_entries = length(entries)
    
    parsed_sequences = Vector{Vector{String}}(undef, num_entries)
    parsed_phis = Vector{Vector{Union{Missing, Float64}}}(undef, num_entries)
    parsed_psis = Vector{Vector{Union{Missing, Float64}}}(undef, num_entries)
    parsed_accs = Vector{Vector{Union{Missing, Float64}}}(undef, num_entries)
    parsed_lengths = Vector{Int}(undef, num_entries)

    for (i,j) in enumerate(entries)
        parsed_sequences[i] = string.(j["fasta_seq"])
        parsed_phis[i] = map(remapnan, j["dssp_phi"])
        parsed_psis[i] = map(remapnan, j["dssp_psi"])
        parsed_accs[i] = map(remapnan, j["dssp_acc"])
        parsed_lengths[i] = length(parsed_sequences[i])
    end

    order_idx = sortperm(parsed_lengths)

    global test_sequences_ordered = parsed_sequences[order_idx]
    global test_phis_ordered = parsed_phis[order_idx]
    global test_psis_ordered = parsed_psis[order_idx]
    global test_accs_ordered = parsed_accs[order_idx]
    global test_lengths_ordered = parsed_lengths[order_idx]
    
end

## 2.3. Lazy DataLoaders

training_dataset = Flux.Data.DataLoader((train_sequences_ordered, train_phis_ordered, train_psis_ordered, train_accs_ordered, train_lengths_ordered); batchsize=hyperpar["batch_size"], shuffle=true)
eval_train_dataset = Flux.Data.DataLoader((train_sequences_ordered, train_phis_ordered, train_psis_ordered, train_accs_ordered, train_lengths_ordered); batchsize=64,shuffle=false)
eval_test_dataset = Flux.Data.DataLoader((test_sequences_ordered, test_phis_ordered, test_psis_ordered, test_accs_ordered, test_lengths_ordered); batchsize=64,shuffle=false)

#####
# 3. Network Structure
#####

model = Stack(
        @nntopo(x → enc → enc → enc → inc → inc → inc → inc → inc → inc → inc → inc → inc → inc → pe:(inc, pe) → bert_input → bert_output → bert_output → bert_output → y → y → y → y → y),
         # Embedding Layers
        Embed(config["embed_dimension"], config["input_vocab_size"]),
        unsqueeze(3),
        x -> permutedims(x, [3,2,1,4]),
        # Processing Layers
        Inception8(config["embed_dimension"]),
        Dropout(config["bert_transformer_dropout_prob"]),
        Inception8(config["embed_dimension"]),
        Dropout(config["bert_transformer_dropout_prob"]),
        Inception8(config["embed_dimension"]),
        Dropout(config["bert_transformer_dropout_prob"]),
        Inception8(config["embed_dimension"]),
        Dropout(config["bert_transformer_dropout_prob"]),
        x -> permutedims(x, [3,2,1,4]),
        x -> reshape(x, size(x,1), size(x,2), size(x,4)),
        PositionEmbedding(config["embed_dimension"], hyperpar["max_seq_len"]; trainable = false),
        (e, pe) -> e .+ pe,
        # Processing Layers
        Bert(
            config["embed_dimension"],
            config["bert_transformer_layer_heads"],
            config["bert_transformer_layer_hidden"],
            config["bert_num_hidden_layers"];
            act = config["bert_transformer_activation"],
            pdrop = config["bert_transformer_dropout_prob"],
            attn_pdrop = config["bert_transformer_dropout_prob"]
        ),
        # Detection Layers
        unsqueeze(3),
        bert_output -> permutedims(bert_output, [3,2,1,4]),
        Conv((1,7), config["embed_dimension"] => config["detection_num_filters"], relu; pad = (0,3)),
        Conv((1,7), config["detection_num_filters"] => config["detection_num_filters"], relu; pad = (0,3)),
        Conv((1,7), config["detection_num_filters"] => config["output_size"], tanh; pad = (0,3)),
        x -> permutedims(x, [3,2,1,4]),
        x -> reshape(x, size(x,1), size(x,2), size(x,4)),
    ) |> gpu

#####
# 4. Model training
#####

opt = Flux.Optimise.ADAM(hyperpar["learning_rate"])     # Initializes the ADAM optimizer
ps = params(model)      # Collects model parameters

function regressor_train_loss(model, x, y, mask)
    sum( ( (model(x) .- y) .^ 2 ) .* mask ) / sum(mask)
end

function regressor_eval_loss(predictions, y, mask)
    sum( ( (predictions .- y) .^ 2 ) .* mask ) / sum(mask)
end

function build_regressor_target_array(
    phis::Vector{Vector{Union{Missing, Float64}}},
    psis::Vector{Vector{Union{Missing, Float64}}},
    accs::Vector{Vector{Union{Missing, Float64}}},
    seqlen::Int64)

    y = zeros(5, seqlen, length(phis))

    for seq_i in 1:length(phis)
        for aa_j in 1:length(phis[seq_i])
            if !ismissing(phis[seq_i][aa_j])
                y[1, aa_j, seq_i] = sind(phis[seq_i][aa_j])
                y[2, aa_j, seq_i] = cosd(phis[seq_i][aa_j])
            end

            if !ismissing(psis[seq_i][aa_j])
                y[3, aa_j, seq_i] = sind(psis[seq_i][aa_j])
                y[4, aa_j, seq_i] = cosd(psis[seq_i][aa_j])
            end

            if !ismissing(accs[seq_i][aa_j])
                y[5, aa_j, seq_i] = (accs[seq_i][aa_j] - 100) / 100
            end

        end
    end

    return(y)

end

function prepare_regressor_mask(batchofseqs::Vector{Vector{String}},
                                phis::Vector{Vector{Union{Missing, Float64}}},
                                psis::Vector{Vector{Union{Missing, Float64}}},
                                accs::Vector{Vector{Union{Missing, Float64}}},
                                seqlen::Int64)

    mask = getmask(preprocess(batchofseqs, seqlen)) # Uses getmask() from Transformers.Basic

    for i in 1:length(batchofseqs)
        for j in 1:length(batchofseqs[i])
            if ( ismissing(phis[i][j]) | ismissing(psis[i][j]) | ismissing(accs[i][j]) )
                mask[1, j, i] = 0.0
            end
        end
    end

    return(mask)

end

#define training loop
function train!(hyperpar, training_dataset,
                model, ps, opt,
                train_loss=regressor_train_loss, eval_loss=regressor_eval_loss,
                eval_train_dataset=eval_train_dataset, eval_test_dataset=eval_test_dataset,
                fastaVocab=fastaVocab)
    @info "start training"
    for epoch in 1:hyperpar["number_of_epochs"]
    i = 0
        for batch in training_dataset
            i += 1
            x, phi, psi, acc, len = batch

            train_x_processed = fastaVocab(preprocess(x, hyperpar["max_seq_len"])) |> gpu
            train_y_processed = build_regressor_target_array(phi, psi, acc, hyperpar["max_seq_len"]) |> gpu
            train_mask_processed = prepare_regressor_mask(x, phi, psi, acc, hyperpar["max_seq_len"]) |> gpu

#            grad = gradient(()->train_loss(model, train_x_processed, train_y_processed, train_mask_processed), ps)

            l, back = Flux.pullback(ps) do
                train_loss(model, train_x_processed, train_y_processed, train_mask_processed)
            end
            grad = back(Flux.Zygote.sensitivity(l))

            update!(opt, ps, grad)

            if i%64 == 0

                Flux.testmode!(model)

                let 
                batch_loss_ideal_train_set = 0.0
                batch_loss_ideal_test_set = 0.0

                for eval_train_batch in eval_train_dataset

                    eval_x, eval_phi, eval_psi, eval_acc, eval_len = eval_train_batch

                    eval_x_processed = fastaVocab(preprocess(x, hyperpar["max_seq_len"])) |> gpu
                    eval_y_processed = build_regressor_target_array(phi, psi, acc, hyperpar["max_seq_len"]) |> gpu
                    eval_mask_processed = prepare_regressor_mask(x, phi, psi, acc, hyperpar["max_seq_len"]) |> gpu

                    network_outputs = model(eval_x_processed)
                    batch_loss_ideal_train_set += eval_loss(network_outputs, eval_y_processed, eval_mask_processed)

                end #end IDEAL_TRAIN_EVAL loop

                for eval_test_batch in eval_test_dataset

                    eval_x, eval_phi, eval_psi, eval_acc, eval_len = eval_test_batch

                    eval_x_processed = fastaVocab(preprocess(x, hyperpar["max_seq_len"])) |> gpu
                    eval_y_processed = build_regressor_target_array(phi, psi, acc, hyperpar["max_seq_len"]) |> gpu
                    eval_mask_processed = prepare_regressor_mask(x, phi, psi, acc, hyperpar["max_seq_len"]) |> gpu

                    network_outputs = model(eval_x_processed)
                    batch_loss_ideal_test_set += eval_loss(network_outputs, eval_y_processed, eval_mask_processed)
                    
                end #end IDEAL_TEST_EVAL loop

            ideal_train_loss = batch_loss_ideal_train_set
            ideal_test_loss = batch_loss_ideal_test_set

            Flux.trainmode!(model)

            checkpoint_model = cpu(model)
            @save "TintiNet-IGBT-Regressor-Checkpoint-$(Dates.format(Dates.now(), dformat)).bson" checkpoint_model opt
            @printf("Epoch: %d\t Batch: %d\t IdTrLos: %2.6f\t IdTsLoss: %2.6f\t Time: %s\n",
                    epoch, i, ideal_train_loss, ideal_test_loss, Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))

            end # end let

            end # endif

        end # end for batch

    end # end for epoch

end # end function train!()

#####
# Uncomment next line to allow the script to undergo training.
# train!(hyperpar, training_dataset, model, ps, opt)
