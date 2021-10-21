#####
#
# Example script for TintiNet.jl
# Model IGBT (InceptGOR-Bert): 3-state SS Classifier
#
# To run training, revise your code and uncomment the last line.
# On slower GPUs, an epoch may take several minutes with the default configuration.
# Adjust model size using the `hyperpar` and `config` dictionaries.
#
#####

#####
# 0. Loading Packages and defining Hyperparameters
#####

using CUDA, Flux, Transformers, JSON, Random, Dates, Printf
using IterTools: ncycle;
using Transformers.Basic, Transformers.Pretrain, Transformers.Datasets, Transformers.BidirectionalEncoder
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
    "max_seq_len"       => 192,     # Maximum sequence length on the model (default 192)
    "batch_size"        => 128,     # Batch size for training (default 128)
    "number_of_epochs"  => 10,      # Number of epochs for training (10 for the example file)
    "learning_rate"     => 1e-3,    # Learning rate for optimizer (default 0.001)
    "smooth_epsilon"    => 1e-6     # Small value to smooth onehot result before logcrossentropy (default 1e-6)
)

##
# Model Configuration

const config = Dict(
  "input_vocab_size"                => 22,      # Size of the SEQUENCE/FASTA vocabulary, usually 20 aminoacids + "X" + "-" 
  "output_vocab_size"               => 4,       # Size of the STRUCTURE/SS3 vocabulary, usually 3 classes + "-" 
  "embed_dimension"                 => 64,      # Size of the individual Embed vectors
  "bert_num_hidden_layers"          => 2,       # Number of consecutive Transformer layers on the BERT encoder
  "bert_transformer_layer_heads"    => 8,       # Number of heads in each Transformer layer
  "bert_transformer_layer_hidden"   => 128,     # Hidden size of the positionwise linear layer in each Transformer layer
  "bert_transformer_dropout_prob"   => 0.1,     # Dropout probability for the Transformer layers
  "bert_transformer_activation"     => gelu,    # Activation function for each Transformer layer (default: gelu)
  "detection_num_filters"           => 64       # Number of filters in each convolutional decoder layer
)

#####
# 2. Obtaining and shaping the data
#####

## 2.1. Training sequences, structures and lengths

training_set_datapath = "../data/fold01training_dataset.json"

open(training_set_datapath, "r") do file
    
    entries = JSON.parse(file)
    num_entries = length(entries)
    
    parsed_sequences = Vector{Vector{String}}(undef, num_entries)
    parsed_structures = Vector{Vector{String}}(undef, num_entries)
    parsed_lengths = Vector{Int}(undef, num_entries)

    for (i,j) in enumerate(entries)
        parsed_sequences[i] = string.(j["fasta_seq"])
        parsed_structures[i] = string.(j["dssp_ss3"])
        parsed_lengths[i] = length(parsed_sequences[i])
    end

    order_idx = sortperm(parsed_lengths)

    global train_sequences_ordered = parsed_sequences[order_idx]
    global train_structures_ordered = parsed_structures[order_idx]
    global train_lengths_ordered = parsed_lengths[order_idx]
    
end

## 2.2. Testing sequences, structures and lengths

testing_set_datapath = "../data/fold01testing_dataset.json"

open(testing_set_datapath, "r") do file
    
    entries = JSON.parse(file)
    num_entries = length(entries)
    
    parsed_sequences = Vector{Vector{String}}(undef, num_entries)
    parsed_structures = Vector{Vector{String}}(undef, num_entries)
    parsed_lengths = Vector{Int}(undef, num_entries)

    for (i,j) in enumerate(entries)
        parsed_sequences[i] = string.(j["fasta_seq"])
        parsed_structures[i] = string.(j["dssp_ss3"])
        parsed_lengths[i] = length(parsed_sequences[i])
    end

    order_idx = sortperm(parsed_lengths)

    global test_sequences_ordered = parsed_sequences[order_idx]
    global test_structures_ordered = parsed_structures[order_idx]
    global test_lengths_ordered = parsed_lengths[order_idx]
    
end

## 2.3. Lazy DataLoaders

training_dataset = Flux.Data.DataLoader((train_sequences_ordered, train_structures_ordered, train_lengths_ordered); batchsize=hyperpar["batch_size"], shuffle=true)
eval_train_dataset = Flux.Data.DataLoader((train_sequences_ordered, train_structures_ordered, train_lengths_ordered); batchsize=64,shuffle=false)
eval_test_dataset = Flux.Data.DataLoader((test_sequences_ordered, test_structures_ordered, test_lengths_ordered); batchsize=64,shuffle=false)

const train_acc_denominator = floor(sum(train_lengths_ordered)*0.9615) # Number of non-gap residues in the training samples. Adjust when using other dataset.
const test_acc_denominator = floor(sum(test_lengths_ordered)*0.9625) # Number of non-gap residues in the testing samples. Adjust when using other dataset.

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
        Conv((1,5), config["embed_dimension"] => config["detection_num_filters"], sigmoid; pad = (0,2)),
        Conv((1,5), config["detection_num_filters"] => config["output_vocab_size"], identity; pad = (0,2)),
        x -> permutedims(x, [3,2,1,4]),
        x -> reshape(x, size(x,1), size(x,2), size(x,4)),
        logsoftmax
    ) |> gpu

#####
# 4. Model training
#####

smooth(et) = generic_smooth(et, hyperpar["smooth_epsilon"], config["input_vocab_size"])     # Creates an alias for smooth() obeying our hyperparameters
opt = Flux.Optimise.ADAM(hyperpar["learning_rate"])                                 # Initializes the ADAM optimizer
ps = params(model)      # Collects model parameters

#define training loop
function train!(hyperpar, training_dataset,
                model, ps, opt,
                smooth=smooth, train_loss=train_loss, eval_loss=eval_loss, accuracy_hits=accuracy_hits,
                eval_train_dataset=eval_train_dataset, eval_test_dataset=eval_test_dataset,
                fastaVocab=fastaVocab, ss3Vocab=ss3Vocab)
    @info "start training"
    for epoch in 1:hyperpar["number_of_epochs"]
    i = 0
        for batch in training_dataset
            i += 1
            x, y, l = batch

            train_x_processed = fastaVocab(preprocess(x)) |> gpu
            train_y_processed = ss3Vocab(preprocess(y)) |> gpu
            train_mask_processed = prepare_mask(y) |> gpu

            grad = gradient(()->train_loss(model, train_x_processed, train_y_processed, train_mask_processed, smooth), ps)
            update!(opt, ps, grad)

            if i%64 == 0

                Flux.testmode!(model)

                let 
                batch_loss_ideal_train_set = 0.0
                batch_loss_ideal_test_set = 0.0
                batch_hits_ideal_train_set = 0
                batch_hits_ideal_test_set = 0

                for eval_train_batch in eval_train_dataset

                    eval_x, eval_y, eval_length = eval_train_batch

                    eval_x_processed = fastaVocab(preprocess(eval_x)) |> gpu
                    eval_y_processed = ss3Vocab(preprocess(eval_y)) |> gpu
                    eval_mask_processed = prepare_mask(eval_y) |> gpu

                    network_outputs = model(eval_x_processed)
                    network_predictions = Flux.onecold(network_outputs)

                    batch_loss_ideal_train_set += eval_loss(network_outputs, eval_y_processed, eval_mask_processed, smooth)
                    batch_hits_ideal_train_set += accuracy_hits(cpu(network_predictions), cpu(eval_y_processed), eval_length)

                end #end IDEAL_TRAIN_EVAL loop

                for eval_test_batch in eval_test_dataset

                    eval_x, eval_y, eval_length = eval_test_batch

                    eval_x_processed = fastaVocab(preprocess(eval_x)) |> gpu
                    eval_y_processed = ss3Vocab(preprocess(eval_y)) |> gpu
                    eval_mask_processed = prepare_mask(eval_y) |> gpu

                    network_outputs = model(eval_x_processed)
                    network_predictions = Flux.onecold(network_outputs)

                    batch_loss_ideal_test_set += eval_loss(network_outputs, eval_y_processed, eval_mask_processed, smooth)
                    batch_hits_ideal_test_set += accuracy_hits(cpu(network_predictions), cpu(eval_y_processed), eval_length)
                    
                end #end IDEAL_TEST_EVAL loop

            ideal_train_accuracy = batch_hits_ideal_train_set/train_acc_denominator
            ideal_test_accuracy = batch_hits_ideal_test_set/test_acc_denominator
            ideal_train_loss = batch_loss_ideal_train_set
            ideal_test_loss = batch_loss_ideal_test_set

            Flux.trainmode!(model)

            checkpoint_model = cpu(model)
            @save "TintiNet-IGBT-Classifier-$(Dates.format(Dates.now(), dformat)).bson" checkpoint_model opt
            @printf("Epoch: %d\t Batch: %d\t IdTrLos: %2.6f\t IdTsLoss: %2.6f\t IdTrAcc: %2.6f\t IdTsAcc: %2.6f\t Time: %s\n",
                    epoch, i, ideal_train_loss, ideal_test_loss, ideal_train_accuracy, ideal_test_accuracy, Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))

            end # end let

            end # endif

        end # end for batch

    end # end for epoch

end # end function train!()

#####
# Uncomment next line to allow the script to undergo training.
# train!(hyperpar, training_dataset, model, ps, opt)
