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

using CUDA, Flux, Transformers, JSON, BSON, Random, Dates, Printf
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
)

#####
# 2. Obtaining and shaping the data
#####

## 2.2. Testing sequences, structures and lengths

testing_set_datapath = "/home/guilherme/Documentos/TINTI/ext/data_folds_128/fold_01_128/fold_01_128_test_dataset.json"

open(testing_set_datapath, "r") do file
    global entries = JSON.parse(file)
end

sequences = [ string.(entries[i]["fasta_seq"]) for i in 1:length(entries) ]

#####
# 3. Loading the saved models
#####

@load "./TintiNet_Classifier_checkpoint.bson" checkpoint_model
classifier_model = checkpoint_model |> gpu
Flux.testmode!(classifier_model)

@load "./TintiNet_Regressor_checkpoint.bson" checkpoint_model
regressor_model = checkpoint_model |> gpu
Flux.testmode!(regressor_model)

#####
# 4. Model evaluation
#####

ss_predictions = compute_predictions(
    classifier_model,
    sequences;
    batched=true,
    eval_batch_size=64,
    use_gpu=true,
    sleep_time_seconds=0.01)

(phi_predictions, psi_predictions, acc_predictions) = compute_regressor_predictions(
    regressor_model,
    sequences;
    batched=false,
    eval_batch_size=64,
    use_gpu=true,
    sleep_time_seconds=0.01)

#####
# 5. Saving Prediction Results as JSON file
#####

for i in 1:length(sequences)

    entries[i]["tinti_ss3_prediction"] = ss_predictions[i]
    entries[i]["tinti_phi_prediction"] = phi_predictions[i]
    entries[i]["tinti_psi_prediction"] = psi_predictions[i]
    entries[i]["tinti_acc_prediction"] = acc_predictions[i]

end

open("./TintiNet_prediction_results.json", "w") do dbFile
    JSON.print(dbFile, entries)
end
