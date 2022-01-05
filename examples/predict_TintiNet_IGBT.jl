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

inputfile_path = ARGS[1] ## First argument is the location of the sequence input file
output_path = ARGS[2] ## Second argument is the directory to write the outputs
#print_header_prediction()

#####
# 1. Obtaining the data
#####

headers, sequences = read_sequences_from_file(seq_file_path, hyperpar["max_seq_len"]; filetype="fasta")

#####
# 2. Loading the saved models
#####

@load "./example_classifier_model.bson" checkpoint_model
classifier_model = checkpoint_model |> gpu
Flux.testmode!(classifier_model)

@load "./example_regressor_model.bson" checkpoint_model
regressor_model = checkpoint_model |> gpu
Flux.testmode!(regressor_model)

#####
# 3. Model evaluation
#####

ss_predictions = compute_classifier_predictions(
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
# 4. Saving Prediction Results
#####

write_csv_predictions(output_path, headers, sequences, ss_predictions, phi_predictions, psi_predictions, acc_predictions)
