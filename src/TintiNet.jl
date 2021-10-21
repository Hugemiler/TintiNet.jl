module TintiNet

    #####
    # 0. Package Dependencies
    #####    

    using CUDA
    using Dates, Printf, Random, BSON, JSON, Flux;
    using IterTools: ncycle;
    using Transformers, Transformers.Basic, Transformers.Pretrain, Transformers.Datasets, Transformers.BidirectionalEncoder;
    using BSON: @load, @save
    using Flux: @functor, onehot, onecold, mse, shuffle, sigmoid, gradient, unsqueeze
    using Flux.Optimise: update!
    using Statistics: mean, std

    #####
    # 1. Source Files
    #####    

    include("./definitions.jl")     # Stactic dictionaries
    include("./preprocessing.jl")   # Functions to prepare data for training/evaluation
    include("./architectures.jl")   # Layer architecture definitions
    include("./training.jl")        # Functions for training the neural network
    include("./evaluation.jl")      # Functions for evaluating samples using a model

    #####
    # 2. Namespace export
    #####    

    # From definitions.jl
    export fastaCategories,
        fastaVocab,
        ss3Categories,
        ss3Vocab,
        ss8Categories,
        ss8Vocab

    # From preprocessing.jl
    export pad_aa_seq,
        preprocess,
        prepare_mask,
        split_train_test

    # From architectures.jl
    export Inception4,
        Inception8

    # From training.jl
    export generic_smooth,
        train_loss,
        eval_loss,
        accuracy_hits

    # From evaluation.jl
    export network_pass_single_GPU,
        network_pass_single_CPU,
        network_pass_batched_GPU,
        network_pass_batched_CPU,
        compute_predictions

end
