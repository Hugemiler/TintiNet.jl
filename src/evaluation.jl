#####
## Static definitions for use with TintiNet.jl
#

# 1. Functions to pass a single sample through the network

"""
    network_pass_single_GPU(model, x::Vector{String})

    This function receives a single sequence `x` as a `Vector{String}`,
    applies a preprocessing function and sends it to the GPU, passing it through the network `model`,
    bringing the result back to CPU and returning it as a CPU object.
"""
function network_pass_single_GPU(model, x::Vector{String})

    x_gpu = fastaVocab(x)
    x_gpu = reshape(x_gpu, :, 1) |> gpu
    network_outputs = model(x_gpu) |> cpu
    CUDA.unsafe_free!(x_gpu)

    return(network_outputs)

end

"""
    network_pass_single_CPU(model, x::Vector{String})

    This function receives a single sequence `x` as a `Vector{String}`,
    applies a preprocessing function and passes through the network `model` in the CPU,
    returning the result.
"""
function network_pass_single_CPU(model, x::Vector{String})

    x = fastaVocab(x)
    x = reshape(x_gpu, :, 1)
    network_outputs = model(x_gpu)

    return(network_outputs)

end

# 2. Functions to pass a batch of samples through the network

"""
    network_pass_batched_GPU(model, x::Vector{String})

    This function receives a batch of sequences `x` as a `Vector{Vector{String}}`,
    applies a preprocessing function and sends it to the GPU, passing it through the network `model`,
    bringing the result back to CPU and returning it as a CPU object.
"""
function network_pass_batched_GPU(model, x::Vector{Vector{String}})

    x_gpu = fastaVocab(preprocess(x, 128)) |> gpu
    network_outputs = model(x_gpu) |> cpu

    return(network_outputs)

end

"""
    network_pass_single_CPU(model, x::Vector{String})

    This function receives a batch of sequences `x` as a `Vector{Vector{String}}`,
    applies a preprocessing function and passesng it through the network `model`,
    returning the network outputs.
"""
function network_pass_batched_CPU(model, x::Vector{Vector{String}})

    x_cpu = fastaVocab(preprocess(x, 128))
    network_outputs = model(x_cpu)

    return(network_outputs)

end

# 3. Function to compute classifier predictions using the network_pass functions

"""
    compute_classifier_predictions(model, sequences; batched=true, batch_size=64, use_gpu=true, sleep_time_seconds=0.02)

    This function receives a `Vector{String}` `v` and returns a same-length `Vector{String}`
    with every element of `v` padded to length `n` with symbol `p`
"""
function compute_classifier_predictions(
            model,
            sequences::Vector{Vector{String}};
            batched=false,
            eval_batch_size=64,
            use_gpu=true,
            sleep_time_seconds=0.02)

    # Pre-allocate the vector containing the prediction strings
    predictions_vector = Vector{Vector{String}}(undef, length(sequences))

    if batched # Utilize a batched network pass

        prediction_position = 1
        sequence_batches = Flux.Data.DataLoader(sequences; batchsize=eval_batch_size, shuffle=false)

        for batch in sequence_batches

            seq_lens = length.(batch)
            batch_outputs = use_gpu ? network_pass_batched_GPU(model, batch) : network_pass_batched_CPU(model, batch)
            batch_prediction = Flux.onecold(batch_outputs)

            for i in 1:length(batch)
                predictions_vector[prediction_position] = ss3Vocab[vec(batch_prediction[1:seq_lens[i], i])]
                prediction_position += 1
            end

#            sleep(sleep_time_seconds)
        end
    
    else # Utilize a single-sequence network pass

        for i in 1:length(sequences)
            single_prediction = use_gpu ? network_pass_single_GPU(model, sequences[i]) : network_pass_single_CPU(model, sequences[i])
            predictions_vector[i] = ss3Vocab[vec(Flux.onecold(single_prediction))]
            sleep(sleep_time_seconds)
        end
    
    end # end if batched

    return(predictions_vector)

end

# 3. Function to compute regressor predictions using the network_pass functions

"""
    compute_regressor_predictions(model, sequences; batched=true, batch_size=64, use_gpu=true, sleep_time_seconds=0.02)

    This function receives a `Vector{String}` `v` and returns a same-length `Vector{String}`
    with every element of `v` padded to length `n` with symbol `p`
"""
function compute_regressor_predictions(
            model,
            sequences::Vector{Vector{String}};
            batched=false,
            eval_batch_size=64,
            use_gpu=true,
            sleep_time_seconds=0.02)

    # Pre-allocate the vector containing the prediction strings
    predicted_phis_vector = Vector{Vector{Float64}}(undef, length(sequences))
    predicted_psis_vector = Vector{Vector{Float64}}(undef, length(sequences))
    predicted_accs_vector = Vector{Vector{Float64}}(undef, length(sequences))

    if batched # Utilize a batched network pass

        prediction_position = 1
        sequence_batches = Flux.Data.DataLoader(sequences; batchsize=eval_batch_size, shuffle=false)

        for batch in sequence_batches

            seq_lens = length.(batch)
            batch_outputs = use_gpu ? network_pass_batched_GPU(model, batch) : network_pass_batched_CPU(model, batch)

            for i in 1:length(batch)

                predicted_phis_vector[prediction_position] = atand.(single_prediction[1, j, 1:seq_lens[prediction_position]], single_prediction[2, j, 1:seq_lens[prediction_position]])
                predicted_psis_vector[prediction_position] = atand.(single_prediction[3, j, 1:seq_lens[prediction_position]], single_prediction[4, j, 1:seq_lens[prediction_position]])
                predicted_accs_vector[prediction_position] = (single_prediction[5, j, 1:seq_lens[prediction_position]] .* 100) .+ 100
                prediction_position += 1

            end

#            sleep(sleep_time_seconds)
        end

    else # Utilize a single-sequence network pass

        for i in 1:length(sequences)
            single_prediction = use_gpu ? network_pass_single_GPU(model, sequences[i]) : network_pass_single_CPU(model, sequences[i])

#        @show single_prediction[1,1] # For debugging

            predicted_phis_vector[i] = [ atand(single_prediction[1, j], single_prediction[2, j]) for j in 1:length(sequences[i]) ]
            predicted_psis_vector[i] = [ atand(single_prediction[3, j], single_prediction[4, j]) for j in 1:length(sequences[i]) ] 
            predicted_accs_vector[i] = [ (single_prediction[5, j]*100)+100 for j in 1:length(sequences[i]) ]
            sleep(sleep_time_seconds)
        end
    
    end # end if batched

    return(predicted_phis_vector, predicted_psis_vector, predicted_accs_vector)

end
