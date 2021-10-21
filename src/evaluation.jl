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
    network_pass_single_GPU(model, x::Vector{String})

    This function receives a single sequence `x` as a `Vector{String}`,
    applies a preprocessing function and sends it to the GPU, passing it through the network `model`,
    bringing the result back to CPU and returning it as a CPU object.
"""
function network_pass_batched_GPU(model, x::Vector{String})

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
function network_pass_batched_CPU(model, x::Vector{String})

    x = fastaVocab(x)
    x = reshape(x_gpu, :, 1)
    network_outputs = model(x_gpu)

    return(network_outputs)

end

# 3. Function to compute predictions using the network_pass functions

"""
    compute_predictions(model, sequences; batched=true, batch_size=64, use_gpu=true, sleep_time_seconds=0.02)

    This function receives a `Vector{String}` `v` and returns a same-length `Vector{String}`
    with every element of `v` padded to length `n` with symbol `p`
"""
function compute_predictions(
            model,
            sequences::Vector{Vector{String}};
            batched=true,
            eval_batch_size=64,
            use_gpu=true,
            sleep_time_seconds=0.02)

    # Pre-allocate the vector containing the prediction strings
    predictions_vector = Vector{Vector{String}}(undef, length(sequences))

    if batched # Utilize a batched network pass

        next_pos = 1
        sequence_batches = Flux.Data.DataLoader(sequences; batchsize=eval_batch_size, shuffle=false)

        for batch in sequence_batches
            batch_prediction = use_gpu ? network_pass_batch_GPU(model, batch) : network_pass_batch_CPU(model, batch)
            predictions_vector[next_pos:minimum([next_pos+eval_batch_size-1, length(sequences)])] = ss3Vocab[Flux.onecold(single_prediction)]
            sleep(sleep_time_seconds)
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
