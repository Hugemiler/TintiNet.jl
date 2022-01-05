#####
## Functions for network training in TintiNet.jl
#

"""
    pad_aa_seq(v::Vector{String}, n::Integer, p::String)

    This function receives a `Vector{String}` `v` and returns a same-length `Vector{String}`
    with every element of `v` padded to length `n` with symbol `p`
"""
function generic_smooth(et, epsilon, vocabsize)
    global hyperpar, config
    sm = fill!(similar(et, Float32),epsilon/vocabsize)
    p = sm .* (1 .+ -et)
    label = p .+ et .* (1 - convert(Float32, epsilon)) # Conversion to Float32 for GPU calculations
    label
end

Flux.@nograd generic_smooth

"""
    pad_aa_seq(v::Vector{String}, n::Integer, p::String)

    This function receives a `Vector{String}` `v` and returns a same-length `Vector{String}`
    with every element of `v` padded to length `n` with symbol `p`
"""
function train_loss(model, x, y, mask, smooth, structureVocab)
    labels = smooth(onehot(structureVocab, y))
    l = Transformers.Basic.logcrossentropy(model(x), labels, mask)
    return(l)
end

"""
    pad_aa_seq(v::Vector{String}, n::Integer, p::String)

    This function receives a `Vector{String}` `v` and returns a same-length `Vector{String}`
    with every element of `v` padded to length `n` with symbol `p`
"""
function eval_loss(predictions, y, mask, smooth, structureVocab)
    labels = smooth(onehot(structureVocab, y))
    l = Transformers.Basic.logcrossentropy(predictions, labels, mask)
    return(l)
end

"""
    pad_aa_seq(v::Vector{String}, n::Integer, p::String)

    This function receives a `Vector{String}` `v` and returns a same-length `Vector{String}`
    with every element of `v` padded to length `n` with symbol `p`
"""
function accuracy_hits(predictions, y, lengths)

    hits = 0
    
    for j in 1:size(y,2)
        for i in 1:lengths[j]
            if predictions[i,j] == y[i,j]
                hits += 1
            end
        end
    end
    return(hits)
end
