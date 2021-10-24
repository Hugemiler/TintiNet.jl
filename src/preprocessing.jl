#####
## Functions for preprocessing data in TintiNet.jl
#

"""
    pad_aa_seq(v::Vector{String}, n::Integer, p::String)

    This function receives a `Vector{String}` `v` and returns a same-length `Vector{String}`
    with every element of `v` padded to length `n` with symbol `p`
"""
pad_aa_seq(v::Vector{String}, n::Integer, p::String) = [v; fill(p, max(n - length(v), 0))]

"""
    preprocess(batchofseqs::Vector{Vector{String}}, n::Integer, p::String)

    This function applies `pad_aa_seq` to every element of `batchofseqs` with arguments
    `n` and `p` (default 120 and \"-\")
"""
preprocess(batchofseqs::Vector{Vector{String}}, n = 128, p = "-") = [ pad_aa_seq([singleseq...], n, "-") for singleseq in batchofseqs ]

"""
    prepare_mask(batchofseqs::Vector{Vector{String}})

    This function returns a mask to be used on the masked loss function for training.
    The mask is an `Array{Float}` filled with ones in positions different from `unk` and zeros otherwise.
    This function makes use of `Transformers.Basic.getmask`
"""
function prepare_mask(batchofseqs::Vector{Vector{String}}, seqlen)

    mask = getmask(preprocess(batchofseqs, seqlen)) # Uses getmask() from Transformers.Basic

    for i in 1:length(batchofseqs)
        for j in 1:length(batchofseqs[i])
            if batchofseqs[i][j] == "-"
                mask[1, j, i] = 0.0
            end
        end
    end

    return(mask)

end

"""
    split_train_test(
        sequences::Vector{Vector{String}},
        structures::Vector{Vector{String}},
        maximum_length::Int,
        minimum_length=32;
        compute_complete_only=true,
        test_all_incomplete_obs=false,
        training_percentage=0.6)

    This function splits a collection of target inputs (`sequences`) and outputs (`structures`) into a test set and training set.
    Sequences/structures are filtered off if their length is smaller than `minimum_length`.
    Sequences/structures are eith length greater than `maximum_length` are trimmed.

    If `compute_complete_only` is `false` and `test_all_incomplete_obs` is `false`,
    all samples will be split according to the fraction `training_percentage`, independant of their completeness, by default.

    If `compute_complete_only` is `true` and `test_all_incomplete_obs` is `false`,
    samples with `unk` positions will be removed and the remaining samples will be split according to the fraction `training_percentage`.

    If `compute_complete_only` is `false` and `test_all_incomplete_obs` is `true`,
    samples with `unk` positions will be placed entirely on the test set, with remaining samples placed on train set, _regardless of_ `training_percentage`.

    If `compute_complete_only` is `true` and `test_all_incomplete_obs` is `true`,
    all samples will be placed on the training set. This is not an adequate use of split_train_test for practical purposes, but can be useful for debugging.
"""
function split_train_test(
            sequences::Vector{Vector{String}},
            structures::Vector{Vector{String}},
            maximum_length::Int,
            minimum_length=32;
            compute_complete_only=false,
            test_on_incomplete_obs=false,
            training_percentage=0.7)

    ## Step 1. Size filters

    sequences = [ el[1:minimum([length(el), maximum_length])] for el in sequences ]
    structures = [ el[1:minimum([length(el), maximum_length])] for el in structures ]

    equal_length_passed_idx = [ size(el) for el in sequences ] .== [ size(el) for el in structures ]
    max_length_passed_idx = [ size(el)[1] for el in sequences ] .<= maximum_length
    min_length_passed_idx = [ size(el)[1] for el in sequences ] .>= minimum_length

    filtered_selection = equal_length_passed_idx .& max_length_passed_idx .& min_length_passed_idx

    filtered_sequences = sequences[filtered_selection]
    filtered_structures = structures[filtered_selection]

    ## Step 2. Splitting between training and test sets.

    if (compute_complete_only & test_all_incomplete_obs) # This is just to handle misunderstading of function arguments
    
        @warn "Function called with compute_complete_only = true and train_all_complete_obs = true.\nTest set will contain zero observations."

        train_sequences = filtered_sequences
        test_sequences = Vector{Vector{String}}(undef, 0)
        train_structures = filtered_structures
        test_structures = Vector{Vector{String}}(undef, 0)

    elseif (compute_complete_only) # The complete sequences/structures selected in Step 1 will be split between training and test sets.

        n = length(filtered_sequences)
        idx = shuffle(1:n)
        train_idx = view(idx, 1:floor(Int, training_percentage*n))
        test_idx = view(idx, (floor(Int, training_percentage*n)+1):n)
        lengths = length.(filtered_sequences)

        train_sequences = filtered_sequences[train_idx]
        test_sequences = filtered_sequences[test_idx]
        train_structures = filtered_structures[train_idx]
        test_structures = filtered_structures[test_idx]

    else # The complete sequences/structures will be the training set, and the incomplete sequences/structures will be the testing set.

        train_idx = [ sum(el .== "-") for el in filtered_structures ] .== 0
        test_idx = .!train_idx

        train_sequences = filtered_sequences[train_idx]
        test_sequences = filtered_sequences[test_idx]
        train_structures = filtered_structures[train_idx]
        test_structures = filtered_structures[test_idx]

    end

    ## Step 3. Ordering the sets by length

    train_order = sortperm(length.(train_sequences))
    test_order = sortperm(length.(test_sequences))
    
    train_sequences_ordered = train_sequences[train_order]
    test_sequences_ordered = test_sequences[test_order]
    train_structures_ordered = train_structures[train_order]
    test_structures_ordered = test_structures[test_order]
    train_lengths_ordered = length.(train_sequences_ordered)
    test_lengths_ordered = length.(test_sequences_ordered)

    return(train_sequences_ordered, test_sequences_ordered, train_structures_ordered, test_structures_ordered, train_lengths_ordered, test_lengths_ordered)

end
