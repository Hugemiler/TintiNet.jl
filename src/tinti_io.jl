#####
## Functions to deal with reading/writing from/to files
#

# 1. Function to read a batch of fasta sequences from file

"""
    read_sequences_from_file(seq_file_path::String, max_seq_len::Int; filetype::String)

    This function receives a String `seq_file_path` pointing to the input file
    formatted according to `filetype`. Sequences will be trimmed to max_seq_len.
    Currently supporting only multi-sequence FASTA files, but support to other
    types of input may be added in the future.
"""
function read_sequences_from_file(
    seq_file_path::String,
    max_seq_len::Int;
    filetype="fasta",
    headertype="cath")

    if lowercase(filetype) == "fasta"

        fastaFile = open(FASTA.Reader, seq_file_path)

        headers = Vector{String}()
        splitFastaSequences = Vector{Vector{String}}()

            for record in fastaFile

                headertype=="simple" && push!(headers, FASTA.header(record))
                headertype=="cath" && push!(headers, split(split(FASTA.header(record),"|")[end], "/")[1])

                thisSequence = [ string(x) for x in FASTA.sequence(String, record) ]
                thisSeqLen = length(thisSequence)
                push!(
                    splitFastaSequences,
                    thisSequence[1:minimum([max_seq_len, thisSeqLen])]
                )
            end

        close(fastaFile)

        return(headers, splitFastaSequences)

    end

end

# 2. Function to write CSV outputs for the predictions

"""
    write_csv_predictions(output_folder::String, headers::Vector{String}, sequences::Vector{Vector{String}},
    ss_predictions::Vector{Vector{String}}, phi_predictions::Vector{Vector{Float64}},
    psi_predictions::Vector{Vector{Float64}}, acc_predictions::Vector{Vector{Float64}})


"""
function write_csv_predictions(
    output_folder::String,
    headers::Vector{String},
    sequences::Vector{Vector{String}},
    ss_predictions::Vector{Vector{String}},
    phi_predictions::Vector{Vector{Float64}},
    psi_predictions::Vector{Vector{Float64}},
    acc_predictions::Vector{Vector{Float64}})

    N_sequences = length(headers) 

    for i in 1:N_sequences

        thisOutputFilePath = output_folder*"/"*headers[i]*".csv"

        predictionMatrix = hcat(
            1:length(sequences[i]),
            sequences[i],
            ss_predictions[i],
            phi_predictions[i],
            psi_predictions[i],
            acc_predictions[i]
        )

        open(thisOutputFilePath, "w") do outputfile
            println(outputfile, "position,aa,pred_SS,pred_PHI,pred_PSI,pred_ACC")
            writedlm(outputfile, predictionMatrix, ',')
        end

    end

end

function write_csv_predictions_debug(
    output_folder::String,
    headers::Vector{String},
    sequences::Vector{Vector{String}},
    ss_predictions::Vector{Vector{String}},
    phi_sin_predictions::Vector{Vector{Float64}},
    phi_cos_predictions::Vector{Vector{Float64}},
    psi_sin_predictions::Vector{Vector{Float64}},
    psi_cos_predictions::Vector{Vector{Float64}},
    acc_predictions::Vector{Vector{Float64}})

    N_sequences = length(headers) 

    for i in 1:N_sequences

        thisOutputFilePath = output_folder*"/"*headers[i]*".csv"

        predictionMatrix = hcat(
            1:length(sequences[i]),
            sequences[i],
            ss_predictions[i],
            phi_sin_predictions[i],
            phi_cos_predictions[i],
            psi_sin_predictions[i],
            psi_cos_predictions[i],
            acc_predictions[i]
        )

        open(thisOutputFilePath, "w") do outputfile
            println(outputfile, "position,aa,pred_SS,pred_PHI_sin,pred_PHI_cos,pred_PSI_sin,pred_PSI_cos,pred_ACC")
            writedlm(outputfile, predictionMatrix, ',')
        end

    end

end
