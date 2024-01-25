using Flux, Transformers, JSON, BSON, Random, Dates, Printf
using IterTools: ncycle;
using Transformers.Basic, Transformers.Pretrain, Transformers.Datasets, Transformers.Stacks, Transformers.BidirectionalEncoder
using BSON: @load, @save
using Flux: @functor, onehot, onecold, mse, shuffle, sigmoid, gradient, unsqueeze
using Flux.Optimise: update!
using Statistics: mean, std
using TintiNet

using CSV, DataFrames, Chain
using CairoMakie

logfiles = [
    "examples/saved_outputs/output_classifier_1.txt",
    "examples/saved_outputs/output_classifier_2.txt",
    "examples/saved_outputs/output_classifier_3.txt",
    "examples/saved_outputs/output_classifier_4.txt",
    "examples/saved_outputs/output_classifier_5.txt",
    "examples/saved_outputs/output_classifier_6.txt",
    "examples/saved_outputs/output_classifier_7.txt",
    "examples/saved_outputs/output_classifier_8.txt",
    "examples/saved_outputs/output_classifier_9.txt",
    "examples/saved_outputs/output_classifier_10.txt"
    ]

fig = Figure(resolution = (1000,600))
ax1 = Axis(
    fig[1,1],
    xlabel = "Epoca",
    ylabel = "Erro de Classificacao (entropia cruzada)",
    title = "Regressao do angulo PHI"
)
ax2 = Axis(
    fig[1,2],
    xlabel = "Epoca",
    ylabel = "Acuracia de Classificacao (Q3)",
    title = "Regressao do angulo PSI"
)
ax3 = Axis(
    fig[1,2],
    xlabel = "Epoca",
    ylabel = "Acuracia de Classificacao (Q3)",
    title = "Regressao da Acessibilidade (ACC)"
)

for i in eachindex(logfiles)

    @show i

    log = @chain CSV.read(logfiles[i], DataFrame;delim = ' ', header = false) begin
        select!( [ :Column2, :Column4, :Column6, :Column8, :Column10, :Column12 ])
        rename!( [ :Column2 => "Epoch", :Column4 => "Batch", :Column6 => "Train_Error", :Column8 => "Test_Error", :Column10 => "Train_Acc", :Column12 => "Test_Acc" ])
        transform!(:Train_Error => (x -> x ./ 9) => :Train_Error; renamecols = false)
        transform!( [:Epoch, :Batch] => ( (x,y) -> x .+ (y ./ 200)) => :X_Axis; renamecols = false)
        insertcols!(1, :Fold => i)
    end

    lines!(ax1, log.X_Axis, log.Train_Error; color = :red )
    lines!(ax1, log.X_Axis, log.Test_Error; color = :blue )

    lines!(ax2, log.X_Axis, log.Train_Acc; color = :red )
    lines!(ax2, log.X_Axis, log.Test_Acc; color = :blue )

end

xlims!(ax1, [0, 40])
xlims!(ax2, [0, 40])

Legend(
    fig[2, :],
    [
        LineElement(;color=:red, strokewidth=1),
        LineElement(;color=color = :blue, strokewidth=1),
    ],
    [
        "Folds de Treino",
        "Folds de Validacao"
    ];
    orientation = :horizontal
)

Label(fig[0,:], "Curvas de aprendizagem para o modelo de Classificacao")

save("thesis_notebooks/figures/classification_learning_curves.png", fig)