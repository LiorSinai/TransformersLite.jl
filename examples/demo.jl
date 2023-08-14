using Random
using DataFrames
using Arrow
using Printf
using BSON, JSON
using Flux
using Flux.CUDA
using Flux: DataLoader
using Unicode
using Dates
using StatsBase: mean

#using TokenizersLite # Uncomment if using bpe or affixer tokenizers below
using TransformersLite
include("utilities.jl")
include("training.jl")

path = normpath(joinpath(@__DIR__, "..", "datasets", "amazon_reviews_multi", "en", "1.0.0"))
filename = "amazon_reviews_multi-train.arrow"
to_device = gpu # gpu or cpu

fingerprint = readdir(path)[1]
filepath = joinpath(path, fingerprint, filename)

df = DataFrame(Arrow.Table(filepath))
display(first(df, 20))
println("")

hyperparameters = Dict(
    "seed" => 2718,
    "tokenizer" => "none", # options: none bpe affixes
    "nlabels" => 5,
    "pdrop" => 0.1,
    "dim_embedding" => 32
)
nlabels = hyperparameters["nlabels"]
n_epochs = 10

## Tokenizers

if hyperparameters["tokenizer"] == "bpe"
    directory = joinpath("vocab", "bpe")
    path_rules = joinpath(directory, "amazon_reviews_train_en_rules.txt")
    path_vocab = joinpath(directory, "amazon_reviews_train_en_vocab.txt")
    tokenizer = load_bpe(path_rules, startsym="⋅")
elseif hyperparameters["tokenizer"] == "affixes"
    directory = joinpath("vocab","affixes")
    path_vocab = joinpath(directory, "amazon_reviews_train_en_vocab.txt")
    tokenizer = load_affix_tokenizer(path_vocab)
elseif hyperparameters["tokenizer"] == "none"
    path_vocab = joinpath("vocab", "amazon_reviews_train_en.txt")
    tokenizer = identity
end

vocab = load_vocab(joinpath(@__DIR__, path_vocab))
indexer = IndexTokenizer(vocab, "[UNK]")

display(tokenizer)
println("")
display(indexer)
println("")

## Tokens

documents = df[!, :review_body]
labels = df[!, :stars]
max_length = 50
indices_path = joinpath(@__DIR__, "outputs", "indices_" * hyperparameters["tokenizer"] * ".bson")
@time tokens = map(d->preprocess(d, tokenizer, max_length=max_length), documents)
@time indices = indexer(tokens)

y_labels = Int.(labels)
if nlabels == 1
    y_labels[labels .≤ 2] .= 0
    y_labels[labels .≥ 4] .= 1
    idxs = labels .!= 3
    y_labels = reshape(y_labels, 1, :)
else
    idxs = Base.OneTo(length(labels))
    y_labels = Flux.onehotbatch(y_labels, 1:nlabels)
end

X_train, y_train = indices[:, idxs], y_labels[:, idxs];
rng = MersenneTwister(hyperparameters["seed"])
train_data, val_data = split_validation(rng, X_train, y_train)

println("train samples:      ", size(train_data[1]), " ", size(train_data[2]))
println("validation samples: ", size(val_data[1]), " ", size(val_data[2]))
println("")

## Model 
dim_embedding = hyperparameters["dim_embedding"]
pdrop = hyperparameters["pdrop"]
# position_encoding = PositionEncoding(dim_embedding) |> to_device
# add_position_encoding(x) = x .+ position_encoding(x)
# model = Chain(
#     Embed(dim_embedding, length(indexer)), 
#     add_position_encoding, 
#     Dropout(pdrop),
#     TransformerEncoderBlock(4, dim_embedding, dim_embedding * 4; pdrop=pdrop),
#     Dense(dim_embedding, 1),
#     TransformersLite.FlattenLayer(),
#     Dense(max_length, nlabels)
#     )
model = TransformersLite.TransformerClassifier(
    Embed(dim_embedding, length(indexer)), 
    PositionEncoding(dim_embedding), 
    Dropout(pdrop),
    TransformerEncoderBlock[
        TransformerEncoderBlock(4, dim_embedding, dim_embedding * 4; pdrop=pdrop)
    ],
    Dense(dim_embedding, 1), 
    FlattenLayer(),
    Dense(max_length, nlabels)
    )
display(model)
println("")
model = to_device(model) 

hyperparameters["model"] = "$(typeof(model).name.wrapper)"
hyperparameters["trainable parameters"] = sum(length, Flux.params(model));

if nlabels == 1
    loss(x, y) = Flux.logitbinarycrossentropy(x, y)
    accuracy(ŷ, y) = mean((Flux.sigmoid.(ŷ) .> 0.5) .== y)
else
    loss(x, y) = Flux.logitcrossentropy(x, y)
    accuracy(ŷ, y) = mean(Flux.onecold(ŷ) .== Flux.onecold(y))
end

## Training
opt_state = Flux.setup(Adam(), model)

batch_size = 32
train_data_loader = DataLoader(train_data |> to_device; batchsize=batch_size, shuffle=true)
val_data_loader = DataLoader(val_data |> to_device; batchsize=batch_size, shuffle=false)

val_acc = batched_metric(model, accuracy, val_data_loader)
val_loss = batched_metric(model, loss, val_data_loader)

@printf "val_acc=%.4f%% ; " val_acc * 100
@printf "val_loss=%.4f \n" val_loss
println("")

directory = normpath( joinpath(@__DIR__, "..", "outputs", Dates.format(now(), "yyyymmdd_HHMM")))
mkpath(directory)
output_path = joinpath(directory, "model.bson")
history_path = joinpath(directory, "history.json")
hyperparameter_path = joinpath(directory, "hyperparameters.json")

open(hyperparameter_path, "w") do f
    JSON.print(f, hyperparameters)
end
println("saved hyperparameters to $(hyperparameter_path).")
println("")

start_time = time_ns()
history = train!(
    loss, model, train_data_loader, opt_state, val_data_loader
    ; num_epochs=n_epochs)
end_time = time_ns() - start_time
println("done training")
@printf "time taken: %.2fs\n" end_time/1e9

## Save 

model = model |> cpu
if hasproperty(tokenizer, :cache)
    # empty cache
    tokenizer = similar(tokenizer)
end
BSON.bson(
    output_path, 
    Dict(
        :model=> model, 
        :tokenizer=>tokenizer,
        :indexer=>indexer
    )
    )
println("saved model to $(output_path).")

open(history_path,"w") do f
  JSON.print(f, history)
end
println("saved history to $(history_path).")