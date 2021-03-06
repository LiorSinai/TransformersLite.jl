using DataFrames
using Arrow
using Printf
using BSON, JSON
using Flux
using Unicode
using Dates

using TokenizersLite
using TransformersLite

using Revise
includet("training.jl")

path = "path\\to\\amazon_reviews_multi\\en\\1.0.0\\"
filename = "amazon_reviews_multi-train.arrow"

checksum = readdir(path)[1]
filepath = joinpath(path, checksum, filename)

df = DataFrame(Arrow.Table(filepath))
display(df)

hyperparameters = Dict(
    "seed" => 2718,
    "tokenizer" => "bpe",
    "nlabels" => 5,
    "model" => "TransformerClassifier",
    "pdrop" => 0.1,
    "dim_embedding" => 32
)
nlabels = hyperparameters["nlabels"]

## Tokenizers

if hyperparameters["tokenizer"] == "bpe"
    directory = "vocab\\bpe"
    path_rules = joinpath(directory, "amazon_reviews_train_en_rules.txt")
    path_vocab = joinpath(directory, "amazon_reviews_train_en_vocab.txt")
    tokenizer = load_bpe(path_rules, startsym="⋅")
elseif hyperparameters["tokenizer"] == "affixes"
    directory = "vocab\\affixes"
    path_vocab = joinpath(directory, "amazon_reviews_train_en_vocab.txt")
    tokenizer = load_affix_tokenizer(path_vocab)
elseif hyperparameters["tokenizer"] == "none"
    path_vocab = joinpath("vocab", "amazon_reviews_train_en.txt")
    tokenizer = identity
end

vocab = load_vocab(path_vocab)
indexer = IndexTokenizer(vocab, "[UNK]")

display(tokenizer)
display(indexer)

## Pipeline
function clean(s::AbstractString)
    s = lowercase(s)
    s = Unicode.normalize(s, :NFD)
    s = replace(s, r"['`’\u200d\p{M}]" => "") # contractions, zero width joiner and marks from normalization
    s = replace(s, r"\n" => " ")
end

function preprocess(document, tokenizer; pattern = r"[A-Za-z][A-Za-z]+\b", max_length::Union{Nothing, Int}=nothing)
    document = clean(document)
    words = map(m->string(m.match), eachmatch(pattern, document))
    tokens = tokenizer(words)
    if !isnothing(max_length)
        if length(tokens) > max_length
            tokens = tokens[1:max_length]
        end
    end
    tokens
end

## Tokens

documents = df[!, :review_body]
labels = df[!, :stars]
max_length = 50
indices_path = "outputs/indices_" * hyperparameters["tokenizer"] * ".bson"
@time tokens = map(d->preprocess(d, tokenizer, max_length=max_length), documents) #takes about 30 seconds for all documents
@time indices = indexer(tokens) #takes about 12 seconds for all documents

# BSON.@save indices_path indices
# BSON.@load indices_path indices

y_train = copy(labels)
if nlabels == 1
    y_train[labels .≤ 2] .= 0
    y_train[labels .≥ 4] .= 1
    idxs = labels .!= 3
    y_train = reshape(y_train, 1, :)
else
    idxs = Base.OneTo(length(labels))
    y_train = Flux.onehotbatch(y_train, 1:nlabels)
end

X_train, y_train = indices[:, idxs], y_train[:, idxs];
train_data, val_data = split_validation(X_train, y_train; rng=MersenneTwister(hyperparameters["seed"]))

println("train samples:      ", size(train_data[1]), " ", size(train_data[2]))
println("validation samples: ", size(val_data[1]), " ", size(val_data[2]))

## Model 
dim_embedding = hyperparameters["dim_embedding"]
pdrop = hyperparameters["pdrop"]
# position_encoding = PositionEncoding(dim_embedding)
# add_position_encoding(x) = x .+ position_encoding(x)
# model = Chain(
#     Embed(dim_embedding, length(indexer)), 
#     add_position_encoding, 
#     Dropout(pdrop),
#     TransformerEncoderBlock(4, dim_embedding, dim_embedding * 4; pdrop=pdrop),
#     Dense(dim_embedding, 1),
#     TransformersLite.FlattenLayer(),
#     Dense(max_length, nlabels)
#      )
model = TransformersLite.TransformerClassifier(
    Embed(dim_embedding, length(indexer)), 
    PositionEncoding(dim_embedding), 
    Dropout(pdrop),
    TransformerEncoderBlock[TransformerEncoderBlock(4, dim_embedding, dim_embedding * 4; pdrop=pdrop)],
    Dense(dim_embedding, 1), 
    FlattenLayer(),
    Dense(max_length, nlabels)
    )
display(model)

hyperparameters["trainable parameters"] = sum(length, Flux.params(model));

if nlabels == 1
    loss(x, y) = Flux.logitbinarycrossentropy(model(x), y)
    accuracy(ŷ, y) = mean((Flux.sigmoid.(ŷ) .> 0.5) .== y)
else
    loss(x, y) = Flux.logitcrossentropy(model(x), y)
    accuracy(ŷ, y) = mean(Flux.onecold(ŷ) .== Flux.onecold(y))
end

loss(x::Tuple) = loss(x[1], x[2])

## Training 
opt = ADAM()

val_acc = batched_metric(accuracy, val_data[1], val_data[2]; g=model)
val_loss = batched_metric(loss, val_data[1], val_data[2])

@printf "val_acc=%.4f%% ; " val_acc * 100
@printf "val_loss=%.4f \n" val_loss

directory = "outputs\\" * Dates.format(now(), "yyyymmdd_HHMM")
mkdir(directory)
output_path = joinpath(directory, "model.bson")
history_path = joinpath(directory, "history.json")
hyperparameter_path = joinpath(directory, "hyperparameters.json")

open(hyperparameter_path, "w") do f
    JSON.print(f, hyperparameters)
end

start_time = time_ns()
history = train!(loss, Flux.params(model), train_data, opt, val_data; n_epochs=10, batch_size=128)
end_time = time_ns() - start_time
println("done training")
@printf "time taken: %.2fs\n" end_time/1e9

## Save 

BSON.@save output_path model

open(history_path,"w") do f
  JSON.print(f, history)
end
