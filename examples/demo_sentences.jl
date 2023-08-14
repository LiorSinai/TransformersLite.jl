using DataFrames
using Arrow
using Printf
using BSON, JSON
using Flux
using Flux: DataLoader
using Unicode
using Dates
using StatsBase: mean

using TokenizersLite
using TransformersLite
include("utilities.jl")
include("training.jl")
include("SentenceClassifier.jl")

path = normpath(joinpath(@__DIR__, "..", "datasets", "amazon_reviews_multi", "en", "1.0.0"))
filename = "amazon_reviews_multi-train.arrow"
file_test = "amazon_reviews_multi-test.arrow" ;

checksum = readdir(path)[1]
filepath = joinpath(path, checksum, filename)

df = DataFrame(Arrow.Table(filepath))
display(first(df, 20))
println("")

hyperparameters = Dict(
    "seed" => 2718,
    "tokenizer" => "sentences+bpe",
    "nlabels" => 1,
    "pdrop" => 0.1,
    "dim_embedding" => 32,
    "max_length" => 30,
)
nlabels = hyperparameters["nlabels"]

## Tokenizers
sentence_splitter = RuleBasedSentenceSplitter()

directory = "vocab\\bpe"
path_rules = joinpath(directory, "amazon_reviews_train_en_rules.txt")
path_vocab = joinpath(directory, "amazon_reviews_train_en_vocab.txt")
tokenizer = load_bpe(path_rules, startsym="⋅")

vocab = load_vocab(path_vocab)
indexer = IndexTokenizer(vocab, "[UNK]")

display(sentence_splitter)
display(tokenizer)
display(indexer)
println("")

## Tokens

function pad!(v::Vector{String}, sym::String, max_length::Int)
    if length(v) < max_length
        padding = [sym for i in 1:(max_length - length(v))]
        append!(v, padding)
    end
end

documents = df[!, :review_body]
labels = df[!, :stars]
max_length = hyperparameters["max_length"]
tokens = Vector{Vector{String}}[]
@time for doc in documents
    sentences = sentence_splitter(doc)
    tokens_doc = map(s->preprocess(s, tokenizer, max_length=max_length), sentences)
    pad!(tokens_doc[1], tokenizer.unksym, max_length) # hack to ensure all indices have common length
    push!(tokens, tokens_doc)
end
@time indices = map(t->indexer(t), tokens) 

y_train = copy(labels)
y_train[labels .≤ 2] .= 0
y_train[labels .≥ 4] .= 1
idxs = labels .!= 3

X_train, y_train = indices[idxs], y_train[idxs];
train_data, val_data = split_validation(X_train, y_train; rng=MersenneTwister(hyperparameters["seed"]))

println("train samples:      ", size(train_data[1]), " ", size(train_data[2]))
println("validation samples: ", size(val_data[1]), " ", size(val_data[2]))
println("")

## Model 
dim_embedding = hyperparameters["dim_embedding"]
pdrop = hyperparameters["pdrop"]

base_model = TransformersLite.TransformerClassifier(
    Embed(dim_embedding, length(indexer)), 
    PositionEncoding(dim_embedding), 
    Dropout(pdrop),
    TransformerEncoderBlock[TransformerEncoderBlock(4, dim_embedding, dim_embedding * 4; pdrop=pdrop)],
    Dense(dim_embedding, 1), 
    FlattenLayer(),
    Dense(max_length, nlabels)
    )

model = SentenceClassifier(base_model, Flux.sigmoid, parabolic_weighted_average)

hyperparameters["model"] = "$(typeof(model).name.wrapper)-$(typeof(model.base_model).name.wrapper)"
hyperparameters["trainable parameters"] = sum(length, Flux.params(model));

loss(x, y) = Flux.binarycrossentropy(model(x), y)
loss(x::Tuple) = loss(x[1], x[2])
accuracy(ŷ, y) = mean((ŷ .> 0.5) .== y)

## Training 
opt_state = Flux.setup(Adam(), model)

batch_size = 32

train_data_loader = DataLoader(train_data; batchsize=batch_size, shuffle=true)
val_data_loader = DataLoader(val_data; batchsize=batch_size, shuffle=false)

val_acc = batched_metric(model, accuracy, val_data_loader)
val_loss = batched_metric(model, loss, val_data_loader)

@printf "val_acc=%.4f%%; " val_acc * 100
@printf "val_loss=%.4f \n" val_loss
println("")

directory = "..\\outputs\\" * Dates.format(now(), "yyyymmdd_HHMM")
mkdir(directory)
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
    ; num_epochs=10
    )
end_time = time_ns() - start_time
println("done training")
@printf "time taken: %.2fs\n" end_time/1e9

## Save 

if hasproperty(tokenizer, :cache)
    tokenizer = similar(tokenizer)
end
BSON.bson(
    output_path, 
    Dict(
        :model=> model, 
        :tokenizer=>tokenizer,
        :indexer=>indexer,
        :sentence_splitter=>sentence_splitter
    )
    )
println("saved model to $(output_path).")

open(history_path,"w") do f
  JSON.print(f, history)
end
println("saved history to $(history_path).")