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
includet("SentenceClassifier.jl")

path = "path\\to\\amazon_reviews_multi\\en\\1.0.0\\"
filename = "amazon_reviews_multi-train.arrow"
file_test = "amazon_reviews_multi-test.arrow" ;

checksum = readdir(path)[1]
filepath = joinpath(path, checksum, filename)

df = DataFrame(Arrow.Table(filepath))
display(df)

hyperparameters = Dict(
    "seed" => 2718,
    "tokenizer" => "sentences+bpe",
    "nlabels" => 1,
    "model" => "TransformerClassifier",
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

## Pipeline
function clean(s::AbstractString)
    s = lowercase(s)
    s = Unicode.normalize(s, :NFD)
    s = replace(s, r"['`’\u200d\p{M}]" => "") # contractions, zero width joiner and marks from normalization
    s = replace(s, r"\n" => " ")
end

function preprocess(document, tokenizer; 
    pattern = r"[A-Za-z][A-Za-z]+\b", 
    max_length::Union{Nothing, Int}=nothing
    )
    tokens_all = Vector{String}[]
    for sentence in document
        sentence = clean(sentence)
        words = map(m->string(m.match), eachmatch(pattern, sentence))
        tokens = tokenizer(words)
        if !isnothing(max_length)
            if length(tokens) > max_length
                tokens = tokens[1:max_length]
            end
        end
        push!(tokens_all, tokens)
    end
    tokens_all
end

function pad!(v::Vector{String}, sym::String, max_length::Int)
    if length(v) < max_length
        padding = [sym for i in 1:(max_length - length(v))]
        append!(v, padding)
    end
end

## Tokens

documents = df[!, :review_body]
documents = [sentence_splitter(d) for d in documents]
labels = df[!, :stars]
max_length = hyperparameters["max_length"]
@time tokens = map(d->preprocess(d, tokenizer, max_length=max_length), documents)
# hack to make sure all the indices have the same number of rows:
for i in 1:length(tokens)
    pad!(tokens[i][1], tokenizer.unksym, max_length)
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

hyperparameters["trainable parameters"] = sum(length, Flux.params(model));

loss(x, y) = Flux.binarycrossentropy(model(x), y)
loss(x::Tuple) = loss(x[1], x[2])
accuracy(ŷ, y) = mean((ŷ .> 0.5) .== y)

## Training 
opt = ADAM()

val_acc = batched_metric(accuracy, val_data[1], val_data[2]; g=model)
val_loss = batched_metric(loss, val_data[1], val_data[2])

@printf "val_acc=%.4f%%; " val_acc * 100
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
