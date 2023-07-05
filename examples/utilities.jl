using Flux: DataLoader
using Unicode

### Tokenizer

function load_vocab(filepath::AbstractString)
    vocab = String[]
    open(filepath, "r") do file
        for line in eachline(file)
            push!(vocab, line)
        end
    end
    vocab
end

#### Preprocessing text

function simplify(s::AbstractString)
    s = lowercase(s)
    s = Unicode.normalize(s, :NFD)
    s = replace(s, r"['`’\u200d\p{M}]" => "") # contractions, zero width joiner and marks from normalization
    s = replace(s, r"\n" => " ")
end

function preprocess(document::AbstractString, tokenizer; pattern::Regex = r"[A-Za-z][A-Za-z]+\b", max_length::Union{Nothing, Int}=nothing)
    document = simplify(document)
    words = map(m->string(m.match), eachmatch(pattern, document))
    tokens = tokenizer(words)
    if !isnothing(max_length)
        if length(tokens) > max_length
            tokens = tokens[1:max_length]
        end
    end
    tokens
end
