using Unicode
using DataStructures

### Tokenizer

function select_vocabulary(corpus::AbstractVector{<:AbstractString}; 
    min_document_frequency::Int=10, pattern::Regex=r"\w\w+\b", transform=simplify)
    document_frequencies = DefaultDict{String, Int}(0)
    for document in corpus
        words = Set{String}()
        for m in eachmatch(pattern, transform(document))
            word = m.match
            if !(word in words)
                push!(words, word)
                document_frequencies[word] += 1
            end
        end
    end
    filter!(x->x[2] ≥ min_document_frequency, document_frequencies)
    vocab = collect(document_frequencies)
    sort!(vocab, by=x->x[2], rev=true)
    [v[1] for v in vocab]
end

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

function preprocess(document::AbstractString, tokenizer;
    pattern::Regex = r"\w\w+\b", max_length::Union{Nothing, Int}=nothing, transform=simplify
    )
    words = map(m->string(m.match), eachmatch(pattern, transform(document)))
    tokens = tokenizer(words)
    if !isnothing(max_length)
        if length(tokens) > max_length
            tokens = tokens[1:max_length]
        end
    end
    tokens
end
