"""
    IndexTokenizer(vocab::Vector{T}, unksym::T) where T

Convert words/tokens to indices.

Usage:
```
indexer = IndexTokenizer(["UNK","this","book","recommend","highly"], "[UNK]")
indexer(["i","highly","recommend","this","book","by","brandon","sanderson"])
# [1, 6, 5, 3, 4, 1, 1, 1]
```
"""
struct IndexTokenizer{T}
    vocabulary::Vector{T}
    unksym::T
    unkidx::Int
    function IndexTokenizer(vocab::Vector{T}, unksym::T) where T
        if !(unksym ∈ vocab)
            pushfirst!(vocab, unksym)
            unkidx = 1
        else
            unkidx = findfirst(isequal(unksym), vocab)
        end
        new{T}(vocab, unksym, unkidx)
    end
end

Base.length(tokenizer::IndexTokenizer) = length(tokenizer.vocabulary)

function Base.show(io::IO, tokenizer::IndexTokenizer) 
    T = eltype(tokenizer.vocabulary)
    print(io, "IndexTokenizer{$(T)}(length(vocabulary)=$(length(tokenizer)), unksym=$(tokenizer.unksym))")
end

"""
    encode(tokenizer::IndexTokenizer, x)

Encode the tokens to indices.
If a vector of sequences, output is maximum_sequence_length × batch_size  
"""
encode(tokenizer::IndexTokenizer{T}, x::T) where T = something(findfirst(isequal(x), tokenizer.vocabulary), tokenizer.unkidx)

function encode(tokenizer::IndexTokenizer{T}, seq::AbstractVector{T}) where T
    map(x->encode(tokenizer, x), seq)
end

function encode(tokenizer::IndexTokenizer{T}, batch::AbstractVector{Vector{T}}) where T
    lengths = map(length, batch)
    indices = fill(tokenizer.unkidx, maximum(lengths), length(batch))
    for (j, seq) ∈ enumerate(batch)
        for (i, x) ∈ enumerate(seq)
            @inbounds indices[i, j] = encode(tokenizer, x)
        end
    end
    indices
end

(tokenizer::IndexTokenizer)(x) = encode(tokenizer, x)

"""
    decode(tokenizer::IndexTokenizer, x)

Decode indices to tokens.
"""
decode(tokenizer::IndexTokenizer{T}, x::Int) where T = 0 <= x <= length(tokenizer) ? tokenizer.vocabulary[x] : tokenizer.unksym

function decode(tokenizer::IndexTokenizer{T}, seq::Vector{Int}) where T
    map(x->decode(tokenizer, x), seq)
end

function decode(tokenizer::IndexTokenizer{T}, indices::AbstractMatrix{Int}) where T
    nrow, ncol = size(indices)
    tokens = Vector{Vector{T}}(undef, ncol)
    for col ∈ 1:ncol
        token = Vector{T}(undef, nrow)
        for row ∈ 1:nrow
            token[row] = decode(tokenizer, indices[row, col])
        end
        tokens[col] = token
    end
    tokens
end