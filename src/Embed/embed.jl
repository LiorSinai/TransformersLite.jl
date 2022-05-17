"""
    Embed(output_dim::Int, vocab_size::Int)

The Embedding Layer. 
"""
struct Embed{W <: AbstractArray}
    embedding::W
end

# tell Flux which parameters are trainable
Flux.@functor Embed

Embed(output_dim::Int, vocab_size::Int) = Embed(randn(Float32, output_dim, vocab_size))

Base.size(e::Embed) = size(e.embedding)

function (e::Embed)(x::AbstractArray{Int})
    gather(e.embedding, x) # see rrule using scatter
end

Base.show(io::IO, e::Embed) = print(io, "Embed($(size(e.embedding, 1)), $(size(e.embedding, 2)))")

function Base.show(io::IO, m::MIME"text/plain", e::Embed)
    Flux._layer_show(io, e)
end