"""
    Embed(output_dim::Int, vocab_size::Int)

An Embed Layer. 

See `Flux.Embedding`.
"""
struct Embed{W <: AbstractArray}
    weight::W
end

# tell Flux which parameters are trainable
Flux.@functor Embed

Embed(output_dim::Int, vocab_size::Int) = Embed(randn(Float32, output_dim, vocab_size))

Base.size(e::Embed) = size(e.weight)

function (e::Embed)(x::AbstractArray{Int})
    gather(e.weight, x) # see rrule using scatter
end

Base.show(io::IO, e::Embed) = print(io, "Embed($(size(e.weight)))")

function Base.show(io::IO, m::MIME"text/plain", e::Embed)
    Flux._layer_show(io, e)
end