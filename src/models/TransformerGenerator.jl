struct TransformerGenerator{
    E<:Embed, 
    PE<:PositionEncoding, 
    DO<:Dropout, 
    TB<:Vector{<:TransformerBlock}, 
    D<:Dense,
    M<:Union{Nothing, AbstractMatrix{Bool}},
    } 
    embedding::E
    position_encoding::PE
    dropout::DO
    blocks::TB
    head::D
    mask::M # optional buffer
end

Flux.@functor TransformerGenerator
Flux.trainable(m::TransformerGenerator) = (; m.embedding, m.blocks, m.dropout, m.head)

function (t::TransformerGenerator)(x::A; mask::M=t.mask) where {
    A<:AbstractArray, M<:Union{Nothing, AbstractMatrix{Bool}}}
    x = t.embedding(x)              # (dm, N, B)
    x = x .+ t.position_encoding(x) # (dm, N, B)
    x = t.dropout(x)                # (dm, N, B)
    for block in t.blocks
        x = block(x; mask=mask)     # (dm, N, B)
    end
    x = t.head(x)                   # (vocab_size, N, B)
    x
end

"""
    generate([rng,] transformer, context; context_size, max_tokens=100)

Generate batches of tokens starting from a given context.
"""
function generate(
    rng::AbstractRNG, model::TransformerGenerator, context::AbstractMatrix{T}
    ; context_size::Int, max_tokens::Int=100,
    ) where T
    for i in 1:max_tokens
        context_crop = tail(context, context_size) # forget everything before the current context
        n = size(context_crop, 1)
        mask = isnothing(model.mask) ? nothing : view(model.mask, Base.OneTo(n), Base.OneTo(n))
        logits = model(context_crop; mask=mask) |> cpu # (vocab_size, n, B)
        # only focus on the last token
        # This means that some of the work done in the last block and in model.head is discarded
        logits = logits[:, end, :] # (vocab_size, B) 
        context_next = multinomial_sampling(rng, logits)
        context = cat(context, transpose(context_next); dims=1) 
    end
    context
end

function generate(model::TransformerGenerator, context::AbstractMatrix; kwargs...)
    generate(Random.default_rng(), model, context; kwargs...)
end

function multinomial_sampling(rng::AbstractRNG, logits::AbstractMatrix)
    probs = softmax(logits; dims=1)
    tokens = [sample(rng, Weights(p)) for p in eachcol(probs)]
    tokens
end

## Show
function Base.show(io::IO, m::MIME"text/plain", t::TransformerGenerator)
    _show_transformer_generator(io, t)
end

function _show_transformer_generator(io::IO, t::TransformerGenerator, indent::Int=0)
    inner_indent = indent + 2
    print(io, " "^indent, "TransformerGenerator(\n")
    for layer in [t.embedding, t.position_encoding, t.dropout, t.blocks..., t.head]
        if typeof(layer) <: TransformerBlock
            _show_transformer_block(io, layer, inner_indent)
        else
            Flux._layer_show(io, layer, inner_indent)
        end
    end
    Flux._layer_show(io, t.mask, inner_indent, "mask")
    print(io, " "^indent, ")")
    if indent == 0
        Flux._big_finale(io, t)
    else
        println(io, ",")
    end
end
