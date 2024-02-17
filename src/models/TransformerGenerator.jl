struct TransformerGenerator{
    E<:Embed, 
    PE<:PositionEncoding, 
    DO<:Dropout, 
    TB<:Vector{<:TransformerBlock}, 
    D<:Dense
    } 
    embedding::E
    position_encoding::PE
    dropout::DO
    blocks::TB
    head::D
end

Flux.@functor TransformerGenerator

function (t::TransformerGenerator)(x::A) where {T, N, A<:AbstractArray{T, N}}
    x = t.embedding(x)              # (dm, N, B)
    x = x .+ t.position_encoding(x) # (dm, N, B)
    x = t.dropout(x)                # (dm, N, B)
    for block in t.blocks
        x = block(x)                # (dm, N, B)
    end
    x = t.head(x)                   # (vocab_size, N, B)
    x
end

"""
    generate([rng,] transformer, context; context_size, max_tokens=100)

Generate batches of tokens starting from a given context.
"""
function generate(
    rng::AbstractRNG, model::TransformerGenerator, context::Matrix{T}
    ; context_size::Int, max_tokens::Int=100
    ) where T
    num_batches = size(context, 2)
    vocab_size = size(model.head.weight, 1)
    for i in 1:max_tokens
        context_crop = tail(context, context_size) # forget everything before the current context
        logits = model(context_crop) |> cpu # (vocab_size, T, B)
        # only focus on the last token
        # This means that some of the work done in the last block and in model.head is discarded
        logits = logits[:, end, :] # (vocab_size, B) 
        probs = softmax(logits; dims=1) # (vocab_size, B)
        context_next = Matrix{T}(undef, 1, num_batches) # (vocab_size, B)
        for b in 1:num_batches
            context_next[:, b] = sample(rng, 1:vocab_size, Weights(probs[:, b]), 1)
        end
        context = cat(context, context_next; dims=1) 
    end
    context
end

function generate(model::TransformerGenerator, context::Matrix; kwargs...)
    generate(Random.default_rng(), model, context; kwargs...)
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
    print(io, " "^indent, ")")
    if indent == 0
        Flux._big_finale(io, t)
    else
        println(io, ",")
    end
end
