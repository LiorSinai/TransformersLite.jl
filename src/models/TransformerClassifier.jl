struct TransformerClassifier{
        E<:Union{Embed, Flux.Embedding}, 
        PE<:Union{Embed, Flux.Embedding, PositionEncoding}, 
        DO<:Dropout, 
        TB<:Vector{<:TransformerBlock}, 
        A, 
        f<:FlattenLayer, 
        D<:Dense
    } 
    embedding::E
    position_encoding::PE
    dropout::DO
    blocks::TB
    agg_layer::A
    flatten_layer::f
    head::D
end

Flux.@functor TransformerClassifier

function (t::TransformerClassifier)(x::A; mask::M=nothing) where {
    A<:AbstractArray, M<:Union{Nothing, AbstractMatrix{Bool}}}
    x = t.embedding(x)              # (dm, N, B)
    N = size(x, 2)
    x = x .+ t.position_encoding(1:N) # (dm, N, B)
    x = t.dropout(x)                # (dm, N, B)
    for block in t.blocks
        x = block(x; mask=mask)     # (dm, N, B)
    end
    x = t.agg_layer(x)              # (1, N, B)
    x = t.flatten_layer(x)          # (N, B)
    x = t.head(x)                   # (n_labels, B)
    x
end

## Show

function Base.show(io::IO, m::MIME"text/plain", t::TransformerClassifier)
    _show_transformer_classifier(io, t)
end

function _show_transformer_classifier(io::IO, t::TransformerClassifier, indent::Int=0)
    inner_indent = indent + 2
    print(io, " "^indent, "TransformerClassifier(\n")
    for layer in [
        t.embedding,
        t.position_encoding,
        t.dropout,
        t.blocks...,
        t.agg_layer,
        t.flatten_layer,
        t.head
        ]
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
