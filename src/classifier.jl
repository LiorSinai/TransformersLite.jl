struct TransformerClassifier{
        E<:Embed, 
        PE<:PositionEncoding, 
        DO<:Dropout, 
        TEB<:Vector{<:TransformerEncoderBlock}, 
        A, 
        f<:FlattenLayer, 
        D<:Dense
    } 
    embed::E
    position_encoding::PE
    dropout::DO
    encoder_layers::TEB
    agg_layer::A
    flatten_layer::f
    classifier::D
end

Flux.@functor TransformerClassifier

function Base.show(io::IO, m::MIME"text/plain", t::TransformerClassifier)
    _show_transformer_classifier(io, t)
end

function _show_transformer_classifier(io::IO, t::TransformerClassifier; indent=0)
    inner_indent = indent + 2
    print(io, " "^indent, "TransformerClassifier(\n")
    for layer in [t.embed, t.position_encoding, t.dropout, t.encoder_layers..., t.agg_layer, t.flatten_layer, t.classifier]
        if typeof(layer) <: TransformerEncoderBlock
            _show_transformer_encoder(io, layer, inner_indent)
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

function (t::TransformerClassifier)(x::A) where {T, N, A<:AbstractArray{T, N}}
    x = t.embed(x)
    x = x .+ t.position_encoding(x)
    x = t.dropout(x)
    for e in t.encoder_layers
        x = e(x)
    end
    x = t.agg_layer(x)
    x = t.flatten_layer(x)
    x = t.classifier(x)
    x
end