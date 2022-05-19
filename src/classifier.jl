struct TransformerClassifier{
    E<:Embed, 
    PE<:PositionEncoding, 
    DO<:Dropout, 
    TEB<:Vector{TransformerEncoderBlock}, 
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
inner_indent = 5
print(io, " "^indent, "TransformerClassifier")
print(io, "(")
print(io, "\n")
Flux._layer_show(io, t.embed, inner_indent)
Flux._layer_show(io, t.position_encoding, inner_indent)
Flux._layer_show(io, t.dropout, inner_indent)
for e in t.encoder_layers
    _show_transformer_encoder(io, e, indent=inner_indent)
end
Flux._layer_show(io, t.agg_layer, inner_indent)
Flux._layer_show(io, t.flatten_layer, inner_indent)
Flux._layer_show(io, t.classifier, inner_indent)
print(io, " "^indent, ")")
if indent==0
    Flux._big_finale(io, t)
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
