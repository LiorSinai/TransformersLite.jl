struct TransformerEncoderBlock{MA<:MultiheadAttention, L1<:LayerNorm, D1<:Dense, D2<:Dense, L2<:LayerNorm, DO<:Dropout}
    multihead_attention::MA
    layer_norm_attention::L1
    dense1::D1
    dense2::D2
    layer_norm_feedforward::L2
    dropout::DO
end

# make whole TransformerEncoder trainable
Flux.@functor TransformerEncoderBlock
"""
    TransformerEncoder(nhead::Int, dm::Int, dhid::Int)

Create a transfomer encoder block based on "Attention is all you need" (https://arxiv.org/abs/1706.03762).
`nhead` is the number of heads for the multi-head attention. 
`dm` is the model dimension also known as the embedding layer dimension. The input and output are both of size `dm`.     
`dhid` is the size of the hidden layer between the two feedforwards after the attention layer.

"""
TransformerEncoderBlock(nhead::Int, dm::Int, dhid::Int; pdrop::Float64=0.1, act=relu) = TransformerEncoderBlock(
    MultiheadAttention(nhead, dm, dm),
    LayerNorm(dm),
    Dense(dm, dhid, act),
    Dense(dhid, dm),
    LayerNorm(dm),
    Dropout(pdrop)
)

function Base.show(io::IO, te::TransformerEncoderBlock)
    print(io, "TransformerEncoderBlock(")
    print(io, te.multihead_attention)
    print(io, ", ", te.layer_norm_attention)
    print(io, ", ", te.dense1)
    print(io, ", ", te.dense2)
    print(io, ", ", te.layer_norm_feedforward)
    print(io, ")")
end

function Base.show(io::IO, m::MIME"text/plain", te::TransformerEncoderBlock)
    if get(io, :typeinfo, nothing) === nothing  # e.g. top level in REPL
        _show_transformer_encoder(io, te)
    elseif !get(io, :compact, false)  # e.g. printed inside a Vector, but not a Matrix
      _layer_show(io, x)
    else
      show(io, x)
    end
end

function _show_transformer_encoder(io::IO, t::TransformerEncoderBlock, indent=0)
    inner_indent = indent + 2
    print(io, " "^indent, "TransformerEncoderBlock(\n")
    _show_multiheadattention(io, t.multihead_attention, inner_indent)
    for layer in [t.dropout, t.layer_norm_attention, t.dense1, t.dense2, t.dropout, t.layer_norm_feedforward]
        Flux._layer_show(io, layer, inner_indent)
    end
    print(io, " "^indent, ")")
    if indent == 0
        Flux._big_finale(io, t)
    else
        println(io, ",")
    end
end

function (t::TransformerEncoderBlock)(x::A) where {T, N, A<:AbstractArray{T, N}}
    a = t.multihead_attention(x, x, x)
    a = t.dropout(a)
    res_a = x + a
    res_a = t.layer_norm_attention(res_a)
    z_ff = t.dense1(res_a)
    z_ff = t.dense2(z_ff)
    z_ff = t.dropout(z_ff)
    res_ff = res_a + z_ff
    res_ff = t.layer_norm_feedforward(res_ff)
    res_ff
end
