struct TransformerEncoderBlock{MA<:MultiheadAttention, L1<:LayerNorm, D1<:Dense, D2<:Dense, L2<:LayerNorm}
    multihead_attention::MA
    layer_norm_attention::L1
    dense1::D1
    dense2::D2
    layer_norm_feedforward::L2
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
TransformerEncoderBlock(nhead::Int, dm::Int, dhid::Int) = TransformerEncoderBlock(
    MultiheadAttention(nhead, dm, dm),
    LayerNorm(dm),
    Dense(dm, dhid, relu),
    Dense(dhid, dm),
    LayerNorm(dm)
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
    _show_transformer_encoder(io, te)
end

function _show_transformer_encoder(io::IO, t::TransformerEncoderBlock; indent=0)
    inner_indent = indent + 5
    print(io, " "^indent, "TransformerEncoderBlock")
    print(io, "(")
    print(io, "\n")
    _show_multiheadattention(io, t.multihead_attention, indent=inner_indent)
    Flux._layer_show(io, t.layer_norm_attention, inner_indent)
    Flux._layer_show(io, t.dense1, inner_indent)
    Flux._layer_show(io, t.dense2, inner_indent)
    Flux._layer_show(io, t.layer_norm_attention, inner_indent)
    print(io, " "^indent, ")")
    if indent==0
        Flux._big_finale(io, t)
    else
        println("")
    end
end

function (t::TransformerEncoderBlock)(x::A) where {T, N, A<:AbstractArray{T, N}}
    a = t.multihead_attention(x, x, x)
    res_a = x + a
    res_a = t.layer_norm_attention(res_a)
    z_ff = t.dense1(res_a)
    z_ff = t.dense2(z_ff)
    res_ff = res_a + z_ff
    res_ff = t.layer_norm_feedforward(res_ff)
    res_ff
end


