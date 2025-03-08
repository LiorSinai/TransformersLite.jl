struct TransformerBlock{
    MHA<:MultiHeadAttention,
    N1<:LayerNorm,
    D1<:Dense,
    D2<:Dense,
    N2<:LayerNorm,
    DO<:Dropout}
    multihead_attention::MHA
    norm_attention::N1
    dense1::D1
    dense2::D2
    norm_feedforward::N2
    dropout::DO
end

# make whole layer trainable
Flux.@layer TransformerBlock

"""
    TransformerBlock(
        nhead, dim_model, dim_hidden
        ; act=relu, mask=nothing, pdrop=0.1
    )

Create a transfomer block based on "Attention is all you need" (https://arxiv.org/abs/1706.03762).
It consists of a multi-head attention layer followed by two fully connected layers, along with
layer norms and residual connections.

`nhead` is the number of heads for the multi-head attention. 
`dim_model` is the model dimension also known as the embedding layer dimension. 
The input and output are both of size `dim_model`.     
`dim_hidden` is the size of the hidden layer between the two dense layers.
"""
TransformerBlock(
    nhead::Int,
    dim_model::Int,
    dim_hidden::Int;
    act=relu,
    pdrop::Float64=0.1,
    ) = TransformerBlock(
    MultiHeadAttention(nhead, dim_model, dim_model),
    LayerNorm(dim_model),
    Dense(dim_model, dim_hidden, act),
    Dense(dim_hidden, dim_model),
    LayerNorm(dim_model),
    Dropout(pdrop),
)

function (t::TransformerBlock)(x::A; mask::M=nothing) where {
    A<:AbstractArray, M<:Union{Nothing, AbstractArray{Bool}}}
    h, scores = t.multihead_attention(x, x, x; mask=mask) # (dm, N, B)
    h = t.dropout(h) 
    h = x + h
    h = t.norm_attention(h)            # (dm, N, B)
    hff = t.dense1(h)                  # (dh, N, B)
    hff = t.dense2(hff)                # (dm, N, B)
    hff = t.dropout(hff)
    h = h + hff
    h = t.norm_feedforward(h)          # (dm, N, B)
    h
end

## Show

function Base.show(io::IO, block::TransformerBlock)
    print(io, "TransformerBlock(")
    print(io, block.multihead_attention)
    print(io, ", ", block.norm_attention)
    print(io, ", ", block.dense1)
    print(io, ", ", block.dense2)
    print(io, ", ", block.norm_feedforward)
    print(io, ")")
end

