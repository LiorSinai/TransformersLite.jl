struct MultiHeadAttention{Q<:Dense, K<:Dense, V<:Dense, O<:Dense, M<:Union{Nothing, AbstractMatrix{Bool}}}
    nhead::Int
    denseQ::Q
    denseK::K
    denseV::V
    denseO::O
    mask::M # optional buffer
end

# tell Flux which parameters are trainable
Flux.@functor MultiHeadAttention
Flux.trainable(m::MultiHeadAttention) = (; m.denseQ, m.denseK, m.denseV, m.denseO)

"""
    MultiHeadAttention(nhead, dim_model, [dim_head,] dim_out)

Multi-head dot product attention Layer. `nhead` is the number of heads, 
`dim_model` is the model embedding dimension size, `dim_head` is the size of each head, 
`dim_out` is the output size. If `dim_head` is not specified, it defaults to `dim_model ÷ nhead`.
"""
function MultiHeadAttention(
    nhead::Int, dim_model::Int, dim_head::Int, dim_out::Int
    ; mask::Union{Nothing, Matrix{Bool}}=nothing
    )
    MultiHeadAttention(
        nhead,
        Dense(dim_model, dim_head*nhead; bias=false),
        Dense(dim_model, dim_head*nhead; bias=false),
        Dense(dim_model, dim_head*nhead; bias=false),
        Dense(dim_head*nhead, dim_out),
        isnothing(mask) ? nothing : copy(mask)
    )
end

function MultiHeadAttention(
    nhead::Int, dim_model::Int, dim_out::Int
    ; mask::Union{Nothing, Matrix{Bool}}=nothing
    )
    if dim_model % nhead != 0 
        error("embedding dimension=$dim_model is not divisible by number of heads=$nhead")
    end
    MultiHeadAttention(nhead, dim_model, div(dim_model, nhead), dim_out; mask=mask)
end

function (mha::MultiHeadAttention)(
    query::A1, key::A2, value::A3
    ) where {
    T, A1 <: AbstractArray{T, 3}, A2 <: AbstractArray{T, 3}, A3 <: AbstractArray{T, 3}}
    # batch multiplication version. Input is dm × N × B
    #size(Q) == (dh*nhead, N, B)
    Q = mha.denseQ(query)
    K = mha.denseK(key)
    V = mha.denseV(value)
    n = size(Q, 2)
    mask = isnothing(mha.mask) ? nothing : view(mha.mask, Base.OneTo(n), Base.OneTo(n))
    A = multi_head_scaled_dot_attention(mha.nhead, Q, K, V; mask=mask)
    mha.denseO(A)
end

function (mha::MultiHeadAttention)(query::A1, key::A2, value::A3) where {
    T, A1 <: AbstractMatrix{T}, A2 <: AbstractMatrix{T}, A3 <: AbstractMatrix{T}}
    # single sample. Make it a batch of 1
    query = reshape(query, size(query, 1), size(query, 2), 1)
    key = reshape(key, size(key, 1), size(key, 2), 1)
    value = reshape(value, size(value, 1), size(value, 2), 1)
    A = mha(query, key, value)
    reshape(A, size(A, 1), size(A, 2))
end

## Show

function Base.show(io::IO, mha::MultiHeadAttention)
    dh = div(size(mha.denseQ.weight)[1], mha.nhead)
    dm = size(mha.denseQ.weight)[2]
    dout = size(mha.denseO.weight)[1]
    print(io, "MultiHeadAttention(")
    print(io, "num_heads=$(mha.nhead), ")
    print(io, "head_size=$(dh), ")
    print(io, "$(dm)=>$(dout)")
    print(io, ")")
end

function Base.show(io::IO, m::MIME"text/plain", mha::MultiHeadAttention)
    _show_multiheadattention(io, mha)
end

function _show_multiheadattention(io::IO, mha::MultiHeadAttention, indent=0)
    inner_indent = indent + 2
    print(io, " "^indent, mha, "(\n") 
    Flux._layer_show(io, mha.denseQ, inner_indent, "denseQ")
    Flux._layer_show(io, mha.denseK, inner_indent, "denseK")
    Flux._layer_show(io, mha.denseV, inner_indent, "denseV")
    Flux._layer_show(io, mha.denseO, inner_indent, "denseO")
    Flux._layer_show(io, mha.mask, inner_indent, "mask")
    print(io, " "^indent, ")")
    if indent == 0
        Flux._big_finale(io, mha)
    else 
        println(io, ",")
    end
end
