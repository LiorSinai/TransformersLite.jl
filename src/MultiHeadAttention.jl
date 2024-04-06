struct MultiHeadAttention{Q<:Dense, K<:Dense, V<:Dense, O<:Dense}
    nhead::Int
    denseQ::Q
    denseK::K
    denseV::V
    denseO::O
end

# tell Flux which parameters are trainable
Flux.@functor MultiHeadAttention
Flux.trainable(m::MultiHeadAttention) = (; m.denseQ, m.denseK, m.denseV, m.denseO)

"""
    MultiHeadAttention(nhead, dim_in, [dim_head,] dim_out)

Multi-head dot product attention Layer. `nhead` is the number of heads, 
`dim_in` is the input dimension size, `dim_head` is the size of each head, 
`dim_out` is the output size. If `dim_head` is not specified, it defaults to `dim_in ÷ nhead`.
"""
function MultiHeadAttention(
    nhead::Int, dim_in::Int, dim_head::Int, dim_out::Int
    )
    MultiHeadAttention(
        nhead,
        Dense(dim_in, dim_head*nhead; bias=false),
        Dense(dim_in, dim_head*nhead; bias=false),
        Dense(dim_in, dim_head*nhead; bias=false),
        Dense(dim_head*nhead, dim_out),
    )
end

function MultiHeadAttention(
    nhead::Int, dim_in::Int, dim_out::Int
    )
    if dim_in % nhead != 0 
        error("input dimension=$dim_in is not divisible by number of heads=$nhead")
    end
    MultiHeadAttention(nhead, dim_in, div(dim_in, nhead), dim_out)
end

function (mha::MultiHeadAttention)(
    query::A3, key::A3, value::A3; kwargs...) where {T, A3 <: AbstractArray{T, 3}}
    # batch multiplication version. Input is dm × N × B
    #size(Q) == (dh*nhead, N, B)
    Q = mha.denseQ(query)
    K = mha.denseK(key)
    V = mha.denseV(value)
    A, scores = multi_head_scaled_dot_attention(mha.nhead, Q, K, V; kwargs...)
    mha.denseO(A), scores
end

function (mha::MultiHeadAttention)(query::A2, key::A2, value::A2
    ; kwargs...) where {T, A2 <: AbstractMatrix{T}}
    # single sample. Make it a batch of 1
    query = reshape(query, size(query, 1), size(query, 2), 1)
    key = reshape(key, size(key, 1), size(key, 2), 1)
    value = reshape(value, size(value, 1), size(value, 2), 1)
    A, scores = mha(query, key, value; kwargs...)
    reshape(A, size(A, 1), size(A, 2)), reshape(scores, size(scores)[1:3]...)
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
    print(io, " "^indent, ")")
    if indent == 0
        Flux._big_finale(io, mha)
    else 
        println(io, ",")
    end
end
