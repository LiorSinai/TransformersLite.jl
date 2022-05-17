

struct MultiheadAttention{Q<:Dense, K<:Dense, V<:Dense, O<:Dense}
    nhead::Int
    denseQ::Q
    denseK::K
    denseV::V
    denseO::O
end

# tell Flux which parameters are trainable
Flux.@functor MultiheadAttention (denseQ, denseK, denseV, denseO, )

"""
    MultiheadAttention(nhead::Int, dm::Int, hs::Int, dout::Int)
    MultiheadAttention(nhead::Int, dm::Int, dout::Int)

Multihead dot product attention Layer. `nhead` is the number of heads, 
`dm` is the model embedding dimension size, `hs` is the size of each head, 
`dout` is the output size.
"""
function MultiheadAttention(nhead::Int, dm::Int, hs::Int, dout::Int)
    MultiheadAttention(
        nhead,
        Dense(dm, hs*nhead),
        Dense(dm, hs*nhead),
        Dense(dm, hs*nhead),
        Dense(hs*nhead, dout),
    )
end

function MultiheadAttention(nhead::Int, dm::Int, dout::Int)
    if dm % nhead != 0 
        error("embedding dimension=$dm is not divisible by number of heads=$nhead")
    end
    MultiheadAttention(nhead, dm, div(dm, nhead), dout)
end

function Base.show(io::IO, mha::MultiheadAttention)
    hs = div(size(mha.denseQ.weight)[1], mha.nhead)
    dm = size(mha.denseQ.weight)[2]
    dout = size(mha.denseO.weight)[1]
    print(io, "MultiheadAttention(")
    print(io, "num_heads=$(mha.nhead), ")
    print(io, "head_size=$(hs), ")
    print(io, "$(dm)=>$(dout)")
    print(io, ")")
end

function Base.show(io::IO, m::MIME"text/plain", mha::MultiheadAttention)
    _show_multiheadattention(io, mha)
end

function _show_multiheadattention(io::IO, mha::MultiheadAttention; indent=0)
    inner_indent = indent + 5
    print(io, " "^indent, mha)
    print(io,"(")
    print(io, "\n")
    Flux._layer_show(io, mha.denseQ, inner_indent, "denseQ")
    Flux._layer_show(io, mha.denseK, inner_indent, "denseK")
    Flux._layer_show(io, mha.denseV, inner_indent, "denseV")
    Flux._layer_show(io, mha.denseO, inner_indent, "denseO")
    print(io, " "^indent, ")")
    if indent==0
        Flux._big_finale(io, mha)
    else 
        println(io, "")
    end
end

function (mha::MultiheadAttention)(query::A1, key::A2, value::A3) where {
    T, A1 <: AbstractArray{T, 3}, A2 <: AbstractArray{T, 3}, A3 <: AbstractArray{T, 3}}
    # batch multiplication version. Input is dm × N × B
    qs = size(query)
    ks = size(key)
    vs = size(value)

    #size(Q) == (hs*nhead, N, B)
    Q = mha.denseQ(query)
    K = mha.denseK(key)
    V = mha.denseV(value)

    dm = size(Q, 1)
    hs = div(dm, mha.nhead)
    #size(Q) == (hs*nhead, N, B) => (hs, nhead, N, B) => (hs, N, nhead, B)
    Q = permutedims(reshape(Q, hs, mha.nhead, qs[2], qs[3]), [1, 3, 2, 4])
    K = permutedims(reshape(K, hs, mha.nhead, ks[2], ks[3]), [1, 3, 2, 4])
    V = permutedims(reshape(V, hs, mha.nhead, vs[2], vs[3]), [1, 3, 2, 4])
    #size(A) == (hs, N, nhead, B)
    A = scaled_dot_attention(Q, K, V)
    #size(A) == (hs, N, nhead, B) => (hs, nhead, N, B) => (dm, N, B)
    A = permutedims(A, [1, 3, 2, 4])
    A = reshape(A, dm, size(A, 3), size(A, 4))

    mha.denseO(A)
end

function (mha::MultiheadAttention)(query::A1, key::A2, value::A3) where {
    T, A1 <: AbstractMatrix{T}, A2 <: AbstractMatrix{T}, A3 <: AbstractMatrix{T}}
    # single sample. Make it a batch of 1
    query = reshape(query, size(query, 1), size(query, 2), 1)
    key = reshape(key, size(key, 1), size(key, 2), 1)
    value = reshape(value, size(value, 1), size(value, 2), 1)
    A = mha(query, key, value)
    reshape(A, size(A, 1), size(A, 2))
end

"""
    attention(query, key, value)

Attention calculation for a transformer. If the inputs are 4D arrays, the output is 
    
    A[:, : h, b] = 1/sqrt(dh) * value[:, : h, b] * softmax(transpose(key[:, : h, b]) * query[:, : h, b]))

If the inputs are 3D arrays, the output is

    A[:, : h] = 1/sqrt(dh) * value[:, : h] * softmax(transpose(key[:, : h]) * query[:, : h]))
"""
function scaled_dot_attention(query::A1, key::A2, value::A3) where {
    T, A1 <: AbstractArray{T, 4}, A2 <: AbstractArray{T, 4}, A3 <: AbstractArray{T, 4}}
    # Batched version. Input is (hs, N, nhead, B)
    dh = size(query, 1)
    #size(score) == (N, N, nhead, B)
    atten = one(T)/convert(T, sqrt(dh)) .* batched_mul(key, query, transA=true)
    score = softmax(atten; dims=1)
    #size(score) == (hs, N, nhead, B)
    batched_mul(value, score)
end

function scaled_dot_attention(query::A1, key::A2, value::A3) where {
    T, A1 <: AbstractArray{T, 3}, A2 <: AbstractArray{T, 3}, A3 <: AbstractArray{T, 3}}
    # Input is (hs, N, nhead)
    dh = size(query, 1)
    #size(score) == (N, N, nhead)
    atten = one(T)/convert(T, sqrt(dh)) .* batched_mul(batched_transpose(key), query)
    score = softmax(atten; dims=1)
    #size(score) == (hs, N, nhead)
    batched_mul(value, score)
end

function scaled_dot_attention(query::A1, key::A2, value::A3) where {
    T, A1 <: AbstractMatrix{T}, A2 <: AbstractMatrix{T}, A3 <: AbstractMatrix{T}}
    ## Matrix version, only for checking answers. Input is (hs, N)
    dh = size(query, 1)
    #size(score) == (N, N)
    atten = one(T)/convert(T, sqrt(dh)) .* transpose(key) * query
    score = softmax(atten; dims=1)
    #size(score) == (hs, N)
    value * score
end