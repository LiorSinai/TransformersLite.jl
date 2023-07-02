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
    MultiheadAttention(nhead::Int, dm::Int, dh::Int, dout::Int)
    MultiheadAttention(nhead::Int, dm::Int, dout::Int)

Multihead dot product attention Layer. `nhead` is the number of heads, 
`dm` is the model embedding dimension size, `dh` is the size of each head, 
`dout` is the output size.
"""
function MultiheadAttention(nhead::Int, dm::Int, dh::Int, dout::Int)
    MultiheadAttention(
        nhead,
        Dense(dm, dh*nhead),
        Dense(dm, dh*nhead),
        Dense(dm, dh*nhead),
        Dense(dh*nhead, dout),
    )
end

function MultiheadAttention(nhead::Int, dm::Int, dout::Int)
    if dm % nhead != 0 
        error("embedding dimension=$dm is not divisible by number of heads=$nhead")
    end
    MultiheadAttention(nhead, dm, div(dm, nhead), dout)
end

function Base.show(io::IO, mha::MultiheadAttention)
    dh = div(size(mha.denseQ.weight)[1], mha.nhead)
    dm = size(mha.denseQ.weight)[2]
    dout = size(mha.denseO.weight)[1]
    print(io, "MultiheadAttention(")
    print(io, "num_heads=$(mha.nhead), ")
    print(io, "head_size=$(dh), ")
    print(io, "$(dm)=>$(dout)")
    print(io, ")")
end

function Base.show(io::IO, m::MIME"text/plain", mha::MultiheadAttention)
    _show_multiheadattention(io, mha)
end

function _show_multiheadattention(io::IO, mha::MultiheadAttention, indent=0)
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

function (mha::MultiheadAttention)(query::A1, key::A2, value::A3) where {
    T, A1 <: AbstractArray{T, 3}, A2 <: AbstractArray{T, 3}, A3 <: AbstractArray{T, 3}}
    # batch multiplication version. Input is dm × N × B
    #size(Q) == (dh*nhead, N, B)
    Q = mha.denseQ(query)
    K = mha.denseK(key)
    V = mha.denseV(value)
    A = multi_head_scaled_dot_attention(mha.nhead, Q, K, V)
    mha.denseO(A)
end

function multi_head_scaled_dot_attention(nhead::Int, Q::A1, K::A2, V::A3) where {
    T, A1 <: AbstractArray{T, 3}, A2 <: AbstractArray{T, 3}, A3 <: AbstractArray{T, 3}}
    qs = size(Q)
    ks = size(K)
    vs = size(V)
    dm = size(Q, 1)
    dh = div(dm, nhead)
    #size(Q) == (dh*nhead, N, B) => (dh, nhead, N, B) => (dh, N, nhead, B)
    Q = permutedims(reshape(Q, dh, nhead, qs[2], qs[3]), [1, 3, 2, 4])
    K = permutedims(reshape(K, dh, nhead, ks[2], ks[3]), [1, 3, 2, 4])
    V = permutedims(reshape(V, dh, nhead, vs[2], vs[3]), [1, 3, 2, 4])
    #size(A) == (dh, N, nhead, B)
    A = scaled_dot_attention(Q, K, V)
    #size(A) == (dh, N, nhead, B) => (dh, nhead, N, B) => (dm, N, B)
    A = permutedims(A, [1, 3, 2, 4])
    A = reshape(A, dm, size(A, 3), size(A, 4))
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
    scaled_dot_attention(query, key, value)

Scaled dot attention as proposed in [Attention Is All You Need](https://arxiv.org/abs/1706.03762). 

If the inputs are matrices, the output is:
    
    A = 1/sqrt(dh) * value * softmax(transpose(key) * query))

If the inputs are 3D arrays, the output is

    A[:, :, h] = 1/sqrt(dh) * value[:, :, h] * softmax(transpose(key[:, :, h]) * query[:, :, h]))   

If the inputs are 4D arrays, the output is 
    
    A[:, :, h, b] = 1/sqrt(dh) * value[:, :, h, b] * softmax(transpose(key[:, :, h, b]) * query[:, :, h, b]))
"""
function scaled_dot_attention(query::A1, key::A2, value::A3) where {
    T, A1 <: AbstractArray{T, 4}, A2 <: AbstractArray{T, 4}, A3 <: AbstractArray{T, 4}}
    # Batched version. Input is (dh, N, nhead, B)
    dh = size(query, 1)
    keyT = permutedims(key, (2, 1, 3, 4))
    atten = one(T)/convert(T, sqrt(dh)) .* batched_mul(keyT, query)
    score = softmax(atten; dims=1) #size(score) == (N, N, nhead, B)
    batched_mul(value, score) #size(attention) == (dh, N, nhead, B)
end

function scaled_dot_attention(query::A1, key::A2, value::A3) where {
    T, A1 <: AbstractArray{T, 3}, A2 <: AbstractArray{T, 3}, A3 <: AbstractArray{T, 3}}
    # Input is (dh, N, nhead)
    dh = size(query, 1)
    keyT = permutedims(key, (2, 1, 3))
    atten = one(T)/convert(T, sqrt(dh)) .* batched_mul(keyT, query)
    score = softmax(atten; dims=1) #size(score) == (N, N, nhead)
    batched_mul(value, score)  #size(attention) == (dh, N, nhead) 
end

function scaled_dot_attention(query::A1, key::A2, value::A3) where {
    T, A1 <: AbstractMatrix{T}, A2 <: AbstractMatrix{T}, A3 <: AbstractMatrix{T}}
    ## Matrix version for a single head. Input is (dh, N)
    dh = size(query, 1)
    atten = one(T)/convert(T, sqrt(dh)) .* transpose(key) * query
    score = softmax(atten; dims=1) #size(score) == (N, N)
    value * score #size(attention) == (dh, N)
end