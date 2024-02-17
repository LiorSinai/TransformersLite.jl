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

"""
    multi_head_scaled_dot_attention(nhead, query, key, value; mask=nothing)

Apply `scaled_dot_attention` in parallel across `nhead` heads. 
Each array is shaped from `(dm, N, B)` to `(dh, N, nhead, B)` and scaled dot attention is applied 
independently along the batch dimensions of `nhead` and `B`.
The result is then shaped back into a `(dm, N, B)` array.
"""
function multi_head_scaled_dot_attention(nhead::Int, Q::A1, K::A2, V::A3; kwargs...) where {
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
    A = scaled_dot_attention(Q, K, V; kwargs...)
    #size(A) == (dh, N, nhead, B) => (dh, nhead, N, B) => (dm, N, B)
    A = permutedims(A, [1, 3, 2, 4])
    A = reshape(A, dm, size(A, 3), size(A, 4))
end

"""
    scaled_dot_attention(query, key, value; mask=nothing)

Scaled dot attention as proposed in [Attention Is All You Need](https://arxiv.org/abs/1706.03762). 

If the inputs are matrices, the output is:
    
    A = 1/sqrt(dh) * value * softmax(transpose(key) * query))

If the inputs are 3D or 4D arrays, the above equation holds for each `A[:, :, i, j]` with inputs indexed at `X[:, :, i, j]` 
where `X` is the `query`, `key` or `value`.
"""
function scaled_dot_attention(
    query::A1, key::A2, value::A3; mask::Union{Nothing, M}=nothing
    ) where {
    T, A1 <: AbstractArray{T, 4}, A2 <: AbstractArray{T, 4}, A3 <: AbstractArray{T, 4}, M <: AbstractArray{Bool}}
    # Batched version. Input is (dh, N, nhead, B)
    dh = size(query, 1)
    keyT = permutedims(key, (2, 1, 3, 4)) # (dkv, dh, nhead, B)
    atten = one(T)/convert(T, sqrt(dh)) .* batched_mul(keyT, query) # (dkv, dq, nhead, B)
    atten = apply_mask(atten, mask) # (dkv, dq, nhead, B)
    score = softmax(atten; dims=1)  # (dkv, dq, nhead, B)
    batched_mul(value, score)       # (dkv, dq, nhead, B)
end

function scaled_dot_attention(
    query::A1, key::A2, value::A3
    ; mask::Union{Nothing, M}=nothing
    ) where {
    T, A1 <: AbstractArray{T, 3}, A2 <: AbstractArray{T, 3}, A3 <: AbstractArray{T, 3}, M <: AbstractArray{Bool}}
    # Input is (dh, N, nhead)
    dh = size(query, 1)
    keyT = permutedims(key, (2, 1, 3)) # (dkv, dh, nhead)
    atten = one(T)/convert(T, sqrt(dh)) .* batched_mul(keyT, query) # (dkv, dh, nhead)*(dh, dq, nhead) => (dkv, dq, nhead)
    atten = apply_mask(atten, mask) # (dkv, dq, nhead)
    score = softmax(atten; dims=1)  # (dkv, dq, nhead)
    batched_mul(value, score)       # (dh, dkv, nhead)*(dkv, dq, nhead) => (dh, dq, nhead)
end

function scaled_dot_attention(
    query::A1, key::A2, value::A3
    ; mask::Union{Nothing, M}=nothing
    ) where {
    T, A1 <: AbstractMatrix{T}, A2 <: AbstractMatrix{T}, A3 <: AbstractMatrix{T}, M <: AbstractArray{Bool}}
    ## Matrix version for a single head. Input is (dh, N)
    dh = size(query, 1) 
    atten = one(T)/convert(T, sqrt(dh)) .* transpose(key) * query # (dkv, dh)*(dh, dq) => (dkv, dq)
    atten = apply_mask(atten, mask) # (dkv, dq)
    score = softmax(atten; dims=1)  # (dkv, dq)
    value * score                   # (dh, dkv)*(dkv, dq) => (dh, dq)
end
