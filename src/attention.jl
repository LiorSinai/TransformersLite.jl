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
