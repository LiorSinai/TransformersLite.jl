"""
    multi_head_scaled_dot_attention(nhead, query, key, value; mask=nothing)

Apply `scaled_dot_attention` in parallel across `nhead` heads. 
Each array is shaped from `(dm, N, B)` to `(dh, N, nhead, B)` and scaled dot attention is applied 
independently along the batch dimensions of `nhead` and `B`.
The result is then shaped back into an `(dm, N, B)` array.

Returns the transformed input sequence and the attention scores.
"""
function multi_head_scaled_dot_attention(nhead::Int, Q::A3, K::A3, V::A3
    ; kwargs...) where {T, A3 <: AbstractArray{T, 3}}
    qs = size(Q)
    ks = size(K)
    vs = size(V)
    dm = size(Q, 1)
    dh = div(dm, nhead)
    #size(Q) == (dh*nhead, N, B) => (dh, nhead, N, B) => (dh, N, nhead, B)
    Q = permutedims(reshape(Q, dh, nhead, qs[2], qs[3]), [1, 3, 2, 4]);
    K = permutedims(reshape(K, dh, nhead, ks[2], ks[3]), [1, 3, 2, 4]);
    V = permutedims(reshape(V, dh, nhead, vs[2], vs[3]), [1, 3, 2, 4]);
    A, scores = scaled_dot_attention(Q, K, V; kwargs...)
    #size(A) == (dh, N, nhead, B) => (dh, nhead, N, B) => (dm, N, B)
    #size(scores) == (N, N, nhead, B)
    A = permutedims(A, [1, 3, 2, 4])
    A = reshape(A, dm, size(A, 3), size(A, 4))
    A, scores
end

"""
    scaled_dot_attention(query, key, value; mask=nothing)

Scaled dot attention as proposed in [Attention Is All You Need](https://arxiv.org/abs/1706.03762).
Returns the transformed input sequence and the attention scores.

If the inputs are matrices, the outputs are equivalent to:

    scores = softmax(transpose(key) * query))
    A = 1/sqrt(dh) * value * scores

along with masking if the `mask` is not `nothing`.


If the inputs are 3D or 4D arrays, the above equations holds for each `A[:, :, k, l]` with inputs indexed at `X[:, :, k, l]` 
where `X` is the `query`, `key` or `value`.
"""
function scaled_dot_attention(
    query::A3, key::A3, value::A3
    ; mask::Union{Nothing, M}=nothing
    ) where {T, A3 <: AbstractArray{T, 3}, M <: AbstractArray{Bool}}
    # Input is (dh, N, nhead)
    dh = size(query, 1)
    keyT = permutedims(key, (2, 1, 3)) # (dkv, dh, nhead)
    atten = one(T)/convert(T, sqrt(dh)) .* batched_mul(keyT, query) # (dkv, dh, nhead)*(dh, dq, nhead) => (dkv, dq, nhead)
    atten = apply_mask(atten, mask) # (dkv, dq, nhead)
    scores = softmax(atten; dims=1) # (dkv, dq, nhead)
    batched_mul(value, scores), scores  # (dh, dkv, nhead)*(dkv, dq, nhead) => (dh, dq, nhead)
end

function scaled_dot_attention(query::A4, key::A4, value::A4; kwargs...) where {T, A4 <: AbstractArray{T, 4}}
    batch_size = size(query)[3:end]
    Q, K, V = map(x -> reshape(x, size(x, 1), size(x, 2), :), (query, key, value))
    A, scores = scaled_dot_attention(Q, K, V; kwargs...)
    A = reshape(A, (size(A, 1), size(A, 2), batch_size...))
    scores = reshape(scores, (size(scores, 1), size(scores, 2), batch_size...))
    A, scores
end

function scaled_dot_attention(
    query::A2, key::A2, value::A2
    ; mask::Union{Nothing, M}=nothing
    ) where {T, A2 <: AbstractMatrix{T}, M <: AbstractArray{Bool}}
    ## Matrix version for a single head. Input is (dh, N)
    dh = size(query, 1) 
    atten = one(T)/convert(T, sqrt(dh)) .* transpose(key) * query # (dkv, dh)*(dh, dq) => (dkv, dq)
    atten = apply_mask(atten, mask) # (dkv, dq)
    scores = softmax(atten; dims=1) # (dkv, dq)
    value * scores, scores          # (dh, dkv)*(dkv, dq) => (dh, dq)
end
