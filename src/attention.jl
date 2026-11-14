"""
    multi_head_scaled_dot_attention(nhead, query, key, value[, q_idxs]; embedding=identity, mask=nothing)

Apply `scaled_dot_attention` in parallel across `nhead` heads.
Each array is shaped from `(dm, N, B)` to `(dh, N, nhead, B)` and scaled dot attention is applied
independently along the batch dimensions of `nhead` and `B`.
The result is then shaped back into an `(dm, N, B)` array.

The `embedding` is applied to the query and key of each head before the attention, for example `RoPE(dh, max_length)`.

Returns the transformed input sequence and the attention scores.
"""
function multi_head_scaled_dot_attention(nhead::Int, Q::A3, K::A3, V::A3
    ; embedding=identity, kwargs...) where {T, A3 <: AbstractArray{T, 3}}
    qs = size(Q)
    ks = size(K)
    vs = size(V)
    dhkq = div(size(Q, 1), nhead)
    dhv = div(size(V, 1), nhead)
    #size(Q) == (dh*nhead, N, B) => (dh, nhead, N, B) => (dh, N, nhead, B)
    Q = permutedims(reshape(Q, dhkq, nhead, qs[2], qs[3]), (1, 3, 2, 4));
    K = permutedims(reshape(K, dhkq, nhead, ks[2], ks[3]), (1, 3, 2, 4));
    V = permutedims(reshape(V, dhv, nhead, vs[2], vs[3]), (1, 3, 2, 4));
    Q = embedding(Q) # (dh, N, nhead, B)
    K = embedding(K) # (dh, N, nhead, B)
    A, scores = scaled_dot_attention(Q, K, V; kwargs...)
    #size(A) == (dh, N, nhead, B) => (dh, nhead, N, B) => (dm, N, B)
    #size(scores) == (N, N, nhead, B)
    A = permutedims(A, (1, 3, 2, 4))
    A = reshape(A, :, size(A, 3), size(A, 4))
    A, scores
end

function multi_head_scaled_dot_attention(nhead::Int, Q::A3, K::A3, V::A3, q_idxs::UnitRange{Int}
    ; embedding=identity, kwargs...) where {T, A3 <: AbstractArray{T, 3}}
    qs = size(Q)
    ks = size(K)
    vs = size(V)
    dhkq = div(size(Q, 1), nhead)
    dhv = div(size(V, 1), nhead)
    #size(Q) == (dh*nhead, N, B) => (dh, nhead, N, B) => (dh, N, nhead, B)
    Q = permutedims(reshape(Q, dhkq, nhead, qs[2], qs[3]), [1, 3, 2, 4]);
    K = permutedims(reshape(K, dhkq, nhead, ks[2], ks[3]), [1, 3, 2, 4]);
    V = permutedims(reshape(V, dhv, nhead, vs[2], vs[3]), [1, 3, 2, 4]);
    Qr = embedding(Q, q_idxs) # this is needed if using KV caching
    Kr = embedding(K)
    A, scores = scaled_dot_attention(Qr, Kr, V; kwargs...)
    #size(A) == (dh, N, nhead, B) => (dh, nhead, N, B) => (dm, N, B)
    #size(scores) == (N, N, nhead, B)
    A = permutedims(A, [1, 3, 2, 4])
    A = reshape(A, :, size(A, 3), size(A, 4))
    A, scores
end

"""
    scaled_dot_attention(query, key, value; mask=nothing)

Scaled dot attention as proposed in [Attention Is All You Need](https://arxiv.org/abs/1706.03762).
Returns the transformed input sequence and the attention scores.

If the inputs are matrices, the outputs are equivalent to:

    scores = softmax(transpose(key) * query / sqrt(dh); dims=1)
    A = value * scores

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
    scale = T(1 / sqrt(dh))
    keyT = batched_transpose(key)             # (nkv, dhkq, nhead)
    atten = scale .* batched_mul(keyT, query) # (nkv, nq, nhead)
    atten = apply_mask(atten, mask)           # (nkv, nq, nhead)
    scores = softmax(atten; dims=1)           # (nkv, nq, nhead)
    batched_mul(value, scores), scores        # (dhv, nq, nhead)
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
    scale = T(1 / sqrt(dh))
    atten = scale .* transpose(key) * query # (nkv, dh)*(dh, nq) => (nkv, nq)
    atten = apply_mask(atten, mask)         # (nkv, nq)
    scores = softmax(atten; dims=1)         # (nkv, nq)
    value * scores, scores                  # (dh, nkv)*(nkv, nq) => (dh, nq)
end

"""
    absorbed_attention(Cq, Ckv, denseKQ[, Kr, Qr]; mask=nothing, scale=one(T))

Latent attention with absorbed weights.

The latents `Cq` and `Ckv` are shared by all heads and only the weight `W_KQ` of `denseKQ` differs per head.
So instead of batching over heads, the heads are folded into a matrix dim of the per-head operand:

    [W_1; W_2; …] * X = [W_1 * X; W_2 * X; …]    # shared operand on the right: heads as rows
    X * [U_1  U_2  …] = [X * U_1  X * U_2  …]    # shared operand on the left: heads as columns

Heads are placed between the feature and token dims, `(dc, nhead, N)`,
so that a `reshape` merges them for free into either the rows `(dc*nhead, N)` or the columns `(dc, nhead*N)`.
This replaces the `nhead*B` small matrix multiplications of `absorbed_attention_broadcasted`
with one large multiplication and `B` medium ones.

Inputs:
- `Cq`: latent query `(dc, nq, B)`.
- `Ckv`: latent key-value `(dc, nkv, B)`.
- `denseKQ`: `Dense(dc => dc*nhead)`.
- `Kr`: key embedding `(dr, nkv, B)`, shared across heads.
- `Qr`: query embedding `(dr, nhead*nq, B)`, heads first.

Returns `A` with size `(dc, nhead, nq, B)` and `scores` with size `(nkv, nq, nhead, B)`.
The calculation is:

    U = denseKQ(Cq)                                       # (dc*nhead, nq, B)
    U = reshape(U, dc, nhead*nq, B)                       # (dc, nhead*nq, B)
    S = softmax(scale * (Ckv^T * U + Kr^T * Qr); dims=1)  # (nkv, nhead*nq, B)
    A = Ckv * S                                           # (dc, nhead*nq, B)

where `*` is batched over `B` and the `mask` is applied before the softmax if it is not `nothing`.
"""
function absorbed_attention(
    Cq::AbstractArray{T, 3},
    Ckv::AbstractArray{T, 3},
    denseKQ::Dense,
    ; mask::Union{Nothing, M}=nothing, scale::T=one(T)
    ) where {T, M <: AbstractArray{Bool}}
    # Input is (dc, nq, B) and (dc, nkv, B)
    dc, nq, B = size(Cq)
    nhead = div(size(denseKQ.weight, 1), dc)
    U = denseKQ(Cq) # (dc*nhead, nq, B)
    _absorbed_attention(nhead, nq, Ckv, U; mask=mask, scale=scale)
end

function _absorbed_attention(
    nhead::Int, nq::Int,
    Ckv::AbstractArray{T, 3},
    U::AbstractArray{T, 3}
    ; mask::Union{Nothing, M}=nothing, scale::T=one(T)
    ) where {T, M <: AbstractArray{Bool}}
    dc, nkv, B = size(Ckv)
    CkvT = batched_transpose(Ckv)  # (nkv, dc, B)
    atten = scale .* batched_mul(CkvT, reshape(U, dc, nhead * nq, B))  # (nkv, nhead*nq, B)
    if !isnothing(mask)
        atten = reshape(atten, nkv, nhead, nq, B)
        atten = apply_mask(atten, _heads_first_mask(mask))
        atten = reshape(atten, nkv, :, B)
    end
    scores = softmax(atten; dims=1)
    scores_out = permutedims(
        reshape(scores, nkv, nhead, nq, B), (1, 3, 2, 4))    # (nkv, nq, nhead, B)
    A = reshape(batched_mul(Ckv, scores), dc, nhead, nq, B)  # (dc, nhead, nq, B)
    A, scores_out
end

# with decoupled embeddings
function absorbed_attention(
    Cq::AbstractArray{T, 3},
    Ckv::AbstractArray{T, 3},
    denseKQ::Dense,
    Kr::AbstractArray{T, 3},
    Qr::AbstractArray{T, 3},
    ; mask::Union{Nothing, M}=nothing, scale::T=one(T)
    ) where {T, M <: AbstractArray{Bool}}
    # Input is (dc, N, B)
    dc, nq, B = size(Cq)
    nhead = div(size(denseKQ.weight, 1), dc)
    U = denseKQ(Cq)               # (dc*nhead, nq, B)
    _absorbed_attention(nhead, nq, Ckv, U, Kr, Qr; mask=mask, scale=scale)
end

function _absorbed_attention(
    nhead::Int, nq::Int,
    Ckv::AbstractArray{T, 3},
    U::AbstractArray{T, 3},
    Kr::AbstractArray{T, 3},
    Qr::AbstractArray{T, 3},
    ; mask::Union{Nothing, M}=nothing, scale::T=one(T)
    ) where {T, M <: AbstractArray{Bool}}
    dc, nkv, B = size(Ckv)
    CkvT = batched_transpose(Ckv) # (nkv, dc, B)
    atten = scale .* batched_mul(CkvT, reshape(U, dc, nhead * nq, B)) # (nkv, nhead*nq, B)
    # embeddings
    KrT = permutedims(Kr, (2, 1, 3))            # (nkv, dr, B)
    atten_embed = scale .* batched_mul(KrT, Qr) # (nkv, nhead*nq, B)
    atten += atten_embed
    if !isnothing(mask)
        atten = reshape(atten, nkv, nhead, nq, B)
        atten = apply_mask(atten, _heads_first_mask(mask))
        atten = reshape(atten, nkv, :, B)
    end
    scores = softmax(atten; dims=1)
    scores_out = permutedims(
        reshape(scores, nkv, nhead, nq, B), (1, 3, 2, 4))   # (nkv, nq, nhead, B)
    A = reshape(batched_mul(Ckv, scores), dc, nhead, nq, B)
    A, scores_out
end

"""
    absorbed_attention(nhead, Cq, Ckv, denseUQ, denseUK[, Kr, Qr]; mask=nothing, scale=one(T))

Latent attention with absorbed weights, following the method of the DeepSeek source code.

Instead of precomputing `W_KQ = (W_UK)^T * W_UQ` per head, each factor is multiplied separately.
Therefore this works with two `dc => dh*nhead` weights rather than one `dc => dc*nhead`.

    Ckv * softmax(scale * Ckv^T * (W_UK_h)^T * W_UQ_h * Cq)

Inputs:
- `Cq`: latent query `(dc, nq, B)`.
- `Ckv`: latent key-value `(dc, nkv, B)`.
- `denseUQ`: `Dense(dc => dh*nhead)`.
- `denseUK`: `Dense(dc => dh*nhead)`. Only its weight is used.
- `Kr`: key embedding `(dr, nkv, B)`, shared across heads.
- `Qr`: query embedding `(dr, nhead*nq, B)`, heads first.

Returns `A` with size `(dc, nhead, nq, B)` and `scores` with size `(nkv, nq, nhead, B)`.
"""
function absorbed_attention(
    nhead::Int,
    Cq::AbstractArray{T, 3},
    Ckv::AbstractArray{T, 3},
    denseUQ::Dense,
    denseUK::Dense,
    ; mask::Union{Nothing, M}=nothing, scale::T=one(T)
    ) where {T, M <: AbstractArray{Bool}}
    dc, nq, B = size(Cq)
    U = _mul_WKQ_factored(nhead, Cq, denseUQ, denseUK) # (dc, nhead*nq, B)
    _absorbed_attention(nhead, nq, Ckv, U; mask=mask, scale=scale)
end

# with decoupled embeddings
function absorbed_attention(
    nhead::Int,
    Cq::AbstractArray{T, 3},
    Ckv::AbstractArray{T, 3},
    denseUQ::Dense,
    denseUK::Dense,
    Kr::AbstractArray{T, 3},
    Qr::AbstractArray{T, 3},
    ; mask::Union{Nothing, M}=nothing, scale::T=one(T)
    ) where {T, M <: AbstractArray{Bool}}
    dc, nq, B = size(Cq)
    U = _mul_WKQ_factored(nhead, Cq, denseUQ, denseUK) # (dc, nhead*nq, B)
    _absorbed_attention(nhead, nq, Ckv, U, Kr, Qr; mask=mask, scale=scale)
end

"""
    _mul_WKQ_factored(nhead, Cq, denseUQ, denseUK)

Calculate `W_KQ * Cq` per head as `(W_UK_h)^T * (W_UQ_h * Cq)`, without forming `W_KQ = (W_UK)^T * W_UQ`.
Equivalent to `denseKQ(Cq)` with `denseKQ = Dense(_absorb_WUK_WUQ(nhead, W_UK, W_UQ), false)`.
Returns size `(dc, nhead*nq, B)`
"""
function _mul_WKQ_factored(nhead::Int, Cq::AbstractArray{T, 3}, denseUQ::Dense, denseUK::Dense) where T
    dc, nq, B = size(Cq)
    dh = div(size(denseUQ.weight, 1), nhead)
    Q = denseUQ(Cq)                                             # (dh*nhead, nq, B)
    Q = permutedims(reshape(Q, dh, nhead, nq, B), (1, 3, 4, 2)) # (dh, nq, B, nhead)
    Q = reshape(Q, dh, nq * B, nhead)                           # (dh, nq*B, nhead)
    W_UK = reshape(denseUK.weight, dh, nhead, dc)               # (dh, nhead, dc)
    # # einsum version: q[c, h, nb] = Σ_d W_UK[d, h, c] * q0[d, h, nb]
    # Q = reshape(Q, dh, nhead, nq * B)   # (dh, nhead, nq*B)
    # U = similar(Q, T, dc, nhead, nq * B)
    # for h in 1:nhead
    #     mul!(view(U, :, h, :), transpose(view(W_UK, :, h, :)), view(q0, :, h, :))
    # end
    W_UKT = permutedims(W_UK, (3, 1, 2))                        # (dc, dh, nhead)
    U = batched_mul(W_UKT, Q)                                   # (dc, nq*B, nhead)
    U = permutedims(reshape(U, dc, nq, B, nhead), (1, 4, 2, 3)) # (dc, nhead, nq, B)
    reshape(U, dc, nhead * nq, B)
end

"""
    absorbed_attention_broadcasted(Cq, Ckv, W_KQ[, Kr, Qr]; mask=nothing, scale=one(T))

Latent attention with broadcasting.

Inputs:
- `Cq`: latent query `(dc, N, B)`.
- `Ckv`: latent key-value `(dc, N, B)`.
- `W_KQ`: absorbed weight `(dc, dc, nhead, 1)`
- `Kr`: latent embedding `(dr, N, B)`
- `Qr`: latent embedding `(dr, N, H, B)`
"""
function absorbed_attention_broadcasted(
    Cq::AbstractArray{T, 3}, 
    Ckv::AbstractArray{T, 3},
    W_KQ::AbstractArray{T, 4},
    ; mask::Union{Nothing, M}=nothing, scale::T=one(T)
    ) where {T, M <: AbstractArray{Bool}}
    Ckv = Flux.unsqueeze(Ckv, dims=3)
    Cq = Flux.unsqueeze(Cq, dims=3)
    keyT = permutedims(Ckv, (2, 1, 3, 4))    # (N, dc, 1, B)
    U = broadcasted_batched_mul(W_KQ, Cq)    # (dc, N, nhead, B)
    atten = scale .* broadcasted_batched_mul(keyT, U) # (N, N, nhead, B)
    atten = apply_mask(atten, mask)       
    scores = softmax(atten; dims=1)
    broadcasted_batched_mul(Ckv, scores), scores # (dc, N, nhead, B)
end

function absorbed_attention_broadcasted(
    Cq::AbstractArray{T, 3}, 
    Ckv::AbstractArray{T, 3},
    W_KQ::AbstractArray{T, 4},
    Kr::AbstractArray{T, 3},
    Qr::AbstractArray{T, 4},
    ; mask::Union{Nothing, M}=nothing, scale::T=one(T)
    ) where {T, M <: AbstractArray{Bool}}
    Ckv = Flux.unsqueeze(Ckv, dims=3)
    Cq = Flux.unsqueeze(Cq, dims=3)
    keyT = permutedims(Ckv, (2, 1, 3, 4))    # (N, dc, 1, B)
    U = broadcasted_batched_mul(W_KQ, Cq)    # (dc, N, nhead, B) 
    atten = scale .* broadcasted_batched_mul(keyT, U) # (N, N, nhead, B)
    # embeddings
    KrT = Flux.unsqueeze(permutedims(Kr, (2, 1, 3)), dims=3) # (N, dr, 1, B)
    atten_embed = scale .* broadcasted_batched_mul(KrT, Qr)  # (N, N, nhead, B)
    atten += atten_embed
    # merge
    atten = apply_mask(atten, mask)       
    scores = softmax(atten; dims=1)
    broadcasted_batched_mul(Ckv, scores), scores # (dc, N, nhead, B)
end
