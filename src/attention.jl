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
    keyT = batched_transpose(key)             # (dkv, dhkq, nhead)
    atten = scale .* batched_mul(keyT, query) # (dkv, dq, nhead)
    atten = apply_mask(atten, mask)           # (dkv, dq, nhead)
    scores = softmax(atten; dims=1)           # (dkv, dq, nhead)
    batched_mul(value, scores), scores        # (dhv, dq, nhead)
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
    atten = scale .* transpose(key) * query # (dkv, dh)*(dh, dq) => (dkv, dq)
    atten = apply_mask(atten, mask)         # (dkv, dq)
    scores = softmax(atten; dims=1)         # (dkv, dq)
    value * scores, scores                  # (dh, dkv)*(dkv, dq) => (dh, dq)
end

"""
    absorbed_attention(Cq, Ckv, W_KQ[, Kr, Qr]; mask=nothing)

Latent attention with absorbed weights.
Heads are placed between the feature and token dims so that a `reshape` merges them into a
matrix dim.

Inputs:
- `Cq`: latent query `(dc, N, B)`.
- `Ckv`: latent key-value `(dc, N, B)`.
- `W_KQ`: absorbed weight `(dc, nhead, dc)`
- `Kr`: latent embedding `(dr, N, B)`
- `Qr`: latent embedding `(dr, H*N, B)`

The outputs are equivalent to:

    U = reshape(W_KQ, dc * nhead, dc) * reshape(Cq, dc, Nq * B),
    U = reshape(U, dc, nhead * Nq, B)
    scores = softmax(transpose(Ckv) * U / sqrt(dh); dims=1)
    A = Ckv * scores

along with masking if the `mask` is not `nothing`.
"""
function absorbed_attention(
    Cq::AbstractArray{T, 3},
    Ckv::AbstractArray{T, 3},
    W_KQ::AbstractArray{T, 3},
    ; mask::Union{Nothing, M}=nothing, scale::T=one(T)
    ) where {T, M <: AbstractArray{Bool}}
    # Input is (dc, N, nhead)
    dc, N, B = size(Cq)
    nhead = size(W_KQ, 2)
    U = reshape(W_KQ, dc * nhead, dc) * reshape(Cq, dc, N * B) # (dc*nhead, N*B)
    CkvT = batched_transpose(Ckv)                               # (N, dc, nhead)
    atten = scale .* batched_mul(CkvT, reshape(U, dc, nhead * N, B))  # (N, nhead*N, B)
    if !isnothing(mask)
        atten = permutedims(reshape(atten, N, nhead, N, B), (1, 3, 2, 4))
        atten = apply_mask(atten, mask)
        atten = reshape(permutedims(atten, (1, 3, 2, 4)), N, :, B)
    end
    scores = softmax(atten; dims=1)
    scores_out = permutedims(
        reshape(scores, N, nhead, N, B), (1, 3, 2, 4))          # (N, N, nhead, B)
    A = reshape(batched_mul(Ckv, scores), dc, nhead, N, B)      # (dc, nhead, N, B)
    A, scores_out                        
end

# with decoupled embeddings
function absorbed_attention(
    Cq::AbstractArray{T, 3},
    Ckv::AbstractArray{T, 3},
    W_KQ::AbstractArray{T, 3},
    Kr::AbstractArray{T, 3},
    Qr::AbstractArray{T, 3},
    ; mask::Union{Nothing, M}=nothing, scale::T=one(T)
    ) where {T, M <: AbstractArray{Bool}}
    # Input is (dc, N, nhead)
    dc, dq, B = size(Cq)
    dc, dkv, B = size(Ckv)
    nhead = size(W_KQ, 2)
    U = reshape(W_KQ, dc * nhead, dc) * reshape(Cq, dc, dq * B) # (dc*nhead, dq*B)
    CkvT = batched_transpose(Ckv)                               # (dkv, dc, nhead)
    atten = scale .* batched_mul(CkvT, reshape(U, dc, nhead * dq, B)) # (dkv, nhead*dq, B)
    # embeddings
    KrT = permutedims(Kr, (2, 1, 3))            # (dkv, dr, B)
    atten_embed = scale .* batched_mul(KrT, Qr) # (dkv, nhead*dq, B)
    atten += atten_embed
    if !isnothing(mask)
        atten = permutedims(reshape(atten, dkv, nhead, dq, B), (1, 3, 2, 4))
        atten = apply_mask(atten, mask)
        atten = reshape(permutedims(atten, (1, 3, 2, 4)), dkv, :, B)
    end
    scores = softmax(atten; dims=1)
    scores_out = permutedims(
        reshape(scores, dkv, nhead, dq, B), (1, 3, 2, 4))     # (dkv, dq, nhead, B)
    A = reshape(batched_mul(Ckv, scores), dc, nhead, dq, B)
    A, scores_out                        
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
