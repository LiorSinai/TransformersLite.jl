"""
    MultiHeadLatentAttentionV2(;
    nhead, dim_in, dim_head, dim_lora, dim_out, dim_rope,
    max_seq_length, max_batch_size,
    norm_layer=RMSNorm, embedding_layer=RoPE
    )

DeepSeek's Multi-Head Latent Attention with decoupled RoPE.
The position embedding is applied to separate query and key vectors of size `dim_rope`.
The key embedding `kr` is shared across all heads.
Both the key-value latent `ckv` and `kr` are cached,
so the cache size is `(dim_lora + dim_rope) × max_seq_length × max_batch_size`.

See also `MultiHeadLatentAttentionV3`, which is the same layer with weight absorption.
This can be layer converted to `MultiHeadLatentAttentionV3` using `absorb_weights`:
```julia
mla_v3 = absorb_weights(mla_v2)
```

The forward pass calculates per head:
```math
  cq = norm_cq(W^DQ * query)
    ckv = norm_ckv(W^DKV * key)
    Q = W^UQ * cq
    K = W^UK * ckv
    V = W^UV * ckv
    qr = embedding(W^QR * cq, idx)
    kr = embedding(W^KR * key, idx)
    Q = vcat(Q, qr)
    K = vcat(K, kr)
    scores = softmax(1/sqrt(dh + dr) * K^T * Q; dims=1)
    A = V * scores
    W^O * A, scores
```
along with masking before the softmax if the `mask` is not `nothing`.
Here `idx = start_pos:(start_pos + N - 1)` are the positions of the query.
If `use_cache=true`, `ckv` and `kr` are the cached values for positions `1:(start_pos + N - 1)`.

Reference: https://arxiv.org/abs/2405.04434
"""
struct MultiHeadLatentAttentionV2{D1<:Dense, D2<:Dense, N, A<:AbstractArray{T, 3} where T, R} 
    nhead::Int
    denseDQ::D1
    denseUQ::D1
    denseDKV::D1
    denseUK::D1
    denseUV::D1
    denseO::D2
    # normalise
    norm_cq::N
    norm_ckv::N
    # embedding
    embedding::R
    denseQR::D1
    denseKR::D1
    # cache
    cache_ckv::A
    cache_kr::A
end

Flux.@layer MultiHeadLatentAttentionV2 trainable=(denseDQ, denseUQ, denseDKV, denseUK, denseUV, denseO, norm_cq, norm_ckv, embedding, denseQR, denseKR)

function MultiHeadLatentAttentionV2(;
    nhead::Int, dim_in::Int, dim_head::Int, dim_lora, dim_out::Int, dim_rope::Int,
    max_seq_length::Int, max_batch_size::Int, norm_layer=RMSNorm, embedding_layer=RoPE
    )
    denseDQ = Dense(dim_in => dim_lora; bias=false) # dim_in => dim_head*nhead
    denseUQ = Dense(dim_lora => dim_head * nhead; bias=false)
    denseDKV = Dense(dim_in => dim_lora; bias=false)
    denseUK = Dense(dim_lora => dim_head*nhead; bias=false)
    denseUV = Dense(dim_lora => dim_head*nhead; bias=false)
    denseO = Dense(dim_head*nhead => dim_out; bias=false)
    # normalise
    norm_cq = norm_layer(dim_lora)
    norm_ckv = norm_layer(dim_lora)
    # embedding 
    embedding = embedding_layer(dim_rope, max_seq_length)
    denseQR = Dense(dim_lora => dim_rope * nhead; bias=false)
    denseKR = Dense(dim_in => dim_rope; bias=false)
    # cache
    T = eltype(denseDKV.weight)
    cache_ckv = similar(denseDKV.weight, T, dim_lora, max_seq_length, max_batch_size)
    cache_kr = similar(denseKR.weight, T, dim_rope, max_seq_length, max_batch_size)
    MultiHeadLatentAttentionV2(
        nhead,
        denseDQ, denseUQ,
        denseDKV, denseUK, denseUV,
        denseO,
        # normalise
        norm_cq, norm_ckv,
        # embedding
        embedding, denseQR, denseKR,
        # cache
        cache_ckv, cache_kr
    )
end

function _apply_embeddings(mla::MultiHeadLatentAttentionV2, key::A3, cq::A3, idx::UnitRange{Int}) where {T, A3 <: AbstractArray{T, 3}}
    dc, nq, batch_dim = size(cq)
    kr = mla.denseKR(key)
    qr = mla.denseQR(cq)
    kr = mla.embedding(kr, idx) # size(kr) == (dr, nkv, B)
    qr = permutedims(reshape(qr, :, mla.nhead, nq, batch_dim), (1, 3, 2, 4)) # (dr*nhead, nq, B) => (dr, nq, nhead, B)
    qr = mla.embedding(qr, idx)
    kr, qr
end

function _cat_decoupled_embedding(nhead::Int, Q::A3, qr::A4, K::A3, kr::A3) where {T, A3 <: AbstractArray{T, 3}, A4 <: AbstractArray{T, 4}}
    nq, nq, B = size(Q)
    dr, nkv, _ = size(kr)
    qr = permutedims(qr, (1, 3, 2, 4))                      # (dr, nq, nhead, B) => (dr, nhead, nq, B)
    kr = repeat(reshape(kr, dr, 1, nkv, B), 1, nhead, 1, 1) # (dr, nkv, B) => (dr, nhead, nkv, B)
    Q = cat(reshape(Q, :, nhead, nq, B), qr; dims=1)        # (dh+dr, nhead, nq, B)
    K = cat(reshape(K, :, nhead, nkv, B), kr; dims=1)       # (dh+dr, nhead, nkv, B)
    reshape(Q, :, nq, B), reshape(K, :, nkv, B)             # (nhead*(dh+dr), N, B)
end

function (mla::MultiHeadLatentAttentionV2)(query::A3, key::A3; kwargs...) where {T, A3 <: AbstractArray{T, 3}}
    mla_naive(mla, query, key; kwargs...)
end

function mla_naive(
    mla::MultiHeadLatentAttentionV2, query::A3, key::A3
    ; start_pos::Int=1, use_cache::Bool=false, mask::Union{Nothing, M}=nothing
    ) where {T, A3 <: AbstractArray{T, 3}, M <: AbstractArray{Bool}}
    # batch multiplication version. Input is dm × N × nhead*B
    dm, seq_length, batch_dim = size(key)
    end_pos = start_pos + seq_length - 1
    cq = mla.norm_cq(mla.denseDQ(query))  # size(cq) == (dc, nq, B)
    ckv = mla.norm_ckv(mla.denseDKV(key)) # size(ckv) == (dc, nkv, B)
    kr, qr = _apply_embeddings(mla, key, cq, start_pos:end_pos)
    if use_cache
        mla.cache_ckv[:, start_pos:end_pos, 1:batch_dim] = ckv
        mla.cache_kr[:, start_pos:end_pos, 1:batch_dim] = kr
        ckv = mla.cache_ckv[:, 1:end_pos, 1:batch_dim]
        kr = mla.cache_kr[:, 1:end_pos, 1:batch_dim]
    end
    K = mla.denseUK(ckv) # size(k) == (dh*nhead, nkv, B)
    V = mla.denseUV(ckv) # size(v) == (dh*nhead, nkv, B)
    Q = mla.denseUQ(cq)  # size(q) == (dh*nhead, nq, B)
    Q, K = _cat_decoupled_embedding(mla.nhead, Q, qr, K, kr)
    A, scores = multi_head_scaled_dot_attention(
        mla.nhead, Q, K, V
        ; mask=mask, embedding=identity # already applied embeddings
    )
    mla.denseO(A), scores
end
