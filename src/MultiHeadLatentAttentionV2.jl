using Flux

"""
    MultiHeadLatentAttentionV2(;
    nhead, dim_in, dim_head, dim_lora, dim_out, dim_rope,
    max_seq_length, max_batch_size, norm_layer=RMSNorm
    )

DeepSeek's Multi-Head Latent Attention with decoupled RoPE and weight absorption.

The forward pass calculates:
```math
  Q = W^UQ * norm_cq(W^DQ * query)
    ckv = norm_ckv(W^DK * key)
    K = W^UK * ckv
    V = W^UV * ckv
    kr = embedding(W^KR * key, idx)
    qr = embedding(W^QR * query, idx)
    Q = cat(Q, qr)
    K = cat(K, kr)
    scores = 1/sqrt(dh+dr) * (K^T * Q) .* mask
    A = V * scores
    W^O * A, scores
```

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
    # absorb
    W_KQ::A
    denseOV::D1
    # cache
    cache_ckv::A
    cache_kr::A
end

Flux.@layer MultiHeadLatentAttentionV2 trainable=(denseDQ, denseUQ, denseDKV, denseUK, denseUV, denseO, norm_cq, norm_ckv, embedding, denseQR, denseKR)

function MultiHeadLatentAttentionV2(;
    nhead::Int, dim_in::Int, dim_head::Int, dim_lora, dim_out::Int, dim_rope::Int,
    max_seq_length::Int, max_batch_size::Int, norm_layer=RMSNorm
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
    embedding = RoPE(dim_rope, max_seq_length)
    denseQR = Dense(dim_lora => dim_rope * nhead; bias=false)
    denseKR = Dense(dim_in => dim_rope; bias=false)
    # W_KQ = (W_UK)^T * W_UQ
    W_KQ = _absorb_WUK_WUQ(nhead, denseUK.weight, denseUQ.weight)
    # W_OV = W_O * W_UV
    W_OV = _absorb_WO_WUV(nhead, denseO.weight, denseUV.weight)
    denseOV = Dense(W_OV, false, denseO.σ)
    # cache
    T = eltype(denseDKV.weight)
    cache_ckv = Array{T, 3}(undef, dim_lora, max_seq_length, max_batch_size)
    cache_kr = Array{T, 3}(undef, dim_rope, max_seq_length, max_batch_size)
    MultiHeadLatentAttentionV2(
        nhead,
        denseDQ, denseUQ,
        denseDKV, denseUK, denseUV,
        denseO,
        # normalise
        norm_cq, norm_ckv,
        # embedding
        embedding, denseQR, denseKR,
        # absorb
        W_KQ, denseOV,
        # cache
        cache_ckv, cache_kr
    )
end

function _absorb_WUK_WUQ(nhead::Int, W_UK::AbstractMatrix, W_UQ::AbstractMatrix)
    # W_KQ = (W_UK)^T * W_UQ
    dh = div(size(W_UK, 1), nhead)
    dim_lora = size(W_UK, 2)
    # (dh*nhead, dc) => (dh, dc, nhead)
    W_UQ = permutedims(reshape(W_UQ, dh, nhead, dim_lora), (1, 3, 2)) 
    W_UK = permutedims(reshape(W_UK, dh, nhead, dim_lora), (1, 3, 2))
    W_UKT = permutedims(W_UK, (2, 1, 3)) # (dh, dc, nhead)^T => (dc, dh, nhead)
    batched_mul(W_UKT, W_UQ) 
end

function _absorb_WO_WUV(nhead::Int, W_O::AbstractMatrix, W_UV::AbstractMatrix)
    #  W_VO = W_O * W_UV
    dh = div(size(W_UV, 1), nhead)
    dim_lora = size(W_UV, 2)
    dout = size(W_O, 1)
    W_UVh = permutedims(reshape(W_UV, dh, nhead, dim_lora), (1, 3, 2)) # (dh*nhead, dc) => (dh, dc, nhead)
    W_Oh = reshape(W_O, dout, dh, nhead) # (dout, dh*nhead) => (dout, dh, nhead)
    W_OVh = batched_mul(W_Oh, W_UVh) # (dout, dh, nhead) * (dh, dc, nhead)
    reshape(W_OVh, dout, dim_lora*nhead) # (dout, dc, nhead) => (dout, dc*nhead)
end

function (mla::MultiHeadLatentAttentionV2)(query::A3, key::A3; kwargs...) where {T, A3 <: AbstractArray{T, 3}}
    mla_naive(mla, query, key; kwargs...) # this is faster than the absorb
end

function mla_naive(
    mla::MultiHeadLatentAttentionV2, query::A3, key::A3
    ; start_pos::Int=1, use_cache::Bool=true, mask::Union{Nothing, M}=nothing
    ) where {T, A3 <: AbstractArray{T, 3}, M <: AbstractArray{Bool}}
    dm, seq_length, batch_dim = size(key)
    end_pos = start_pos + seq_length - 1
    cq = mla.norm_cq(mla.denseDQ(query))   # size(cq) == (dc, dq, B)
    ckv = mla.norm_ckv(mla.denseDKV(key)) # size(ckv) == (dc, dkv, B)
    kr, qr = _apply_embeddings(mla, key, cq, start_pos:end_pos)
    if use_cache
        mla.cache_ckv[:, start_pos:end_pos, 1:batch_dim] = ckv
        mla.cache_kr[:, start_pos:end_pos, 1:batch_dim] = kr
        ckv = mla.cache_ckv[:, 1:end_pos, 1:batch_dim]
        kr = mla.cache_kr[:, 1:end_pos, 1:batch_dim]
    end
    K = mla.denseUK(ckv) # size(k) == (dh*nhead, dkv, B)
    V = mla.denseUV(ckv) # size(v) == (dh*nhead, dkv, B)
    Q = mla.denseUQ(cq)  # size(q) == (dh*nhead, dq, B)
    Q, K = _cat_decoupled_embedding(mla.nhead, Q, qr, K, kr)
    A, scores = multi_head_scaled_dot_attention(mla.nhead, Q, K, V; mask=mask)
    mla.denseO(A), scores
end

function _apply_embeddings(mla::MultiHeadLatentAttentionV2, key::A3, cq::A3, idx::UnitRange{Int}) where {T, A3 <: AbstractArray{T, 3}}
    dim_lora, dq, batch_dim = size(cq)
    kr = mla.denseKR(key)
    qr = mla.denseQR(cq)
    kr = mla.embedding(kr, idx) # size(kr) == (dr, dkv, B)
    qr = permutedims(reshape(qr, :, mla.nhead, dq, batch_dim), (1, 3, 2, 4)) # (dr*nhead, dq, B) => (dr, dq, nhead, B)
    qr = mla.embedding(qr, idx)
    kr, qr
end

function _cat_decoupled_embedding(nhead::Int, Qin::A3, Qr::A4, Kin::A3, kr::A3) where {T, A3 <: AbstractArray{T, 3}, A4 <: AbstractArray{T, 4}}
    dhq, dq, B = size(Qin)
    dhk, dkv, B = size(Kin)
    Q = reshape(
        cat(reshape(Qin, :, nhead, dq, B), permutedims(Qr, (1, 3, 2, 4)), dims=1),
        : , dq, B)
    Kr = repeat(Flux.unsqueeze(kr, dims=2), outer=(1, 2, 1, 1))
    K = reshape(
        cat(reshape(Kin, :, nhead, dkv, B), reshape(Kr, :, nhead, dkv, B), dims=1),
        :, dkv, B)
    Q, K
end

function mla_absorb(
    mla::MultiHeadLatentAttentionV2, query::A3, key::A3
    ; start_pos::Int=1, use_cache::Bool=true, mask::Union{Nothing, M}=nothing
    ) where {T, A3 <: AbstractArray{T, 3}, M <: AbstractArray{Bool}}
    # batch multiplication version. Input is dm × N × nhead*B
    dm, seq_length, batch_dim = size(key)
    end_pos = start_pos + seq_length - 1
    dq = size(query, 2)
    dh = div(dm, mla.nhead)
    cq = mla.norm_cq(mla.denseDQ(query))  # size(cq) == (dc, dq, B)
    ckv = mla.norm_ckv(mla.denseDKV(key)) # size(ckv) == (dc, dkv, B)
    kr, qr = _apply_embeddings(mla, key, cq, start_pos:end_pos)
    dr = size(kr, 1)
    if use_cache
        mla.cache_ckv[:, start_pos:end_pos, 1:batch_dim] = ckv
        mla.cache_kr[:, start_pos:end_pos, 1:batch_dim] = kr
        ckv = mla.cache_ckv[:, 1:end_pos, 1:batch_dim]
        kr = mla.cache_kr[:, 1:end_pos, 1:batch_dim]
    end
    ## softmax(1/sqrt(dh) * K^T * Q)
    ## = softmax(1/sqrt(dh) * ckv^T * W_UK^T * W_QU * cq)
    ckv_ = Flux.unsqueeze(ckv, dims=3)
    keyT = permutedims(ckv_, (2, 1, 3, 4)) # (dkv, dc, B) => (dkv, dc, 1, B)
    cq_ = Flux.unsqueeze(cq, dims=3) # (dkv, dc, B) => (dkv, dc, 1, B)
    W_KQ = Flux.unsqueeze(mla.W_KQ, dims=4); # (dc, dc, nhead) => (dc, dc, nhead, 1)
    krT = Flux.unsqueeze(permutedims(kr, (2, 1, 3)), dims=3) # (dr, dkv, B) => (dkv, dr, 1, 1B)
    atten_base = broadcasted_batched_mul(keyT, broadcasted_batched_mul(W_KQ, cq_))
    atten_embed = broadcasted_batched_mul(krT, qr)
    atten = one(T)/convert(T, sqrt(dh + dr)) .* (atten_base + atten_embed)
    atten = apply_mask(atten, mask)
    scores = softmax(atten; dims=1)
    ## W_O * V * atten
    ## = W_O * (W_V * ckv) * atten
    A = broadcasted_batched_mul(ckv_, scores) # (dc, dq, nhead, B)
    # (dc, dq, nhead, B) => (dc*nhead, dq, B)
    A = permutedims(A, [1, 3, 2, 4])
    A = reshape(A, :, size(A, 3), size(A, 4))
    mla.denseOV(A), scores # (dout, dq, B)
end
