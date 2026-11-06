"""
    MultiHeadLatentAttentionV3(;
    nhead, dim_in, dim_head, dim_lora, dim_out, dim_rope,
    max_seq_length, max_batch_size,
    norm_layer=RMSNorm, embedding_layer=RoPE
    )

DeepSeek's Multi-Head Latent Attention with decoupled RoPE and weight absorption.
This is mathematically equivalent to `MultiHeadLatentAttentionV2` but the up-projections are absorbed
into the other weights per head:
```math
  W^KQ = (W^UK)^T * W^UQ
    W^OV = W^O * W^UV
```
so that the keys and values never need to be uncompressed from the latent `ckv`.
Use `absorb_weights` to convert a `MultiHeadLatentAttentionV2` layer to this layer.

The forward pass calculates per head:
```math
  cq = norm_cq(W^DQ * query)
    ckv = norm_ckv(W^DKV * key)
    qr = embedding(W^QR * cq, idx)
    kr = embedding(W^KR * key, idx)
    scores = softmax(1/sqrt(dh + dr) * (ckv' * W^KQ * cq + kr' * qr); dims=1)
    A = ckv * scores
    W^OV * A, scores
```
along with masking before the softmax if the `mask` is not `nothing`.
Here `idx = start_pos:(start_pos + N - 1)` are the positions of the query.
If `use_cache=true`, `ckv` and `kr` are the cached values for positions `1:(start_pos + N - 1)`.

Reference: https://arxiv.org/abs/2405.04434
"""
struct MultiHeadLatentAttentionV3{D1<:Dense, N, A<:AbstractArray{T, 3} where T, R} 
    nhead::Int
    dim_head::Int
    denseDQ::D1
    denseDKV::D1
    denseKQ::D1
    denseOV::D1
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

Flux.@layer MultiHeadLatentAttentionV3 trainable=(denseDQ, denseDKV, denseKQ, denseOV, norm_cq, norm_ckv, embedding, denseQR, denseKR)

function MultiHeadLatentAttentionV3(;
    nhead::Int, dim_in::Int, dim_head::Int, dim_lora, dim_out::Int, dim_rope::Int,
    max_seq_length::Int, max_batch_size::Int, norm_layer=RMSNorm, embedding_layer=RoPE
    )
    denseDQ = Dense(dim_in => dim_lora; bias=false) # dim_in => dim_head*nhead
    denseDKV = Dense(dim_in => dim_lora; bias=false)
    # W_KQ = (W_UK)^T * W_UQ
    denseUK = Dense(dim_lora => dim_head * nhead; bias=false)
    denseUQ = Dense(dim_lora => dim_head * nhead; bias=false)
    W_KQ = _absorb_WUK_WUQ(nhead, denseUK.weight, denseUQ.weight)
    denseKQ = Dense(W_KQ, false)
    # W_OV = W_O * W_UV
    denseUV = Dense(dim_lora => dim_head*nhead; bias=false)
    denseO = Dense(dim_head*nhead => dim_out; bias=false)
    W_OV = _absorb_WO_WUV(nhead, denseO.weight, denseUV.weight)
    denseOV = Dense(W_OV, false, denseO.σ)
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
    MultiHeadLatentAttentionV3(
        nhead, dim_head,
        denseDQ, denseDKV, denseKQ, denseOV,
        # normalise
        norm_cq, norm_ckv,
        # embedding
        embedding, denseQR, denseKR,
        # cache
        cache_ckv, cache_kr
    )
end

"""
    absorb_weights(mla::MultiHeadLatentAttentionV2)

Convert a `MultiHeadLatentAttentionV2` instance to `MultiHeadLatentAttentionV3` by absorbing the up-projections per head:
```math
  W^KQ = (W^UK)^T * W^UQ
    W^OV = W^O * W^UV
```

Outputs of `mla(query, key; start_pos, mask, use_cache)` are identical up to floating point error.
`MultiHeadLatentAttentionV3` attends to the latent cache directly instead of uncompressing it with `W^UK` and `W^UV`,
which reduces the cost per decoding step by a factor of about `dim_head`.

Note that the absorbed weights usually have more parameters, because absorption replaces `dim_head` with `dim_lora` in every head
and usually `dim_head < dim_lora < nhead*dim_head`:
- `W^KQ` is `(dim_lora*nhead) × dim_lora` instead of `2 × (dim_head*nhead) × dim_lora`.
- `W^OV` is `dim_out × (dim_lora*nhead)` instead of `(dim_out+dim_lora) × (dim_head*nhead)`.
After conversion each head's block of the absorbed weights has rank at most `dim_head`, but this is not preserved by training.
So the layer should be converted after training.
"""
function absorb_weights(mla::MultiHeadLatentAttentionV2)
    W_KQ = _absorb_WUK_WUQ(mla.nhead, mla.denseUK.weight, mla.denseUQ.weight)
    denseKQ = Dense(W_KQ, false)
    W_OV = _absorb_WO_WUV(mla.nhead, mla.denseO.weight, mla.denseUV.weight)
    denseOV = Dense(W_OV, false, mla.denseO.σ)
    MultiHeadLatentAttentionV3(
        mla.nhead, div(size(mla.denseUK.weight, 1), mla.nhead),
        mla.denseDQ, mla.denseDKV, denseKQ, denseOV,
        # normalise
        mla.norm_cq, mla.norm_ckv,
        # embedding
        mla.embedding, mla.denseQR, mla.denseKR,
        # cache
        mla.cache_ckv, mla.cache_kr
    )
end

function _absorb_WUK_WUQ(nhead::Int, W_UK::AbstractMatrix, W_UQ::AbstractMatrix)
    # W_KQ = (W_UK)^T * W_UQ
    dh = div(size(W_UK, 1), nhead)
    dc = size(W_UK, 2)
    # (dh*nhead, dc) => (dh, dc, nhead)
    W_UQ = permutedims(reshape(W_UQ, dh, nhead, dc), (1, 3, 2)) # (dh, dc, nhead)
    W_UK = permutedims(reshape(W_UK, dh, nhead, dc), (1, 3, 2)) # (dh, dc, nhead)
    W_UKT = permutedims(W_UK, (2, 1, 3))                        # (dc, dh, nhead)
    W_KQ = batched_mul(W_UKT, W_UQ)                             # (dc, dc, nhead)
    W_KQ = permutedims(W_KQ, (1, 3, 2))                         # (dc, nhead, dc)
    reshape(W_KQ, dc*nhead, dc) # absorbed_attention works with head dim in the middle
end

function _absorb_WO_WUV(nhead::Int, W_O::AbstractMatrix, W_UV::AbstractMatrix)
    #  W_VO = W_O * W_UV
    dh = div(size(W_UV, 1), nhead)
    dc = size(W_UV, 2)
    dout = size(W_O, 1)
    W_UVh = permutedims(reshape(W_UV, dh, nhead, dc), (1, 3, 2)) # (dh, dc, nhead)
    W_Oh = reshape(W_O, dout, dh, nhead)                         # (dout, dh, nhead)
    W_OVh = batched_mul(W_Oh, W_UVh)                             # (dout, dc, nhead)
    reshape(W_OVh, dout, dc*nhead)                               # (dout, dc*nhead)
end

function _apply_embeddings(mla::MultiHeadLatentAttentionV3, key::A3, cq::A3, idx::UnitRange{Int}) where {T, A3 <: AbstractArray{T, 3}}
    dc, dq, batch_dim = size(cq)
    kr = mla.denseKR(key)
    qr = mla.denseQR(cq)
    kr = mla.embedding(kr, idx) # size(kr) == (dr, dkv, B)
    # qr will be reshaped later for absorbed_attention
    # but need it in the expected shape for embedding (RoPE) now
    qr = permutedims(reshape(qr, :, mla.nhead, dq, batch_dim), (1, 3, 2, 4)) # (dr*nhead, dq, B) => (dr, dq, nhead, B)
    qr = mla.embedding(qr, idx)
    kr, qr
end

function (mla::MultiHeadLatentAttentionV3)(query::A3, key::A3; kwargs...) where {T, A3 <: AbstractArray{T, 3}}
    mla_absorb(mla, query, key; kwargs...)
end

function mla_absorb(
    mla::MultiHeadLatentAttentionV3, query::A3, key::A3
    ; start_pos::Int=1, use_cache::Bool=false, mask::Union{Nothing, M}=nothing
    ) where {T, A3 <: AbstractArray{T, 3}, M <: AbstractArray{Bool}}
    # batch multiplication version. Input is dm × N × nhead*B
    dm, seq_length, batch_dim = size(key)
    end_pos = start_pos + seq_length - 1
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
    # softmax(1/sqrt(dh + dr) * K^T * Q)
    # = softmax(1/sqrt(dh + dr) * ckv^T * W_UK^T * W_QU * cq)
    # = softmax(1/sqrt(dh + dr) * ckv^T * W_KQ * cq)
    dc, dq, _ = size(cq)
    scale = T(1 / sqrt(mla.dim_head + dr))
    qr = reshape(permutedims(qr, (1, 3, 2, 4)), dr, :, batch_dim)
    A, scores = absorbed_attention(
        cq, ckv, mla.denseKQ, kr, qr; mask=mask, scale=scale) # (dc, nhead, N, B), (N, dq, nhead, B)
    A = reshape(A, :, dq, batch_dim)
    # W_O * V * A
    # = W_O * (W_V * ckv) * A
    # = W_OV * A
    mla.denseOV(A), scores # (dout, dq, B), (N, dq, nhead, B)
end
