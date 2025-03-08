using Flux

"""
    MultiHeadLatentAttention(;
    nhead, dim_in, dim_head, dim_lora, dim_out,
    max_seq_length, max_batch_size,
    norm_layer=RMSNorm, embedding_layer=RoPE
    )

DeepSeek's Multi-Head Latent Attention with RoPE.
This layer compresses the `key` and `query` before uncompressing them.
This results in a smaller cache size `dim_lora × max_seq_length × max_batch_size`
as well as a significant performance boost.

See also `MultiHeadLatentAttentionV2` with decoupled embeddings and weight absorption.

The forward pass calculates:
```math
  Q = embedding(W^UQ * norm_cq(W^DQ * query))
    ckv = norm_ckv(W^DK * key)
    K = embedding(W^UK * ckv)
    V = W^UV * ckv
    scores = 1/sqrt(dh) * (K^T * Q) .* mask
    A = V * scores
    W^O * A, scores
```

Reference: https://arxiv.org/abs/2405.04434
"""
struct MultiHeadLatentAttention{D1<:Dense, D2<:Dense, N, A<:AbstractArray{T, 3} where T, R} 
    nhead::Int
    denseDQ::D1
    denseUQ::D1
    denseDKV::D1
    denseUK::D1
    denseUV::D1
    denseO::D2
    # extras
    norm_cq::N
    norm_ckv::N
    embedding::R
    cache_ckv::A
end

Flux.@layer MultiHeadLatentAttention trainable=(denseDQ, denseUQ, denseDKV, denseUK, denseUV, denseO, norm_cq, norm_ckv, embedding)

function MultiHeadLatentAttention(;
    nhead::Int, dim_in::Int, dim_head::Int, dim_lora, dim_out::Int,
    max_seq_length::Int, max_batch_size::Int, norm_layer=RMSNorm, embedding_layer=RoPE
    )
    denseDQ = Dense(dim_in => dim_lora; bias=false) # dim_in => dim_head*nhead
    denseUQ = Dense(dim_lora => dim_head * nhead; bias=false)
    denseDKV = Dense(dim_in => dim_lora; bias=false)
    denseUK = Dense(dim_lora => dim_head*nhead; bias=false)
    denseUV = Dense(dim_lora => dim_head*nhead; bias=false)
    denseO = Dense(dim_head*nhead => dim_out; bias=false)
    # extras
    norm_cq = norm_layer(dim_lora)
    norm_ckv = norm_layer(dim_lora)
    embedding = embedding_layer(dim_head, max_seq_length)
    # cache
    T = eltype(denseDKV.weight)
    cache_ckv = Array{T, 3}(undef, dim_lora, max_seq_length, max_batch_size)
    MultiHeadLatentAttention(
        nhead,
        denseDQ, denseUQ,
        denseDKV, denseUK, denseUV,
        denseO,
        # extras
        norm_cq, norm_ckv,
        embedding,
        cache_ckv,
    )
end

function (mla::MultiHeadLatentAttention)(query::A3, key::A3
    ; start_pos::Int=1, use_cache::Bool=true, mask::Union{Nothing, M}=nothing
    ) where {T, A3 <: AbstractArray{T, 3}, M <: AbstractArray{Bool}}
    dm, seq_length, batch_dim = size(key)
    end_pos = start_pos + seq_length - 1
    cq = mla.norm_cq(mla.denseDQ(query))  # size(cq) == (dc, dq, B)
    ckv = mla.norm_ckv(mla.denseDKV(key)) # size(ckv) == (dc, dkv, B)
    if use_cache
        mla.cache_ckv[:, start_pos:end_pos, 1:batch_dim] = ckv
        ckv = mla.cache_ckv[:, 1:end_pos, 1:batch_dim]
    end
    K = mla.denseUK(ckv) # size(k) == (dh*nhead, dkv, B)
    V = mla.denseUV(ckv) # size(v) == (dh*nhead, dkv, B)
    Q = mla.denseUQ(cq)  # size(q) == (dh*nhead, dq, B)
    A, scores = multi_head_scaled_dot_attention(
        mla.nhead, Q, K, V, mla.embedding, start_pos:end_pos; mask=mask
    )
    mla.denseO(A), scores
end
