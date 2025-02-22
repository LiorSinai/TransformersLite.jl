using Flux

struct MultiHeadLatentAttention{D1<:Dense, D2<:Dense, N, A<:AbstractArray{T, 3} where T} 
    nhead::Int
    denseDQ::D1
    denseUQ::D1
    denseDKV::D1
    denseUK::D1
    denseUV::D1
    denseO::D2
    norm_cq::N
    norm_ckv::N
    # absorb
    W_KQ::A
    denseOV::D1
    # cache
    cache_ckv::A
end

Flux.@layer MultiHeadLatentAttention

function MultiHeadLatentAttention(;
    nhead::Int, dim_in::Int, dim_head::Int, dim_lora, dim_out::Int,
    max_seq_length::Int, max_batch_size::Int, norm_layer=RMSNorm
    )
    denseDQ = Dense(dim_in => dim_lora; bias=false) # dim_in => dim_head*nhead
    denseUQ = Dense(dim_lora => dim_head * nhead; bias=false)
    denseDKV = Dense(dim_in => dim_lora; bias=false)
    denseUK = Dense(dim_lora => dim_head*nhead; bias=false)
    denseUV = Dense(dim_lora => dim_head*nhead; bias=false)
    denseO = Dense(dim_head*nhead => dim_out; bias=false)
    norm_cq = norm_layer(dim_lora)
    norm_ckv = norm_layer(dim_lora)
    # W_KQ = (W_UK)^T * W_UQ
    W_KQ = _absorb_WUK_WUQ(nhead, denseUK.weight, denseUQ.weight)
    # W_OV = W_O * W_UV
    W_OV = _absorb_WO_WUV(nhead, denseO.weight, denseUV.weight)
    denseOV = Dense(W_OV, false, denseO.σ)
    # cache
    T = eltype(denseDKV.weight)
    cache_ckv = Array{T, 3}(undef, dim_lora, max_seq_length, max_batch_size)
    MultiHeadLatentAttention(
        nhead,
        denseDQ,
        denseUQ,
        denseDKV,
        denseUK,
        denseUV,
        denseO,
        norm_cq,
        norm_ckv,
        # absorb
        W_KQ,
        denseOV,
        # cache
        cache_ckv
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

function (mla::MultiHeadLatentAttention)(query::A3, key::A3; kwargs...) where {T, A3 <: AbstractArray{T, 3}}
    mla_naive(mla, query, key; kwargs...) # this is faster than the absorb
end

function mla_naive(
    mla::MultiHeadLatentAttention, query::A3, key::A3
    ; start_pos::Int=1, use_cache::Bool=true, mask::Union{Nothing, M}=nothing
    ) where {T, A3 <: AbstractArray{T, 3}, M <: AbstractArray{Bool}}
    dm, seq_length, batch_dim = size(key)
    cq = mla.norm_cq(mla.denseDQ(key))      # size(cq) == (dc, dq, B)
    ckv = mla.norm_ckv(mla.denseDKV(query)) # size(ckv) == (dc, dkv, B)
    if use_cache
        end_pos = start_pos + seq_length - 1
        mla.cache_ckv[:, start_pos:end_pos, 1:batch_dim] = ckv
        ckv = mla.cache_ckv[:, 1:end_pos, 1:batch_dim]
    end
    K = mla.denseUK(ckv) # size(k) == (dh*nhead, dkv, B)
    V = mla.denseUV(ckv) # size(v) == (dh*nhead, dkv, B)
    Q = mla.denseUQ(cq)  # size(q) == (dh*nhead, dq, B)
    A_naive, scores_naive = multi_head_scaled_dot_attention(mla.nhead, Q, K, V; mask=mask)
    A_naive = mla.denseO(A_naive)
    A_naive, scores_naive
end

function mla_absorb(
    mla::MultiHeadLatentAttention, query::A3, key::A3
    ; start_pos::Int=1, use_cache::Bool=true, mask::Union{Nothing, M}=nothing
    ) where {T, A3 <: AbstractArray{T, 3}, M <: AbstractArray{Bool}}
    # batch multiplication version. Input is dm × N × nhead*B
    dm, seq_length, batch_dim = size(key)
    dh = div(dm, mla.nhead)
    cq = mla.norm_cq(mla.denseDQ(key))      # size(cq) == (dc, dq, B)
    ckv = mla.norm_ckv(mla.denseDKV(query)) # size(ckv) == (dc, dkv, B)
    if use_cache
        end_pos = start_pos + seq_length - 1
        mla.cache_ckv[:, start_pos:end_pos, 1:batch_dim] = ckv
        ckv = mla.cache_ckv[:, 1:end_pos, 1:batch_dim]
    end
    ## softmax(1/sqrt(dh) * K^T * Q)
    ## = softmax(1/sqrt(dh) * ckv^T * W_UK^T * W_QU * cq)
    ckv_ = Flux.unsqueeze(ckv, dims=3)
    keyT = permutedims(ckv_, (2, 1, 3, 4)) # (dkv, dc, B) => (dkv, dc, 1, B)
    cq_ = Flux.unsqueeze(cq, dims=3) # (dkv, dc, B) => (dkv, dc, 1, B)
    W_KQ = Flux.unsqueeze(mla.W_KQ, dims=4); # (dc, dc, nhead) => (dc, dc, nhead, 1)
    atten = one(T)/convert(T, sqrt(dh)) .* 
        broadcasted_batched_mul(keyT, broadcasted_batched_mul(W_KQ, cq_)) # (dkv, dq, nhead, B)
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
