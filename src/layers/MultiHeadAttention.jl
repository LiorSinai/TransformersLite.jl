struct MultiHeadAttention{Q<:Dense, K<:Dense, V<:Dense, O<:Dense}
    nhead::Int
    denseQ::Q
    denseK::K
    denseV::V
    denseO::O
end

# tell Flux which parameters are trainable
Flux.@layer :ignore MultiHeadAttention trainable=(denseQ, denseK, denseV, denseO)

"""
    MultiHeadAttention(nhead, dim_in, [dim_head,] dim_out)

Multi-head dot product attention Layer. `nhead` is the number of heads, 
`dim_in` is the input dimension size, `dim_head` is the size of each head, 
`dim_out` is the output size. If `dim_head` is not specified, it defaults to `dim_in ÷ nhead`.
"""
function MultiHeadAttention(
    nhead::Int, dim_in::Int, dim_head::Int, dim_out::Int
    )
    MultiHeadAttention(
        nhead,
        Dense(dim_in, dim_head*nhead; bias=false),
        Dense(dim_in, dim_head*nhead; bias=false),
        Dense(dim_in, dim_head*nhead; bias=false),
        Dense(dim_head*nhead, dim_out),
    )
end

function MultiHeadAttention(
    nhead::Int, dim_in::Int, dim_out::Int
    )
    if dim_in % nhead != 0 
        error("input dimension=$dim_in is not divisible by number of heads=$nhead")
    end
    MultiHeadAttention(nhead, dim_in, div(dim_in, nhead), dim_out)
end

function (mha::MultiHeadAttention)(
    query::A3, key::A3, value::A3
    ; start_pos::Int=1, use_cache::Bool=false, kwargs...
    ) where {T, A3 <: AbstractArray{T, 3}}
    # batch multiplication version. Input is dm × N × B
    #size(Q) == (dh*nhead, N, B)
    Q = mha.denseQ(query)
    K = mha.denseK(key)
    V = mha.denseV(value)
    A, scores = multi_head_scaled_dot_attention(mha.nhead, Q, K, V; kwargs...)
    mha.denseO(A), scores
end

function (mha::MultiHeadAttention)(
    query::A2, key::A2, value::A2
    ; kwargs...
    ) where {T, A2 <: AbstractMatrix{T}}
    # single sample. Make it a batch of 1
    query = reshape(query, size(query, 1), size(query, 2), 1)
    key = reshape(key, size(key, 1), size(key, 2), 1)
    value = reshape(value, size(value, 1), size(value, 2), 1)
    A, scores = mha(query, key, value; kwargs...)
    reshape(A, size(A, 1), size(A, 2)), reshape(scores, size(scores)[1:3]...)
end

## Show

function Base.show(io::IO, mha::MultiHeadAttention)
    dh = div(size(mha.denseQ.weight)[1], mha.nhead)
    dm = size(mha.denseQ.weight)[2]
    dout = size(mha.denseO.weight)[1]
    print(io, "MultiHeadAttention(")
    print(io, "nhead=$(mha.nhead), ")
    print(io, "head_size=$(dh), ")
    print(io, "$(dm)=>$(dout)")
    print(io, ")")
end

function Flux._big_show(io::IO, mha::MultiHeadAttention, indent::Int=0)
    inner_indent = indent + 2
    print(io, " "^indent, "MultiHeadAttention(\n") 
    println(io, " "^inner_indent, "nhead=$(mha.nhead),")
    Flux._layer_show(io, mha.denseQ, inner_indent, "denseQ")
    Flux._layer_show(io, mha.denseK, inner_indent, "denseK")
    Flux._layer_show(io, mha.denseV, inner_indent, "denseV")
    Flux._layer_show(io, mha.denseO, inner_indent, "denseO")
    print(io, " "^indent, ")")
    if indent == 0
        Flux._big_finale(io, mha)
    else 
        println(io, ",")
    end
end

# With KV Cache
struct MultiHeadAttentionKVCache{
    Q<:Dense, K<:Dense, V<:Dense, O<:Dense, C<:Array{T, 3}  where T
    }
    nhead::Int
    denseQ::Q
    denseK::K
    denseV::V
    denseO::O
    cache_k::C
    cache_v::C
end

Flux.@layer :ignore MultiHeadAttentionKVCache trainable=(denseQ, denseK, denseV, denseO)

function (mha::MultiHeadAttentionKVCache)(
    query::A3, key::A3, value::A3
    ; start_pos::Int=1, use_cache::Bool=true, kwargs...
    ) where {T, A3 <: AbstractArray{T, 3}}
    # batch multiplication version. Input is dm × N × B
    #size(q) == (dm, dq, B)
    q = mha.denseQ(query)
    k = mha.denseK(key)
    v = mha.denseV(value)
    # size(K) == size(V) == (dim, end_pos, B)
    if use_cache
        dim, seq_length, batch_dim = size(query)
        end_pos = start_pos + seq_length - 1
        mha.cache_k[:, start_pos:end_pos, 1:batch_dim] = k
        mha.cache_v[:, start_pos:end_pos, 1:batch_dim] = v
        K = mha.cache_k[:, 1:end_pos, 1:batch_dim]
        V = mha.cache_v[:, 1:end_pos, 1:batch_dim]
    else
        K = k
        V = v
    end
    A, scores = multi_head_scaled_dot_attention(mha.nhead, q, K, V; kwargs...)
    mha.denseO(A), scores
end

function (mha::MultiHeadAttentionKVCache)(
    query::A2, key::A2, value::A2
    ; kwargs...
    ) where {T, A2 <: AbstractMatrix{T}}
    # single sample. Make it a batch of 1
    query = reshape(query, size(query, 1), size(query, 2), 1)
    key = reshape(key, size(key, 1), size(key, 2), 1)
    value = reshape(value, size(value, 1), size(value, 2), 1)
    A, scores = mha(query, key, value; kwargs...)
    reshape(A, size(A, 1), size(A, 2)), reshape(scores, size(scores)[1:3]...)
end

function clone_add_kv_cache(mha::MultiHeadAttention, max_seq_length::Int, max_batch_size::Int)
    dkv = size(mha.denseK.weight, 1)
    T = eltype(mha.denseK.weight)
    cache_k = Array{T, 3}(undef, dkv, max_seq_length, max_batch_size)
    cache_v = Array{T, 3}(undef, dkv, max_seq_length, max_batch_size)
    MultiHeadAttentionKVCache(
        mha.nhead,
        deepcopy(mha.denseQ),
        deepcopy(mha.denseK),
        deepcopy(mha.denseV),
        deepcopy(mha.denseO),
        cache_k,
        cache_v,
    )
end

function Flux._big_show(io::IO, mha::MultiHeadAttentionKVCache, indent::Int=0)
    inner_indent = indent + 2
    print(io, " "^indent, "MultiHeadAttention(\n") 
    println(io, " "^inner_indent, "nhead=$(mha.nhead),")
    Flux._layer_show(io, mha.denseQ, inner_indent, "denseQ")
    Flux._layer_show(io, mha.denseK, inner_indent, "denseK")
    Flux._layer_show(io, mha.denseV, inner_indent, "denseV")
    Flux._layer_show(io, mha.denseO, inner_indent, "denseO")
    Flux._layer_show(io, mha.cache_k, inner_indent, "cache_k")
    Flux._layer_show(io, mha.cache_v, inner_indent, "cache_v")
    print(io, " "^indent, ")")
    if indent == 0
        Flux._big_finale(io, mha)
    else 
        println(io, ",")
    end
end
