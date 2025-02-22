using TransformersLite
using TransformersLite: make_causal_mask
using Flux: pullback
using Test

@testset "MultiHeadAttention" begin
    nhead, dim_model, dim_out = 4, 32, 13
    mask = make_causal_mask(ones(10, 10))
    mha = TransformersLite.MultiHeadAttention(nhead, dim_model, dim_out)

    x = randn(Float32, 32, 10, 5)
    y, scores = mha(x, x, x; mask=mask)
    @test size(y) == (13, 10, 5)

    y2, scores = mha(x, x, x)
    @test !isapprox(y, y2) # the mask changes output

    # gradients
    # use sum as a dummy loss function
    y0, back = pullback(m->sum(m(x, x, x)[1]), mha)
    grads = back(1.0)
    @test length(grads[1]) == 5

    y1, back = pullback(m->sum(m(x, x, x; mask=mask)[1]), mha)
    grads = back(1.0)
    @test length(grads[1]) == 5

    nhead, dim_model, dim_head, dim_out = 4, 32, 11, 13
    mha = TransformersLite.MultiHeadAttention(nhead, dim_model, dim_head, dim_out)

    x = randn(Float32, 32, 10, 5)
    y, scores = mha(x, x, x)
    @test size(y) == (13, 10, 5)
end

@testset "MultiHeadAttentionKVCache" begin
    nhead, dim_model, dim_out = 4, 32, 13
    max_batch_size = 8
    max_seq_length = 16
    mha = TransformersLite.MultiHeadAttentionKVCache(
        nhead,
        Dense(dim_model, dim_model; bias=false),
        Dense(dim_model, dim_model; bias=false),
        Dense(dim_model, dim_model; bias=false),
        Dense(dim_model, dim_out),
        Array{Float32, 3}(undef, dim_model, max_seq_length, max_batch_size),
        Array{Float32, 3}(undef, dim_model, max_seq_length, max_batch_size),
    )
    # Fill cache
    X = randn(Float32, 32, 10, 5)
    mask = make_causal_mask(ones(10, 10))
    A, scores = mha(X, X, X; mask=mask, start_pos=1, use_cache=true)
    @test size(A) == (13, 10, 5)

    # Use cache
    x = randn(Float32, 32, 1, 5)
    X = cat(X, x, dims=2)
    mask = make_causal_mask(ones(11, 11))
    AX, scoresX = mha(X, X, X; mask=mask, start_pos=1, use_cache=false)
    mask = repeat([true], inner=(11, 1))
    Ax, scoresx = mha(x, x, x; mask=mask, start_pos=11, use_cache=true)
    @test size(AX) == (13, 11, 5)
    @test size(Ax) == (13, 1, 5)
    @test isapprox(AX[:, end, :], Ax[:, end, :])
end
