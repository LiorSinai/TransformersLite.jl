using TransformersLite
using TransformersLite: make_causal_mask, mla_naive, mla_absorb
using Flux: pullback
using Test

@testset "MultiHeadLatentAttention" begin
    nhead, dim_head, dim_lora, dim_out = 2, 4, 7, 9
    dim_model = nhead * dim_head
    N, max_seq_length, batch_dim = 10, 16, 3
    mla = MultiHeadLatentAttention(
        nhead=nhead, dim_in=dim_model, dim_head=dim_head,
        dim_lora=dim_lora, dim_out=dim_out,
        max_seq_length=max_seq_length, max_batch_size=batch_dim
        )
    
    X0 = randn(Float32, dim_model, N, batch_dim)
    # Fill cache and compare with naive
    mask = make_causal_mask(ones(N, N));
    A, scores = mla(X0, X0; mask=mask, use_cache=true);
    @test size(A) == (9, 10, 3)
    @test size(scores) ==(10, 10, 2, 3)

    # use cache
    x = randn(Float32, dim_model, 1, batch_dim)
    X = cat(X0, x, dims=2)
    mask = repeat([true], inner=(N + 1, 1))
    Ax, scoresx = mla(x, x; mask=mask, start_pos=11, use_cache=true)
    mask = make_causal_mask(ones(N + 1, N + 1))
    AX, scoresX = mla(X, X; mask=mask, use_cache=false)
    @test size(AX) == (9, 11, 3)
    @test size(Ax) == (9, 1, 3)
    @test isapprox(AX[:, end, :], Ax[:, end, :])
end

@testset "MultiHeadLatentAttentionV2" begin
    nhead, dim_head, dim_rope, dim_lora, dim_out = 2, 4, 4, 7, 9
    dim_model = nhead * dim_head
    N, max_seq_length, batch_dim = 10, 16, 3
    mla = MultiHeadLatentAttentionV2(
        nhead=nhead, dim_in=dim_model, dim_head=dim_head, dim_rope=dim_rope,
        dim_lora=dim_lora, dim_out=dim_out,
        max_seq_length=max_seq_length, max_batch_size=batch_dim
        )
    
    X0 = randn(Float32, dim_model, N, batch_dim)
    # Fill cache and compare with naive
    mask = make_causal_mask(ones(N, N));
    A_naive, scores_naive = mla_naive(mla, X0, X0; mask=mask, use_cache=true);
    A_absorb, scores_absorb = mla_absorb(mla, X0, X0; mask=mask, use_cache=true);
    @test isapprox(A_absorb, A_naive)
    @test maximum(abs.(A_absorb - A_naive)) < 1e-5
    @test isapprox(scores_absorb, scores_naive)

    # use cache
    x = randn(Float32, dim_model, 1, batch_dim)
    X = cat(X0, x, dims=2)
    mask = repeat([true], inner=(N + 1, 1))
    Ax_naive, scoresx_naive = mla_naive(mla, x, x; mask=mask, start_pos=11, use_cache=true)
    Ax_absorb, scoresx_absorb = mla_absorb(mla, x, x; mask=mask, start_pos=11, use_cache=true)
    mask = make_causal_mask(ones(N + 1, N + 1))
    AX, scoresX = mla_naive(mla, X, X; mask=mask, use_cache=false)
    @test size(AX) == (9, 11, 3)
    @test size(Ax_naive) == (9, 1, 3)
    @test size(Ax_absorb) == (9, 1, 3)
    @test isapprox(Ax_absorb, Ax_naive)
    @test isapprox(AX[:, end, :], Ax_absorb[:, end, :])
end

# nhead != 2 and dim_in != nhead * dim_head
@testset "MultiHeadLatentAttention odd sizes" begin
    nhead, dim_in, dim_head, dim_lora, dim_out = 3, 10, 4, 7, 9
    N, max_seq_length, batch_dim = 10, 16, 3
    mla = MultiHeadLatentAttention(
        nhead=nhead, dim_in=dim_in, dim_head=dim_head,
        dim_lora=dim_lora, dim_out=dim_out,
        max_seq_length=max_seq_length, max_batch_size=batch_dim
        )

    X0 = randn(Float32, dim_in, N, batch_dim)
    mask = make_causal_mask(ones(N, N))
    A, scores = mla(X0, X0; mask=mask, use_cache=true)
    @test size(A) == (dim_out, N, batch_dim)
    @test size(scores) == (N, N, nhead, batch_dim)

    x = randn(Float32, dim_in, 1, batch_dim)
    X = cat(X0, x, dims=2)
    mask = repeat([true], inner=(N + 1, 1))
    Ax, scoresx = mla(x, x; mask=mask, start_pos=N + 1, use_cache=true)
    mask = make_causal_mask(ones(N + 1, N + 1))
    AX, scoresX = mla(X, X; mask=mask, use_cache=false)
    @test isapprox(AX[:, end, :], Ax[:, end, :])
end

@testset "MultiHeadLatentAttentionV2 odd sizes" begin
    nhead, dim_in, dim_head, dim_rope, dim_lora, dim_out = 3, 10, 4, 6, 7, 9
    N, max_seq_length, batch_dim = 10, 16, 3
    mla = MultiHeadLatentAttentionV2(
        nhead=nhead, dim_in=dim_in, dim_head=dim_head, dim_rope=dim_rope,
        dim_lora=dim_lora, dim_out=dim_out,
        max_seq_length=max_seq_length, max_batch_size=batch_dim
        )

    X0 = randn(Float32, dim_in, N, batch_dim)
    mask = make_causal_mask(ones(N, N))
    A_naive, scores_naive = mla_naive(mla, X0, X0; mask=mask, use_cache=true)
    A_absorb, scores_absorb = mla_absorb(mla, X0, X0; mask=mask, use_cache=true)
    @test size(A_naive) == (dim_out, N, batch_dim)
    @test size(scores_naive) == (N, N, nhead, batch_dim)
    @test isapprox(A_absorb, A_naive)
    @test isapprox(scores_absorb, scores_naive)

    x = randn(Float32, dim_in, 1, batch_dim)
    X = cat(X0, x, dims=2)
    mask = repeat([true], inner=(N + 1, 1))
    Ax_naive, _ = mla_naive(mla, x, x; mask=mask, start_pos=N + 1, use_cache=true)
    Ax_absorb, _ = mla_absorb(mla, x, x; mask=mask, start_pos=N + 1, use_cache=true)
    mask = make_causal_mask(ones(N + 1, N + 1))
    AX, _ = mla_naive(mla, X, X; mask=mask, use_cache=false)
    @test isapprox(Ax_absorb, Ax_naive)
    @test isapprox(AX[:, end, :], Ax_naive[:, end, :])
end

# default arguments must not mutate the cache so that Zygote can differentiate them
@testset "MultiHeadLatentAttention gradients" begin
    nhead, dim_in, dim_head, dim_rope, dim_lora, dim_out = 3, 10, 4, 6, 7, 9
    N, batch_dim = 10, 3
    X = randn(Float32, dim_in, N, batch_dim)
    mask = make_causal_mask(ones(N, N))

    mla = MultiHeadLatentAttention(
        nhead=nhead, dim_in=dim_in, dim_head=dim_head,
        dim_lora=dim_lora, dim_out=dim_out,
        max_seq_length=N, max_batch_size=batch_dim
        )
    _, back = pullback(m -> sum(first(m(X, X; mask=mask))), mla)
    grads = back(1.0f0)[1]
    @test size(grads.denseUQ.weight) == size(mla.denseUQ.weight)
    @test size(grads.denseDKV.weight) == size(mla.denseDKV.weight)

    mla2 = MultiHeadLatentAttentionV2(
        nhead=nhead, dim_in=dim_in, dim_head=dim_head, dim_rope=dim_rope,
        dim_lora=dim_lora, dim_out=dim_out,
        max_seq_length=N, max_batch_size=batch_dim
        )
    _, back = pullback(m -> sum(first(m(X, X; mask=mask))), mla2)
    grads = back(1.0f0)[1]
    @test size(grads.denseUQ.weight) == size(mla2.denseUQ.weight)
    @test size(grads.denseKR.weight) == size(mla2.denseKR.weight)
end