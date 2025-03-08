using TransformersLite
using TransformersLite: make_causal_mask, mla_naive, mla_absorb
using Test

@testset "broadcasted_batched_mul" begin
    x = rand(3, 2, 1, 3)
    y = rand(2, 3, 2, 1)
    z1 = broadcasted_batched_mul(x, y)
    z2 = broadcasted_mul4d(x, y)
    @test z1 == z2

    # vector
    x = rand(3, 1, 1, 3)
    y = rand(1, 3, 2, 1)
    z1 = broadcasted_batched_mul(x, y)
    z2 = broadcasted_mul4d(x, y)
    @test z1 == z2
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