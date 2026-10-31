using TransformersLite
using TransformersLite: make_causal_mask
using Flux: pullback
using Test

@testset "MultiHeadAttention" begin
    nhead, dim_model, dim_out = 4, 32, 13
    mask = make_causal_mask(ones(10, 10))
    mha = TransformersLite.MultiHeadAttention(nhead, dim_model, dim_out)
    @test mha.embedding === identity

    x = randn(Float32, 32, 10, 5)
    y, scores = mha(x, x, x; mask=mask)
    @test size(y) == (13, 10, 5)

    y2, scores = mha(x, x, x)
    @test !isapprox(y, y2) # the mask changes output

    # gradients
    # use sum as a dummy loss function
    y0, back = pullback(m->sum(m(x, x, x)[1]), mha)
    grads = back(1.0)
    @test length(grads[1]) == 6

    y1, back = pullback(m->sum(m(x, x, x; mask=mask)[1]), mha)
    grads = back(1.0)
    @test length(grads[1]) == 6

    nhead, dim_model, dim_head, dim_out = 4, 32, 11, 13
    mha = TransformersLite.MultiHeadAttention(nhead, dim_model, dim_head, dim_out)

    x = randn(Float32, 32, 10, 5)
    y, scores = mha(x, x, x)
    @test size(y) == (13, 10, 5)
end

@testset "MultiHeadAttention with RoPE" begin
    nhead, dim_model, dim_out = 4, 32, 13
    dim_head = dim_model ÷ nhead
    N, batch_size = 10, 5
    rope = RoPE(dim_head, 1000)
    mha = TransformersLite.MultiHeadAttention(nhead, dim_model, dim_out; embedding=rope)

    x = randn(Float32, dim_model, N, batch_size)
    y, scores = mha(x, x, x)
    @test size(y) == (dim_out, N, batch_size)
    @test size(scores) == (N, N, nhead, batch_size)

    # same weights without RoPE
    mha_no_rope = TransformersLite.MultiHeadAttention(
        nhead, mha.denseQ, mha.denseK, mha.denseV, mha.denseO, identity
    )
    y_no_rope, _ = mha_no_rope(x, x, x)
    @test !isapprox(y, y_no_rope)

    # without positions attention is permutation equivariant: shuffling the tokens shuffles the output
    # RoPE encodes the positions so this no longer holds
    p = [2, 1, 3, 4, 5, 6, 7, 8, 9, 10]
    y_shuffled, _ = mha_no_rope(x[:, p, :], x[:, p, :], x[:, p, :])
    @test y_shuffled ≈ y_no_rope[:, p, :]
    y_shuffled, _ = mha(x[:, p, :], x[:, p, :], x[:, p, :])
    @test !isapprox(y_shuffled, y[:, p, :])

    # gradients
    mask = make_causal_mask(ones(N, N))
    _, back = pullback(m->sum(m(x, x, x; mask=mask)[1]), mha)
    grads = back(1.0)
    @test size(grads[1].denseQ.weight) == size(mha.denseQ.weight)
    @test size(grads[1].denseK.weight) == size(mha.denseK.weight)

    # Require rope `dim == dim_head`
    rope = RoPE(10, 1000)
    @test_throws ArgumentError TransformersLite.MultiHeadAttention(nhead, dim_model, dim_out; embedding=rope)
end
