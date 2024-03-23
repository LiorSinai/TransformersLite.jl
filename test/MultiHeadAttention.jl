using TransformersLite
using TransformersLite: make_causal_mask
using Flux: pullback
using Test

@testset "MultiHeadAttention" begin
    nhead, dim_model, dim_out = 4, 32, 13
    mask = make_causal_mask(ones(10, 10))
    mha = TransformersLite.MultiHeadAttention(nhead, dim_model, dim_out)

    x = randn(Float32, 32, 10, 5)
    y = mha(x, x, x; mask=mask)
    @test size(y) == (13, 10, 5)

    y2 = mha(x, x, x)
    @test !isapprox(y, y2) # the mask changes output

    # gradients
    # use sum as a dummy loss function
    y0, back = pullback(m->sum(m(x, x, x)), mha)
    grads = back(1.0)
    @test length(grads[1]) == 5

    y1, back = pullback(m->sum(m(x, x, x; mask=mask)), mha)
    grads = back(1.0)
    @test length(grads[1]) == 5

    nhead, dim_model, dim_head, dim_out = 4, 32, 11, 13
    mha = TransformersLite.MultiHeadAttention(nhead, dim_model, dim_head, dim_out)

    x = randn(Float32, 32, 10, 5)
    y = mha(x, x, x)
    @test size(y) == (13, 10, 5)
end
