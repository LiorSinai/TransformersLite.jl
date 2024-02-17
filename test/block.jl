using Test
using TransformersLite
using TransformersLite: make_causal_mask
using Flux
using Flux: pullback

@testset "TransformerBlock" begin
    block = TransformerBlock(4, 32, 32 * 4; pdrop=0.1)
    x = randn(Float32, 32, 10, 5)
    y = block(x)
    @test size(y) == (32, 10, 5)

    # gradients
    # use sum as a dummy loss function
    _, back = pullback(m->sum(m(x)), block)
    grads = back(1.0)
    @test length(grads[1]) == 6
end

@testset "TransformerBlock - mask" begin
    mask = make_causal_mask(ones(20, 20))
    block = TransformerBlock(4, 32, 32 * 4; pdrop=0.1, mask=mask)
    x = randn(Float32, 32, 10, 5)
    y = block(x)
    @test size(y) == (32, 10, 5)

    # gradients
    # use sum as a dummy loss function
    _, back = pullback(m->sum(m(x)), block)
    grads = back(1.0)
    @test length(grads[1]) == 6
end