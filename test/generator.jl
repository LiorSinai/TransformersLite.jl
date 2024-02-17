using Test
using TransformersLite
using Flux
using Flux: pullback

@testset "TransformerGenerator" begin
    mask = make_causal_mask(ones(16, 16))
    model = TransformersLite.TransformerGenerator(
        Embed(32, 65), # vocab_size is 65
        PositionEncoding(32), 
        Dropout(0.1),
        TransformerBlock[
            TransformerBlock(4, 32, 32 * 4; pdrop=0.1, mask=mask),
            TransformerBlock(4, 32, 32 * 4; pdrop=0.1, mask=mask),
        ],
        Dense(32, 65), # vocab_size is 65 
    )
    x = rand(1:65, 16, 5) 
    y = model(x)
    @test size(y) == (65, 16, 5)

    # gradients
    # use sum as a dummy loss function
    _, back = pullback(m->sum(m(x)), model) # dummy loss function
    grads = back(1.0)
    @test length(grads[1]) == 5
end