using Test
using TransformersLite
using Flux
using Flux: pullback

@testset "TransformerGenerator" begin
    model = TransformersLite.TransformerGenerator(
        Embedding(65 => 32), # vocab_size is 65
        PositionEncoding(32), 
        Dropout(0.1),
        TransformerBlock[
            TransformerBlock(4, 32, 32 * 4; pdrop=0.1),
            TransformerBlock(4, 32, 32 * 4; pdrop=0.1),
        ],
        Dense(32, 65), # vocab_size is 65 
        make_causal_mask(ones(16, 16)),
    )
    x = rand(1:65, 16, 5) 
    y = model(x)
    @test size(y) == (65, 16, 5)

    # gradients
    # use sum as a dummy loss function
    _, back = pullback(m->sum(m(x)), model) # dummy loss function
    grads = back(1.0)
    @test length(grads[1]) == 6
end