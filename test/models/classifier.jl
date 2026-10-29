using Test
using TransformersLite
using Flux
using Flux: pullback

@testset "TransformerClassifier" begin
    model = TransformersLite.TransformerClassifier(
        Embedding(100 => 32), # vocab length is 100
        PositionEncoding(32), 
        Dropout(0.1),
        TransformerBlock[
            TransformerBlock(4, 32, 32 * 4; pdrop=0.1),
            TransformerBlock(4, 32, 32 * 4; pdrop=0.1)
        ],
        Dense(32, 1), 
        FlattenLayer(),
        Dense(10, 3) # sentence length is 10, 3 labels
    )
    x = rand(1:100, 10, 5)
    y = model(x)
    @test size(y) == (3, 5)

    # gradients
    # use sum as a dummy loss function
    _, back = pullback(m->sum(m(x)), model)
    grads = back(1.0)
    @test length(grads[1]) == 7
end

@testset "TransformerClassifier with RoPE" begin
    rope = RoPE(8, 1000)
    model = TransformersLite.TransformerClassifier(
        Embedding(100 => 32), # vocab length is 100
        ZeroPositionEncoding(), # positions are encoded by RoPE in the attention layers
        Dropout(0.1),
        TransformerBlock[
            TransformerBlock(4, 32, 32 * 4; pdrop=0.1, embedding=rope),
            TransformerBlock(4, 32, 32 * 4; pdrop=0.1, embedding=rope)
        ],
        Dense(32, 1),
        FlattenLayer(),
        Dense(10, 3) # sentence length is 10, 3 labels
    )
    @test model.blocks[1].multihead_attention.embedding isa RoPE
    x = rand(1:100, 10, 5)
    y = model(x)
    @test size(y) == (3, 5)

    # gradients
    # use sum as a dummy loss function
    y, back = pullback(m->sum(m(x)), model)
    grads = back(1.0)
    @test length(grads[1]) == 7
end
