using Test
using TransformersLite
using Flux
using Flux: pullback
using Random
using TransformersLite: make_causal_mask, clone_add_kv_cache, generate_with_cache

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

    # inputs shorter than the mask
    x_short = rand(1:65, 8, 5)
    y_short = model(x_short)
    @test size(y_short) == (65, 8, 5)

    # generate
    context = generate(model, reshape([1], 1, 1); context_size=16, max_tokens=20)
    @test size(context) == (21, 1)

    # gradients
    # use sum as a dummy loss function
    _, back = pullback(m->sum(m(x)), model) # dummy loss function
    grads = back(1.0)
    @test length(grads[1]) == 6
end

@testset "TransformerGenerator with KV cache" begin
    vocab_size, context_size, batch_size = 65, 16, 5
    model = TransformersLite.TransformerGenerator(
        Embedding(vocab_size => 32),
        PositionEncoding(32),
        Dropout(0.1),
        TransformerBlock[
            TransformerBlock(4, 32, 32 * 4; pdrop=0.1),
            TransformerBlock(4, 32, 32 * 4; pdrop=0.1),
        ],
        Dense(32, vocab_size),
        make_causal_mask(ones(context_size, context_size)),
    )
    model_cached = clone_add_kv_cache(model, context_size, batch_size)

    # fill the cache in chunks: the second chunk has start_pos > 1 and more than one token
    x = rand(1:vocab_size, 9, batch_size)
    y = model(x)
    y1 = model_cached(x[1:4, :]; start_pos=1, use_cache=true)
    y2 = model_cached(x[5:8, :]; start_pos=5, use_cache=true)
    y3 = model_cached(x[9:9, :]; start_pos=9, use_cache=true)
    @test isapprox(y1, y[:, 1:4, :])
    @test isapprox(y2, y[:, 5:8, :])
    @test isapprox(y3, y[:, 9:9, :])

    # prompt longer than one token gives the same result as uncached generation
    context = rand(1:vocab_size, 4, batch_size)
    max_tokens = 12
    out = generate(Random.Xoshiro(1), model, context; context_size=context_size, max_tokens=max_tokens)
    model_cached = clone_add_kv_cache(model, context_size, batch_size)
    out_cached = generate_with_cache(Random.Xoshiro(1), model_cached, context; max_tokens=max_tokens)
    @test size(out_cached) == (4 + max_tokens, batch_size)
    @test out_cached[1:4, :] == context
    @test out_cached == out

    # too long for the cache
    @test_throws ErrorException generate_with_cache(model_cached, context; max_tokens=14)
end