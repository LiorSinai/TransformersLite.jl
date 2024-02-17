using TransformersLite: multi_head_scaled_dot_attention, scaled_dot_attention, make_causal_mask
using LinearAlgebra
using Flux: pullback
using Flux: MultiHeadAttention

@testset "multi-head attention" verbose=true begin

@testset "head independence" begin
    key = randn(12, 5, 1)
    query = randn(12, 5, 1)
    value = randn(12, 5, 1)
    A = multi_head_scaled_dot_attention(3, key, query, value)

    K1 = key[1:4, :, :]
    Q1 = query[1:4, :, :]
    V1 = value[1:4, :, :]
    A1 = scaled_dot_attention(K1, Q1, V1)
    @test A[1:4, :, :] == A1

    K2 = key[5:8, :, :]
    Q2 = query[5:8, :, :]
    V2 = value[5:8, :, :]
    A2 = scaled_dot_attention(K2, Q2, V2)
    @test A[5:8, :, :] == A2

    # multiple samples
    key = randn(12, 5, 3)
    query = randn(12, 5, 3)
    value = randn(12, 5, 3)
    A = multi_head_scaled_dot_attention(3, key, query, value)

    K13 = key[1:4, :, 3]
    Q13 = query[1:4, :, 3]
    V13 = value[1:4, :, 3]
    A13 = scaled_dot_attention(K13, Q13, V13)
    @test A[1:4, :, 3] == A13

    K21 = key[5:8, :, 1]
    Q21 = query[5:8, :, 1]
    V21 = value[5:8, :, 1]
    A21 = scaled_dot_attention(K21, Q21, V21)
    @test A[5:8, :, 1] == A21
end

@testset "masking" begin
    # single sample
    x = [
        1.0 2.0 3.0 ;
        4.0 5.0 6.0 ;
        7.0 8.0 9.0
    ]
    mask = make_causal_mask(x)

    A, pull = pullback(
        (k, q, v) -> scaled_dot_attention(k, q, v; mask=mask),
        x, x, x)
    @test_nowarn grads = pull(ones(size(A)))

    # batch sample
    key = randn(12, 5, 3)
    query = randn(12, 5, 3)
    value = randn(12, 5, 3)
    mask = make_causal_mask(key)

    A, pull = pullback(
        (k, q, v) -> scaled_dot_attention(k, q, v; mask=mask),
        key, query, value)
    @test_nowarn grads = pull(ones(size(A)))

    # multi-head sample
    key = randn(12, 5, 3)
    query = randn(12, 5, 3)
    value = randn(12, 5, 3)
    mask = make_causal_mask(key)
    A, pull = pullback(
        (k, q, v) -> multi_head_scaled_dot_attention(3, k, q, v; mask=mask)
        , key, query, value)
    @test_nowarn grads = pull(ones(size(A)))
end

@testset "MultiHeadAttention" begin
    nhead, dim_model, dim_out = 4, 32, 13
    mask = make_causal_mask(ones(10, 10))
    mha = TransformersLite.MultiHeadAttention(nhead, dim_model, dim_out; mask=mask)

    x = randn(Float32, 32, 10, 5)
    y = mha(x, x, x)
    @test size(y) == (13, 10, 5)

    # gradients
    # use sum as a dummy loss function
    _, back = pullback(m->sum(m(x, x, x)), mha)
    grads = back(1.0)
    @test length(grads[1]) == 6

    nhead, dim_model, dim_head, dim_out = 4, 32, 11, 13
    mha = TransformersLite.MultiHeadAttention(nhead, dim_model, dim_head, dim_out)

    x = randn(Float32, 32, 10, 5)
    y = mha(x, x, x)
    @test size(y) == (13, 10, 5)
end

end