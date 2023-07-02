using TransformersLite: multi_head_scaled_dot_attention, scaled_dot_attention

@testset "multi-head attention" begin
    ## independence of heads
    # single sample
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