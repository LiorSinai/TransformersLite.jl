using Flux: Dense, gradient, pullback, unsqueeze
using LinearAlgebra
using Test
using TransformersLite: multi_head_scaled_dot_attention, scaled_dot_attention, make_causal_mask
using TransformersLite: absorbed_attention, absorbed_attention_broadcasted
using TransformersLite: _absorb_WUK_WUQ

@testset "attention" verbose=true begin

@testset "values" begin
    X = [
        0.971084  0.632     0.363706  0.704496  0.757392  0.942686   0.522516  0.719285;
        0.121592  0.481155  0.34992   0.111048  0.153215  0.0320891  0.845669  0.152237;
        0.638092  0.171303  0.191171  0.759539  0.912153  0.295568   0.406384  0.127703;
        0.369658  0.376206  0.246995  0.69837   0.540543  0.619199   0.631959  0.629104;;;
    ]
    A, scores = multi_head_scaled_dot_attention(2, X, X, X)
    expected = [
        0.723074  0.706491  0.702856  0.716756  0.717117  0.72436   0.695074  0.716198;
        0.264675  0.289068  0.28896   0.270156  0.270857  0.261412  0.310158  0.271669;
        0.476473  0.450162  0.450251  0.485985  0.493597  0.458975  0.465315  0.449687;
        0.524952  0.521392  0.519503  0.530866  0.529673  0.526141  0.527181  0.52502;;;
    ]
    @test maximum(abs.(expected - A)) < 1e-6
end

@testset "grads" begin
    X = randn(Float32, 10, 10, 2)
    A, pull = pullback(
        (k, q, v) -> scaled_dot_attention(k, q, v)[1],
        X, X, X)
    @test_nowarn grads = pull(ones(size(A)))

    A_scores, pull = pullback(
        (k, q, v) -> scaled_dot_attention(k, q, v),
        X, X, X)
    @test_nowarn grads = pull((ones(size(A_scores[1])), nothing))

    A, pull = pullback(
        (k, q, v) -> multi_head_scaled_dot_attention(2, k, q, v)[1],
        X, X, X)
    @test_nowarn grads = pull(ones(size(A)))
end

@testset "head independence" begin
    key = randn(12, 5, 1)
    query = randn(12, 5, 1)
    value = randn(12, 5, 1)
    A, scores = multi_head_scaled_dot_attention(3, key, query, value)

    K1 = key[1:4, :, :]
    Q1 = query[1:4, :, :]
    V1 = value[1:4, :, :]
    A1, scores1 = scaled_dot_attention(K1, Q1, V1)
    @test A[1:4, :, :] == A1
    @test scores[:, :, 1, :] == scores1

    K2 = key[5:8, :, :]
    Q2 = query[5:8, :, :]
    V2 = value[5:8, :, :]
    A2, scores2 = scaled_dot_attention(K2, Q2, V2)
    @test A[5:8, :, :] == A2
    @test scores[:, :, 2, :] == scores2

    # multiple samples
    key = randn(12, 5, 3)
    query = randn(12, 5, 3)
    value = randn(12, 5, 3)
    A, scores = multi_head_scaled_dot_attention(3, key, query, value)

    K13 = key[1:4, :, 3]
    Q13 = query[1:4, :, 3]
    V13 = value[1:4, :, 3]
    A13, scores13 = scaled_dot_attention(K13, Q13, V13)
    @test A[1:4, :, 3] == A13
    @test scores[:, :, 1, 3] == scores13

    K21 = key[5:8, :, 1]
    Q21 = query[5:8, :, 1]
    V21 = value[5:8, :, 1]
    A21, scores21 = scaled_dot_attention(K21, Q21, V21)
    @test A[5:8, :, 1] == A21
    @test scores[:, :, 2, 1] == scores21
end

@testset "masking" begin
    # single sample
    x = [
        1.0 2.0 3.0 ;
        4.0 5.0 6.0 ;
        7.0 8.0 9.0
    ]
    mask = make_causal_mask(x)

    y, scores = scaled_dot_attention(x, x, x; mask=mask)
    @test scores[2, 1] == 0.0
    @test scores[3, 1] == 0.0
    @test scores[3, 2] == 0.0

    A, pull = pullback(
        (k, q, v) -> scaled_dot_attention(k, q, v; mask=mask)[1],
        x, x, x)
    @test_nowarn grads = pull(ones(size(A)))

    # batch sample
    key = randn(12, 5, 3)
    query = randn(12, 5, 3)
    value = randn(12, 5, 3)
    mask = make_causal_mask(key)

    Y, scores = scaled_dot_attention(key, query, value; mask=mask)
    tril_indices = [
        false false false false false;
        true false false false false;
        true true false false false;
        true true true false false;
        true true true true false;
    ]
    @test all(scores[tril_indices, :] .== 0)

    A, pull = pullback(
        (k, q, v) -> scaled_dot_attention(k, q, v; mask=mask)[1],
        key, query, value)
    @test_nowarn grads = pull(ones(size(A)))

    # multi-head sample
    key = randn(12, 5, 3)
    query = randn(12, 5, 3)
    value = randn(12, 5, 3)
    mask = make_causal_mask(key)
    A, pull = pullback(
        (k, q, v) -> multi_head_scaled_dot_attention(3, k, q, v; mask=mask)[1]
        , key, query, value)
    @test_nowarn grads = pull(ones(size(A)))
end

@testset "absorbed_attention" begin
    n, dc, nhead, B = 6, 5, 3, 4
    Cq = randn(dc, n, B)
    Ckv = randn(dc, n, B)
    W_KQ = randn(dc, nhead, dc)
    denseKQ = Dense(reshape(W_KQ, dc*nhead, dc), false)
    A1, scores1 = absorbed_attention(Cq, Ckv, denseKQ)
    @test size(A1) == (dc, nhead, n, B) && size(scores1) == (n, n, nhead, B)
    W2 = unsqueeze(permutedims(W_KQ, (1, 3, 2)), dims=4)
    A2, scores2 = absorbed_attention_broadcasted(Cq, Ckv, W2)
    @test size(A2) == (dc, n, nhead, B) && size(scores2) == (n, n, nhead, B)
    A11 = permutedims(A1, (1, 3, 2, 4))
    @test A11 ≈ A2 && scores1 ≈ scores2

    # loss(f) = (Cq, CKv, W_KQ) -> sum(abs2, f(Cq, Ckv, W_KQ))
    # grads1 = gradient(loss(absorbed_attention), Cq, Ckv, W_KQ)
    # grads2 = gradient(loss(absorbed_attention_broadcasted), Cq, Ckv, W2)
    # @test all(grads1 .≈ grads2)
end

@testset "absorbed_attention with decoupled embeddings" begin
    n, dc, dr, nhead, B = 6, 5, 7, 3, 4
    Cq = randn(dc, n, B)
    Ckv = randn(dc, n, B)
    W_KQ = randn(dc, nhead, dc)
    denseKQ = Dense(reshape(W_KQ, dc*nhead, dc), false)
    Kr = randn(dr, n, B)
    Qr = randn(dr, nhead * n, B)
    A1, scores1 = absorbed_attention(Cq, Ckv, denseKQ, Kr, Qr)
    @test size(A1) == (dc, nhead, n, B) && size(scores1) == (n, n, nhead, B)
    W2 = unsqueeze(permutedims(W_KQ, (1, 3, 2)), dims=4)
    Qr2 = permutedims(reshape(Qr, dr, nhead, n, B), (1, 3, 2, 4))
    A2, scores2 = absorbed_attention_broadcasted(Cq, Ckv, W2, Kr, Qr2)
    @test size(A2) == (dc, n, nhead, B) && size(scores2) == (n, n, nhead, B)
    A11 = permutedims(A1, (1, 3, 2, 4))
    @test A11 ≈ A2 && scores1 ≈ scores2
end

@testset "absorbed_attention factors" begin
    n, dc, dh, dr, nhead, B = 6, 5, 2, 7, 3, 4
    Cq = randn(Float32, dc, n, B)
    Ckv = randn(Float32, dc, n, B)
    denseUQ = Dense(dc => dh * nhead, bias=false)
    denseUK = Dense(dc => dh * nhead, bias=false)
    W_KQ = _absorb_WUK_WUQ(nhead, denseUK.weight, denseUQ.weight)
    denseKQ = Dense(W_KQ, false)
    mask = make_causal_mask(ones(n, n))
    # without embeddings
    A1, scores1 = absorbed_attention(Cq, Ckv, denseKQ; mask=mask)
    A2, scores2 = absorbed_attention(nhead, Cq, Ckv, denseUQ, denseUK; mask=mask)
    @test size(A2) == (dc, nhead, n, B) && size(scores2) == (n, n, nhead, B)
    @test A1 ≈ A2 && scores1 ≈ scores2
    # with embeddings
    Kr = randn(Float32, dr, n, B)
    Qr = randn(Float32, dr, nhead * n, B)
    A1, scores1 = absorbed_attention(Cq, Ckv, denseKQ, Kr, Qr; mask=mask)
    A2, scores2 = absorbed_attention(nhead, Cq, Ckv, denseUQ, denseUK, Kr, Qr; mask=mask)
    @test A1 ≈ A2 && scores1 ≈ scores2
end

end