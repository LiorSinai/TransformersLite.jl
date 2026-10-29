using Test
using Flux: gradient
using TransformersLite: broadcasted_batched_mul_fast

shapes = [
    ((7, 5, 3, 4), (5, 6, 3, 4)), # equal
    ((7, 5, 3, 4), (5, 6, 1, 1)), # B single matrix
    ((7, 5, 1, 1), (5, 6, 3, 4)), # A single matrix
    ((7, 5, 1, 4), (5, 6, 3, 1)), # mixed, both copied
    ((7, 5, 1, 4), (5, 6, 3, 4)), # mixed, only A copied
]

@testset "broadcasted_batchmul" begin
    @testset "forward $sA x $sB" for (sA, sB) in shapes
        A = randn(sA...)
        B = randn(sB...)
        C1 = broadcasted_mul4d(A, B)
        C2 = broadcasted_batched_mul(A, B)
        @test C1 ≈ C2
        C3 = broadcasted_batched_mul_fast(A, B)
        @test C1 ≈ C3
    end

    @testset "fast $sA x $sB" for (sA, sB) in [
        ((7, 5, 3, 4), (5, 6, 1, 4)), # B is 1 leading, loops
        ((7, 5, 1, 4, 2), (5, 6, 3, 4, 2)), # A is 1 leading, folded
        ((7, 5, 3, 1, 2), (5, 6, 1, 4, 2)), # B is 1, A is 1, equal
        ((7, 5, 1, 1, 2), (5, 6, 3, 4, 1)), # A is 1 merged, then B is 1
        ((7, 5, 2, 1, 1), (5, 6, 2, 1, 1)), # trailing 1s dropped
        ((7, 5, 1, 1, 1), (5, 6, 1, 1, 1)), # all 1
        ((7, 5, 3), (5, 6, 1)), # 3D
    ]
        A = randn(sA...)
        B = randn(sB...)
        @test broadcasted_batched_mul_fast(A, B) ≈ broadcasted_batched_mul(A, B)
    end

    @testset "gradient $sA x $sB" for (sA, sB) in shapes
        A = randn(sA...)
        B = randn(sB...)
        loss(f) = (a, b) -> sum(abs2, f(a, b))
        dA1, dB1 = gradient(loss(broadcasted_batched_mul), A, B)
        @test size(dA1) == size(A)
        @test size(dB1) == size(B)
        @test_throws ErrorException gradient(loss(broadcasted_batched_mul_fast), A, B)
    end
end
