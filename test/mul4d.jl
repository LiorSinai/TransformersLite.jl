using Test
using TransformersLite: mul4d
using Flux: pullback

@testset "mul4d" begin
    # Integer
    A = rand(-100:100, 2, 2, 3, 4)
    B = rand(-100:100, 2, 2, 3, 4)
    @test multiply_test(A, B, mul4d(A, B))

    # Real
    A = randn(7, 5, 3, 4)
    B = randn(5, 7, 3, 4)
    C = randn(7, 6, 3, 4)
    AT = PermutedDimsArray(A, (2, 1, 3, 4))
    BT = PermutedDimsArray(B, (2, 1, 3, 4))

    @test multiply_test(A, B, mul4d(A, B))
    @test multiply_test(AT, BT, mul4d(AT, BT))
    @test multiply_test(AT, C, mul4d(AT, C))
    @test multiply_test(AT, A, mul4d(AT, A))

    # Errors
    A = randn(7, 5, 3, 4)
    B = randn(4, 7, 3, 4)
    @test_throws DimensionMismatch mul4d(A, B)
    B = randn(5, 7, 2, 4)
    @test_throws DimensionMismatch mul4d(A, B)
    B = randn(5, 7, 3, 2)
    @test_throws DimensionMismatch mul4d(A, B)
end

@testset "mul4d grad" begin
    # Real
    A = randn(7, 5, 3, 4)
    B = randn(5, 7, 3, 4)
    C = randn(7, 6, 3, 4)
    AT = PermutedDimsArray(A, (2, 1, 3, 4))
    BT = PermutedDimsArray(B, (2, 1, 3, 4))

    @test grad_test_analytical(mul4d, A, B, randn(7, 7, 3, 4))
    @test grad_test_analytical(mul4d, AT, BT, randn(5, 5, 3, 4))
    @test grad_test_analytical(mul4d, AT, C, randn(5, 6, 3, 4))
    @test grad_test_analytical(mul4d, AT, A, randn(5, 5, 3, 4))
end
