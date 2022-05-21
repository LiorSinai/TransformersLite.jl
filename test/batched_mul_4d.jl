using Test
using TransformersLite: batched_mul
using Flux: pullback

include("utils.jl")

@testset "batched_mul 4d" begin
    # Real
    A = randn(7,5,3,4)
    B = randn(5,7,3,4)
    C = randn(7,6,3,4)
    AT = PermutedDimsArray(A, (2, 1, 3, 4))
    BT = PermutedDimsArray(B, (2, 1, 3, 4))

    @test multiply_test(A, B, batched_mul(A, B))
    @test multiply_test(AT, BT, batched_mul(AT, BT))
    @test multiply_test(AT, C, batched_mul(AT, C))
    @test multiply_test(AT, A, batched_mul(AT, A))
end

@testset "batched_mul 4d grad" begin
    # Real
    A = randn(7,5,3,4)
    B = randn(5,7,3,4)
    C = randn(7,6,3,4)
    AT = PermutedDimsArray(A, (2, 1, 3, 4))
    BT = PermutedDimsArray(B, (2, 1, 3, 4))

    @test grad_test_analytical(batched_mul, A, B, randn(7, 7, 3, 4))
    @test grad_test_analytical(batched_mul, AT, BT, randn(5, 5, 3, 4))
    @test grad_test_analytical(batched_mul, AT, C, randn(5, 6, 3, 4))
    @test grad_test_analytical(batched_mul, AT, A, randn(5, 5, 3, 4))
end



