using Test
using TransformersLite: mul4d
using Flux: pullback

include("utils.jl")

@testset "mul4d" begin
    # Real
    A = randn(7,5,3,4)
    B = randn(5,7,3,4)
    C = randn(7,6,3,4)

    @test multiply_test(A, B, mul4d(A, B))
    @test multiply_test(A, B, mul4d(A, B; transA=true, transB=true); transA=true, transB=true)
    @test multiply_test(A, C, mul4d(A, C; transA=true); transA=true)
    @test multiply_test(A, A, mul4d(A, A; transA=true); transA=true)
end

@testset "mul4d grad" begin
    # Real
    A = randn(7,5,3,4)
    B = randn(5,7,3,4)

    @test grad_test_analytical(mul4d, A, B, randn(7, 7, 3, 4))
end

