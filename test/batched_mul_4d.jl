using Test
using TransformersLite: batched_mul
using Flux: pullback

include("utils.jl")

@testset "batched_mul 4d" begin
    # Real
    A = randn(7,5,3,4)
    B = randn(5,7,3,4)
    C = randn(7,6,3,4)
    D = randn(5, 7,3,4)

    @test multiply_test(A, B, batched_mul(A, B))
    @test multiply_test(A, B, batched_mul(A, B; transA=true, transB=true); transA=true, transB=true)
    @test multiply_test(A, C, batched_mul(A, C; transA=true); transA=true)
    @test multiply_test(A, A, batched_mul(A, A; transA=true); transA=true)
end

@testset "batched_mul 4d grad" begin
    # Real
    A = randn(7,5,3,4)
    B = randn(5,7,3,4)

    @test grad_test_analytical(batched_mul, A, B, randn(7, 7, 3, 4))
end



