using TransformersLite
using Test

@testset "PositionEncoding" begin
    pe = TransformersLite.PositionEncoding(8)
    x = rand(Float32, 4, 10)
    z = pe(x)
    @test size(z) == (8, 10)
    z2 = pe(1:10)
    @test z2 == z

    @test_throws ErrorException PositionEncoding(5)
end