using Test
using TransformersLite

@testset verbose = true "TransformersLite" begin
    include("utilities.jl")
    include("indexer.jl")
    include("batched_mul_4d.jl")
    include("mul4d.jl")
    include("attention.jl")
    include("block.jl")
    include("classifier.jl")
end
