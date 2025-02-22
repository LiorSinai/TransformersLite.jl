using Test
using TransformersLite

@testset verbose = true "TransformersLite" begin
    include("utilities.jl")
    include("indexer.jl")
    include("mul4d.jl")
    include("attention.jl")
    include("MultiHeadAttention.jl")
    include("MultiHeadLatentAttention.jl")
    include("block.jl")
    include("classifier.jl")
    include("generator.jl")
end
