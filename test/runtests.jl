using Test
using TransformersLite

@testset verbose = true "TransformersLite" begin
    include("utilities.jl")
    include("embedding/indexer.jl")
    include("embedding/position_encoding.jl")
    include("mul4d.jl")
    include("attention.jl")
    include("layers/MultiHeadAttention.jl")
    include("layers/block.jl")
    include("models/classifier.jl")
    include("models/generator.jl")
end
