module TransformersLite

using ChainRulesCore: @thunk
import ChainRulesCore: rrule, NoTangent
import NNlib: batched_mul
using NNlib: gather, batched_transpose, softmax
using Flux
using StatsBase

include("Embed/tokenizer.jl")
include("Embed/embed.jl")
include("Embed/position_encoding.jl")
export IndexTokenizer, encode, decode
export Embed, PositionEncoding, add_position_encoding

include("batched_mul_4d.jl")
include("mul4d.jl")
export batched_mul, mul4d

include("attention.jl")
export MultiheadAttention, scaled_dot_attention

include("mean_layer.jl")
export MeanLayer

include("encoder.jl")
export TransformerEncoderBlock

include("classifier.jl")


end