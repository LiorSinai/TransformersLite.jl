module TransformersLite

using ChainRulesCore: @thunk
using Flux
using LinearAlgebra
using NNlib: gather, softmax
using Random
using StatsBase

import ChainRulesCore: rrule, NoTangent
import NNlib: batched_mul

## Functions
include("attention.jl")
include("batched_mul_4d.jl")
include("mask.jl")
include("mul4d.jl")
export batched_mul, mul4d
export scaled_dot_attention, multi_head_scaled_dot_attention

## Layers
include("Embed/tokenizer.jl")
include("Embed/embed.jl")
include("Embed/position_encoding.jl")
export IndexTokenizer, encode, decode
export Embed, PositionEncoding

include("MultiHeadAttention.jl")
export MultiHeadAttention

include("aggregate_layer.jl")
export MeanLayer, FlattenLayer

include("block.jl")
export TransformerBlock

## Models
include("models/TransformerClassifier.jl")

end