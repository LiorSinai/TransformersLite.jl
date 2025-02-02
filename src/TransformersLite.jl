module TransformersLite

using ChainRulesCore: @thunk
using Flux
using LinearAlgebra
using NNlib: gather, softmax, batched_mul
using Random
using StatsBase

import ChainRulesCore: rrule, NoTangent
import Flux: _big_show

## Functions
include("attention.jl")
include("mask.jl")
include("mul4d.jl")
include("tail.jl")
export mul4d
export scaled_dot_attention, multi_head_scaled_dot_attention

## Layers
include("Embed/tokenizer.jl")
include("Embed/position_encoding.jl")
export IndexTokenizer, encode, decode
export PositionEncoding

include("MultiHeadAttention.jl")
export MultiHeadAttention

include("aggregate_layer.jl")
export MeanLayer, FlattenLayer

include("block.jl")
export TransformerBlock

## Models
include("models/TransformerClassifier.jl")
include("models/TransformerGenerator.jl")
export generate

end