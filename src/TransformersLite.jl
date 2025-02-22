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
include("broadcasted_batched_mul.jl")
include("tail.jl")
export mul4d, broadcasted_mul4d, broadcasted_batched_mul
export scaled_dot_attention, multi_head_scaled_dot_attention

## Layers
include("Embed/tokenizer.jl")
include("Embed/position_encoding.jl")
include("normalise.jl")
export IndexTokenizer, encode, decode
export PositionEncoding
export RMSNorm, rms_norm

include("MultiHeadAttention.jl")
include("MultiHeadLatentAttention.jl")
export MultiHeadAttention, MultiHeadAttentionKVCache, MultiHeadLatentAttention

include("aggregate_layer.jl")
export MeanLayer, FlattenLayer

include("block.jl")
export TransformerBlock

## Models
include("models/TransformerClassifier.jl")
include("models/TransformerGenerator.jl")
export generate

end