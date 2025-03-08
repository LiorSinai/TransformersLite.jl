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
include("embedding/tokenizer.jl")
include("embedding/position_encoding.jl")
include("embedding/RoPE.jl")
include("layers/normalise.jl")
export IndexTokenizer, encode, decode
export PositionEncoding
export RoPE, apply_rope
export RMSNorm, rms_norm

include("layers/MultiHeadAttention.jl")
include("layers/MultiHeadLatentAttention.jl")
include("layers/MultiHeadLatentAttentionV2.jl")
export MultiHeadAttention, MultiHeadAttentionKVCache
export MultiHeadLatentAttention, MultiHeadLatentAttentionV2

include("layers/aggregate_layer.jl")
export MeanLayer, FlattenLayer

include("layers/block.jl")
export TransformerBlock

## Models
include("models/TransformerClassifier.jl")
include("models/TransformerGenerator.jl")
export generate

end