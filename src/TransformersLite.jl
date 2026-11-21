module TransformersLite

using ChainRulesCore: @thunk, @non_differentiable, unthunk
using Flux
using Flux: @layer, unsqueeze
using LinearAlgebra
using NNlib: gather, softmax, batched_mul, batched_mul!, batched_transpose
using Random
using StatsBase

import ChainRulesCore: rrule, NoTangent
import Flux: _big_show

## Functions
include("attention.jl")
include("mask.jl")
include("mul4d.jl")
include("batched_mul.jl")
include("tail.jl")
export mul4d, broadcasted_mul4d, broadcasted_batched_mul
export scaled_dot_attention, multi_head_scaled_dot_attention
export absorbed_attention, absorbed_attention_broadcasted

## Layers
include("embedding/tokenizer.jl")
include("embedding/position_encoding.jl")
include("embedding/rope.jl")
include("layers/normalise.jl")
export IndexTokenizer, encode, decode
export PositionEncoding, ZeroPositionEncoding
export RoPE, apply_rope
export RMSNorm, rms_norm

include("layers/MultiHeadAttention.jl")
include("layers/MultiHeadLatentAttention.jl")
include("layers/MultiHeadLatentAttentionV2.jl")
include("layers/MultiHeadLatentAttentionV3.jl")
export MultiHeadAttention, MultiHeadAttentionKVCache
export MultiHeadLatentAttention, MultiHeadLatentAttentionV2, MultiHeadLatentAttentionV3
export absorb_weights

include("layers/aggregate_layer.jl")
export MeanLayer, FlattenLayer

include("layers/block.jl")
export TransformerBlock

## Models
include("models/TransformerClassifier.jl")
include("models/TransformerGenerator.jl")
export generate

end