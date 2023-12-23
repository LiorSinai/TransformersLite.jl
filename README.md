# TransformersLite

A basic transformer package. This repository is meant for learning
and is paired with this [blog post](https://liorsinai.github.io/coding/2022/05/18/transformers.html). For a much more comprehensive package with APIs for HuggingFace, optimizations and more, please see Transformers.jl at [github.com/chengchingwen/Transformers.jl](https://github.com/chengchingwen/Transformers.jl).

This package is designed to work with [Flux](https://github.com/FluxML/Flux.jl). It provides a multi-head attention layer as described in the paper [Attention is all you need](https://arxiv.org/abs/1706.03762).
It also provides 
- A simple index tokenizer for mapping words to indices.
- A wrapper for an embedding layer.
- A wrapper for a mean layer.
- A position encoding layer.
- Two encompassing layers to chain these together: `TransformerEncoderBlock` and `TransformerClassifier`. Flux's `chain` function can also be used to chain the layers together.

Two implementations are provided for the 4D batch multiplication such that `A×B` results in `C[:,:,k,l] == A[:,:,k,l] * B[:,:,k,l]`.
These are `mul4d` and an extension to NNlib's `batched_mul`. The extension to `batched_mul` is about 1.5× faster than `mul4d`.

## Examples

Create a model with `Flux.chain`:
```julia
using TransformersLite, Flux
position_encoding = PositionEncoding(32)
add_position_encoding(x) = x .+ position_encoding(x)
model = Chain(
    Embed(32, 1000), # vocab length is 1000
    add_position_encoding, # can also make anonymous
    Dropout(0.1),
    TransformerEncoderBlock(4, 32, 32 * 4; pdrop=0.1),
    TransformerEncoderBlock(4, 32, 32 * 4; pdrop=0.1),
    Dense(32, 1),
    FlattenLayer(),
    Dense(10, 3) # sentence length is 10, 3 labels
    )
```

Create a model with `TransformersLite.TransformerClassifier`:
```julia
using TransformersLite, Flux
model = TransformersLite.TransformerClassifier(
    Embed(32, 1000), # vocab length is 1000
    PositionEncoding(32), 
    Dropout(0.1),
    TransformerEncoderBlock[
        TransformerEncoderBlock(4, 32, 32 * 4; pdrop=0.1),
        TransformerEncoderBlock(4, 32, 32 * 4; pdrop=0.1)
    ],
    Dense(32, 1), 
    FlattenLayer(),
    Dense(10, 3) # sentence length is 10, 3 labels
    )
```

Output looks like:
```julia
TransformerClassifier(
  Embed((32, 1000)),                    # 32_000 parameters
  PositionEncoding(32),
  Dropout(0.1),
  TransformerEncoderBlock(
    MultiheadAttention(num_heads=4, head_size=8, 32=>32)(
      denseQ = Dense(32 => 32),         # 1_056 parameters
      denseK = Dense(32 => 32),         # 1_056 parameters
      denseV = Dense(32 => 32),         # 1_056 parameters
      denseO = Dense(32 => 32),         # 1_056 parameters
    ),
    Dropout(0.1),
    LayerNorm(32),                      # 64 parameters
    Dense(32 => 128, relu),             # 4_224 parameters
    Dense(128 => 32),                   # 4_128 parameters
    Dropout(0.1),
    LayerNorm(32),                      # 64 parameters
  ),
  TransformerEncoderBlock(
    MultiheadAttention(num_heads=4, head_size=8, 32=>32)(
      denseQ = Dense(32 => 32),         # 1_056 parameters
      denseK = Dense(32 => 32),         # 1_056 parameters
      denseV = Dense(32 => 32),         # 1_056 parameters
      denseO = Dense(32 => 32),         # 1_056 parameters
    ),
    Dropout(0.1),
    LayerNorm(32),                      # 64 parameters
    Dense(32 => 128, relu),             # 4_224 parameters
    Dense(128 => 32),                   # 4_128 parameters
    Dropout(0.1),
    LayerNorm(32),                      # 64 parameters
  ),
  Dense(32 => 1),                       # 33 parameters
  FlattenLayer(),
  Dense(10 => 3),                       # 33 parameters
)        # Total: 37 trainable arrays, 57_474 parameters,
         # plus 1 non-trainable, 32_000 parameters, summarysize 352.125 KiB
```

Usage:
```julia
sentence_length = size(model.classifier.weight, 2)
x = rand(1:1000, sentence_length)
y = model(x) # 3×1 Matrix{Float32}

batch_size = 8
X = rand(1:1000, sentence_length, batch_size)
Y = model(X) # 3×8 Matrix{Float32}
```

GPU support:
```julia
using CUDA, cuDNN # As of Julia 1.9, these must be loaded separately to FLux
model = gpu(model) 
X = gpu(X)   # 10×8 CuArray{Int64, 2, CUDA.Mem.DeviceBuffer}
Y = model(X) # 3×8 CuArray{Float32, 2, CUDA.Mem.DeviceBuffer}
```

## Installation

Download the GitHub repository (it is not registered). Then in the Julia REPL:
```julia-repl
julia> ] # enter package mode
(@v1.x) pkg> dev path\\to\\TransformersLite.jl
julia> using Revise # for dynamic editing of code
julia> using TransformersLite
```
