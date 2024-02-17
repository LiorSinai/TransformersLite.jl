# TransformersLite

A basic transformer package. This repository is meant for learning
and is paired with this [blog post](https://liorsinai.github.io/coding/2022/05/18/transformers.html). For a much more comprehensive package with APIs for HuggingFace, optimizations and more, please see Transformers.jl at [github.com/chengchingwen/Transformers.jl](https://github.com/chengchingwen/Transformers.jl).

This package is designed to work with [Flux](https://github.com/FluxML/Flux.jl). It implements multi-head attention as described in the paper [Attention is all you need](https://arxiv.org/abs/1706.03762).

It comes with the following layers:
- `Indexer`: map tokens to indices.
- `Embed`: and embedding layer, similar to `Flux.Embedding`.
- `PositionEncoding`: implements a fix sinusodial layer as described in [Attention is all you need](https://arxiv.org/abs/1706.03762) paper.
- `MultiHeadAttention`: similar to but differs from `Flux.MultiHeadAttention`.
- `MeanLayer`
- `FlattenLayer`
- `TransformerBlock`: this encompasses a `MultiHeadAttention` layer followed by a dense feed forward network, with dropout and normalization.

Two implementations are provided for the 4D batch multiplication such that `A×B` results in `C[:,:,k,l] == A[:,:,k,l] * B[:,:,k,l]`.
These are `mul4d` and an extension to `NNlib.batched_mul`. The extension to `batched_mul` is about 1.5× faster than `mul4d`.

These layers can be used together with `Flux.Chain`. For convenience, the following models are also provided:
- `TransformerClassifier`: a transformer encoder followed by some aggregation layer (use `MeanLayer` or `Dense`), a `FlattenLayer` and a `Dense` layer for the head.
- `TransformerGenerator`: a transformer encoder with masking followed by a `Dense` layer for the head.

## Examples
### Classifier

Create a model with `Flux.chain`:
```julia
using TransformersLite, Flux
position_encoding = PositionEncoding(32)
add_position_encoding(x) = x .+ position_encoding(x)
model = Chain(
    Embed(32, 1000), # vocab length is 1000
    add_position_encoding, # can also make anonymous
    Dropout(0.1),
    TransformerBlock(4, 32, 32 * 4; pdrop=0.1),
    TransformerBlock(4, 32, 32 * 4; pdrop=0.1),
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
    TransformerBlock[
        TransformerBlock(4, 32, 32 * 4; pdrop=0.1),
        TransformerBlock(4, 32, 32 * 4; pdrop=0.1)
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
  TransformerBlock(
    MultiHeadAttention(num_heads=4, head_size=8, 32=>32)(
      denseQ = Dense(32 => 32; bias=false),  # 1_024 parameters
      denseK = Dense(32 => 32; bias=false),  # 1_024 parameters
      denseV = Dense(32 => 32; bias=false),  # 1_024 parameters
      denseO = Dense(32 => 32),         # 1_056 parameters
      mask = nothing,
    ),
    Dropout(0.1),
    LayerNorm(32),                      # 64 parameters
    Dense(32 => 128, relu),             # 4_224 parameters
    Dense(128 => 32),                   # 4_128 parameters
    Dropout(0.1),
    LayerNorm(32),                      # 64 parameters
  ),
  TransformerBlock(
    MultiHeadAttention(num_heads=4, head_size=8, 32=>32)(
      denseQ = Dense(32 => 32; bias=false),  # 1_024 parameters
      denseK = Dense(32 => 32; bias=false),  # 1_024 parameters
      denseV = Dense(32 => 32; bias=false),  # 1_024 parameters
      denseO = Dense(32 => 32),         # 1_056 parameters
      mask = nothing,
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
)        # Total: 31 trainable arrays, 57_282 parameters,
          # plus 1 non-trainable, 32_000 parameters, summarysize 351.141 KiB.
```

Usage:
```julia
vocab_size = 1000
sentence_length = size(model.classifier.weight, 2)
x = rand(1:vocab_size, sentence_length) 
y = model(x) # 3×1 Matrix{Float32}

batch_size = 8
X = rand(1:vocab_size, sentence_length, batch_size)
Y = model(X) # 3×8 Matrix{Float32}
```

### Generator

Create a model with `TransformersLite.TransformerGenerator`:
```julia
using TransformersLite, Flux
using TransformersLite: make_causal_mask
mask = make_causal_mask(ones(16, 16))
model = TransformersLite.TransformerGenerator(
    Embed(32, 65), # vocab_size is 65
    PositionEncoding(32), 
    Dropout(0.1),
    TransformerBlock[
        TransformerBlock(4, 32, 32 * 4; pdrop=0.1, mask=mask),
        TransformerBlock(4, 32, 32 * 4; pdrop=0.1, mask=mask),
    ],
    Dense(32, 65), # vocab_size is 65 
    )
```

Output looks like:
```julia
TransformerGenerator(
  Embed((32, 65)),                      # 2_080 parameters
  PositionEncoding(32),
  Dropout(0.1),
  TransformerBlock(
    MultiHeadAttention(num_heads=4, head_size=8, 32=>32)(
      denseQ = Dense(32 => 32),         # 1_056 parameters
      denseK = Dense(32 => 32),         # 1_056 parameters
      denseV = Dense(32 => 32),         # 1_056 parameters
      denseO = Dense(32 => 32),         # 1_056 parameters
      mask = Bool[1 1 … 1 1; 0 1 … 1 1; … ; 0 0 … 1 1; 0 0 … 0 1],  # 256 parameters
    ),
    Dropout(0.1),
    LayerNorm(32),                      # 64 parameters
    Dense(32 => 128, relu),             # 4_224 parameters
    Dense(128 => 32),                   # 4_128 parameters
    Dropout(0.1),
    LayerNorm(32),                      # 64 parameters
  ),
  TransformerBlock(
    MultiHeadAttention(num_heads=4, head_size=8, 32=>32)(
      denseQ = Dense(32 => 32),         # 1_056 parameters
      denseK = Dense(32 => 32),         # 1_056 parameters
      denseV = Dense(32 => 32),         # 1_056 parameters
      denseO = Dense(32 => 32),         # 1_056 parameters
      mask = Bool[1 1 … 1 1; 0 1 … 1 1; … ; 0 0 … 1 1; 0 0 … 0 1],  # 256 parameters
    ),
    Dropout(0.1),
    LayerNorm(32),                      # 64 parameters
    Dense(32 => 128, relu),             # 4_224 parameters
    Dense(128 => 32),                   # 4_128 parameters
    Dropout(0.1),
    LayerNorm(32),                      # 64 parameters
  ),
  Dense(32 => 65),                      # 2_145 parameters
)        # Total: 35 trainable arrays, 29_633 parameters,
          # plus 3 non-trainable, 32_512 parameters, summarysize 243.871 KiB.
```

Usage:
```julia
vocab_size = 65
context_size = 16
x = rand(1:vocab_size, context_size) 
y = model(x) # 65×16 Matrix{Float32}

batch_size = 3
X = rand(1:vocab_size, context_size, batch_size)
Y = model(X) # 65×16×3 Matrix{Float32}
```

Generate:
```julia
context = reshape([1], 1, 1)
context = generate(model, context; context_size=context_size, max_tokens=100)
context # 101×1 Matrix{Int64}
```

### GPU support

```julia
using CUDA, cuDNN # As of Julia 1.9, these must be loaded separately to FLux
model = gpu(model) # using the classifier above
X = gpu(X)   # 10×8 CuArray{Int64, 2, CUDA.Mem.DeviceBuffer}
Y = model(X) # 3×8 CuArray{Float32, 2, CUDA.Mem.DeviceBuffer}
```

More extensive examples were part of this repository.
They have since been moved to [github.com/LiorSinai/TransformersLite-Examples](https://github.com/LiorSinai/TransformersLite-Examples).

## Installation

Download the GitHub repository (it is not registered). Then in the Julia REPL:
```julia-repl
julia> ] # enter package mode
(@v1.x) pkg> dev path\\to\\TransformersLite.jl
julia> using Revise # for dynamic editing of code
julia> using TransformersLite
```
