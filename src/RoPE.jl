"""
    RoPE(dim, max_seq_length; base=10_000)
Rotary Position Embeddings (RoPE) as proposed in 
[RoFormer: Enhanced Transformer with Rotary Position Embedding (2021)][https://arxiv.org/abs/2104.09864].

Each column vector `X[:, m, ... ]` is multiplied by a rotation matrix `Rm` where 
```
     | cos(mθ₁) -sin(mθ₁)   0          …       |
Rm = | sin(mθ₁)  cos(mθ₁)   0          …       |
     |  0          0    cos(mθ₂) -sin(mθ₂)  …  |
     |  ⋮          ⋮        ⋮          ⋱       |
```

So that the output is: `[R1*X[:, 1] R2*X[:, 2] … ]`

This product can also be calculated with complex numbers:
```
             | cos(mθ₁)-sin(mθ₁)*im |    | X[1, m] + X[2, m]*im |
Rm*X[:, m] = | cos(mθ₂)-sin(mθ₂)*im | .* | X[3, m] + X[4, m]*im |
             |         ⋮            |    |          ⋮           |
```
"""
struct RoPE{T, M <: AbstractMatrix{<:Complex{T}}}
    base::Int
    dim::Int
    seq_length::Int
    freqs_complex::M
end

Flux.@layer RoPE trainable=()

RoPE(dim::Int, max_seq_length::Int; base::Int=10_000) = RoPE(Float32, dim, max_seq_length; base=base)

function RoPE(T::DataType, dim::Int, max_seq_length::Int; base::Int=10_000)
    @assert dim % 2 == 0 "Require even dim"
    θ = (1 ./ (base .^ ((0:2:(dim - 2)) / dim)))
    angles = θ * transpose(0:(max_seq_length-1))
    freqs = map(x -> reverse(T.(sincos(x))), angles)
    freqs_complex = map(cs -> Complex(cs...), freqs)
    RoPE(base, dim, max_seq_length, freqs_complex)
end

(r::RoPE)(x::AbstractArray) = apply_rope(x, r.freqs_complex[:, 1:size(x, 2)])
(r::RoPE)(x::AbstractArray, indices) = apply_rope(x, r.freqs_complex[:, indices])

"""
    apply_rope(x, freqs_complex)

Apply the complex product version of RoPE.
"""
function apply_rope(x::AbstractArray{T}, freqs_complex::AbstractMatrix{<:Complex{T}}) where T
    x_complex = reinterpret(Complex{T}, x)
    rx_complex = freqs_complex .* x_complex
    T.(reinterpret(T, rx_complex))
end

"""
    apply_rope(x, freqs)

Apply a real value product version of RoPE.

On a CPU this is comparable in speed to the complex version.
However on a GPU it much slower because of scalar indexing.
"""
function apply_rope(x::AbstractArray{T, 3}, freqs::AbstractMatrix{<:Tuple{T, T}}) where T
    Rx = similar(x)
    for k in axes(x, 3)
        for j in axes(x, 2)
            for i in 1:2:size(x, 1)
                c, s = freqs[ceil(Int, i/2), j]
                x1 = x[i, j, k]
                x2 = x[i + 1, j, k]
                Rx[i, j, k] = x1 * c - x2 * s
                Rx[i + 1, j, k] = x2 * c + x1 * s
            end
        end
    end
    Rx
end

function apply_rope(x::AbstractArray{T}, freqs::AbstractMatrix{<:Tuple{T, T}}) where T
    x3 = reshape(x, size(x, 1), size(x, 2), :)
    Rx = apply_rope(x3, freqs)
    reshape(Rx, size(x)...)
end

function Base.show(io::IO, r::RoPE)
    print(io, "RoPE(")
    print(io, "base=$(r.base)")
    print(io, ", dim=$(r.dim)")
    print(io, ", seq_length=$(r.seq_length)")
    print(io, ", freqs_complex=", Base.dims2string(size(r.freqs_complex)), " ", typeof(r.freqs_complex))
    print(io, ")")
end
