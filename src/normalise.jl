using Flux
using StatsBase

"""
    rms_norm(x; dims=1, eps=1f-5)

Applies the root mean square layer normalisation, where:

```
y[i] = scale .* x[i] / sqrt(ϵ + mean(abs2, x))
```

across each layer `i`.

Reference: https://arxiv.org/abs/1910.07467
"""
function rms_norm(x::AbstractArray; dims=1, eps=1f-5)
    x ./ sqrt.(eps .+ mean(abs2, x, dims=dims))
end

"""
    RMSNorm(size....; affine=true, eps=1f-5)

Applies the Root Mean Square layer normalisation, where:

```
y[i] = scale .* x[i] / sqrt(ϵ + mean(abs2, x))
```

across each layer `i`.

Modelled on `Flux.LayerNorm`.
    
Reference: https://arxiv.org/abs/1910.07467
"""
struct RMSNorm{D, T, N}
    diag::D
    ϵ::T
    size::NTuple{N,Int}
    affine::Bool
end

Flux.@layer RMSNorm

function RMSNorm(size::Integer...; affine::Bool=true, eps::Real=1f-5) 
    diag = affine ? Flux.Scale(size..., bias=false, init=Flux.ones32, _act = identity) : identity
    RMSNorm(diag, eps, Int.(size), affine)
end

function (a::RMSNorm)(x::AbstractArray)
    a.diag(rms_norm(x; dims=1:length(a.size), eps=a.ϵ))
end

function Base.show(io::IO, l::RMSNorm)
    print(io, "RMSNorm(", join(l.size, ", "))
    print(io, ", affine=", l.affine)
    print(io, ")")
end

