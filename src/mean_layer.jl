"""
    MeanLayer(x)

Reduce to the mean along dims=1 and return in an array of nlayers × nbatch.
For an array of size 3, equivalent to Flux.flatten(Flux.GlobalMeanPool()(x)).

Compare Flux.GlobalMeanPool()
"""
struct MeanLayer end

Flux.@functor MeanLayer

function (m::MeanLayer)(x::AbstractArray{T, 3}) where T
  means = mean(x, dims = 1)
  reshape(means, :, size(x, 3))
end

function (m::MeanLayer)(x::AbstractArray{T, 2}) where T
    means = mean(x, dims = 1)
    reshape(means, :, 1)
end

function Base.show(io::IO, g::MeanLayer)
  print(io, "MeanLayer()")
end
