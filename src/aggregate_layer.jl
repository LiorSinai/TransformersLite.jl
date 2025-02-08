"""
    MeanLayer(x)

Reduce to the mean along dims=1.

Compare `Flux.GlobalMeanPool()`.
"""
struct MeanLayer end

function (m::MeanLayer)(x::AbstractArray)
  mean(x, dims = 1)
end

"""
  FlattenLayer(x)

Return a matrix of nlayers × nbatch.
"""
struct FlattenLayer end

function (f::FlattenLayer)(x::AbstractArray{T, 3}) where T
  reshape(x, :, size(x, 3)) # same as Flux.flatten
end

function (f::FlattenLayer)(x::AbstractArray{T, 2}) where T
    reshape(x, :, 1) # returns a column vector
end
