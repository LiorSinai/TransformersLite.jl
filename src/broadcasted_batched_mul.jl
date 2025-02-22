# Very slow -> Julia does not optimise batched_mul well with views
# using LazyArrays, FillArrays

# function repeatview(A::AbstractArray{T, 4}; outer=nothing, inner=nothing) where T
#     isnothing(outer) && throw("require a value for outer")
#     !isnothing(inner) && throw("inner not supported")
#     dims = outer
#     f = Fill(true, dims...)
#     LazyArray(@~ A .* f)
# end

"""
    broadcasted_batched_mul(A, B) -> C

4D matrix multiplication. Broadcast over batch dims (dims ≥ 3) if they are 1.

Inefficient implementation which copies the arrays in memory.

Result has `C[:,:,...] == A[:,:,...] * B[:,:,...]`.
"""
function broadcasted_batched_mul(x::AbstractArray{T, N}, y::AbstractArray{T, N}) where {T, N}
    batch_dims_x = Tuple(size(x, idx) == 1 ? size(y, idx) : 1 for idx in 3:N)
    dims_x = (1, 1, batch_dims_x...)
    batch_dims_y = Tuple(size(y, idx) == 1 ? size(x, idx) : 1 for idx in 3:N)
    dims_y = (1, 1, batch_dims_y...)
    xb = repeat(x; outer=dims_x)
    yb = repeat(y; outer=dims_y)
    batched_mul(xb, yb)
end

"""
    broadcasted_mul4d(A, B) -> C

4D matrix multiplication. Broadcast over batch dims (dims 3 and 4) if they are 1.

It uses scalar indexing which results in very slow performance on a GPU.

Result has `C[:,:,k,l] == A[:,:,k,l] * B[:,:,k,l]`.
"""
function broadcasted_mul4d(A::AbstractArray{T, 4}, B::AbstractArray{T, 4}) where {T}
    @assert(
        all(dx==dy || dx==1 || dy==1 for (dx, dy) in zip(size(A)[3:4], size(B)[3:4])),
        "batched dimensions must match or be 1 for broadcasting"
    )
    d3 = max(size(A, 3), size(B, 3))
    d4 = max(size(A, 4), size(B, 4))
    broadcast_dimsA = size(A) .== 1
    broadcast_dimsB = size(B) .== 1
    C = Array{T, 4}(undef, size(A, 1), size(B, 2), d3, d4)
    for l in 1:d4
        lA = broadcast_dimsA[4] ? 1 : l
        lB = broadcast_dimsB[4] ? 1 : l
        for k in 1:d3
            kA = broadcast_dimsA[3] ? 1 : k
            kB = broadcast_dimsB[3] ? 1 : k
            C[:, :, k, l] = A[:, :, kA, lA] * B[:, :, kB, lB]
        end
    end
    C
end
