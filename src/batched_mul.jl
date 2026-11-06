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
    broadcasted_mul4d(A, B) -> C

4D matrix multiplication. Broadcast over batch dims (dims 3 and 4) if they are 1.

Not GPU friendly (scalar indexing) and not differentiable (mutation).

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

"""
    broadcasted_batched_mul(A, B) -> C

N-D matrix multiplication. Broadcast over batch dims (dims ≥ 3) if they are 1.

GPU friendly (no scalar indexing) and differentiable (no mutation).

No copies are made if the batch dims are equal or if one input has all batch dims equal to 1.
Otherwise only the batch dims that need broadcasting are copied in memory with `repeat`.

Result has `C[:,:,...] == A[:,:,...] * B[:,:,...]`.
"""
function broadcasted_batched_mul(A::AbstractArray{T, N}, B::AbstractArray{T, N}) where {T, N}
    batch_A = size(A)[3:N]
    batch_B = size(B)[3:N]
    batch_C, outer_A, outer_B = _broadcast_batch_sizes(batch_A, batch_B)
    # batched_mul broadcasts natively if either batch size is 1, so only mixed cases need copies
    if batch_A != batch_B && !all(==(1), batch_A) && !all(==(1), batch_B)
        A = all(==(1), outer_A) ? A : repeat(A; outer=(1, 1, outer_A...))
        B = all(==(1), outer_B) ? B : repeat(B; outer=(1, 1, outer_B...))
    end
    A3 = reshape(A, size(A, 1), size(A, 2), :)
    B3 = reshape(B, size(B, 1), size(B, 2), :)
    C3 = batched_mul(A3, B3)
    reshape(C3, size(C3, 1), size(C3, 2), batch_C...)
end

"""
    _broadcast_batch_sizes(batch_A, batch_B) -> (batch_C, outer_A, outer_B)

Return the broadcasted batch dims and how many times each input must be repeated along them.
"""
function _broadcast_batch_sizes(batch_A::Tuple, batch_B::Tuple)
    @assert(
        all(dA==dB || dA==1 || dB==1 for (dA, dB) in zip(batch_A, batch_B)),
        "batched dimensions must match or be 1 for broadcasting"
    )
    batch_C = max.(batch_A, batch_B)
    outer_A = map((dA, dC) -> dA == 1 ? dC : 1, batch_A, batch_C)
    outer_B = map((dB, dC) -> dB == 1 ? dC : 1, batch_B, batch_C)
    batch_C, outer_A, outer_B
end

# integer size calculations have no derivatives
@non_differentiable _broadcast_batch_sizes(::Any, ::Any)

"""
    broadcasted_batched_mul_fast(A, B) -> C

N-D matrix multiplication. Broadcast over batch dims (dims ≥ 3) if they are 1.

GPU friendly (no scalar indexing) and never copies the inputs, but not differentiable.
See `broadcasted_batched_mul!`.

This only beats `broadcasted_batched_mul` when there are mixed batch dimensions e.g. (n, d, 1, B) x (d, n, H, 1)

Result has `C[:,:,...] == A[:,:,...] * B[:,:,...]`.
"""
function broadcasted_batched_mul_fast(A::AbstractArray{T, N}, B::AbstractArray{T, N}) where {T, N}
    batch_C, _, _ = _broadcast_batch_sizes(size(A)[3:N], size(B)[3:N])
    C = similar(A, T, size(A, 1), size(B, 2), batch_C...)
    broadcasted_batched_mul!(C, A, B)
end

"""
    broadcasted_batched_mul!(C, A, B) -> C

In-place N-D matrix multiplication. Broadcast over batch dims (dims ≥ 3) if they are 1.

No copies are made. Batch dims are merged into groups by reshaping:
- dims where both are 1 are dropped and adjacent dims of the same kind are merged.
- a leading group where `A` is 1 is folded into the columns of `B` and `C`.
- the next group is the native batch dim of `batched_mul!`.
- the remaining groups are looped over with one `batched_mul!` call per index.

Result has `C[:,:,...] == A[:,:,...] * B[:,:,...]`.
"""
function broadcasted_batched_mul!(
    C::AbstractArray{T, N}, A::AbstractArray{T, N}, B::AbstractArray{T, N}
    ) where {T, N}
    batch_A = size(A)[3:N]
    batch_B = size(B)[3:N]
    batch_C, _, _ = _broadcast_batch_sizes(batch_A, batch_B)
    m, k, n = size(A, 1), size(A, 2), size(B, 2)
    @assert size(C) == (m, n, batch_C...) "output has the wrong size"
    groups_A, groups_B = _group_batch_dims(batch_A, batch_B)
    n_fold = 1
    if !isempty(groups_A) && groups_A[1] == 1 # leading group where A is 1
        n_fold = popfirst!(groups_B)
        popfirst!(groups_A)
    end
    if isempty(groups_A)
        push!(groups_A, 1)
        push!(groups_B, 1)
    end
    groups_C = max.(groups_A, groups_B)
    A_ = reshape(A, m, k, groups_A...)
    B_ = reshape(B, k, n * n_fold, groups_B...)
    C_ = reshape(C, m, n * n_fold, groups_C...)
    for I in CartesianIndices(Tuple(groups_C[2:end]))
        idx = Tuple(I)
        idx_A = map((i, d) -> d == 1 ? 1 : i, idx, Tuple(groups_A[2:end]))
        idx_B = map((i, d) -> d == 1 ? 1 : i, idx, Tuple(groups_B[2:end]))
        batched_mul!(
            view(C_, :, :, :, idx...),
            view(A_, :, :, :, idx_A...),
            view(B_, :, :, :, idx_B...)
        )
    end
    C
end

"""
    _group_batch_dims(batch_A, batch_B) -> (groups_A, groups_B)

Drop batch dims where both are 1 and merge adjacent dims of the same kind (equal, A is 1, B is 1).
"""
function _group_batch_dims(batch_A::Tuple, batch_B::Tuple)
    groups_A = Int[]
    groups_B = Int[]
    prev_kind = :none
    for (dA, dB) in zip(batch_A, batch_B)
        (dA == 1 && dB == 1) && continue
        kind = dA == dB ? :equal : (dA == 1 ? :broadcast_A : :broadcast_B)
        if kind == prev_kind
            groups_A[end] *= dA
            groups_B[end] *= dB
        else
            push!(groups_A, dA)
            push!(groups_B, dB)
            prev_kind = kind
        end
    end
    groups_A, groups_B
end

