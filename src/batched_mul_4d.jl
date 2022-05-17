
"""
    batched_mul(A, B; transA, transB) -> C

Batched matrix multiplication in 4D. Result has `C[:,:,k,l] == A[:,:,k,l] * B[:,:,k,l]`
"""
function batched_mul(A::AbstractArray{T, 4}, B::AbstractArray{T, 4}; transA=false, transB=false) where {T}
    if (size(B, 4) != size(A, 4)) 
        throw(DimensionMismatch("4th dimension mismatch: $(size(A)) != $(size(B))"))
    end
    new_A = reshape(A, size(A, 1), size(A, 2), :)
    new_B = reshape(B, size(B, 1), size(B, 2), :)
    if transA
        new_A = batched_transpose(new_A)
    end
    if transB
        new_B = batched_transpose(new_B)
    end
    C = batched_mul(new_A, new_B)
    new_C = reshape(C, (size(C, 1), size(C, 2), size(A, 3), size(A, 4)))
    new_C
end