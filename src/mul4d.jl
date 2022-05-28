"""
    mul4d(A, B) -> C

4D matrix multiplication. Result has `C[:,:,k,l] == A[:,:,k,l] * B[:,:,k,l]`
"""
function mul4d(A::AbstractArray{T, 4}, B::AbstractArray{T, 4}) where T
    if (size(A, 2) != size(B, 1)) || (size(A, 3) != size(B, 3)) || (size(A, 4) != size(B, 4))
        message = "A has dimensions $(size(A)) but B has dimensions $(size(B))"
        throw(DimensionMismatch(message))
    end
    C = Array{Float64, 4}(undef, size(A, 1), size(B, 2), size(A, 3), size(A, 4))
    for l in 1:size(A, 4)
        for k in 1:size(A, 3)
            C[:, :, k, l] = A[:, :, k, l] * B[:, :, k, l]
        end
    end
    C
end

function rrule(::typeof(mul4d), A::AbstractArray{T, 4}, B::AbstractArray{T, 4}) where T
    C = mul4d(A, B)
    function mul4d_pullBack(C̄)
            Ā = @thunk mul4d(C̄, PermutedDimsArray(B, (2, 1, 3, 4)))
            B̄ = @thunk mul4d(PermutedDimsArray(A, (2, 1, 3, 4)), C̄)
        return NoTangent(), Ā, B̄
    end
    return C, mul4d_pullBack
end
