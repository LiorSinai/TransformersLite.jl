"""
    mul4d(A, B; transA, transB) -> C

4D matrix multiplication. Result has `C[:,:,k,l] == A[:,:,k,l] * B[:,:,k,l]`
"""
function mul4d(A::AbstractArray{T, 4}, B::AbstractArray{T, 4}; transA::Bool=false, transB::Bool=false) where T
    mB = transB ? size(B, 2) : size(B, 1)
    nB = transB ? size(B, 1) : size(B, 2)
    mA = transA ? size(A, 2) : size(A, 1)
    nA = transA ? size(A, 1) : size(A, 2)
    if (nA != mB) || (size(A, 3) != size(B, 3)) || (size(A, 4) != size(B, 4))
        message_transA = transA ? "transposed " : ""
        message_transB = transB ? "transposed " : ""
        message = "A " * message_transA * "has dimensions ($mA, $nA, $(size(A, 3)), $(size(A, 4))) " *
              "but B " * message_transB * "has dimensions ($mB, $nB, $(size(B, 3)), $(size(B, 4)))"
        throw(DimensionMismatch(message))
    end
    C = Array{Float64, 4}(undef, mA, nB, size(A, 3), size(A, 4))
    for l in 1:size(A, 4)
        for k in 1:size(A, 3)
            ar = A[:, :, k, l]
            br = B[:, :, k, l]
            if transA
                ar = transpose(ar)
            end
            if transB
                br = transpose(br)
            end
            C[:, :, k, l] = ar * br
        end
    end
    C
end

function rrule(::typeof(mul4d), A::AbstractArray{T, 4}, B::AbstractArray{T, 4}; transA::Bool=false, transB::Bool=false) where T
    C = mul4d(A, B, transA=transA, transB=transB)
    function mul4d_pullBack(C̄)
        if transA
            if transB
                Ā = @thunk mul4d(B, C̄; transA=true, transB=true)
                B̄ = @thunk mul4d(C̄, A; transA=true, transB=true)
            else
                Ā = @thunk mul4d(B, C̄; transB=true)
                B̄ = @thunk mul4d(A, C̄)
            end
        else
            if transB
                Ā = @thunk mul4d(C̄, B)
                B̄ = @thunk mul4d(C̄, A; transA=true)
            else
                Ā = @thunk mul4d(C̄, B; transB=true)
                B̄ = @thunk mul4d(A, C̄; transA=true)
            end
        end
        return NoTangent(), Ā, B̄
    end
    return C, mul4d_pullBack
end
