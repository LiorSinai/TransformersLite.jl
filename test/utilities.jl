function multiply_test(A::AbstractArray{T, 4}, B::AbstractArray{T, 4}, C::AbstractArray{T, 4}) where T
    for l in 1:size(A, 4)
        for k in 1:size(A, 3)
            if !(A[:, :, k, l]*B[:, :, k, l] ≈ C[:, :, k, l])
                return false
            end
        end
    end
    true
end

function grad_test(f, A::AbstractArray{T, 4}, B::AbstractArray{T, 4}, loss::AbstractArray{T, 4}) where T
    dA, dB = batched_mul_grad(A, B, loss)
    C, pull = pullback(f, A, B)
    grads = pull(loss)
    return (grads[1] ≈ dA) && (grads[2] ≈ dB)
end

function batched_mul_grad(A::AbstractArray{T, 4}, B::AbstractArray{T, 4}, loss::AbstractArray{T, 4}) where T
    # analytical gradient
    dA = Array{Float64, 4}(undef, size(A)...)
    BT = permutedims(B, [2,1,3,4])
    for l in 1:size(A, 4)
        for k in 1:size(A, 3)
            dA[:, :, k, l] = loss[:, :, k, l] * BT[:, :, k, l]
        end
    end

    dB = Array{Float64, 4}(undef, size(B)...)
    AT = permutedims(A, [2,1,3,4])
    for l in 1:size(B, 4)
        for k in 1:size(B, 3)
            dB[:, :, k, l] = AT[:, :, k, l] * loss[:, :, k, l]
        end
    end
    dA, dB
end