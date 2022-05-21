function multiply_test(A, B, C)
    for l in 1:size(A, 4)
        for k in 1:size(A, 3)
            if !(A[:, :, k, l]*B[:, :, k, l] ≈ C[:, :, k, l])
                return false
            end
        end
    end
    true
end

function grad_test_analytical(f, A, B, loss)
    dA = []
    BT = permutedims(B, [2,1,3,4])
    for l in 1:size(A, 4)
        ks = []
        for k in 1:size(A, 3)
            push!(ks, loss[:, :, k, l] * BT[:, :, k, l])
        end
        push!(dA, cat(ks...; dims=3))
    end
    dA = cat(dA...; dims = 4)

    dB = []
    AT = permutedims(A, [2,1,3,4])
    for l in 1:size(B, 4)
        ks = []
        for k in 1:size(B, 3)
            push!(ks, AT[:, :, k, l] * loss[:, :, k, l])
        end
        push!(dB, cat(ks...; dims=3))
    end
    dB = cat(dB...; dims = 4)

    C, pull = pullback(f, A, B)
    grads = pull(loss)

    return (grads[1] ≈ dA) && (grads[2] ≈ dB)
end