"""
    tail(A, n)

Return the last `n` rows of a matrix.
"""
function tail(A::AbstractMatrix, n::Int)
    n = min(n, size(A, 1))
    A[(end - n + 1):end, :]
end