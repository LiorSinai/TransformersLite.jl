apply_mask(logits, mask::Nothing) = logits

"""
  apply_mask(logits, mask)

Keep the values of `logits` where `mask` is `true`, 
else return negative infinity or a corresponding `typemin`.
"""
function apply_mask(logits, mask)
    neginf = typemin(eltype(logits))
    ifelse.(mask, logits, neginf)
end

"""
    make_causal_mask(x, dims=2)

Return a boolean square matrix `m` of the same type as `x` and of side `size(x, dims)`.
Its elements are set such that `m[i, j] == i ≤ j`.
"""
function make_causal_mask(x::AbstractArray; dims::Int=2)
  len = size(x, dims)
  mask = triu(trues_like(x, (len, len)))
  mask
end

trues_like(x::AbstractArray, sz=size(x)) = fill!(similar(x, Bool, sz), true)
