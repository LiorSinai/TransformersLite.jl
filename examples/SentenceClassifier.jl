struct SentenceClassifier
    base_model
    activation
    aggregate
end

Flux.@functor SentenceClassifier

function Base.show(io::IO, sc::SentenceClassifier)
    print(io, "SentenceClassifer(")
    print(io, "base_model=")
    print(io, sc.base_model)
    print(io, ", activation=")
    print(io, sc.activation)
    print(io, ", aggregate=")
    print(io, sc.aggregate)
    print(io, ")")
end

# For integration with existing code, return a single value per sample by default.
function (sc::SentenceClassifier)(x::A) where {T, A<:AbstractMatrix{T}}
    sc.aggregate(sc.activation.(sc.base_model(x)))
end

function (sc::SentenceClassifier)(v::V) where {T, A<:AbstractMatrix{T}, V<:AbstractArray{A}}
    [sc(x) for x in v]
end

# The predict_probs function is for cases where the score per sentence is required 
function predict_probs(sc::SentenceClassifier, x::A) where {T, A<:AbstractMatrix{T}}
    sc.activation.(sc.base_model(x))
end

function predict_probs(sc::SentenceClassifier, v::V) where {T, A<:AbstractMatrix{T}, V<:AbstractArray{A}}
    [predict(sc, x) for x in v]
end

function parabolic_weighted_average(x)
    # weights negative and positive values more than neutral values
    w = (x .-0.5) .^2
    sum(x .* w ./ sum(w))
end