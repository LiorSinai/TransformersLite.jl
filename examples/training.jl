using Flux
using Flux: DataLoader
using Random: shuffle, MersenneTwister
using ProgressMeter
using Printf

"""
    batched_metric(f, data; g=identity)

Caculates `f(g(x), y)` for each `(x, y)` in data and returns a weighted sum by batch size.
If `f` takes the mean this will recover the full sample mean.
Reduces memory load for `f` and `g`. 
To automatically batch data, use `Flux.DataLoader`.
"""
function batched_metric(f, data; g=identity)
    result = 0.0f0
    num_observations = 0
    for (x, y) in data
        metric = f(g(x), y) 
        batch_size = count_observations(x) 
        result += metric * batch_size
        num_observations += batch_size
    end
    result / num_observations
end

count_observations(data::D) where {D<:DataLoader} = count_observations(data.data)
count_observations(data::Tuple) = count_observations(data[1]) # assume data[1] are samples and data[2] are labels
count_observations(data::AbstractArray{<:Any,N}) where {N} = size(data, N)
count_observations(data) = length(data)

"""
    split_validation(X, Y, frac=0.1; rng)

Splits a data set into a training and validation data set.
"""
function split_validation(X::AbstractMatrix, Y::AbstractMatrix, frac=0.1; rng=Random.GLOBAL_RNG)
    nsamples = size(X, 2)
    idxs = shuffle(rng, 1:nsamples)
    ntrain = nsamples - floor(Int, frac * nsamples)
    (X[:, idxs[1:ntrain]], Y[:, idxs[1:ntrain]]), (X[:, idxs[ntrain+1:end]], Y[:, idxs[ntrain+1:end]])
end

function split_validation(X::AbstractVector, Y::AbstractVector, frac=0.1; rng=Random.GLOBAL_RNG)
    nsamples = length(X)
    idxs = shuffle(rng, 1:nsamples)
    ntrain = nsamples - floor(Int, frac * nsamples)
    (X[idxs[1:ntrain]], Y[idxs[1:ntrain]]), (X[idxs[ntrain+1:end]], Y[idxs[ntrain+1:end]])
end

function train!(loss, model, train_data, opt_state, val_data; n_epochs=10)
    history = Dict(
        "train_acc"=>Float64[], 
        "train_loss"=>Float64[], 
        "val_acc"=>Float64[], 
        "val_loss"=>Float64[]
        )
    for epoch in 1:n_epochs
        progress = Progress(length(train_data); desc="epoch $epoch/$n_epochs")
        total_loss = 0.0    
        for (i, Xy) in enumerate(train_data)
            batch_loss, grads = Flux.withgradient(model) do m
                loss(m(Xy[1]), Xy[2])
            end
            Flux.update!(opt_state, model, grads[1])
            total_loss += batch_loss
            ProgressMeter.next!(progress; showvalues = [(:mean_loss, total_loss / i), (:batch_loss, batch_loss)])
        end
        update_history!(history, model, loss, train_data, val_data)
    end
    println("")
    history
end

function update_history!(history::Dict, model, loss, train_data, val_data)
    train_acc = batched_metric(accuracy, train_data; g=model)
    train_loss = batched_metric(loss, train_data; g=model)
    val_acc = batched_metric(accuracy, val_data; g=model)
    val_loss = batched_metric(loss, val_data; g=model)
    
    push!(history["train_acc"], train_acc)
    push!(history["train_loss"], train_loss)
    push!(history["val_acc"], val_acc)
    push!(history["val_loss"], val_loss)

    @printf "train_acc=%.4f%%; " train_acc * 100
    @printf "train_loss=%.4f; " train_loss
    @printf "val_acc=%.4f%%; " val_acc * 100
    @printf "val_loss=%.4f ;" val_loss
    println("")
end
