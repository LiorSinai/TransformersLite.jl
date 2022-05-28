using Flux, StatsBase
using Random

function accuracy(ŷ, y) 
    if size(y, 1) == 1 || size(y, 2) == 1
        mean((σ.(ŷ) .> 0.5) .== y)
    else
        mean(Flux.onecold(ŷ) .== Flux.onecold(y))
    end
end

"""
    batch_matrix(X, Y, col::Int; batch_size=128)

Return a batch from a matrix dataset with indices `X[:, col:batch_size]` and `Y[:, col:batch_size]`. 
"""
function batch_matrix(X::AbstractMatrix, Y::AbstractMatrix, col::Int; batch_size=128)
    maxcol = min(col + batch_size - 1, size(X, 2))
    X[:, col:maxcol], Y[:, col:maxcol]
end;

"""
    batched_metric(f, X, Y; batch_size=1024, g=identity)

Caculates `f(g(X), Y)` except in bactches and returns a weighted sum by batch size (all equal except for the final batch).
If `f` takes the mean this will recover the full sample mean.
Reduces memory load for f. 
"""
function batched_metric(f, X::AbstractMatrix, Y::AbstractMatrix; batch_size=128, g=identity)
    result = 0.0
    nsamples = size(X, 2)
    for j in 1:batch_size:nsamples
        batch_ = batch_matrix(X, Y, j, batch_size=batch_size)
        metric = f(g(batch_[1]), batch_[2]) 
        result += metric * size(batch_[1], 2) / nsamples
    end
    result
end

"""
    split_validation(X, Y, frac=0.1; rng=Random.GLOBAL_RNG)

Split a matrix data set into a training and validation set.
"""
function split_validation(X::AbstractMatrix, Y::AbstractMatrix, frac=0.1; rng=Random.GLOBAL_RNG)
    nsamples = size(X, 2)
    idxs = shuffle(rng, 1:nsamples)
    ntrain = nsamples - floor(Int, frac * nsamples)
    (X[:, idxs[1:ntrain]], Y[:, idxs[1:ntrain]]), (X[:, idxs[ntrain+1:end]], Y[:, idxs[ntrain+1:end]])
end

function train!(loss, ps, data, opt, val_data; n_epochs=100, batch_size=128)
    history = Dict(
        "train_acc"=>Float64[], 
        "train_loss"=>Float64[], 
        "val_acc"=>Float64[], 
        "val_loss"=>Float64[]
        )
    nsamples = size(train_data[1], 2)
    update_incr = nsamples/batch_size/100
    for e in 1:n_epochs
        print("\n$e ")
        update_val = 0.0
        idxs = randperm(nsamples)
        X, Y = data[1][:, idxs], data[2][:, idxs]
        ps = Flux.Params(ps)
        for j in 1:batch_size:nsamples
            batch_ = batch_matrix(X, Y, j, batch_size=batch_size)
            gs = gradient(ps) do
                loss(batch_...)
            end
            Flux.update!(opt, ps, gs)
            if (j/batch_size) > update_val
                print('.')
                update_val += update_incr
            end
        end
        update_history!(history, model, loss, data, val_data)
    end
    history
end

function update_history!(history::Dict, model, loss, train_data, val_data)
    train_acc = batched_metric(accuracy, train_data[1], train_data[2]; g=model)
    train_loss = batched_metric(loss, train_data[1], train_data[2])
    val_acc = batched_metric(accuracy, val_data[1], val_data[2]; g=model)
    val_loss = batched_metric(loss, val_data[1], val_data[2])
    
    push!(history["train_acc"], train_acc)
    push!(history["train_loss"], train_loss)
    push!(history["val_acc"], val_acc)
    push!(history["val_loss"], val_loss)

    @printf "\ntrain_acc=%.4f%% ; " train_acc * 100
    @printf "train_loss=%.4f ; " train_loss
    @printf "val_acc=%.4f%% ; " val_acc * 100
    @printf "val_loss=%.4f ; " val_loss
end

