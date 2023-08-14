using Flux
using Flux: DataLoader
using Random: shuffle, MersenneTwister
using ProgressMeter
using Printf

"""
    batched_metric(g, f, data)

Caculates `f(g(x), y)` for each `(x, y)` in data and returns a weighted sum by batch size.
If `f` takes the mean this will recover the full sample mean.
Reduces memory load for `f` and `g`. 
To automatically batch data, use `Flux.DataLoader`.
"""
function batched_metric(g, f, data)
    result = 0.0f0
    num_observations = 0
    for (x, y) in data
        val = f(g(x), y) 
        batch_size = count_observations(x) 
        result += val * batch_size
        num_observations += batch_size
    end
    result / num_observations
end

count_observations(data::D) where {D<:DataLoader} = count_observations(data.data)
count_observations(data::Tuple) = count_observations(data[1]) # assume data[1] are samples and data[2] are labels
count_observations(data::AbstractArray{<:Any,N}) where {N} = size(data, N)
count_observations(data) = length(data)

"""
    split_validation(rng, X, Y; frac=0.1)

Splits a data set into a training and validation data set.
"""
function split_validation(rng::AbstractRNG, data::AbstractArray, labels::AbstractVecOrMat; frac::Float64=0.1)
    nsamples = size(data)[end]
    idxs = randperm(rng, nsamples)
    ntrain = nsamples - floor(Int, frac * nsamples)
    inds_start = ntuple(Returns(:), ndims(data) - 1)
    ## train data
    idxs_train = idxs[1:ntrain]
    train_data = data[inds_start..., idxs_train]
    train_labels = ndims(labels) == 2 ? labels[:, idxs_train] : labels[idxs_train]
    ## validation data
    idxs_val = idxs[(ntrain + 1):end]
    val_data = data[inds_start..., idxs_val]
    val_labels = ndims(labels) == 2 ? labels[:, idxs_val] : labels[idxs_val]
    (train_data, train_labels), (val_data, val_labels)
end

function train!(loss, model, train_data, opt_state, val_data; num_epochs=10)
    history = Dict(
        "train_acc" => Float64[], 
        "train_loss" => Float64[], 
        "val_acc" => Float64[], 
        "val_loss" => Float64[],
        "mean_batch_loss" => Float64[],
        )
    for epoch in 1:num_epochs
        print(stderr, "")
        progress = Progress(length(train_data); desc="epoch $epoch/$n_epochs")
        total_loss = 0.0    
        for (i, Xy) in enumerate(train_data)
            batch_loss, grads = Flux.withgradient(model) do m
                loss(m(Xy[1]), Xy[2])
            end
            Flux.update!(opt_state, model, grads[1])
            total_loss += batch_loss
            ProgressMeter.next!(
                progress; showvalues = 
                [(:mean_loss, total_loss / i), (:batch_loss, batch_loss)]
            )
        end
        mean_batch_loss = total_loss / length(train_data)
        push!(history["mean_batch_loss"], mean_batch_loss)
        update_history!(history, model, loss, train_data, val_data)
    end
    println("")
    history
end

function update_history!(history::Dict, model, loss, train_data, val_data)
    train_acc = batched_metric(model, accuracy, train_data)
    train_loss = batched_metric(model, loss, train_data)
    val_acc = batched_metric(model, accuracy, val_data)
    val_loss = batched_metric(model, loss, val_data)
    
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
