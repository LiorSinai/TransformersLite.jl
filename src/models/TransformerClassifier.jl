struct TransformerClassifier{
        E<:Flux.Embedding, 
        PE<:Union{Flux.Embedding, PositionEncoding}, 
        DO<:Dropout, 
        TB<:Vector{<:TransformerBlock}, 
        A, 
        f<:FlattenLayer, 
        D<:Dense
    } 
    embedding::E
    position_encoding::PE
    dropout::DO
    blocks::TB
    agg_layer::A
    flatten_layer::f
    head::D
end

Flux.@layer TransformerClassifier

function (t::TransformerClassifier)(x::A; mask::M=nothing) where {
    A<:AbstractArray, M<:Union{Nothing, AbstractMatrix{Bool}}}
    x = t.embedding(x)              # (dm, N, B)
    N = size(x, 2)
    x = x .+ t.position_encoding(1:N) # (dm, N, B)
    x = t.dropout(x)                # (dm, N, B)
    for block in t.blocks
        x = block(x; mask=mask)     # (dm, N, B)
    end
    x = t.agg_layer(x)              # (1, N, B)
    x = t.flatten_layer(x)          # (N, B)
    x = t.head(x)                   # (n_labels, B)
    x
end
