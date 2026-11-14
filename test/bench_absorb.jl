using Flux: Dense, unsqueeze
using TransformersLite
using TransformersLite: absorbed_attention, absorbed_attention_broadcasted
using TransformersLite: make_causal_mask, _absorb_WUK_WUQ

function bench(f, args...; n=10, options...)
    f(args...; options...) # compile
    # noise makes it slower, so minimum is a robust indicator
    t = minimum(@elapsed(f(args...; options...)) for _ in 1:n)
    mem = @allocated f(args...; options...)
    t, mem
end

function print_bench(name, t, mem)
    println("  ", rpad(name, 16), round(1e3 * t, digits=2), " ms  ", round(mem / 2^20, digits=1), " MiB")
end

## absorbed_attention
n, dh, nhead, B = 64, 32, 16, 16
for dc in (2dh, 4dh, 8dh, 16dh) # at dc=2nhead*dh the cache is the same size
    Cq = randn(dc, n, B)
    Ckv = randn(dc, n, B)
    denseUQ = Dense(randn(dh*nhead, dc), false)
    denseUK = Dense(randn(dh*nhead, dc), false)
    W_KQ = _absorb_WUK_WUQ(nhead, denseUK.weight, denseUQ.weight) # (dc*nhead, dc)
    denseKQ = Dense(W_KQ, false)
    W_KQ4 = unsqueeze(permutedims(reshape(W_KQ, dc, nhead, dc), (1, 3, 2)), dims=4) # (dc, dc, nhead, 1)

    A1, S1 = absorbed_attention(Cq, Ckv, denseKQ)
    A2, S2 = absorbed_attention(nhead, Cq, Ckv, denseUQ, denseUK)
    A3, S3 = absorbed_attention_broadcasted(Cq, Ckv, W_KQ4)
    @assert S1 ≈ S2 && A1 ≈ A2
    @assert S1 ≈ S3 && permutedims(A1, (1, 3, 2, 4)) ≈ A3

    println("Attention $(size(Cq)) dh*H/dc=$(dh*nhead/dc)")
    print_bench("absorbed", bench(absorbed_attention, Cq, Ckv, denseKQ)...)
    print_bench("factors", bench(absorbed_attention, nhead, Cq, Ckv, denseUQ, denseUK)...)
    print_bench("broadcasted", bench(absorbed_attention_broadcasted, Cq, Ckv, W_KQ4)...)
end

## MLA naive vs absorbed
nhead, dim_head, dim_rope, dim_lora, N, B = 8, 64, 64, 264, 128, 16
dim_in = nhead * dim_head
mla2 = MultiHeadLatentAttentionV2(
    nhead=nhead, dim_in=dim_in, dim_head=dim_head,
    dim_rope=dim_rope,
    dim_lora=dim_lora, dim_out=dim_in,
    max_seq_length=N, max_batch_size=B
)
mla3 = absorb_weights(mla2) # shares the layers and caches of mla2
layers = [("MLA naive", mla2), ("MLA absorb", mla3)]

X = randn(Float32, dim_in, N, B)
mask = make_causal_mask(ones(N, N))
A2, S2 = mla2(X, X; mask=mask)
A3, S3 = mla3(X, X; mask=mask)
@assert A2 ≈ A3 && S2 ≈ S3

println("MLA with X $(size(X))")
for (name, mla) in layers
    print_bench(name, bench(mla, X, X; mask=mask)...)
end

## MLA with cache: n new tokens after N - n cached tokens
n = 1
start_pos = N - n + 1
# the cache is shared so this fills it for both layers
mla2.cache_ckv[:, 1:(N-n), 1:B] = randn(Float32, dim_lora, N - n, B)
mla2.cache_kr[:, 1:(N-n), 1:B] = randn(Float32, dim_rope, N - n, B)
x = X[:, start_pos:end, :]
mask = make_causal_mask(ones(N, N))[:, start_pos:end] # (N, n)
options = (mask=mask, start_pos=start_pos, use_cache=true)
A2, S2 = mla2(x, x; options...)
A3, S3 = mla3(x, x; options...)
@assert A2 ≈ A3 && S2 ≈ S3

println("MLA with cache ($dim_in, $(N-n) + $n, $B)")
for (name, mla) in layers
    print_bench(name, bench(mla, x, x; options...)...)
end
