using Flux: unsqueeze
using TransformersLite: absorbed_attention, absorbed_attention_broadcasted
using TransformersLite: make_causal_mask
using TransformersLite: MultiHeadLatentAttentionV2, mla_naive, mla_absorb

function bench(f, args...; n=10, options...)
    f(args...; options...) # compile
    # noise makes it slower, so minimum is a robust indicator
    t = minimum(@elapsed(f(args...; options...)) for _ in 1:n)
    mem = @allocated f(args...)
    t, mem
end

n, dc, nhead, B = 64, 128, 8,32
Cq = randn(dc, n, B)
Ckv = randn(dc, n, B)
W_KQ = randn(dc, nhead, dc)
functions = [
    ("absorbed", absorbed_attention, W_KQ),
    ("broadcasted", absorbed_attention_broadcasted, unsqueeze(permutedims(W_KQ, (1, 3, 2)), dims=4)),
]
println("Attention $(size(Cq))")
for (fname, f, W) in functions
    t, mem = bench(f, Cq, Ckv, W)
    println("  ", rpad(fname, 16), round(1e3 * t, digits=2), " ms  ", round(mem / 2^20, digits=1), " MiB")
end

functions = [
    ("MLA naive", mla_naive),
    ("MLA absorb", mla_absorb),
]
nhead, dim_head, dim_rope, dim_lora, N, B = 8, 64, 64, 264, 128, 16
dim_in = nhead * dim_head
mla = MultiHeadLatentAttentionV2(
    nhead=nhead, dim_in=dim_in, dim_head=dim_head,
    dim_rope=dim_rope,
    dim_lora=dim_lora, dim_out=dim_in,
    max_seq_length=N, max_batch_size=B
)
X = randn(Float32, dim_in, N, B)
mask = make_causal_mask(ones(N, N))
println("MLA with X $(size(X))")
for (fname, f) in functions
    t, mem = bench(f, mla, X, X; mask=nothing)
    println("  ", rpad(fname, 16), round(1e3 * t, digits=2), " ms  ", round(mem / 2^20, digits=1), " MiB")
end
n = 1
mla.cache_ckv[:, 1:(N-n), 1:B] = randn(Float32, dim_lora, N - n, B)
mla.cache_kr[:, 1:(N-n), 1:B] = randn(Float32, dim_rope, N - n, B)
mask = mask = repeat([true], inner=(N, 1))
x = X[:, (N-n+1):end, :]
println("MLA with cache ($dim_in, $(N-n) + $n, $B)")
for (fname, f) in functions
    t, mem = bench(f, mla, x, x; mask=mask, start_pos=N-n+1, use_cache=true)
    println("  ", rpad(fname, 16), round(1e3 * t, digits=2), " ms  ", round(mem / 2^20, digits=1), " MiB")
end
