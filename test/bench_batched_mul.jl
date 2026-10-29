using Flux: batched_mul
using TransformersLite: broadcasted_batched_mul, broadcasted_mul4d, broadcasted_batched_mul_fast

function bench(f, args...; n=10)
    f(args...) # compile
    # noise makes it slower, so minimum is a robust indicator
    t = minimum(@elapsed(f(args...)) for _ in 1:n)
    mem = @allocated f(args...)
    t, mem
end

cases = [
    ("equal", (512, 64, 8, 16), (64, 512, 8, 16)),
    ("B single matrix", (512, 64, 8, 16), (64, 512, 1, 1)),
    ("A is 1, folded", (512, 64, 1, 16), (64, 512, 8, 16)),
    ("B is 1, loops", (512, 64, 8, 16), (64, 512, 1, 16)),
    ("mixed both", (512, 64, 8, 1), (64, 512, 1, 16)),
]

functions = [
    ("scalar", broadcasted_mul4d),
    ("differentiable", broadcasted_batched_mul),
    ("fast", broadcasted_batched_mul_fast),
]

for (name, sA, sB) in cases
    A = randn(Float32, sA...)
    B = randn(Float32, sB...)
    println(name, ": ", sA, " x ", sB)
    for (fname, f) in functions
        t, mem = bench(f, A, B)
        println("  ", rpad(fname, 16), round(1e3 * t, digits=2), " ms  ", round(mem / 2^20, digits=1), " MiB")
    end
end

