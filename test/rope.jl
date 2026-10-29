using Test
using TransformersLite: RoPE, apply_rope
using Flux: gradient
using LinearAlgebra: dot

@testset "RoPE" begin
    @testset "apply_rope" begin
        dim, max_seq_length, batch_size = 8, 5, 3
        freqs_complex = RoPE(Float64, dim, max_seq_length).freqs_complex
        freqs = map(c -> (real(c), imag(c)), freqs_complex) # (cos, sin)

        x = rand(dim, max_seq_length, batch_size);
        rx1 = apply_rope(x, freqs)
        rx2 = apply_rope(x, freqs_complex)

        # simpler non-differentiable version of apply_rope
        x_complex = reinterpret(Complex{Float64}, x)
        rx_complex = freqs_complex .* x_complex
        rx3 = Float64.(reinterpret(Float64, rx_complex))

        @test size(rx1) == (8, 5, 3)
        @test rx1 ≈ rx2
        @test rx1 ≈ rx3
    end

    @testset "RoPE" begin
        dim, seq_length, nhead, batch_size = 8, 5, 2, 3
        r = RoPE(dim, seq_length)

        x = rand(Float32, dim, seq_length, batch_size);
        rx = r(x)
        @test size(rx) ==  (8, 5, 3)
        x = rand(Float32, dim, seq_length, nhead, batch_size);
        rx = r(x)
        @test size(rx) ==  (8, 5, 2, 3)

        x = rand(Float32, dim, 3, nhead, batch_size);
        rx = r(x, 3:5)
        @test size(rx) ==  (8, 3, 2, 3)
    end

    @testset "properties" begin
        dim, seq_length, nhead, batch_size = 8, 10, 2, 3
        r = RoPE(dim, seq_length)

        # relative position: dot(R_m*q, R_n*k) only depends on the distance m - n
        q = rand(Float32, dim, 1)
        k = rand(Float32, dim, 1)
        score(m, n) = dot(r(q, m:m), r(k, n:n))
        @test score(2, 1) ≈ score(5, 4)
        @test score(7, 3) ≈ score(10, 6)
        @test !(score(2, 1) ≈ score(5, 1))

        # rotations do not change the length of each column
        x = rand(Float32, dim, seq_length, nhead, batch_size)
        rx = r(x)
        @test sum(abs2, rx; dims=1) ≈ sum(abs2, x; dims=1)

        # Float32 stays Float32
        @test eltype(rx) == Float32
    end

    @testset "gradient" begin
        dim, seq_length, nhead, batch_size = 8, 5, 2, 3
        r = RoPE(dim, seq_length)
        x = rand(Float32, dim, seq_length, nhead, batch_size)

        # rotation preserves the norm, so d/dx sum(abs2, R*x) == 2x
        dx = gradient(x -> sum(abs2, r(x)), x)[1]
        @test size(dx) == size(x)
        @test dx ≈ 2 .* x

        # for a weighted sum the gradient is transpose(R) * W, so rotating it gives back W
        W = rand(Float32, size(x)...)
        dx = gradient(x -> sum(W .* r(x)), x)[1]
        @test r(dx) ≈ W
    end

    @testset "errors" begin
        dim, seq_length = 8, 5
        r = RoPE(dim, seq_length)

        # dim must be even to pair up rows
        @test_throws AssertionError RoPE(7, seq_length)
        # sequence longer than the precomputed frequencies
        @test_throws BoundsError r(rand(Float32, dim, seq_length + 1, 1))
        # element types of the input and the frequencies must match
        @test_throws MethodError r(rand(Float64, dim, seq_length, 1))
    end
end
