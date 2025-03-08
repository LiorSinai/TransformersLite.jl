using TransformersLite: RoPE

@testset "RoPE" begin
    @testset "apply_rope" begin
        dim = 8
        batch_size = 3
        max_seq_length = 5
        base = 10_000
        θ = 1 ./ (base .^ ((0:2:(dim - 2)) / dim))
        angles = θ * transpose(0:(max_seq_length-1))
        freqs = map(x -> reverse(sincos(x)), angles)
        freqs_complex = map(cs -> Complex(cs...), freqs)

        x = rand(dim, max_seq_length, batch_size);
        rx1 = apply_rope(x, freqs)
        rx2 = apply_rope(x, freqs_complex)
        @test size(rx1) == (8, 5, 3)
        @test rx1 == rx2
    end

    @testset "RoPE" begin
        dim = 8
        nhead = 2
        batch_size = 3
        seq_length = 5
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
end
