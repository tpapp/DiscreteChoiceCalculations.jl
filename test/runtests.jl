using DiscreteChoiceCalculations
using Test, Distributions, StaticArrays, Sobol

"""
Test function for `noisy_continuation_and_probabilities`, using quasi-random simulation
to calculate the same quantities as `DiscreteChoiceCalculations.ECV_and_probabilities`.
"""
function simulate_noise(noise, V; N = 10000)
    n = length(V)
    s = SobolSeq(n)
    x = zeros(n)
    ΣV = zero(eltype(V))
    cπ = zeros(Int, size(V)...)
    y = similar(V, Float64)
    for _ in 1:N
        next!(s, x)
        for j in 1:n
            y[j] = V[j] + quantile(noise, x[j])
        end
        V̂, i = findmax(y)
        ΣV += V̂
        cπ[i] += 1
    end
    (ECV = ΣV / N, π = cπ ./ N)
end

@testset "Gumbel noise" begin
    for μ in [0.0, 0.5, 1.0, -2.0]
        for σ in [0.5, 0.7, 1.0, 1.5, 2.0]
            noise = Gumbel(μ, σ)
            V = SVector(1.0, 2.0, 3.0)
            z1 = simulate_noise(noise, V)
            z2 = @inferred ECV_and_probabilities(noise, V)
            @test z1.ECV ≈ z2.ECV atol = 1e-2
            @test maximum(abs, z1.π .- z2.π) ≤ 1e-2
            @test isa(z2.π, SVector)
        end
    end
end

# write tests here

## NOTE add JET to the test environment, then uncomment
# using JET
# @testset "static analysis with JET.jl" begin
#     @test isempty(JET.get_reports(report_package(DiscreteChoiceCalculations, target_modules=(DiscreteChoiceCalculations,))))
# end

## NOTE add Aqua to the test environment, then uncomment
# @testset "QA with Aqua" begin
#     import Aqua
#     Aqua.test_all(DiscreteChoiceCalculations; ambiguities = false)
#     # testing separately, cf https://github.com/JuliaTesting/Aqua.jl/issues/77
#     Aqua.test_ambiguities(DiscreteChoiceCalculations)
# end
