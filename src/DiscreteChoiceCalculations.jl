"""
$(README)

# API

$(EXPORTS)
"""
module DiscreteChoiceCalculations

export ECV_and_probabilities

using ArgCheck: @argcheck
using Distributions: Gumbel, location, scale
using DocStringExtensions: SIGNATURES, README, EXPORTS

"""
$(SIGNATURES)

For IID random variables ``εᵢ ∼ \tex{noise}``, return

1. the *expected continuation value (ECV)* `ECV = E[maxᵢ (Vᵢ + εᵢ)]``, and
2. the probabilities of ``πᵢ`` of ``X = Vᵢ + εᵢ``

as a `NamedTuple{(:ECV,:π)}`, where `π` has the same “shape” as `V`, and other
characteristics are retained to the extent possible (eg `SArray`s, `Tuple`s etc are
mapped to similar types).
"""
function ECV_and_probabilities(noise::Gumbel, V)
    @argcheck all(isfinite, V)  # for our application, all inputs are finite
    μ = location(noise)
    σ = scale(noise)
    M = maximum(V)              # subtract maximum for numerical stability
    v = exp.((V .- M) / σ)
    (ECV = σ * (log(sum(v)) + MathConstants.eulergamma) + M + μ, π = v ./ sum(v))
end

end # module
