# Embedded verbatim in candidates and repros under the correctness check: depends only on Enzyme.

"Thrown by `checked_autodiff` when Enzyme's gradient for argument `arg` differs from finite differences by `diff`."
struct WrongGradient <: Exception
    arg::Int
    diff::Float64
end
Base.showerror(io::IO, e::WrongGradient) =
    print(io, "WrongGradient: Enzyme and finite differences disagree on argument $(e.arg) (directional derivatives differ by $(e.diff))")

"""
    checked_autodiff(mode, f, [ret], args...; rtol = 1e-4, atol = 1e-6, directions = 3, elementwise = false)

`autodiff`, then a finite-difference check of the gradient in every `Duplicated` float-array
shadow (reverse mode, scalar return, non-mutating primal).  Compares `directions` fixed random
directions, or every entry with `elementwise`.
"""
function checked_autodiff(mode, f, rest...; rtol = 1e-4, atol = 1e-6, directions = 3, elementwise = false)
    mode isa ReverseMode || error("the correctness check needs reverse mode")
    ret = !isempty(rest) && rest[1] isa Type ? rest[1] : nothing
    args = ret === nothing ? rest : rest[2:end]
    primal = f isa Annotation ? f.val : f
    inputs = map(a -> deepcopy(a.val), args)
    before = map(a -> a isa Duplicated ? deepcopy(a.dval) : nothing, args)

    result = ret === nothing ? autodiff(mode, f, args...) : autodiff(mode, f, ret, args...)
    ret === Const && return result

    for (i, a) in enumerate(args)
        a isa Duplicated && a.val isa AbstractArray{<:AbstractFloat} || continue
        x = inputs[i]
        grad = a.dval .- before[i]                # Enzyme accumulates into the shadow
        h = cbrt(eps(eltype(x))) * max(one(eltype(x)), maximum(abs, x))
        for v in (elementwise ? unit_vectors(x) : pseudo_random_directions(x, directions))
            at(xi) = primal(Base.setindex(deepcopy(inputs), xi, i)...)
            fd = (at(x .+ h .* v) - at(x .- h .* v)) / 2h
            got = sum(grad .* v)
            isapprox(got, fd; rtol, atol) || throw(WrongGradient(i, abs(got - fd)))
        end
    end
    return result
end

"Fixed unit vectors that look random, so runs are reproducible without a RNG."
function pseudo_random_directions(x::AbstractArray{T}, n) where {T}
    map(1:n) do k
        v = [sin(12.9898 * k * j + 78.233) for j in 1:length(x)]     # Float64: Float32 loses the phase for large j
        reshape(T.(v ./ sqrt(sum(abs2, v))), size(x))
    end
end

unit_vectors(x::AbstractArray{T}) where {T} =
    (setindex!(zero(x), one(T), j) for j in eachindex(x))
