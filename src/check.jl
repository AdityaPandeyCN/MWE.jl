# This file is embedded verbatim in candidates and repros run under the
# correctness check, so it must depend on nothing but Enzyme.

"""
    WrongGradient(arg, maxdiff)

Thrown by [`checked_autodiff`](@ref) when Enzyme's gradient for argument
`arg` disagrees with central finite differences by at most `maxdiff`.
"""
struct WrongGradient <: Exception
    arg::Int
    maxdiff::Float64
end
Base.showerror(io::IO, e::WrongGradient) =
    print(io, "WrongGradient: Enzyme and finite differences disagree on argument $(e.arg) (max abs diff $(e.maxdiff))")

"""
    checked_autodiff(mode, f, [ret], args...; rtol = 1e-4, atol = 1e-6)

`autodiff`, followed by a comparison of the gradient Enzyme added to each
`Duplicated` floating-point array shadow against central finite differences
of the primal.  Throws [`WrongGradient`](@ref) on disagreement.

Requires reverse mode and a scalar-valued primal.  Arguments that are
`Const` are not checked, and neither is anything when the return activity
is `Const`, so a candidate that switches those off is a pass, not a
failure.  The primal must not mutate its arguments.
"""
function checked_autodiff(mode, f, rest...; rtol = 1e-4, atol = 1e-6)
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
        grad = a.dval .- before[i]                # Enzyme accumulates into the shadow
        fd = similar(a.val)
        for j in eachindex(a.val)
            h = cbrt(eps(eltype(a.val))) * max(one(eltype(a.val)), abs(inputs[i][j]))
            plus, minus = deepcopy(inputs), deepcopy(inputs)
            plus[i][j] += h
            minus[i][j] -= h
            fd[j] = (primal(plus...) - primal(minus...)) / 2h
        end
        isapprox(grad, fd; rtol, atol) || throw(WrongGradient(i, maximum(abs, grad .- fd)))
    end
    return result
end
