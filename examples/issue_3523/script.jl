# EnzymeAD/Enzyme.jl#3523: reverse-over-forward fails with EnzymeNoTypeError
# when the views written come from an `ntuple`.  Script as posted in the issue.

using Enzyme, LinearAlgebra
println("julia $VERSION   Enzyme $(pkgversion(Enzyme))"); flush(stdout)

const N = 54

steps(A) = ntuple(s -> view(A, :, :, s), size(A, 3))

function f(y, w, A)
    s = 0.0
    for v in steps(A)
        fill!(v, 0)
        @inbounds for k in 1:N; v[k] = Complex(y[k], 0.0); end
        @inbounds for k in 1:N; s += w[k] * abs2(v[k]); end
    end
    return s
end

y = collect(1.0:N)
w = [ifelse(iseven(k), 1.0, -1.0) for k in 1:N]
A = zeros(ComplexF64, N, 1, 2)
dy = zeros(N)

# reverse (over y) of a forward JVP (over y, seeded with w): d/dy [ wᵀ J w ]
Enzyme.autodiff(set_runtime_activity(Reverse),
    (y, w, A) -> only(Enzyme.autodiff(set_runtime_activity(Forward), f, Duplicated,
                                      Duplicated(y, w), Const(w), Duplicated(A, Enzyme.make_zero(A)))),
    Active, Duplicated(y, dy), Const(w), Duplicated(A, Enzyme.make_zero(A)))

println("|dy| = ", norm(dy))
