using Enzyme
using LinearAlgebra

# Unrelated helper and statements surround the actual bug: a branch that
# picks either the active `x` or the constant `y`, which Enzyme cannot
# resolve statically (EnzymeRuntimeActivityError).
scale(a) = a .* 2

function pick(x, y, w)
    a = scale(x)
    b = norm(a)
    c = w' * w
    z = x[1] > 0 ? x : y
    s = sum(z) * w[1]
    return s + b + c[1]
end

x = rand(1000); dx = zero(x)
y = rand(1000)
w = rand(8, 8); dw = zero(w)
unused = rand(100)

autodiff(ReverseWithPrimal, pick, Active, Duplicated(x, dx), Const(y), Duplicated(w, dw))
