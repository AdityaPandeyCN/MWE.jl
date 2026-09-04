using Enzyme
include("helper.jl")

"""
    f(x, p)

Entry function with three assignments and a return.
"""
function f(x, p)
    a = g(x)
    b = sum(a) * p
    c = b + 1
    return c
end

x = [1.0, 2.0, 3.0, 4.0]
r = autodiff(Reverse, f, Active, Duplicated(x, zero(x)), Const(3.0))
