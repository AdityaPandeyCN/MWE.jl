using Enzyme
import Enzyme.EnzymeRules

# A custom rule that is wrong: `mysin` is declared inactive, so Enzyme
# skips its derivative and the gradient of `loss` is off.  Surrounding
# code is correct and should be stripped away.
mysin(x) = sin(x)
EnzymeRules.inactive(::typeof(mysin), args...) = nothing

function loss(x, p)
    a = x .* p
    b = sum(mysin.(a))
    c = sum(abs2, x)
    d = c / 3
    return b + d
end

x = rand(50); dx = zero(x)
p = 2.0

autodiff(Reverse, loss, Active, Duplicated(x, dx), Const(p))
