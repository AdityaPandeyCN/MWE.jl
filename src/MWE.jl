"""
    MWE

Automatically shrink a failing `Enzyme.autodiff` call to a minimal working
example, following the design of PyTorch's FX minifier:

1. [`minify`](@ref) runs the script with every `autodiff` call wrapped in a
   capture hook.  The first call that throws gives the failing invocation.
2. One instrumented run of the primal records the value of every
   statement in the user's functions.
3. Delta debugging shrinks the program while an oracle, running each
   variant in an isolated worker, confirms the failure class is unchanged.
   Statements are not merely deleted: they are replaced by inputs holding
   the recorded values, so downstream code stays type-correct.
4. The smallest program that still fails is written out as `repro.jl`.
"""
module MWE

using Enzyme
using Distributed
using Serialization

include("check.jl")
"Source of `check.jl`, embedded in candidates and repros run under the correctness check."
const CHECK_SOURCE = read(joinpath(@__DIR__, "check.jl"), String)

include("classify.jl")
include("values.jl")
include("program.jl")
include("workers.jl")
include("record.jl")
include("passes.jl")
include("minify.jl")

export minify

end
