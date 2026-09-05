"Shrink a failing `Enzyme.autodiff` call to a minimal reproducer; see [`minify`](@ref)."
module Shrink

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
