# The program is the reducer's "graph": the script's top-level expressions
# plus the captured call.  Nodes are top-level definitions and, inside
# function definitions, the statements of the body.

"""
    Arg

One argument of the captured call: the annotation `kind` (`:Const`,
`:Active`, `:Duplicated`, `:DuplicatedNoNeed`, `:BatchDuplicated`,
`:BatchDuplicatedNoNeed`), the variable `name` used for it in generated
source, the primal `val`, and its `shadows` (none, one, or one per batch
member).
"""
struct Arg
    kind::Symbol
    name::Symbol
    val::Value
    shadows::Vector{Value}
end

"""
    Call

The captured `autodiff(mode, f, ret, args...)`.  `head` is how `autodiff`
was spelled (`autodiff` or `Enzyme.autodiff`), `f` the source expression of
the function argument (a name, or e.g. `Const(f)`), `ret` the return
activity type or `nothing` when it was left for Enzyme to infer.
"""
struct Call
    head::Union{Symbol,Expr}
    mode::Enzyme.Mode
    f::Union{Symbol,Expr}
    ret::Union{Nothing,Type}
    args::Vector{Arg}
end

"""
    Program

What is being reduced.

- `defs`: the script's top-level expressions in order, with `include`s
  inlined, docstrings stripped and function definitions normalised to
  `function ... end` form with a block body.
- `consts`: placeholder constants introduced for statements of callees.
- `call`: the captured invocation.
- `values`: the recorded value of each assignment statement, keyed by the
  statement `Expr` object itself.  Passes rebuild bodies from the same
  statement objects, so identity survives every edit.
"""
struct Program
    defs::Vector{Any}
    consts::Vector{Pair{Symbol,Value}}
    call::Call
    values::IdDict{Any,Value}
end

with(p::Program; defs = p.defs, consts = p.consts, call = p.call) = Program(defs, consts, call, p.values)
with(c::Call; mode = c.mode, ret = c.ret, args = c.args) = Call(c.head, mode, c.f, ret, args)

# ---- parsing the script -----------------------------------------------------

"""
    parse_script(file) -> Vector{Any}

Top-level expressions of `file`, with `include("...")` calls replaced by
the included file's expressions, docstrings stripped, line numbers removed
and function definitions normalised (see [`normalize`](@ref)).
"""
function parse_script(file::AbstractString)
    defs = Any[]
    collect_defs!(defs, Meta.parseall(read(file, String); filename = file), file)
    return defs
end

# `a = 1; b = 2` on one line parses as a nested `:toplevel`; flatten those.
function collect_defs!(defs, e, file)
    e isa LineNumberNode && return
    e = strip_doc(e)
    if e isa Expr && e.head == :toplevel
        foreach(x -> collect_defs!(defs, x, file), e.args)
    elseif e isa Expr && e.head == :call && e.args[1] == :include &&
           length(e.args) == 2 && e.args[2] isa AbstractString
        append!(defs, parse_script(joinpath(dirname(file), e.args[2])))
    else
        push!(defs, normalize(Base.remove_linenums!(e)))
    end
end

strip_doc(e) = e isa Expr && e.head == :macrocall && e.args[1] == GlobalRef(Core, Symbol("@doc")) ? e.args[end] : e

"A function signature: `f(x)`, `f(x) where T`, `f(x)::T`."
issig(x) = x isa Expr && (x.head == :call || (x.head in (:where, :(::)) && issig(x.args[1])))

"""
    normalize(e)

Rewrite `f(x) = body` as `function f(x) body end`, and make every function
body a block, so passes only ever see one form.
"""
function normalize(e)
    e isa Expr || return e
    if e.head == :(=) && issig(e.args[1])
        return Expr(:function, e.args[1], asblock(e.args[2]))
    elseif e.head == :function && length(e.args) == 2
        return Expr(:function, e.args[1], asblock(e.args[2]))
    end
    return e
end
asblock(b) = b isa Expr && b.head == :block ? b : Expr(:block, b)

# ---- function definitions ---------------------------------------------------

isfdef(e) = e isa Expr && e.head == :function && length(e.args) == 2

"The `f(args...)` part of a signature, under any `where`/`::` wrappers."
callpart(sig) = sig.head == :call ? sig : callpart(sig.args[1])

"Rebuild a signature with `g` applied to its `f(args...)` part."
mapcall(g, sig) = sig.head == :call ? g(sig) : Expr(sig.head, mapcall(g, sig.args[1]), sig.args[2:end]...)

"Name of a function definition, or `nothing` if it is not a plain name (e.g. `Base.show`)."
function fname(d)
    n = callpart(d.args[1]).args[1]
    n isa Symbol ? n : nothing
end

fparams(d) = callpart(d.args[1]).args[2:end]
stmts(d) = d.args[2].args

with_body(d, body::Vector) = Expr(:function, d.args[1], Expr(:block, body...))
with_params(d, params::Vector) = Expr(:function, mapcall(c -> Expr(:call, c.args[1], params...), d.args[1]), d.args[2])

"Name bound by a parameter: `x`, `x::T`, `x = 1`, `x...`; `nothing` for anything else."
param_name(p::Symbol) = p
param_name(p::Expr) = p.head in (:(::), :kw, :(...)) && length(p.args) >= 1 ? param_name(p.args[1]) : nothing
param_name(p) = nothing

"An assignment to a plain local, `y = rhs`: the statements that can become placeholders."
is_assignment(s) = s isa Expr && s.head == :(=) && s.args[1] isa Symbol
assigned(s) = s.args[1]

"Whether symbol `sym` occurs anywhere in `e`."
uses(sym::Symbol, e) = e === sym || (e isa Expr && any(a -> uses(sym, a), e.args))

# ---- the entry function ---------------------------------------------------

"Unwrap `Const(f)` / `Duplicated(f, df)` to the function expression."
unwrap_f(f::Symbol) = f
unwrap_f(f::Expr) = f.head == :call && f.args[1] in (:Const, :Active, :Duplicated, :DuplicatedNoNeed, :BatchDuplicated) ? f.args[2] : f

"""
    entry(p::Program) -> Union{Int, Nothing}

Index in `p.defs` of the differentiated function's definition, when it is a
named function with exactly one definition in the script.
"""
function entry(p::Program)
    name = unwrap_f(p.call.f)
    name isa Symbol || return nothing
    js = findall(d -> isfdef(d) && fname(d) == name, p.defs)
    return length(js) == 1 ? only(js) : nothing
end

# ---- source rendering -------------------------------------------------------

"`\"\"` or `\"Enzyme.\"`, depending on how the script spelled `autodiff`."
prefix(c::Call) = c.head == :autodiff ? "" : "Enzyme."

"""
    mode_source(m::Enzyme.Mode, prefix) -> String

Source text that evaluates to `m`.  The common modes are spelled the way a
user would write them; anything exotic (holomorphic, custom ABI, split
modes) falls back to `repr`, which is valid but unreadable.
"""
function mode_source(m::Enzyme.Mode, prefix::String = "")
    stripped = Enzyme.EnzymeCore.clear_strong_zero(Enzyme.EnzymeCore.clear_runtime_activity(m))
    base = stripped == Reverse           ? "Reverse" :
           stripped == ReverseWithPrimal ? "ReverseWithPrimal" :
           stripped == Forward           ? "Forward" :
           stripped == ForwardWithPrimal ? "ForwardWithPrimal" : nothing
    base === nothing && return repr(m)
    base = prefix * base
    Enzyme.EnzymeCore.runtime_activity(m) && (base = "$(prefix)set_runtime_activity($base)")
    Enzyme.EnzymeCore.strong_zero(m)      && (base = "$(prefix)set_strong_zero($base)")
    return base
end

shadow_name(a::Arg, k) = length(a.shadows) == 1 ? Symbol(:d, a.name) : Symbol(:d, a.name, :_, k)

function arg_source(a::Arg, prefix::String)
    kind = prefix * string(a.kind)
    isempty(a.shadows) && return "$kind($(a.name))"
    length(a.shadows) == 1 && return "$kind($(a.name), $(shadow_name(a, 1)))"
    return "$kind($(a.name), ($(join((shadow_name(a, k) for k in eachindex(a.shadows)), ", ")),))"
end

"The `autodiff(...)` line; `checked_autodiff(...)` under the correctness check."
function call_source(c::Call, check::Symbol = :error)
    pre = prefix(c)
    parts = [mode_source(c.mode, pre), string(c.f)]
    c.ret === nothing || push!(parts, pre * string(nameof(c.ret)))
    append!(parts, arg_source.(c.args, pre))
    head = check == :correctness ? "checked_autodiff" : string(c.head)
    return "$head(" * join(parts, ", ") * ")"
end

"The plain `f(x, y)` call, used for the recording run."
primal_source(c::Call) = "$(unwrap_f(c.f))(" * join(string.(getfield.(c.args, :name)), ", ") * ")"

"""
    render(p::Program, store_path::String, check = :error) -> (setup::String, call::String, store::Store)

Source for `p`: `setup` defines everything and binds the arguments, `call`
is the `autodiff` line.  Numeric values too large for literals are placed in
`store` and loaded by `setup` from `store_path` (a source expression) when
there are any.  Under the correctness `check`, `setup` also embeds
[`checked_autodiff`](@ref) and `call` uses it.
"""
function render(p::Program, store_path::String, check::Symbol = :error)
    store = Store("STORE")
    body = IOBuffer()
    for (name, v) in p.consts
        println(body, "const ", name, " = ", render(v, string(name), store))
    end
    for a in p.call.args
        println(body, a.name, " = ", render(a.val, string(a.name), store))
        for (k, s) in enumerate(a.shadows)
            sname = string(shadow_name(a, k))
            println(body, sname, " = ", iszero_shadow(s) ? "zero($(a.name))" : render(s, sname, store))
        end
    end

    setup = IOBuffer()
    for d in p.defs
        println(setup, d)
        println(setup)
    end
    if !isempty(store.entries)
        println(setup, "import Serialization")
        println(setup, "const STORE = Serialization.deserialize(", store_path, ")")
    end
    check == :correctness && println(setup, CHECK_SOURCE)
    write(setup, take!(body))
    return String(take!(setup)), call_source(p.call, check), store
end

"""
    write_program(path, p::Program, class::Class, original::String)

Write `p` as a standalone script at `path`, with its store (if any) at
`<path stem>_data.jls`.  Used for checkpoints and the final `repro.jl`.
"""
function write_program(path::String, p::Program, class::Class, original::String, check::Symbol = :error)
    stem = splitext(path)[1]
    setup, call, store = render(p, "joinpath(@__DIR__, $(repr(basename(stem) * "_data.jls")))", check)
    isempty(store.entries) || serialize(stem * "_data.jls", store.entries)
    open(path, "w") do io
        println(io, "# Reduced automatically by MWE.jl — Julia $VERSION, Enzyme $(pkgversion(Enzyme))")
        println(io, "# Failure: ", class)
        println(io, "# Reduced from: ", original)
        println(io)
        print(io, setup)
        println(io, call)
    end
end

# ---- capture ----------------------------------------------------------------

"""
    Captured

Thrown by [`capture`](@ref) in place of the original error, carrying the
call expression and the live argument values.
"""
struct Captured <: Exception
    expr::Expr
    mode::Enzyme.Mode
    f::Any
    tail::Tuple
    err::Any
end

"""
    capture(expr, mode, f, tail...; kw...)

What every `autodiff` call in the script is rewritten to by
[`hook_autodiff`](@ref).  Runs the real `autodiff`; if it throws, throws
[`Captured`](@ref) instead so the failing invocation can be recovered.
The arguments are copied before the call: `autodiff` writes gradients into
shadows, and the reproducer must start from the values the user passed in.
"""
function capture(expr, mode, f, tail...; kw...)
    saved = deepcopy(tail)
    try
        return CHECK[] == :correctness ? checked_autodiff(mode, f, tail...; kw...) :
                                         autodiff(mode, f, tail...; kw...)
    catch err
        throw(Captured(expr, mode, f, saved, err))
    end
end

"""
Which failures the run is looking for: `:error` (the call throws) or
`:correctness` (the gradient disagrees with finite differences).  Set on
the workers before capture.
"""
const CHECK = Ref(:error)

"Rewrite every `autodiff(...)` call in `e` to go through [`capture`](@ref)."
function hook_autodiff(e)
    e isa Expr || return e
    if e.head == :call && (e.args[1] == :autodiff || e.args[1] == :(Enzyme.autodiff))
        return Expr(:call, GlobalRef(MWE, :capture), QuoteNode(e), map(hook_autodiff, e.args[2:end])...)
    end
    return Expr(e.head, map(hook_autodiff, e.args)...)
end

"""
    Call(c::Captured, m::Module) -> Call

Turn a capture into the reducer's representation, with values captured
relative to module `m`.
"""
function Call(c::Captured, m::Module)
    any(a -> a isa Expr && a.head in (:kw, :parameters), c.expr.args) &&
        error("keyword arguments to autodiff are not supported")
    exprs = c.expr.args[4:end]
    tail = collect(Any, c.tail)
    ret = nothing
    if !isempty(tail) && tail[1] isa Type && tail[1] <: Annotation
        ret = popfirst!(tail)
        popfirst!(exprs)
    end
    args = map(enumerate(zip(tail, exprs))) do (i, (a, e))
        a isa Annotation || error("argument $i is $(typeof(a)); every argument after f must be an Annotation")
        name = e isa Expr && e.head == :call && length(e.args) >= 2 && e.args[2] isa Symbol ? e.args[2] : Symbol(:a, i)
        Arg(nameof(typeof(a)), name, Value(a.val, m), shadows(a, m))
    end
    return Call(c.expr.args[1], c.mode, c.expr.args[3], ret, args)
end

shadows(a::Union{Const,Active}, m) = Value[]
shadows(a::Union{Duplicated,DuplicatedNoNeed}, m) = [Value(a.dval, m)]
shadows(a::Union{BatchDuplicated,BatchDuplicatedNoNeed}, m) = [Value(d, m) for d in a.dval]
shadows(a::Annotation, m) = error("unsupported annotation $(typeof(a))")
