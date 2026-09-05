# The program: the script's top-level expressions plus the captured call.

"One argument of the captured call: annotation `kind`, variable `name`, primal `val` and `shadows`."
struct Arg
    kind::Symbol
    name::Symbol
    val::Value
    shadows::Vector{Value}
end

"The captured `autodiff(mode, f, ret, args...)`; `ret` is `nothing` when left for Enzyme to infer."
struct Call
    head::Union{Symbol,Expr}
    mode::Enzyme.Mode
    f::Union{Symbol,Expr}
    ret::Union{Nothing,Type}
    args::Vector{Arg}
end

"What is being reduced: top-level `defs`, placeholder `consts`, the `call`, and recorded `values` keyed by statement object."
struct Program
    defs::Vector{Any}
    consts::Vector{Pair{Symbol,Value}}
    call::Call
    values::IdDict{Any,Vector{Value}}
end

with(p::Program; defs = p.defs, consts = p.consts, call = p.call) = Program(defs, consts, call, p.values)
with(c::Call; mode = c.mode, ret = c.ret, args = c.args) = Call(c.head, mode, c.f, ret, args)

# ---- parsing the script -----------------------------------------------------

"Top-level expressions of `file`, with literal `include`s inlined, docstrings stripped and definitions normalised."
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
    elseif e isa Expr && e.head in (:using, :import) && length(e.args) > 1
        append!(defs, [Expr(e.head, a) for a in e.args])      # `using A, B`: one definition each

    else
        push!(defs, normalize(Base.remove_linenums!(e)))
    end
end

strip_doc(e) = e isa Expr && e.head == :macrocall && e.args[1] == GlobalRef(Core, Symbol("@doc")) ? e.args[end] : e

"A function signature: `f(x)`, `f(x) where T`, `f(x)::T`."
issig(x) = x isa Expr && (x.head == :call || (x.head in (:where, :(::)) && issig(x.args[1])))

"Rewrite `f(x) = body` as `function f(x) body end` with a block body."
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
"Top-level statements of a function body."
stmts(d) = d.args[2].args

with_body(d, body::Expr) = Expr(:function, d.args[1], body)
with_body(d, body::Vector) = with_body(d, Expr(:block, body...))
with_params(d, params::Vector) = Expr(:function, mapcall(c -> Expr(:call, c.args[1], params...), d.args[1]), d.args[2])

"Name bound by a parameter: `x`, `x::T`, `x = 1`, `x...`; `nothing` for anything else."
param_name(p::Symbol) = p
param_name(p::Expr) = p.head in (:(::), :kw, :(...)) && length(p.args) >= 1 ? param_name(p.args[1]) : nothing
param_name(p) = nothing

"An assignment to plain locals, `y = rhs` or `a, b = rhs`."
is_assignment(s) = s isa Expr && s.head == :(=) &&
    (s.args[1] isa Symbol || (s.args[1] isa Expr && s.args[1].head == :tuple && all(x -> x isa Symbol, s.args[1].args)))

"Names bound by an assignment statement."
assigned_names(s) = s.args[1] isa Symbol ? [s.args[1]] : Symbol[s.args[1].args...]

"Whether symbol `sym` occurs anywhere in `e`."
uses(sym::Symbol, e) = e === sym || (e isa Expr && any(a -> uses(sym, a), e.args))

# ---- nested statements ------------------------------------------------------

"The statement blocks nested directly inside `s`: loop and `let` bodies, `if` branches, `begin`, macro and `do` bodies."
function sub_blocks(s)
    s isa Expr || return Expr[]
    isblock(b) = b isa Expr && b.head == :block
    if s.head in (:for, :while, :let)
        return Expr[s.args[2]]
    elseif s.head in (:if, :elseif)
        out = Expr[]
        for b in s.args[2:end]
            isblock(b) ? push!(out, b) : append!(out, sub_blocks(b))   # an `elseif` chain
        end
        return out
    elseif s.head == :block
        return Expr[s]
    elseif s.head == :macrocall
        return isblock(s.args[end]) ? Expr[s.args[end]] : sub_blocks(s.args[end])
    elseif s.head == :do
        return Expr[s.args[2].args[2]]
    end
    return Expr[]
end

"`s` with each nested block replaced by `g(block)`."
function map_sub_blocks(g, s)
    isempty(sub_blocks(s)) && return s
    isblock(b) = b isa Expr && b.head == :block
    if s.head in (:for, :while, :let)
        return Expr(s.head, s.args[1], g(s.args[2]))
    elseif s.head in (:if, :elseif)
        return Expr(s.head, s.args[1], [isblock(b) ? g(b) : map_sub_blocks(g, b) for b in s.args[2:end]]...)
    elseif s.head == :block
        return g(s)
    elseif s.head == :macrocall
        last = s.args[end]
        return Expr(:macrocall, s.args[1:end-1]..., isblock(last) ? g(last) : map_sub_blocks(g, last))
    elseif s.head == :do
        lam = s.args[2]
        return Expr(:do, s.args[1], Expr(lam.head, lam.args[1], g(lam.args[2])))
    end
    return s
end

"Every statement in `body` at any depth, in source order."
function all_statements(body::Expr)
    out = Any[]
    for s in body.args
        push!(out, s)
        for b in sub_blocks(s)
            append!(out, all_statements(b))
        end
    end
    return out
end

"Rebuild `body` with `f` applied to every statement: a replacement, `nothing` to delete, or the statement itself to descend."
function map_statements(f, body::Expr)
    out = Any[]
    for s in body.args
        r = f(s)
        r === nothing && continue
        r === s && (r = map_sub_blocks(b -> map_statements(f, b), s))
        push!(out, r)
    end
    return Expr(:block, out...)
end

# ---- the entry function ---------------------------------------------------

"Unwrap `Const(f)` / `Duplicated(f, df)` to the function expression."
unwrap_f(f::Symbol) = f
unwrap_f(f::Expr) = f.head == :call && f.args[1] in (:Const, :Active, :Duplicated, :DuplicatedNoNeed, :BatchDuplicated) ? f.args[2] : f

"Index in `p.defs` of the differentiated function, if it is a named function defined exactly once."
function entry(p::Program)
    name = unwrap_f(p.call.f)
    name isa Symbol || return nothing
    js = findall(d -> isfdef(d) && fname(d) == name, p.defs)
    return length(js) == 1 ? only(js) : nothing
end

"Whether the entry function is mentioned outside its own signature, so its arity cannot change."
function entry_called_elsewhere(p::Program)
    j = entry(p)
    j === nothing && return true
    name = fname(p.defs[j])
    return uses(name, p.defs[j].args[2]) || any(d -> d !== p.defs[j] && uses(name, d), p.defs)
end

# ---- source rendering -------------------------------------------------------

"`\"\"` or `\"Enzyme.\"`, depending on whether the script qualified the call."
prefix(c::Call) = c.head isa Symbol ? "" : "Enzyme."

"Source text for `m`; exotic modes fall back to `repr`."
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

"The plain `f(x, y)` call; with `copy`, every argument is `deepcopy`d."
function primal_source(c::Call; copy::Bool = false)
    args = [copy ? "deepcopy($(a.name))" : string(a.name) for a in c.args]
    return "$(unwrap_f(c.f))(" * join(args, ", ") * ")"
end

"`s` on one line, cut to `n` characters."
function oneline(s::AbstractString, n::Int = 200)
    t = join(strip.(split(s, '\n')), " ")
    return length(t) > n ? first(t, n) * "…" : t
end

"Source for `p`: `setup` (definitions and arguments) and the `autodiff` `call`; values too large for literals go in `store`."
function render(p::Program, store_path::String, check::Symbol = :error)
    store = Store("STORE")
    body = IOBuffer()
    for (name, v) in p.consts
        println(body, "const ", name, " = ", render(v, string(name), store), type_note(v))
    end
    for a in p.call.args
        println(body, a.name, " = ", render(a.val, string(a.name), store), type_note(a.val))
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

"Write `label * text` as a comment, every line of a multi-line `text` included."
function comment(io, label, text)
    for (i, line) in enumerate(split(text, '\n'))
        println(io, "# ", i == 1 ? label : " "^length(label), line)
    end
end

"Write `p` as a standalone script at `path`, with its store at `<stem>_data.jls`."
function write_program(path::String, p::Program, class::Class, original::String, check::Symbol = :error)
    stem = splitext(path)[1]
    setup, call, store = render(p, "joinpath(@__DIR__, $(repr(basename(stem) * "_data.jls")))", check)
    isempty(store.entries) || serialize(stem * "_data.jls", store.entries)
    open(path, "w") do io
        println(io, "# Reduced automatically by MWE.jl — Julia $VERSION, Enzyme $(pkgversion(Enzyme))")
        comment(io, "Failure: ", string(class))
        comment(io, "Reduced from: ", oneline(original))
        println(io)
        print(io, setup)
        println(io, call)
    end
end

# ---- capture ----------------------------------------------------------------

"Thrown by `capture` in place of the original error, with the call expression and pre-call argument values."
struct Captured <: Exception
    expr::Expr
    mode::Enzyme.Mode
    f::Any
    tail::Tuple
    err::Any
end

"`:error` or `:correctness`: which failures the run looks for; set on the workers."
const CHECK = Ref(:error)

"Hook for `autodiff`/`autodiff_deferred`: run the real call and rethrow a failure as `Captured` with copies of the arguments."
function capture(expr, fn, mode, f, tail...; kw...)
    fn in (Enzyme.autodiff, Enzyme.autodiff_deferred) && !Enzyme.within_autodiff() ||
        return fn(mode, f, tail...; kw...)
    saved = deepcopy(tail)
    try
        return CHECK[] == :correctness ? checked_autodiff(mode, f, tail...; kw...) :
                                         fn(mode, f, tail...; kw...)
    catch err
        throw(Captured(expr, mode, f, saved, err))
    end
end

"Hook for reverse-mode `Enzyme.gradient` calls, captured in `autodiff` form."
function capture_gradient(expr, fn, mode, f, xs...; kw...)
    fn === Enzyme.gradient && mode isa ReverseMode && isempty(kw) && !Enzyme.within_autodiff() ||
        return fn(mode, f, xs...; kw...)
    anns = map(x -> x isa Number ? Active(x) : Duplicated(x, Enzyme.make_zero(x)), xs)
    saved = deepcopy(anns)
    call = Expr(:call, :autodiff, expr.args[2], expr.args[3], :Active,
                [Expr(:call, x isa Number ? :Active : :Duplicated, e) for (x, e) in zip(xs, expr.args[4:end])]...)
    try
        CHECK[] == :correctness && checked_autodiff(mode, f, Active, anns...)
        return fn(mode, f, xs...)
    catch err
        throw(Captured(call, mode, f, (Active, saved...), err))
    end
end

"Name of the callee in a call's head: `autodiff` for both `autodiff` and `Enzyme.autodiff`."
callee_name(x::Symbol) = x
callee_name(x::Expr) = x.head == :. && x.args[1] == :Enzyme && x.args[2] isa QuoteNode ? x.args[2].value : nothing
callee_name(x) = nothing

"Rewrite `autodiff`, `autodiff_deferred` and `gradient` calls in `e` to go through the capture hooks."
function hook_autodiff(e)
    e isa Expr || return e
    if e.head == :call
        name = callee_name(e.args[1])
        hook = name in (:autodiff, :autodiff_deferred) ? :capture :
               name == :gradient ? :capture_gradient : nothing
        hook === nothing || return Expr(:call, GlobalRef(MWE, hook), QuoteNode(e), e.args...)
    end
    return Expr(e.head, map(hook_autodiff, e.args)...)
end

"Define an anonymous differentiated function as a named top-level function."
function lift_lambda(p::Program)
    f = p.call.f
    f isa Expr && f.head == :-> || return p
    params = f.args[1] isa Expr && f.args[1].head == :tuple ? f.args[1].args : Any[f.args[1]]
    name = fresh(:entry, taken_names(p))
    d = normalize(Expr(:function, Expr(:call, name, params...), f.args[2]))
    return with(p; defs = [p.defs; d], call = Call(p.call.head, p.call.mode, name, p.call.ret, p.call.args))
end

"Turn a capture into the reducer's representation, with values relative to module `m`."
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
