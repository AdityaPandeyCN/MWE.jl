# A pass looks at a program and proposes simpler ones along a single axis,
# as `label => program` pairs in the order they should be tried: coarse
# cuts first.  The reducer accepts the first proposal whose failure class is
# unchanged and re-runs the pass, so a pass never needs to search itself.

const Proposals = Vector{Pair{String,Program}}

# ---- helpers ----------------------------------------------------------------

replace_def(p::Program, j, d) = with(p; defs = [i == j ? d : p.defs[i] for i in eachindex(p.defs)])
replace_arg(c::Call, i, a) = with(c; args = [j == i ? a : c.args[j] for j in eachindex(c.args)])

"Chunk sizes for delta debugging over `n` items: powers of two, largest first."
granularities(n) = n < 1 ? Int[] : [2^k for k in floor(Int, log2(n)):-1:0]

"""
    bisect_order(n) -> Vector{Int}

`1:n` in binary-search order (midpoint first, then the midpoints of each
half, ...), so that accepted cuts converge in O(log n) steps.
"""
function bisect_order(n)
    order = Int[]
    queue = [(1, n)]
    while !isempty(queue)
        lo, hi = popfirst!(queue)
        lo > hi && continue
        mid = (lo + hi) ÷ 2
        push!(order, mid)
        push!(queue, (lo, mid - 1), (mid + 1, hi))
    end
    return order
end

"A name based on `base` not in `taken`."
function fresh(base::Symbol, taken)
    name, k = base, 1
    while name in taken
        k += 1
        name = Symbol(base, :_, k)
    end
    return name
end

"Every symbol bound at top level of `p`: argument names, constants, function names."
function taken_names(p::Program)
    names = Set{Symbol}(a.name for a in p.call.args)
    union!(names, first.(p.consts))
    for d in p.defs
        isfdef(d) && fname(d) !== nothing && push!(names, fname(d))
    end
    return names
end

"""
    placeholder_arg(name, v::Value, mode) -> Arg

The input that stands in for a recorded intermediate: arrays are
`Duplicated` with a zero shadow, floats `Active` in reverse mode and
`Duplicated` with a unit shadow in forward mode, everything else `Const`.
Differentiation is kept flowing through the placeholder so the failure it
participates in can survive; [`drop_activity`](@ref) makes it `Const`
later if that is not needed.
"""
function placeholder_arg(name::Symbol, v::Value, mode::Enzyme.Mode)
    d = v.data
    if d isa Array{<:AbstractFloat}
        return Arg(:Duplicated, name, v, [Value(zero(d), nothing, v.type)])
    elseif d isa AbstractFloat
        return mode isa Enzyme.ReverseMode ? Arg(:Active, name, v, Value[]) :
                                             Arg(:Duplicated, name, v, [Value(one(d), nothing, v.type)])
    end
    return Arg(:Const, name, v, Value[])
end

"Keep the first `k` entries of `x` along dimension `dim`."
take(x::Array, dim, k) = x[ntuple(d -> d == dim ? (1:k) : Colon(), ndims(x))...]

# ---- structural passes ------------------------------------------------------

"""
    remove_suffix(p::Program)

Truncate the entry function after statement `k` and return that
statement's value, for `k` in bisection order.  With an `Active` return
the value must be a scalar, so an array intermediate is returned as its
`sum`.  Finds where in the function the failure is triggered.
"""
function remove_suffix(p::Program)
    out = Proposals()
    j = entry(p)
    j === nothing && return out
    d = p.defs[j]
    s = stmts(d)
    scalar = p.call.ret === Active || p.call.ret === nothing
    for k in bisect_order(length(s) - 1)
        is_assignment(s[k]) && haskey(p.values, s[k]) || continue
        v, y = p.values[s[k]], assigned(s[k])
        ret = if !scalar
            y
        elseif v.data isa Number
            y
        elseif v.data isa Array{<:Number}
            :(sum($y))
        else
            continue
        end
        body = [s[1:k]..., Expr(:return, ret)]
        body == s && continue                     # already truncated here
        push!(out, "$(fname(d)): keep statements 1:$k, return $ret" =>
              replace_def(p, j, with_body(d, body)))
    end
    return out
end

"""
    placeholder_statements(p::Program)

Delta debugging over the statements of each function body (the entry
function first).  A chunk of statements is replaced at once: an assignment
with a recorded value becomes a placeholder — a new argument of the entry
function, or a global constant in a callee — and any other statement is
deleted.  The last statement, the function's result, is never touched.
"""
function placeholder_statements(p::Program)
    out = Proposals()
    e = entry(p)
    order = [j for j in eachindex(p.defs) if isfdef(p.defs[j]) && fname(p.defs[j]) !== nothing]
    e === nothing || (order = [e; filter(!=(e), order)])
    for j in order
        d = p.defs[j]
        n = length(stmts(d)) - 1
        for g in granularities(n), chunk in Iterators.partition(1:n, g)
            q = replace_statements(p, j, chunk, j == e)
            label = length(chunk) == 1 ? "$(fname(d)): statement $(first(chunk))" :
                                         "$(fname(d)): statements $(first(chunk)):$(last(chunk))"
            push!(out, label => q)
        end
    end
    return out
end

"""
    replace_statements(p, j, idxs, is_entry) -> Program

`p` with statements `idxs` of definition `j` turned into placeholders where
a value is available and deleted otherwise.
"""
function replace_statements(p::Program, j, idxs, is_entry::Bool)
    d = p.defs[j]
    s = copy(stmts(d))
    params = copy(fparams(d))
    args = copy(p.call.args)
    consts = copy(p.consts)
    taken = taken_names(p)
    keep = trues(length(s))
    for i in idxs
        st = s[i]
        v = get(p.values, st, nothing)
        if is_assignment(st) && v !== nothing && available(v)
            y = assigned(st)
            if is_entry
                name = fresh(Symbol(y, :_rec), taken)
                push!(params, name)
                push!(args, placeholder_arg(name, v, p.call.mode))
            else
                name = fresh(Symbol(:P_, fname(d), :_, y), taken)
                push!(consts, name => v)
            end
            push!(taken, name)
            s[i] = Expr(:(=), y, name)
        else
            keep[i] = false
        end
    end
    d′ = with_params(with_body(d, s[keep]), params)
    return with(replace_def(p, j, d′); consts, call = with(p.call; args))
end

"""
    eliminate_dead_code(p::Program)

For each function, delete in one go every assignment whose variable no
later statement mentions.  The oracle still has the last word, since a
dead assignment's right-hand side may have side effects.
"""
function eliminate_dead_code(p::Program)
    out = Proposals()
    for (j, d) in enumerate(p.defs)
        isfdef(d) && fname(d) !== nothing || continue
        s = stmts(d)
        dead = [i for i in 1:length(s)-1 if is_assignment(s[i]) &&
                !any(t -> uses(assigned(s[i]), t), s[i+1:end])]
        isempty(dead) && continue
        push!(out, "$(fname(d)): remove dead statements $(join(dead, ","))" =>
              replace_def(p, j, with_body(d, s[setdiff(1:length(s), dead)])))
    end
    return out
end

"""
    remove_unused_args(p::Program)

Drop the positional parameters of the entry function that its body never
mentions, together with the corresponding arguments of the call: all of
them at once first, then one at a time.
"""
function remove_unused_args(p::Program)
    out = Proposals()
    j = entry(p)
    j === nothing && return out
    d = p.defs[j]
    params = fparams(d)
    length(params) == length(p.call.args) || return out
    all(x -> x isa Symbol || (x isa Expr && x.head == :(::)), params) || return out
    unused = [i for (i, x) in enumerate(params) if !uses(param_name(x), d.args[2])]
    groups = [[i] for i in unused]
    length(unused) > 1 && pushfirst!(groups, unused)
    for idxs in groups
        keep = setdiff(1:length(params), idxs)
        names = join((param_name(params[i]) for i in idxs), ", ")
        push!(out, "$(fname(d)): drop unused argument$(length(idxs) > 1 ? "s" : "") $names" =>
              with(replace_def(p, j, with_params(d, params[keep])); call = with(p.call; args = p.call.args[keep])))
    end
    return out
end

"""
    remove_defs(p::Program)

Delta debugging over top-level definitions other than the entry function:
`using`s, data, helpers that are no longer called.
"""
function remove_defs(p::Program)
    out = Proposals()
    idxs = [j for j in eachindex(p.defs) if j != entry(p)]
    for g in granularities(length(idxs)), chunk in Iterators.partition(idxs, g)
        keep = [p.defs[j] for j in eachindex(p.defs) if !(j in chunk)]
        label = "remove definition$(length(chunk) == 1 ? "" : "s") $(join(chunk, ","))"
        push!(out, label => with(p; defs = keep))
    end
    return out
end

"Drop placeholder constants no definition mentions any more."
function remove_unused_consts(p::Program)
    out = Proposals()
    for (i, (name, _)) in enumerate(p.consts)
        any(d -> uses(name, d), p.defs) && continue
        push!(out, "drop unused constant $name" => with(p; consts = [p.consts[1:i-1]; p.consts[i+1:end]]))
    end
    return out
end

# ---- invocation passes ------------------------------------------------------

"Propose the mode without runtime activity, strong zero, and `WithPrimal`, one at a time."
function simplify_mode(p::Program)
    out = Proposals()
    m = p.call.mode
    Enzyme.EnzymeCore.runtime_activity(m) &&
        push!(out, "clear runtime activity" => with(p; call = with(p.call; mode = Enzyme.EnzymeCore.clear_runtime_activity(m))))
    Enzyme.EnzymeCore.strong_zero(m) &&
        push!(out, "clear strong zero" => with(p; call = with(p.call; mode = Enzyme.EnzymeCore.clear_strong_zero(m))))
    m == ReverseWithPrimal && push!(out, "drop WithPrimal" => with(p; call = with(p.call; mode = Reverse)))
    m == ForwardWithPrimal && push!(out, "drop WithPrimal" => with(p; call = with(p.call; mode = Forward)))
    return out
end

"Propose a `Const` return activity."
function simplify_return(p::Program)
    r = p.call.ret
    r === nothing || r === Const ? Proposals() :
        ["return $(nameof(r)) → Const" => with(p; call = with(p.call; ret = Const))]
end

"""
    drop_activity(p::Program)

Propose making each non-`Const` argument `Const`.  A `Const` argument is
removed from differentiation entirely, so if the failure survives, that
argument's activity was irrelevant.
"""
function drop_activity(p::Program)
    out = Proposals()
    for (i, a) in enumerate(p.call.args)
        a.kind == :Const && continue
        push!(out, "arg $(a.name) $(a.kind) → Const" =>
              with(p; call = replace_arg(p.call, i, Arg(:Const, a.name, a.val, Value[]))))
    end
    return out
end

"Propose replacing each batched argument by its first shadow."
function unbatch(p::Program)
    out = Proposals()
    for (i, a) in enumerate(p.call.args)
        kind = a.kind == :BatchDuplicated ? :Duplicated : a.kind == :BatchDuplicatedNoNeed ? :DuplicatedNoNeed : continue
        push!(out, "arg $(a.name) unbatch" =>
              with(p; call = replace_arg(p.call, i, Arg(kind, a.name, a.val, a.shadows[1:1]))))
    end
    return out
end

"""
    targets(n) -> Vector{Int}

Extents to try for a dimension of extent `n`, in order: 1 (most bugs do not
depend on size), then half (converges in O(log n) steps when they do), then
`n - 1` (finds the exact threshold).
"""
targets(n) = unique((1, cld(n, 2), n - 1))

shrink_arg(a::Arg, dim, k) = Arg(a.kind, a.name, mapdata(x -> take(x, dim, k), a.val),
                                 [mapdata(x -> take(x, dim, k), s) for s in a.shadows])

function shrink_dims(p::Program, dims::Vector{Tuple{Int,Int}}, k)
    args = copy(p.call.args)
    for (i, d) in dims
        args[i] = shrink_arg(args[i], d, k)
    end
    return with(p; call = with(p.call; args))
end

"""
    shrink_arrays(p::Program)

Propose smaller array arguments.  Dimensions sharing the same extent are
shrunk together first: shapes are usually coupled (a length-`n` vector and
an `n×n` matrix), and shrinking one alone only produces a
`DimensionMismatch` that gets rejected.  Individual dimensions follow.
"""
function shrink_arrays(p::Program)
    out = Proposals()
    dims = [(i, d) => n for (i, a) in enumerate(p.call.args) if a.val.data isa Array
                       for (d, n) in enumerate(size(a.val.data)) if n > 1]
    for n in sort!(unique(last.(dims)); rev = true)
        group = [first(x) for x in dims if last(x) == n]
        for k in targets(n)
            push!(out, "dims with extent $n → $k" => shrink_dims(p, group, k))
        end
    end
    for ((i, d), n) in dims
        count(x -> last(x) == n, dims) == 1 && continue   # its group above was just this dimension
        for k in targets(n)
            push!(out, "arg $(p.call.args[i].name) dim $d: $n → $k" => shrink_dims(p, [(i, d)], k))
        end
    end
    return out
end

"""
The passes in the order the reducer runs them.  Cheap, informative changes
to the invocation first; then the cuts inside the entry function (suffix,
placeholders); then clean-up of what those left behind.
"""
const PASSES = (simplify_mode, simplify_return, remove_suffix, placeholder_statements,
                eliminate_dead_code, remove_unused_args, drop_activity, unbatch, shrink_arrays,
                remove_defs, remove_unused_consts)

"One-line size of a program: definitions, statements in the entry function, arguments."
function size_summary(p::Program)
    j = entry(p)
    n = j === nothing ? "?" : length(stmts(p.defs[j]))
    "defs $(length(p.defs)), statements $n, args $(length(p.call.args))"
end
