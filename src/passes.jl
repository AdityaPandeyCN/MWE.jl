# A pass proposes simpler programs along one axis, as `label => program` pairs, coarse cuts first.

const Proposals = Vector{Pair{String,Program}}

# ---- helpers ----------------------------------------------------------------

replace_def(p::Program, j, d) = with(p; defs = [i == j ? d : p.defs[i] for i in eachindex(p.defs)])
replace_arg(c::Call, i, a) = with(c; args = [j == i ? a : c.args[j] for j in eachindex(c.args)])

"Chunk sizes for delta debugging over `n` items: powers of two, largest first."
granularities(n) = n < 1 ? Int[] : [2^k for k in floor(Int, log2(n)):-1:0]

"Consecutive chunks of `items` at every granularity, coarsest first, without duplicates."
function chunks(items::AbstractVector)
    seen = Set{Vector{eltype(items)}}()
    out = Vector{eltype(items)}[]
    for g in granularities(length(items)), c in Iterators.partition(items, g)
        c = collect(c)
        c in seen || (push!(seen, c); push!(out, c))
    end
    return out
end

"`1:n` in binary-search order."
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

"Real or complex floating point: what Enzyme differentiates."
isfloatlike(::Type{T}) where {T} = T <: Union{AbstractFloat,Complex{<:AbstractFloat}}

"The input standing in for a recorded intermediate: float arrays `Duplicated`, floats `Active` (`Duplicated` in forward mode), else `Const`."
function placeholder_arg(name::Symbol, v::Value, mode::Enzyme.Mode)
    d = v.data
    if d isa Array && isfloatlike(eltype(d))
        return Arg(:Duplicated, name, v, [Value(zero(d), nothing, v.type)])
    elseif d isa Number && isfloatlike(typeof(d))
        return mode isa Enzyme.ReverseMode ? Arg(:Active, name, v, Value[]) :
                                             Arg(:Duplicated, name, v, [Value(one(d), nothing, v.type)])
    end
    return Arg(:Const, name, v, Value[])
end

"Keep the first `k` entries of `x` along dimension `dim`."
take(x::Array, dim, k) = x[ntuple(d -> d == dim ? (1:k) : Colon(), ndims(x))...]

"Whether a top-level definition brings Enzyme into scope; never worth removing."
is_enzyme_import(d) = d isa Expr && d.head in (:using, :import) && uses(:Enzyme, d)

# ---- structural passes ------------------------------------------------------

"Truncate the entry function after statement `k` and return that statement's value, in bisection order."
function remove_suffix(p::Program)
    out = Proposals()
    j = entry(p)
    j === nothing && return out
    d = p.defs[j]
    s = stmts(d)
    scalar = p.call.ret === Active || p.call.ret === nothing
    for k in bisect_order(length(s) - 1)
        is_assignment(s[k]) && length(assigned_names(s[k])) == 1 && haskey(p.values, s[k]) || continue
        v, y = only(p.values[s[k]]), only(assigned_names(s[k]))
        ret = if !scalar
            y
        elseif v.data isa AbstractFloat
            y
        elseif v.data isa Complex{<:AbstractFloat}
            :(abs2($y))                           # an Active return must be real
        elseif v.data isa Array{<:AbstractFloat}
            :(sum($y))
        elseif v.data isa Array{<:Complex{<:AbstractFloat}}
            :(sum(abs2, $y))
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

"Delta debugging over statements at any depth: assignments with recorded values become placeholders, others are deleted."
function placeholder_statements(p::Program)
    out = Proposals()
    e = entry(p)
    as_args = e !== nothing && !entry_called_elsewhere(p)
    order = [j for j in eachindex(p.defs) if isfdef(p.defs[j]) && fname(p.defs[j]) !== nothing]
    e === nothing || (order = [e; filter(!=(e), order)])
    for j in order
        d = p.defs[j]
        nodes = filter(s -> s !== last(stmts(d)), all_statements(d.args[2]))
        n = length(nodes)
        for chunk in chunks(1:n)
            q = replace_statements(p, j, nodes[chunk], j == e && as_args)
            label = length(chunk) == 1 ? "$(fname(d)): statement $(first(chunk))" :
                                         "$(fname(d)): statements $(first(chunk)):$(last(chunk))"
            push!(out, "$label of $n" => q)
        end
    end
    return out
end

"`p` with statements `targets` of definition `j` turned into placeholders where possible and deleted otherwise."
function replace_statements(p::Program, j, targets, as_args::Bool)
    d = p.defs[j]
    params = copy(fparams(d))
    args = copy(p.call.args)
    consts = copy(p.consts)
    taken = taken_names(p)
    repl = IdDict{Any,Any}()
    for st in targets
        vs = get(p.values, st, nothing)
        if is_assignment(st) && vs !== nothing && all(available, vs)
            names = map(zip(assigned_names(st), vs)) do (y, v)
                if as_args
                    name = fresh(Symbol(y, :_rec), taken)
                    push!(params, name)
                    push!(args, placeholder_arg(name, v, p.call.mode))
                else
                    name = fresh(Symbol(:P_, fname(d), :_, y), taken)
                    push!(consts, name => v)
                end
                push!(taken, name)
                name
            end
            lhs = st.args[1]
            repl[st] = length(names) == 1 ? Expr(:(=), lhs, only(names)) : Expr(:(=), lhs, Expr(:tuple, names...))
        else
            repl[st] = nothing
        end
    end
    body = map_statements(s -> haskey(repl, s) ? repl[s] : s, d.args[2])
    d′ = with_params(with_body(d, body), params)
    return with(replace_def(p, j, d′); consts, call = with(p.call; args))
end

"Delete every top-level assignment no later statement mentions."
function eliminate_dead_code(p::Program)
    out = Proposals()
    for (j, d) in enumerate(p.defs)
        isfdef(d) && fname(d) !== nothing || continue
        s = stmts(d)
        dead = [i for i in 1:length(s)-1 if is_assignment(s[i]) &&
                !any(y -> any(t -> uses(y, t), s[i+1:end]), assigned_names(s[i]))]
        isempty(dead) && continue
        push!(out, "$(fname(d)): remove dead statements $(join(dead, ","))" =>
              replace_def(p, j, with_body(d, s[setdiff(1:length(s), dead)])))
    end
    return out
end

const LITERAL_RETURNS = (Expr(:return, nothing), Expr(:return, 0.0))

"Replace each function's last statement by `return nothing` or `return 0.0`; a literal return is terminal."
function simplify_result(p::Program)
    out = Proposals()
    e = entry(p)
    for (j, d) in enumerate(p.defs)
        isfdef(d) && fname(d) !== nothing || continue
        s = stmts(d)
        (isempty(s) || last(s) in LITERAL_RETURNS) && continue
        for new in LITERAL_RETURNS
            # the entry's result feeds the return activity: `nothing` is only ever `Const`
            j == e && new.args[1] === nothing && !(p.call.ret in (Const, nothing)) && continue
            push!(out, "$(fname(d)): return $(repr(new.args[1]))" => replace_def(p, j, with_body(d, [s[1:end-1]; new])))
        end
    end
    return out
end

"Drop parameters of the entry function its body never mentions, all at once and then one at a time."
function remove_unused_args(p::Program)
    out = Proposals()
    j = entry(p)
    j === nothing && return out
    entry_called_elsewhere(p) && return out
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

"Delta debugging over top-level definitions other than the entry function and `using Enzyme`."
function remove_defs(p::Program)
    out = Proposals()
    idxs = [j for j in eachindex(p.defs) if j != entry(p) && !is_enzyme_import(p.defs[j])]
    for chunk in chunks(idxs)
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

"Propose making each non-`Const` argument `Const`."
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

"Extents to try for a dimension of extent `n`: 1, half, `n - 1`."
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

"Propose smaller arrays: dimensions sharing an extent shrink together first, then individually."
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

"The passes in the order the reducer runs them."
const PASSES = (simplify_mode, simplify_return, remove_suffix, placeholder_statements,
                eliminate_dead_code, simplify_result, remove_unused_args, drop_activity, unbatch,
                shrink_arrays, remove_defs, remove_unused_consts)

"One-line size of a program: definitions, statements in the entry function (at any depth), arguments."
function size_summary(p::Program)
    j = entry(p)
    n = j === nothing ? "?" : length(all_statements(p.defs[j].args[2]))
    "defs $(length(p.defs)), statements $n, args $(length(p.call.args))"
end
