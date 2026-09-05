using Test, Enzyme, Shrink
using Shrink: Class, PASS, classify, classify_run, normalize_key, Value, available, render, Store, type_note,
           Program, Call, Arg, Captured, parse_script, normalize, hook_autodiff, capture, capture_gradient,
           entry, entry_called_elsewhere, stmts, fparams, is_assignment, assigned_names, uses,
           sub_blocks, all_statements, map_statements, call_source, primal_source, mode_source,
           instrument, bisect_order, granularities, remove_suffix, placeholder_statements, replace_statements,
           remove_unused_args, remove_defs, remove_unused_consts, drop_activity, shrink_arrays,
           simplify_mode, simplify_return, targets, take, write_program,
           eliminate_dead_code, simplify_result, checked_autodiff, WrongGradient, size_summary

const SCRIPT = joinpath(@__DIR__, "fixtures", "script.jl")

"A program for `SCRIPT` with the call captured by hand and values recorded by hand."
function program(; mode = Reverse, ret = Active, values = true)
    defs = parse_script(SCRIPT)[1:end-1]                      # everything before the call, as in a real run
    x = [1.0, 2.0, 3.0, 4.0]
    call = Call(:autodiff, mode, :f, ret,
                [Arg(:Duplicated, :x, Value(x, Main), [Value(zero(x), Main)]),
                 Arg(:Const, :p, Value(3.0, Main), Value[])])
    p = Program(defs, Pair{Symbol,Value}[], call, IdDict{Any,Vector{Value}}())
    if values
        d = defs[entry(p)]
        for (i, s) in enumerate(stmts(d))
            is_assignment(s) && (p.values[s] = [Value(i == 1 ? x .* 2 : Float64(i), Main)])
        end
    end
    return p
end

"A program whose entry function has a loop and a tuple assignment."
function looped(; extra = Any[])
    h = normalize(Base.remove_linenums!(:(function h(x)
        s = 0.0
        for i in 1:2
            t = x[i] * 2
            s += t
        end
        a, b = (s, 2s)
        return a
    end)))
    defs = Any[:(using Enzyme), h, extra...]
    call = Call(:autodiff, Reverse, :h, Active, [Arg(:Duplicated, :x, Value([1.0, 2.0], Main), [Value([0.0, 0.0], Main)])])
    p = Program(defs, Pair{Symbol,Value}[], call, IdDict{Any,Vector{Value}}())
    nodes = all_statements(h.args[2])
    p.values[nodes[1]] = [Value(0.0, Main)]
    p.values[nodes[3]] = [Value(2.0, Main)]
    p.values[nodes[5]] = [Value(6.0, Main), Value(12.0, Main)]
    return p
end

@testset "classify" begin
    c = classify(BoundsError([1.0], 3))
    @test c.kind == :BoundsError
    @test c.key == "attempt to access N-element Vector{FloatN} at index [N]"
    @test c == classify(BoundsError([1.0, 2.0], 7))            # sizes do not matter
    @test c != PASS
    @test classify(ErrorException("a")) != classify(ErrorException("b"))   # messages do
    @test hash(c) == hash(classify(BoundsError([1.0, 2.0], 7)))
    @test length(Set([c, classify(BoundsError([1.0], 9)), PASS])) == 2
    @test normalize_key("%12 = load 0xdeadbeef, size 3") == "%N = load 0xX, size N"
    @test classify(Enzyme.Compiler.EnzymeNonScalarReturnException([0.0], "")).key == ""   # typed Enzyme errors: type only
    @test classify(LoadError("f.jl", 1, LoadError("g.jl", 2, BoundsError([1.0], 3)))).kind == :BoundsError
    @test classify(ErrorException("ErrorException: x")).detail == "x"   # showerror's own type prefix is dropped
    @test classify_run(() -> 1) == PASS
    @test classify_run(() -> error("boom")).kind == :ErrorException
end

@testset "values" begin
    v = Value([1.0, 2.0], Main)
    @test v.data == [1.0, 2.0] && v.text === nothing && available(v) && type_note(v) == ""
    @test render(v, "x", Store()) == "[1.0, 2.0]"
    big = Value(rand(1000), Main)
    s = Store()
    @test render(big, "x", s) == "STORE[\"x\"]" && haskey(s.entries, "x")
    t = Value((a = 1, b = "s"), Main)
    @test t.data === nothing && t.text == "(a = 1, b = \"s\")"
    @test !available(Value(nothing, nothing, "Foo"))
    sub = Value(view([1.0, 2.0, 3.0], 1:2), Main)                  # non-Array numeric: collected, noted
    @test sub.data isa Vector{Float64} && sub.data == [1.0, 2.0]
    @test startswith(type_note(sub), "   # was SubArray")
    @test Value(1:3, Main).data == [1, 2, 3]
end

@testset "parse_script" begin
    defs = parse_script(SCRIPT)
    @test length(defs) == 6                                  # include inlined, docstring stripped
    @test all(d -> d isa Expr, defs)
    @test count(Shrink.isfdef, defs) == 3
    @test Base.remove_linenums!(normalize(:(g(x) = x + 1))) == Base.remove_linenums!(:(function g(x) x + 1 end))
    @test Shrink.fname(normalize(:(h(x::T) where {T} = x))) == :h
    @test fparams(defs[4]) == [:x, :p]
    @test uses(:p, defs[4]) && !uses(:nope, defs[4])
    @test is_assignment(Meta.parse("a, b = f(x)")) && assigned_names(Meta.parse("a, b = f(x)")) == [:a, :b]
    @test !is_assignment(:(a[1] = 2)) && !is_assignment(:(s += 1))
    split = mktempdir()
    write(joinpath(split, "s.jl"), "using Enzyme, LinearAlgebra\nimport Enzyme: Reverse, Forward\n")
    @test parse_script(joinpath(split, "s.jl")) == [:(using Enzyme), :(using LinearAlgebra), :(import Enzyme: Reverse, Forward)]
end

@testset "nested statements" begin
    body = looped().defs[2].args[2]
    nodes = all_statements(body)
    @test length(nodes) == 6
    @test nodes[2].head == :for && nodes[3] == :(t = x[i] * 2)
    @test length(sub_blocks(nodes[2])) == 1 && isempty(sub_blocks(nodes[1]))
    ifs = Base.remove_linenums!(Meta.parse("if c; a = 1; elseif d; b = 2; else; e = 3; end"))
    @test length(sub_blocks(ifs)) == 3
    @test length(sub_blocks(Base.remove_linenums!(Meta.parse("@inbounds for i in 1:2; y = i; end")))) == 1

    rebuilt = map_statements(s -> s == :(t = x[i] * 2) ? :(t = t_rec) : s, body)
    @test all_statements(rebuilt)[3] == :(t = t_rec)
    @test all_statements(rebuilt)[1] === nodes[1]                # untouched statements keep identity
    deleted = map_statements(s -> s.head == :for ? nothing : s, body)
    @test length(all_statements(deleted)) == 3
end

@testset "capture" begin
    e = :(r = autodiff(Reverse, f, Active, Duplicated(x, dx), Const(p)))
    h = hook_autodiff(e)
    @test h.args[2].args[1] == GlobalRef(Shrink, :capture)
    @test h.args[2].args[2] == QuoteNode(e.args[2]) && h.args[2].args[3] == :autodiff
    @test hook_autodiff(:(Enzyme.autodiff_deferred(Reverse, f, Const(x)))).args[1] == GlobalRef(Shrink, :capture)
    g = hook_autodiff(:(gradient(Reverse, f, x)))
    @test g.args[1] == GlobalRef(Shrink, :capture_gradient) && g.args[3] == :gradient
    @test hook_autodiff(:(foo(Reverse, f, x))).args[1] == :foo
    nested = hook_autodiff(:(autodiff(Reverse, y -> only(autodiff(Forward, f, Duplicated(y, w))), Active, Duplicated(y, dy))))
    @test occursin("autodiff(Forward", string(nested)) && count("Shrink.capture", string(nested)) == 1   # inner call untouched

    @test capture(:(x), +, 1, 2) == 3                            # not Enzyme's: called unchanged
    @test capture(e.args[2], autodiff, Reverse, sum, Active, Const([1.0])) === ((nothing,),)
    x = [1.0]
    err = try capture(e.args[2], autodiff, Reverse, (x, p) -> x[5] * p, Active, Duplicated(x, zero(x)), Const(2.0)); nothing catch err; err end
    @test err isa Captured && err.err isa BoundsError
    c = Call(err, Main)
    @test c.ret === Active && [a.kind for a in c.args] == [:Duplicated, :Const]
    @test [a.name for a in c.args] == [:x, :p]
    @test call_source(c) == "autodiff(Reverse, f, Active, Duplicated(x, dx), Const(p))"
    @test primal_source(c) == "f(x, p)"
    @test primal_source(c; copy = true) == "f(deepcopy(x), deepcopy(p))"
    @test Shrink.oneline("a\n    b\nc") == "a b c" && Shrink.oneline("abcdef", 3) == "abc…"

    @test capture_gradient(:(x), (m, f, x) -> 42, Reverse, sin, 1.0) == 42
    bad(x) = x[7]
    err = try capture_gradient(:(gradient(Reverse, bad, x)), Enzyme.gradient, Reverse, bad, [1.0]); nothing catch err; err end
    @test err isa Captured && err.err isa BoundsError
    @test err.expr == :(autodiff(Reverse, bad, Active, Duplicated(x))) && err.tail[1] === Active && err.tail[2] isa Duplicated
    @test call_source(Call(err, Main)) == "autodiff(Reverse, bad, Active, Duplicated(x, dx))"
end

@testset "lift_lambda" begin
    p = program()
    @test Shrink.lift_lambda(p) === p
    lam = Call(:autodiff, Reverse, :((x, p) -> sum(x) * p), Active, p.call.args)
    q = Shrink.lift_lambda(Program(p.defs, p.consts, lam, p.values))
    @test q.call.f == :entry && Shrink.fname(q.defs[end]) == :entry
    @test fparams(q.defs[end]) == [:x, :p] && entry(q) == length(q.defs)
    @test call_source(q.call) == "autodiff(Reverse, entry, Active, Duplicated(x, dx), Const(p))"
end

@testset "mode_source" begin
    @test mode_source(Reverse) == "Reverse"
    @test mode_source(set_runtime_activity(Reverse), "Enzyme.") == "Enzyme.set_runtime_activity(Enzyme.Reverse)"
    @test Meta.parse(mode_source(ReverseHolomorphic)) isa Expr   # exotic: falls back to repr
end

@testset "render" begin
    p = program()
    setup, call, store = render(p, "\"data.jls\"")
    @test isempty(store.entries)
    @test occursin("function f(x, p)", setup)
    @test endswith(setup, "x = [1.0, 2.0, 3.0, 4.0]\ndx = zero(x)\np = 3.0\n")
    @test call == "autodiff(Reverse, f, Active, Duplicated(x, dx), Const(p))"
    path = joinpath(mktempdir(), "repro.jl")
    write_program(path, p, Class(:X, "why"), "orig")
    @test Meta.parseall(read(path, String)) isa Expr
    write_program(path, p, Class(:X, "why\nnot"), "autodiff(Reverse, x -> begin\n    f(x)\nend, Active, x)")
    src = read(path, String)
    header = split(src, "\n\n"; limit = 2)[1]
    @test all(l -> startswith(l, "#"), split(header, '\n'))   # header stays a comment
    @test occursin("# Failure: X: why\n#          not\n", src)
    @test occursin("# Reduced from: autodiff(Reverse, x -> begin f(x) end, Active, x)\n", src)
    @test Meta.parseall(src) isa Expr
end

@testset "instrument" begin
    p = program(values = false)
    defs, keys = instrument(p)
    @test length(keys) == 4                                   # a, b, c in f; t in g
    @test any(k -> startswith(k, "f@"), values(keys))
    @test occursin("Main.Shrink.record!(\"f@4#1.a\", a)", string(defs[entry(p)]))
    @test p.defs[entry(p)] !== defs[entry(p)]                  # original untouched

    q = looped()
    defs, keys = instrument(q)
    @test length(keys) == 3                                   # s, t (inside the loop), (a, b)
    @test occursin("record!(\"h@2#3.b\", b)", string(defs[2]))

    # instrumented code can itself be differentiated: recording is inactive
    # and the instrumented statement keeps the assignment's value
    empty!(Shrink.RECORDS)
    rec(x) = begin
        y = (y = 2x; Shrink.record!("k.y", y); y)
        y * x
    end
    @test only(autodiff(Forward, rec, Duplicated, Duplicated(3.0, 1.0))) == 12.0
    @test autodiff(Reverse, rec, Active, Active(3.0))[1][1] == 12.0
    @test Shrink.RECORDS["k.y"] == 6.0
    empty!(Shrink.RECORDS)
end

@testset "search orders" begin
    @test bisect_order(7) == [4, 2, 6, 1, 3, 5, 7]
    @test bisect_order(0) == Int[]
    @test granularities(5) == [4, 2, 1]
    @test Shrink.chunks([2, 3, 5]) == [[2, 3], [5], [2], [3]]
    @test targets(1000) == [1, 500, 999] && targets(2) == [1]
    @test take(collect(reshape(1:12, 3, 4)), 2, 2) == [1 4; 2 5; 3 6]
end

@testset "structural passes" begin
    p = program()
    d = p.defs[entry(p)]
    @test length(stmts(d)) == 4

    s = remove_suffix(p)
    @test first.(s) == ["f: keep statements 1:2, return b", "f: keep statements 1:1, return sum(a)"]
    @test length(stmts(s[2].second.defs[entry(p)])) == 2
    q = s[1].second                                            # truncated after b ...
    @test first.(remove_suffix(q)) == ["f: keep statements 1:1, return sum(a)"]   # ... is not proposed again

    ph = placeholder_statements(p)
    @test first(first.(ph)) == "f: statements 1:2 of 3"        # coarsest chunk of the entry first
    q = ph[1].second
    d′ = q.defs[entry(q)]
    @test fparams(d′) == [:x, :p, :a_rec, :b_rec]              # two placeholders became inputs
    @test [a.kind for a in q.call.args] == [:Duplicated, :Const, :Duplicated, :Active]
    @test stmts(d′)[1] == :(a = a_rec) && stmts(d′)[2] == :(b = b_rec)
    callee = findfirst(x -> startswith(x, "g:"), first.(ph))
    @test callee !== nothing && isempty(ph[callee].second.consts)   # g's `t` has no recorded value: deleted

    @test isempty(remove_unused_args(program()))               # both x and p are used
    u = remove_unused_args(q)                                   # x, p, a_rec, b_rec after placeholders
    @test first.(u) == ["f: drop unused arguments x, p", "f: drop unused argument x", "f: drop unused argument p"]
    @test fparams(u[1].second.defs[entry(u[1].second)]) == [:a_rec, :b_rec]
    @test [a.name for a in u[1].second.call.args] == [:a_rec, :b_rec]

    @test isempty(eliminate_dead_code(p))                       # a, b, c each feed the next statement
    dc = eliminate_dead_code(q)                                 # a = a_rec is dead once b = b_rec
    @test first(first.(dc)) == "f: remove dead statements 1"
    @test length(stmts(dc[1].second.defs[entry(q)])) == 3
    @test size_summary(p) == "defs 5, statements 4, args 2"

    sr = simplify_result(p)
    @test first.(sr)[1:2] == ["g: return nothing", "g: return 0.0"]
    @test stmts(sr[1].second.defs[2])[end] == Expr(:return, nothing)
    @test last(stmts(sr[end].second.defs[entry(p)])) == Expr(:return, 0.0)
    # a literal return is terminal: no proposal swaps `nothing` for `0.0` and back
    @test !any(l -> startswith(l, "g:"), first.(simplify_result(sr[1].second)))
    @test !any(l -> startswith(l, "g:"), first.(simplify_result(sr[2].second)))

    r = remove_defs(p)
    @test first.(r) == ["remove definitions 2,3", "remove definition 5", "remove definition 2", "remove definition 3"]
    @test length(r[1].second.defs) == 3                        # never `using Enzyme` (1) or the entry (4)
    @test isempty(remove_unused_consts(p))
end

@testset "nested placeholders" begin
    p = looped()
    @test !entry_called_elsewhere(p)
    nodes = all_statements(p.defs[2].args[2])
    q = replace_statements(p, 2, [nodes[3]], true)              # t = x[i] * 2, inside the loop
    body = all_statements(q.defs[2].args[2])
    @test body[3] == :(t = t_rec) && fparams(q.defs[2]) == [:x, :t_rec]
    @test q.call.args[end].kind == :Active
    q = replace_statements(p, 2, [nodes[5]], true)              # a, b = (s, 2s)
    @test all_statements(q.defs[2].args[2])[5] == :((a, b) = (a_rec, b_rec))
    @test fparams(q.defs[2]) == [:x, :a_rec, :b_rec]
    q = replace_statements(p, 2, [nodes[2]], true)              # the loop: deleted
    @test length(all_statements(q.defs[2].args[2])) == 3
    @test first(first.(placeholder_statements(p))) == "h: statements 1:4 of 5"

    r = looped(extra = Any[:(callit(x) = h(x))])                 # entry is called elsewhere: arity is fixed
    @test entry_called_elsewhere(r)
    q = replace_statements(r, 2, [all_statements(r.defs[2].args[2])[3]], false)
    @test first(q.consts)[1] == :P_h_t && fparams(q.defs[2]) == [:x]
    @test isempty(remove_unused_args(r))
end

@testset "correctness check" begin
    g(x, p) = sum(x .* p)
    x = [1.0, 2.0]
    @test checked_autodiff(Reverse, g, Active, Duplicated(x, zero(x)), Const(3.0)) == ((nothing, nothing),)
    @test checked_autodiff(Reverse, g, Active, Duplicated(x, zero(x)), Const(3.0); elementwise = true) == ((nothing, nothing),)
    wrong(x, p) = sum(x .* p)
    Enzyme.EnzymeRules.inactive(::typeof(wrong), args...) = nothing
    @test_throws WrongGradient checked_autodiff(Reverse, wrong, Active, Duplicated(x, zero(x)), Const(3.0))
    @test checked_autodiff(Reverse, wrong, Const, Duplicated(x, zero(x)), Const(3.0)) !== nothing   # Const return: not checked
    @test_throws ErrorException checked_autodiff(Forward, g, Duplicated(x, zero(x)), Const(3.0))
    dx = [5.0, 5.0]                                             # pre-filled shadow: only the increment is compared
    @test checked_autodiff(Reverse, g, Active, Duplicated(x, dx), Const(3.0)) !== nothing
    @test dx == [8.0, 8.0]

    Shrink.CHECK[] = :correctness                                  # capture snapshots arguments before the call runs
    dx = zero(x)
    half(x, p) = g(x, p) + wrong(x, p)                          # Enzyme sees p per entry, truth is 2p
    err = try capture(:(autodiff(Reverse, half, Active, Duplicated(x, dx), Const(3.0))), autodiff, Reverse, half, Active, Duplicated(x, dx), Const(3.0)); nothing catch e; e end
    Shrink.CHECK[] = :error
    @test err isa Captured && err.err isa WrongGradient
    @test err.tail[2].dval == [0.0, 0.0] && dx == [3.0, 3.0]
    c = program().call
    @test call_source(c, :correctness) == "checked_autodiff(Reverse, f, Active, Duplicated(x, dx), Const(p))"
    setup, _, _ = render(program(), "\"d.jls\"", :correctness)
    @test occursin("function checked_autodiff", setup)
end

@testset "invocation passes" begin
    p = program(mode = set_runtime_activity(ReverseWithPrimal))
    @test first.(simplify_mode(p)) == ["clear runtime activity"]
    @test only(simplify_return(p)).second.call.ret === Const
    @test only(drop_activity(p)).second.call.args[1].kind == :Const
    s = shrink_arrays(p)
    @test first.(s) == ["dims with extent 4 → 1", "dims with extent 4 → 2", "dims with extent 4 → 3"]
    a = s[1].second.call.args[1]
    @test a.val.data == [1.0] && a.shadows[1].data == [0.0]
end

@testset "oracle precondition" begin
    # a candidate whose primal does not run is :setup, whatever autodiff would say
    setup = "using Enzyme\nf(x) = (x[2]; x[1])\nx = [1.0]\ndx = zero(x)\n"
    call = "autodiff(Reverse, f, Active, Duplicated(x, dx))"
    log = tempname()
    @test Shrink.worker_query("q1", setup, call, log, "f(deepcopy(x))").kind == :setup
    @test Shrink.worker_query("q2", setup, call, log).kind == :BoundsError
    # the primal runs on copies: a mutating primal does not change what the call sees
    setup = "using Enzyme\nf(x) = (x[1] += 1; x[1]^2)\nx = [1.0]\ndx = zero(x)\n"
    call = "autodiff(Reverse, f, Active, Duplicated(x, dx)); dx[1] == 4.0 || error(\"saw mutated x\")"
    @test Shrink.worker_query("q3", setup, call, log, "f(deepcopy(x))") == PASS
end

# Slow: runs the whole pipeline on the examples.  SHRINK_E2E=1 julia --project -e 'using Pkg; Pkg.test()'
if get(ENV, "SHRINK_E2E", "") != ""
    @testset "end to end: $example" for (example, check, target, entryname, maxstmts) in
            [("runtime_activity", :error, "EnzymeRuntimeActivityError", :pick, 2),
             ("wrong_gradient", :correctness, "WrongGradient", :loss, 3)]
        dir = mktempdir()
        script = joinpath(dir, "script.jl")
        cp(joinpath(@__DIR__, "..", "examples", example, "script.jl"), script)
        repro = minify(script; check, workers = 2)
        @test repro !== nothing
        defs = parse_script(repro)
        d = only(filter(d -> Shrink.isfdef(d) && Shrink.fname(d) == entryname, defs))
        @test length(all_statements(d.args[2])) <= maxstmts
        log = joinpath(dir, "repro.log")
        run(pipeline(ignorestatus(`$(Base.julia_cmd()) --project=$(Base.active_project()) $repro`); stdout = log, stderr = log))
        @test occursin(target, read(log, String))
    end
end
