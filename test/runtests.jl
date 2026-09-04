using Test, Enzyme, MWE
using MWE: Class, PASS, classify, classify_run, Value, available, render, Store,
           Program, Call, Arg, Captured, parse_script, normalize, hook_autodiff, capture,
           entry, stmts, fparams, is_assignment, uses, call_source, primal_source, mode_source,
           instrument, bisect_order, granularities, remove_suffix, placeholder_statements,
           remove_unused_args, remove_defs, remove_unused_consts, drop_activity, shrink_arrays,
           simplify_mode, simplify_return, targets, take, write_program,
           eliminate_dead_code, checked_autodiff, WrongGradient, size_summary

const SCRIPT = joinpath(@__DIR__, "fixtures", "script.jl")

"A program for `SCRIPT` with the call captured by hand and values recorded by hand."
function program(; mode = Reverse, ret = Active, values = true)
    defs = parse_script(SCRIPT)
    x = [1.0, 2.0, 3.0, 4.0]
    call = Call(:autodiff, mode, :f, ret,
                [Arg(:Duplicated, :x, Value(x, Main), [Value(zero(x), Main)]),
                 Arg(:Const, :p, Value(3.0, Main), Value[])])
    p = Program(defs, Pair{Symbol,Value}[], call, IdDict{Any,Value}())
    if values
        d = defs[entry(p)]
        for (i, s) in enumerate(stmts(d))
            is_assignment(s) && (p.values[s] = Value(i == 1 ? x .* 2 : Float64(i), Main))
        end
    end
    return p
end

@testset "classify" begin
    c = classify(BoundsError([1.0], 3))
    @test c.kind == :BoundsError
    @test c == Class(:BoundsError, "some other text")   # detail is display-only
    @test c != PASS
    @test classify_run(() -> 1) == PASS
    @test classify_run(() -> error("boom")).kind == :ErrorException
    @test classify(LoadError("f.jl", 1, LoadError("g.jl", 2, BoundsError([1.0], 3)))).kind == :BoundsError
end

@testset "values" begin
    v = Value([1.0, 2.0], Main)
    @test v.data == [1.0, 2.0] && v.text === nothing && available(v)
    @test render(v, "x", Store()) == "[1.0, 2.0]"
    big = Value(rand(1000), Main)
    s = Store()
    @test render(big, "x", s) == "STORE[\"x\"]" && haskey(s.entries, "x")
    t = Value((a = 1, b = "s"), Main)
    @test t.data === nothing && t.text == "(a = 1, b = \"s\")"
    @test !available(Value(nothing, nothing, "Foo"))
end

@testset "parse_script" begin
    defs = parse_script(SCRIPT)
    @test length(defs) == 6                                  # include inlined, docstring stripped
    @test all(d -> d isa Expr, defs)
    @test count(MWE.isfdef, defs) == 3
    @test Base.remove_linenums!(normalize(:(g(x) = x + 1))) == Base.remove_linenums!(:(function g(x) x + 1 end))
    @test MWE.fname(normalize(:(h(x::T) where {T} = x))) == :h
    @test fparams(defs[4]) == [:x, :p]
    @test uses(:p, defs[4]) && !uses(:nope, defs[4])
end

@testset "capture" begin
    e = :(r = autodiff(Reverse, f, Active, Duplicated(x, dx), Const(p)))
    h = hook_autodiff(e)
    @test h.args[2].args[1] == GlobalRef(MWE, :capture)
    @test h.args[2].args[2] == QuoteNode(e.args[2])

    @test capture(e, Reverse, sum, Active, Const([1.0])) === ((nothing,),)
    x = [1.0]
    err = try capture(e.args[2], Reverse, (x, p) -> x[5] * p, Active, Duplicated(x, zero(x)), Const(2.0)); nothing catch err; err end
    @test err isa Captured && err.err isa BoundsError
    c = Call(err, Main)
    @test c.ret === Active && [a.kind for a in c.args] == [:Duplicated, :Const]
    @test [a.name for a in c.args] == [:x, :p]
    @test call_source(c) == "autodiff(Reverse, f, Active, Duplicated(x, dx), Const(p))"
    @test primal_source(c) == "f(x, p)"
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
end

@testset "instrument" begin
    p = program(values = false)
    defs, keys = instrument(p)
    @test length(keys) == 4                                   # a, b, c in f; t in g
    @test any(k -> startswith(k, "f@"), values(keys))
    @test occursin("Main.MWE.record!", string(defs[entry(p)]))
    @test p.defs[entry(p)] !== defs[entry(p)]                  # original untouched
end

@testset "search orders" begin
    @test bisect_order(7) == [4, 2, 6, 1, 3, 5, 7]
    @test bisect_order(0) == Int[]
    @test granularities(5) == [4, 2, 1]
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
    @test first(first.(ph)) == "f: statements 1:2"           # coarsest chunk of the entry first
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

    dc = eliminate_dead_code(p)
    @test isempty(dc)                                           # a, b, c each feed the next statement
    dc = eliminate_dead_code(q)                                 # a = a_rec is dead once b = b_rec
    @test first(first.(dc)) == "f: remove dead statements 1"
    @test length(stmts(dc[1].second.defs[entry(q)])) == 3
    @test size_summary(p) == "defs 6, statements 4, args 2"
    r = remove_defs(p)
    @test first(first.(r)) == "remove definitions 1,2,3,5" && length(r[1].second.defs) == 2
    @test isempty(remove_unused_consts(p))
end

@testset "correctness check" begin
    g(x, p) = sum(x .* p)
    x = [1.0, 2.0]
    @test checked_autodiff(Reverse, g, Active, Duplicated(x, zero(x)), Const(3.0)) == ((nothing, nothing),)
    wrong(x, p) = sum(x .* p)
    Enzyme.EnzymeRules.inactive(::typeof(wrong), args...) = nothing
    @test_throws WrongGradient checked_autodiff(Reverse, wrong, Active, Duplicated(x, zero(x)), Const(3.0))
    @test checked_autodiff(Reverse, wrong, Const, Duplicated(x, zero(x)), Const(3.0)) !== nothing   # Const return: not checked
    @test_throws ErrorException checked_autodiff(Forward, g, Duplicated(x, zero(x)), Const(3.0))
    dx = [5.0, 5.0]                                             # pre-filled shadow: only the increment is compared
    @test checked_autodiff(Reverse, g, Active, Duplicated(x, dx), Const(3.0)) !== nothing
    @test dx == [8.0, 8.0]

    MWE.CHECK[] = :correctness                                  # capture snapshots arguments before the call runs
    dx = zero(x)
    half(x, p) = g(x, p) + wrong(x, p)                          # Enzyme sees p per entry, truth is 2p
    err = try capture(:(autodiff(Reverse, half, Active, Duplicated(x, dx), Const(3.0))), Reverse, half, Active, Duplicated(x, dx), Const(3.0)); nothing catch e; e end
    MWE.CHECK[] = :error
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
