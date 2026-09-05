"""
    minify(file; check = :error, workers = default_workers(), timeout = 600.0) -> path or nothing

Reduce the first failing `autodiff` call in the script `file` to a minimal reproducer,
written to `shrink_<time>/repro.jl` next to it; `nothing` if no call failed.  `check` is
`:error` (the call throws) or `:correctness` (the gradient disagrees with finite differences).
"""
function minify(file::AbstractString; check::Symbol = :error, workers = default_workers(), timeout = 600.0)
    check in (:error, :correctness) || error("check must be :error or :correctness")
    file = abspath(file)
    isfile(file) || error("no such file: $file")
    dir = mkpath(joinpath(dirname(file), "shrink_" * Libc.strftime("%Y%m%d-%H%M%S", time())))
    mkpath(joinpath(dir, "checkpoints"))
    defs = parse_script(file)

    println("starting $workers workers")
    pool = Pool(dir; cwd = dirname(file), workers, timeout)
    try
        r = try
            remote(pool, 1, worker_capture, defs, joinpath(dir, "capture.log"), check)
        catch err
            err isa RemoteException || rethrow()
            error("the script failed before any autodiff call could be captured:\n" *
                  sprint(showerror, err.captured.ex) * "\nsee $(joinpath(dir, "capture.log"))")
        end
        if r isa Class
            error("the script $(r.kind == :crash ? "crashed" : "timed out") before a failing autodiff was found; see $(joinpath(dir, "capture.log"))")
        elseif r === nothing
            println("no autodiff call failed; nothing to reduce")
            return nothing
        end
        k, call, class0 = r
        original = call_source(call)
        println("captured at top-level expression $k: ", original)
        println("  ", class0)

        p = lift_lambda(Program(defs[1:k-1], Pair{Symbol,Value}[], call, IdDict{Any,Vector{Value}}()))
        target = sanity(p, Oracle(pool, dir, check))
        p, primal_ok = record(p, pool, dir)
        oracle = Oracle(pool, dir, check, primal_ok)

        p = minimize(p, oracle, target, dir, original)
        repro = joinpath(dir, "repro.jl")
        write_program(repro, p, target, original, check)
        println("\nrepro written to ", repro)
        return repro
    finally
        shutdown(pool)
    end
end

"Runs candidates on the pool and returns their failure classes; with `primal`, a candidate whose primal fails is `:setup`."
struct Oracle
    pool::Pool
    dir::String
    check::Symbol
    primal::Bool
end
Oracle(pool::Pool, dir::String, check::Symbol) = Oracle(pool, dir, check, false)

function (o::Oracle)(programs::AbstractVector{Program})
    results = Vector{Tuple{String,Class}}(undef, length(programs))
    slots = length(o.pool.pids)
    for chunk in Iterators.partition(eachindex(programs), slots)
        @sync for (slot, i) in enumerate(chunk)
            @async results[i] = query(o, slot, programs[i])
        end
    end
    return results
end

"Run one candidate; anything unexpected is `:setup`, so one bad candidate cannot abort the run."
function query(o::Oracle, slot::Int, p::Program)
    o.pool.count += 1
    id = lpad(o.pool.count, 4, '0')
    base = joinpath(o.dir, id)
    try
        setup, call, store = render(p, repr(base * "_data.jls"), o.check)
        isempty(store.entries) || serialize(base * "_data.jls", store.entries)
        write(base * ".jl", setup, call, "\n")

        primal = o.primal ? primal_source(p.call; copy = true) : nothing
        class = remote(o.pool, slot, worker_query, id, setup, call, base * ".log", primal)
        if class.kind == :crash
            detail = crash_detail(read(base * ".log", String))
            class = Class(:crash, normalize_key(detail), detail)
        end
        return id, class
    catch err
        msg = sprint(showerror, err)
        @warn "candidate $id could not be run" exception = msg
        return id, Class(:setup, "internal error: " * first(msg, 160))
    end
end

"The assertion or LLVM error line from a dead worker's log, if there is one."
function crash_detail(log)
    for line in Iterators.reverse(split(log, '\n'))
        occursin(r"Assertion|LLVM ERROR|EnzymeInternalError|signal \(\d+\)", line) &&
            return String(strip(line))
    end
    return "worker died"
end

"Run the original on every worker: it must fail, and deterministically."
function sanity(p::Program, oracle::Oracle)
    classes = last.(oracle(fill(p, length(oracle.pool.pids))))
    target = classes[1]
    target == PASS && error("the captured call does not fail when run in isolation; the failure depends on something outside the script's top-level definitions")
    target.kind == :setup && error("the program does not run in isolation: $target")
    all(==(target), classes) || error("failure is not deterministic across $(length(classes)) runs: $(join(classes, "; "))")
    println("target: ", target)
    return target
end

"Run the passes to a fixpoint, checkpointing each accepted program to `dir/checkpoints/`."
function minimize(p::Program, oracle::Oracle, target::Class, dir::String, original::String)
    checkpoint(p) = write_program(joinpath(dir, "checkpoints", lpad(oracle.pool.count, 4, '0') * ".jl"), p, target, original, oracle.check)
    checkpoint(p)
    println("start: ", size_summary(p))
    while true
        changed = false
        for pass in PASSES
            while (next = first_accepted(pass(p), oracle, target)) !== nothing
                p = next
                changed = true
                checkpoint(p)
            end
        end
        changed || return p
    end
end

"The first proposal, in order, whose class equals `target`; `nothing` if none."
function first_accepted(proposals::Proposals, oracle::Oracle, target::Class)
    for batch in Iterators.partition(proposals, length(oracle.pool.pids))
        results = oracle(last.(batch))
        for ((label, p), (id, class)) in zip(batch, results)
            accepted = class == target
            println("[", id, "] ", rpad(label, 48), " ", accepted ? "kept  → " * size_summary(p) : "different: $(class.kind)")
            flush(stdout)
            accepted && return p
        end
    end
    return nothing
end
