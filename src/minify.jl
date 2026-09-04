"""
    minify(file; check = :error, workers = default_workers(), timeout = 600.0) -> path or nothing

Reduce the first failing `autodiff` call in the script `file` to a minimal
reproducer.  Nothing in the script needs to change.

The script is run on a worker with every `autodiff` call hooked; the first
one that throws is captured with its live arguments.  The primal is then
run once to record intermediate values, and the program is shrunk by the
passes in [`PASSES`](@ref) while the failure class is unchanged.  Every
accepted step is written to `mwe_<time>/checkpoints/`, and the result to
`mwe_<time>/repro.jl`, next to the script.  Returns the repro path, or
`nothing` if no `autodiff` call failed.

`check` selects the failure being reduced: `:error`, the call throws; or
`:correctness`, the gradient disagrees with finite differences (see
[`checked_autodiff`](@ref)).  `workers` is the number of concurrent worker
processes and `timeout` the seconds allowed per candidate.
"""
function minify(file::AbstractString; check::Symbol = :error, workers = default_workers(), timeout = 600.0)
    check in (:error, :correctness) || error("check must be :error or :correctness")
    file = abspath(file)
    isfile(file) || error("no such file: $file")
    dir = mkpath(joinpath(dirname(file), "mwe_" * Libc.strftime("%Y%m%d-%H%M%S", time())))
    mkpath(joinpath(dir, "checkpoints"))
    defs = parse_script(file)

    println("starting $workers workers")
    pool = Pool(dir; workers, timeout)
    try
        r = remote(pool, 1, worker_capture, defs, joinpath(dir, "capture.log"), check)
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

        p = Program(defs[1:k-1], Pair{Symbol,Value}[], call, IdDict{Any,Value}())
        oracle = Oracle(pool, dir, check)
        target = sanity(p, oracle)
        p = record(p, pool, dir)

        p = minimize(p, oracle, target, dir, original)
        repro = joinpath(dir, "repro.jl")
        write_program(repro, p, target, original, check)
        println("\nrepro written to ", repro)
        return repro
    finally
        shutdown(pool)
    end
end

"""
    Oracle(pool, dir)

Answers "what class does this program fail with?" by rendering it to a
script in `dir` and running it on a worker.  Calling it on a vector of
programs runs up to one per worker concurrently and returns `(id, class)`
pairs in the same order.
"""
struct Oracle
    pool::Pool
    dir::String
    check::Symbol
end

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

function query(o::Oracle, slot::Int, p::Program)
    o.pool.count += 1
    id = lpad(o.pool.count, 4, '0')
    base = joinpath(o.dir, id)
    setup, call, store = render(p, repr(base * "_data.jls"), o.check)
    isempty(store.entries) || serialize(base * "_data.jls", store.entries)
    write(base * ".jl", setup, call, "\n")

    class = remote(o.pool, slot, worker_query, id, setup, call, base * ".log")
    class.kind == :crash && (class = Class(:crash, crash_detail(read(base * ".log", String))))
    return id, class
end

"The assertion or LLVM error line from a dead worker's log, if there is one."
function crash_detail(log)
    for line in Iterators.reverse(split(log, '\n'))
        occursin(r"Assertion|LLVM ERROR|EnzymeInternalError|signal \(\d+\)", line) &&
            return String(strip(line))
    end
    return "worker died"
end

"""
    sanity(p, oracle) -> Class

Run the original program once on every worker.  It must fail, and fail the
same way each time: a nondeterministic failure cannot be bisected.
"""
function sanity(p::Program, oracle::Oracle)
    classes = last.(oracle(fill(p, length(oracle.pool.pids))))
    target = classes[1]
    target == PASS && error("the captured call does not fail when run in isolation; the failure depends on something outside the script's top-level definitions")
    target.kind == :setup && error("the program does not run in isolation: $target")
    all(==(target), classes) || error("failure is not deterministic across $(length(classes)) runs: $(join(classes, "; "))")
    println("target: ", target)
    return target
end

"""
    minimize(p, oracle, target, dir, original) -> Program

Sweep the passes in order, running each to its own fixpoint, until a whole
sweep accepts nothing.  Each accepted program is written to
`dir/checkpoints/` so partial progress is always a valid reproducer.
"""
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

"""
    first_accepted(proposals, oracle, target) -> Union{Program, Nothing}

Query `proposals` in batches of one per worker and return the first, in
proposal order, whose class equals `target`.  Batches are run in full so
the result does not depend on which worker finishes first.
"""
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
