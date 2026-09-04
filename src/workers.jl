"""
    Pool(dir; workers = default_workers(), project = Base.active_project(), timeout = 600.0)

Persistent `Distributed` workers that run candidates.  Each worker loads
Enzyme once; every candidate is then evaluated in a fresh module of its
own, so removed definitions are really gone and Enzyme's compiler is
already warm.  A worker that dies (LLVM assertion, segfault) or exceeds
`timeout` (killed) is replaced, and each worker is recycled after
[`RECYCLE_AFTER`](@ref) candidates to bound memory growth from modules
that can never be freed.

`dir` receives the per-candidate scripts and logs.
"""
mutable struct Pool
    dir::String
    project::String
    timeout::Float64
    pids::Vector{Int}
    served::Vector{Int}
    count::Int
end

"Half the cores: an Enzyme compile is single-threaded but needs a few GB."
default_workers() = max(1, Sys.CPU_THREADS ÷ 2)

"Candidates a worker runs before being replaced."
const RECYCLE_AFTER = 40

function Pool(dir; workers = default_workers(), project = Base.active_project(), timeout = 600.0)
    pool = Pool(dir, project, timeout, Int[], Int[], 0)
    append!(pool.pids, spawn(pool, workers))
    append!(pool.served, zeros(Int, workers))
    return pool
end

"Start `n` workers with Enzyme and MWE loaded."
function spawn(pool::Pool, n::Int)
    pids = addprocs(n; exeflags = "--project=$(pool.project)")
    @sync for pid in pids
        @async remotecall_wait(Core.eval, pid, Main, :(using Enzyme, MWE))
    end
    return pids
end

function replace!(pool::Pool, slot::Int)
    pool.pids[slot] = only(spawn(pool, 1))
    pool.served[slot] = 0
end

shutdown(pool::Pool) = rmprocs(pool.pids; waitfor = 5)

"""
    remote(pool, slot, f, args...)

Run `f(args...)` on the worker in `slot`.  Returns its result, or a `Class`
of kind `:crash` or `:timeout` if the worker died or had to be killed; in
both cases the slot gets a fresh worker.  Errors raised by `f` itself are
rethrown.
"""
function remote(pool::Pool, slot::Int, f, args...)
    if pool.served[slot] >= RECYCLE_AFTER
        replace!(pool, slot)
    end
    pool.served[slot] += 1
    pid = pool.pids[slot]
    task = @async try
        remotecall_fetch(f, pid, args...)
    catch err
        err
    end
    if timedwait(() -> istaskdone(task), pool.timeout; pollint = 0.1) != :ok
        kill(Distributed.worker_from_id(pid).config.process, Base.SIGKILL)
        wait(task)
        replace!(pool, slot)
        return Class(:timeout, "still running after $(pool.timeout)s")
    end
    r = fetch(task)
    if r isa ProcessExitedException
        replace!(pool, slot)
        return Class(:crash, "")
    elseif r isa Exception
        throw(r)
    end
    return r
end

# ---- functions that run on the workers --------------------------------------

"Run `f()` with stdout and stderr going to `logfile`."
function logged(f, logfile::String)
    open(logfile, "w") do log
        redirect_stdio(f; stdout = log, stderr = log)
    end
end

"""
    worker_capture(defs, logfile, check) -> (k, Call, Class) or nothing

Evaluate the script's top-level expressions in the worker's `Main`, with
`autodiff` calls hooked (see [`CHECK`](@ref) for `check`).  Stops at the
first expression `k` during which an `autodiff` call failed and returns
the captured call; `nothing` if the whole script ran without one.
"""
function worker_capture(defs::Vector, logfile::String, check::Symbol)
    CHECK[] = check
    logged(logfile) do
        for (k, e) in enumerate(defs)
            try
                Core.eval(Main, hook_autodiff(e))
            catch err
                err isa Captured || rethrow()
                return k, Call(err, Main), classify(err.err)
            end
        end
        return nothing
    end
end

"""
    worker_query(id, setup, call, logfile) -> Class

Evaluate a candidate in a fresh module: `setup` first (a failure there is
`:setup`, never the target class), then the `call`.
"""
function worker_query(id::String, setup::String, call::String, logfile::String)
    logged(logfile) do
        m = Module(Symbol(:Candidate_, id))
        try
            Base.include_string(m, setup, "candidate_$id.jl")
        catch err
            return Class(:setup, classify(err).detail)
        end
        classify_run(() -> Base.include_string(m, call, "candidate_$id.jl"))
    end
end

"First recorded value of every instrumented statement, by key; see [`record!`](@ref)."
const RECORDS = Dict{String,Any}()

"""
    record!(key, v)

Called by instrumented statements (see [`instrument`](@ref)); keeps the
first value seen for `key` so that statements inside loops record their
first iteration.
"""
record!(key::String, v) = (haskey(RECORDS, key) || (RECORDS[key] = v); v)

"Values with more bytes than this are not recorded."
const RECORD_MAX_BYTES = 64 * 2^20

"""
    worker_record(id, setup, primal, logfile) -> (Class, Dict{String,Value})

Run the instrumented primal once in a fresh module and return the recorded
values, captured relative to that module.
"""
function worker_record(id::String, setup::String, primal::String, logfile::String)
    empty!(RECORDS)
    logged(logfile) do
        m = Module(Symbol(:Record_, id))
        Base.include_string(m, setup, "record_$id.jl")
        class = classify_run(() -> Base.include_string(m, primal, "record_$id.jl"))
        values = Dict{String,Value}()
        for (key, v) in RECORDS
            big = v isa Array && Base.summarysize(v) > RECORD_MAX_BYTES
            values[key] = big ? Value(nothing, nothing, string(typeof(v))) : Value(v, m)
        end
        empty!(RECORDS)
        return class, values
    end
end
