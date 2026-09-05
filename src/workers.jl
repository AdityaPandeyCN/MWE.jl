"""
Persistent workers running candidates in fresh modules; dead or timed-out workers are replaced, each recycled after `RECYCLE_AFTER` candidates.

A worker loads Enzyme once, so its compiler is warm for every candidate after the first.
Each candidate is evaluated in a module of its own so that a removed definition is really
gone rather than shadowed by an earlier candidate's.  Modules can never be freed, so a worker
is replaced after `RECYCLE_AFTER` candidates to bound memory growth.  Workers run in `cwd`,
the script's directory, so relative paths in the script resolve.
"""
mutable struct Pool
    dir::String
    cwd::String
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

function Pool(dir; cwd = pwd(), workers = default_workers(), project = Base.active_project(), timeout = 600.0)
    pool = Pool(dir, cwd, project, timeout, Int[], Int[], 0)
    append!(pool.pids, spawn(pool, workers))
    append!(pool.served, zeros(Int, workers))
    return pool
end

"Start `n` workers with Enzyme and this package loaded, in the pool's working directory."
function spawn(pool::Pool, n::Int)
    pids = addprocs(n; exeflags = "--project=$(pool.project)", dir = pool.cwd)
    @sync for pid in pids
        @async remotecall_wait(Core.eval, pid, Main, Expr(:using, Expr(:., :Enzyme), Expr(:., nameof(@__MODULE__))))
    end
    return pids
end

function respawn!(pool::Pool, slot::Int)
    pool.pids[slot] = only(spawn(pool, 1))
    pool.served[slot] = 0
end

shutdown(pool::Pool) = rmprocs(pool.pids; waitfor = 5)

"Run `f(args...)` on the worker in `slot`; a `Class` of kind `:crash` or `:timeout` if the worker died or was killed."
function remote(pool::Pool, slot::Int, f, args...)
    if pool.served[slot] >= RECYCLE_AFTER
        respawn!(pool, slot)
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
        respawn!(pool, slot)
        return Class(:timeout, "still running after $(pool.timeout)s")
    end
    r = fetch(task)
    if r isa ProcessExitedException
        respawn!(pool, slot)
        return Class(:crash)
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

"Run the script with `autodiff` hooked; `(k, Call, Class)` for the first failing call, or `nothing`."
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

"Evaluate a candidate in a fresh module: `setup`, then `primal` if given (a failure in either is `:setup`), then `call`."
function worker_query(id::String, setup::String, call::String, logfile::String, primal = nothing)
    logged(logfile) do
        m = Module(Symbol(:Candidate_, id))
        try
            Base.include_string(m, setup, "candidate_$id.jl")
        catch err
            return Class(:setup, classify(err).detail)
        end
        if primal !== nothing
            c = classify_run(() -> Base.include_string(m, primal, "candidate_$id.jl"))
            c == PASS || return Class(:setup, "the primal fails: $c")
        end
        classify_run(() -> Base.include_string(m, call, "candidate_$id.jl"))
    end
end

"First recorded value of every instrumented statement, by key; see [`record!`](@ref)."
const RECORDS = Dict{String,Any}()

"Keep the first value seen for `key`, so statements in loops record their first iteration."
record!(key::String, v) = (haskey(RECORDS, key) || (RECORDS[key] = v); v)

# recording is a side effect Enzyme runs but never differentiates
Enzyme.EnzymeRules.inactive(::typeof(record!), args...) = nothing

"Values with more bytes than this are not recorded."
const RECORD_MAX_BYTES = 64 * 2^20

"Run the instrumented primal in a fresh module and return its class and the recorded values."
function worker_record(id::String, setup::String, primal::String, logfile::String)
    empty!(RECORDS)
    logged(logfile) do
        m = Module(Symbol(:Record_, id))
        Base.include_string(m, setup, "record_$id.jl")
        class = classify_run(() -> Base.include_string(m, primal, "record_$id.jl"))
        values = Dict{String,Value}()
        for (key, v) in RECORDS
            big = v isa AbstractArray && Base.summarysize(v) > RECORD_MAX_BYTES
            values[key] = big ? Value(nothing, nothing, string(typeof(v))) : Value(v, m)
        end
        empty!(RECORDS)
        return class, values
    end
end
