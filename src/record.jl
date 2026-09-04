"""
    instrument(p::Program) -> (defs::Vector, keys::IdDict{Any,String})

Copy of `p.defs` in which every assignment statement `y = rhs` of every
named function also records `y` under a key unique to that statement, plus
the map from statement to key.  The analogue of PyTorch's `ConcreteProp`.
"""
function instrument(p::Program)
    keys = IdDict{Any,String}()
    recorder = :(Main.MWE.record!)
    defs = map(enumerate(p.defs)) do (j, d)
        isfdef(d) && fname(d) !== nothing || return d
        body = map(enumerate(stmts(d))) do (i, s)
            is_assignment(s) || return s
            key = "$(fname(d))@$j#$i"
            keys[s] = key
            Expr(:block, s, Expr(:call, recorder, key, assigned(s)))
        end
        with_body(d, body)
    end
    return defs, keys
end

"""
    record(p::Program, pool::Pool, dir) -> Program

Run the instrumented primal on a worker and attach the recorded values to
`p`'s statements.  Warns if the primal itself fails, in which case only the
statements reached are recorded.
"""
function record(p::Program, pool::Pool, dir::String)
    defs, keys = instrument(p)
    setup, _, store = render(with(p; defs), repr(joinpath(dir, "record_data.jls")))
    isempty(store.entries) || serialize(joinpath(dir, "record_data.jls"), store.entries)
    write(joinpath(dir, "record.jl"), setup, primal_source(p.call))

    r = remote(pool, 1, worker_record, "0", setup, primal_source(p.call), joinpath(dir, "record.log"))
    r isa Class && error("the primal could not be run for recording: $r")
    class, recorded = r
    class == PASS || @warn "the primal itself fails, before any differentiation: $class"

    values = IdDict{Any,Value}()
    for (s, key) in keys
        haskey(recorded, key) && (values[s] = recorded[key])
    end
    return Program(p.defs, p.consts, p.call, values)
end
