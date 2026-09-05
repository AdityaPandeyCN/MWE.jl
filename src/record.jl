"Copy of `p.defs` in which every assignment also records the values it binds, plus the statement-to-key map."
function instrument(p::Program)
    keys = IdDict{Any,String}()
    recorder = :(Main.$(nameof(@__MODULE__)).record!)
    defs = map(enumerate(p.defs)) do (j, d)
        isfdef(d) && fname(d) !== nothing || return d
        n = 0
        body = map_statements(d.args[2]) do s
            is_assignment(s) || return s
            key = "$(fname(d))@$j#$(n += 1)"
            keys[s] = key
            # the block evaluates to the assigned value
            Expr(:block, s, [Expr(:call, recorder, "$key.$y", y) for y in assigned_names(s)]..., s.args[1])
        end
        with_body(d, body)
    end
    return defs, keys
end

"Run the instrumented primal on a worker and attach the recorded values to `p`; also whether the primal passed."
function record(p::Program, pool::Pool, dir::String)
    defs, keys = instrument(p)
    setup, _, store = render(with(p; defs), repr(joinpath(dir, "record_data.jls")))
    isempty(store.entries) || serialize(joinpath(dir, "record_data.jls"), store.entries)
    write(joinpath(dir, "record.jl"), setup, primal_source(p.call))

    r = remote(pool, 1, worker_record, "0", setup, primal_source(p.call), joinpath(dir, "record.log"))
    r isa Class && error("the primal could not be run for recording: $r")
    class, recorded = r
    class == PASS || @warn "the primal itself fails, before any differentiation, so candidates are not required to run as plain programs: $class"

    values = IdDict{Any,Vector{Value}}()
    for (s, key) in keys
        names = ["$key.$y" for y in assigned_names(s)]
        all(k -> haskey(recorded, k), names) && (values[s] = [recorded[k] for k in names])
    end
    return Program(p.defs, p.consts, p.call, values), class == PASS
end
