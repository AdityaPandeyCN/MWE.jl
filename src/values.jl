"""
    Value

A recorded value as the reducer holds it.  The reducer only ever holds two
kinds of data, so that values never have to cross process boundaries as
objects of user-defined types (which would be different types in every
candidate module):

- `data`: a number or an array of numbers, kept in memory and shrunk in
  place; written to the candidate as a literal when small, otherwise via a
  Serialization store next to it.
- `text`: a source literal (`Point(1.0, 2.0)`), rendered in the module the
  value was created in.

A value with neither (`data === nothing && text === nothing`) could not be
captured and cannot be used as a placeholder.  `type` is for display.
"""
struct Value
    data::Union{Nothing,Number,Array{<:Number}}
    text::Union{Nothing,String}
    type::String
end

"Literal source longer than this is not embedded; the value is stored instead."
const LITERAL_MAX_CHARS = 400

"""
    Value(v, m::Module) -> Value

Capture the live object `v`, with type and literal names written relative
to module `m`.
"""
function Value(v, m::Module)
    type = sprint(show, typeof(v); context = :module => m)
    (v isa Number || v isa Array{<:Number}) && return Value(v, nothing, type)
    s = sprint(show, v; context = :module => m)
    ok = length(s) <= LITERAL_MAX_CHARS && (try Meta.parse(s); true catch; false end)
    return Value(nothing, ok ? s : nothing, type)
end

available(v::Value) = v.data !== nothing || v.text !== nothing

"Copy of `v` with `g` applied to its numeric data; values without data are returned unchanged."
mapdata(g, v::Value) = v.data === nothing ? v : Value(g(v.data), nothing, v.type)

"""
    Store

Collects numeric values too large to embed as literals while a candidate is
being rendered.  `ref` is the source expression the candidate uses to name
the loaded store (e.g. `STORE`); `entries` are written to a Serialization
file by the caller if non-empty.
"""
struct Store
    ref::String
    entries::Dict{String,Any}
end
Store(ref = "STORE") = Store(ref, Dict{String,Any}())

"""
    render(v::Value, name::String, store::Store) -> String

Source expression for `v`, using `name` as its key in `store` if it has to
be stored.
"""
function render(v::Value, name::String, store::Store)
    v.text === nothing || return v.text
    v.data === nothing && error("value $name of type $(v.type) was not captured")
    s = repr(v.data)
    length(s) <= LITERAL_MAX_CHARS && return s
    store.entries[name] = v.data
    return "$(store.ref)[$(repr(name))]"
end

"Whether a shadow can be written as `zero(primal)`."
iszero_shadow(v::Value) = v.data isa AbstractArray && all(iszero, v.data)
