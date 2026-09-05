"""
A recorded value: numeric `data` (numbers and `Array`s; other arrays are collected) or a source literal `text`; `type` is what it was.

Only these two representations exist because values cross process boundaries: an object of
a user-defined type is a different type in every candidate module and cannot be serialised
between them, but a number, an `Array` of numbers, or a source literal that `repr`
round-trips can.  Anything else (`data === nothing && text === nothing`) is not available
as a placeholder.
"""
struct Value
    data::Union{Nothing,Number,Array{<:Number}}
    text::Union{Nothing,String}
    type::String
end

"Literal source longer than this is not embedded; the value is stored instead."
const LITERAL_MAX_CHARS = 400

"Capture the live object `v`, with names written relative to module `m`."
function Value(v, m::Module)
    type = sprint(show, typeof(v); context = :module => m)
    v isa Number && return Value(v, nothing, type)
    v isa AbstractArray{<:Number} && return Value(v isa Array ? v : collect(v), nothing, type)
    s = sprint(show, v; context = :module => m)
    ok = length(s) <= LITERAL_MAX_CHARS && (try Meta.parse(s); true catch; false end)
    return Value(nothing, ok ? s : nothing, type)
end

available(v::Value) = v.data !== nothing || v.text !== nothing

"Copy of `v` with `g` applied to its numeric data; values without data are returned unchanged."
mapdata(g, v::Value) = v.data === nothing ? v : Value(g(v.data), nothing, v.type)

"`\"  # was T\"` when the data no longer has the type it was captured with, else empty."
function type_note(v::Value)
    v.data === nothing && return ""
    string(typeof(v.data)) == v.type ? "" : "   # was $(v.type)"
end

"Numeric values too large to embed, keyed by name; `ref` is the source expression naming the loaded store."
struct Store
    ref::String
    entries::Dict{String,Any}
end
Store(ref = "STORE") = Store(ref, Dict{String,Any}())

"Source expression for `v`, stored under `name` when too large for a literal."
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
