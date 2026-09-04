"""
    Class(kind::Symbol, detail::String)

Identity of a failure, as compared by the oracle.  Two runs are the same
failure when their `kind` is equal; `detail` (the first line of the error
message) is kept for display only.

Only the exception type takes part in equality on purpose: messages contain
SSA names, pointers, line numbers and array sizes, all of which legitimately
change while the program is being shrunk, and matching on them would reject
valid reductions.

Besides exception type names, `kind` can be `:pass` (no error), `:setup`
(the candidate failed before reaching the call, e.g. a removed definition
was still needed), `:crash` (the worker died: LLVM assertion, segfault) or
`:timeout`.
"""
struct Class
    kind::Symbol
    detail::String
end

Base.:(==)(a::Class, b::Class) = a.kind == b.kind
Base.show(io::IO, c::Class) = print(io, c.kind, isempty(c.detail) ? "" : ": " * c.detail)

"The `Class` of a run that raised no error."
const PASS = Class(:pass, "")

"""
    classify(err::Exception) -> Class

The `Class` of a caught exception: its unparameterised type name plus the
first line of its message.
"""
function classify(err::Exception)
    while err isa LoadError            # include_string wraps what the candidate threw
        err = err.error
    end
    msg = Base.invokelatest(sprint, showerror, err)   # showerror methods defined by the candidate itself
    line = strip(first(split(msg, '\n'; limit = 2)))
    Class(nameof(typeof(err)), first(line, 160))
end

"""
    classify_run(thunk) -> Class

Run `thunk()` and classify the outcome: [`PASS`](@ref) or the class of the
exception it threw.
"""
classify_run(thunk) = try
    thunk()
    PASS
catch err
    classify(err)
end
