"""
Identity of a failure: `kind` (exception type, `:pass`, `:setup`, `:crash`, `:timeout`) and `key`; `detail` is for display only.

Two runs are the same failure when `kind` and `key` agree.  The key is empty for Enzyme's
typed exceptions, whose messages are mostly IR that changes with every edit, so the type
alone identifies them.  For everything else (`ErrorException`, `MethodError`,
`EnzymeInternalError`, crashes) it is the normalised first line of the message, so that
two different internal errors are not merged while sizes changing under `shrink_arrays`
do not split one.
"""
struct Class
    kind::Symbol
    key::String
    detail::String
end
Class(kind::Symbol, detail::String = "") = Class(kind, "", detail)

Base.:(==)(a::Class, b::Class) = a.kind == b.kind && a.key == b.key
Base.hash(c::Class, h::UInt) = hash((c.kind, c.key), h)
Base.show(io::IO, c::Class) = print(io, c.kind, isempty(c.detail) ? "" : ": " * c.detail)

"The `Class` of a run that raised no error."
const PASS = Class(:pass)

"`line` with SSA names, hex literals and numbers replaced by placeholders; two failures differing only in a number share a key."
normalize_key(line::AbstractString) =
    replace(line, r"%\d+" => "%N", r"0x[0-9a-fA-F]+" => "0xX", r"\d+" => "N")

"The `Class` of a caught exception, with `LoadError`s unwrapped."
function classify(err::Exception)
    while err isa LoadError
        err = err.error
    end
    msg = Base.invokelatest(sprint, showerror, err)   # showerror methods defined by the candidate itself
    line = String(strip(first(split(msg, '\n'; limit = 2))))
    pre = "$(nameof(typeof(err))): "
    startswith(line, pre) && (line = line[length(pre)+1:end])   # Enzyme's showerror names the type itself
    typed = (err isa Enzyme.Compiler.EnzymeError || err isa Enzyme.Compiler.CustomRuleError) &&
            !(err isa Enzyme.Compiler.EnzymeInternalError)
    return Class(nameof(typeof(err)), typed ? "" : normalize_key(line), first(line, 160))
end

"Run `thunk()` and return `PASS` or the class of the exception it threw."
classify_run(thunk) = try
    thunk()
    PASS
catch err
    classify(err)
end
