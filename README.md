# MWE.jl

Automatically shrink a failing `Enzyme.autodiff` call to a minimal working
example.

Enzyme bug reports usually start as a large script. Turning that into the
ten-line reproducer a maintainer can act on is slow, manual work. MWE.jl
does it the way PyTorch's [FX minifier](https://pytorch.org/functorch/stable/notebooks/minifier.html)
does: capture the failing call, record every intermediate value, then cut
the program down piece by piece while checking after each cut that it still
fails the same way.

## Usage

Nothing in the script needs to change. Point `minify` at it:

```julia
using MWE
minify("script.jl")
```

Every `autodiff` call in the script is hooked; the first one that throws is
the one that gets reduced. The result is written next to the script:

```
mwe_20260904-220429/
├── repro.jl          # the reduced, standalone reproducer
├── checkpoints/      # every accepted intermediate step, each a valid reproducer
├── 0001.jl, 0001.log # every candidate that was tried, with its output
└── capture.log
```

For wrong-gradient bugs, where nothing throws, use the correctness check.
The failure is then "Enzyme's gradient disagrees with central finite
differences":

```julia
minify("script.jl"; check = :correctness)
```

Options: `workers` (concurrent worker processes, default half the cores)
and `timeout` (seconds per candidate, default 600).

## Example

From this script, where only the branch matters:

```julia
using Enzyme
using LinearAlgebra
scale(a) = a .* 2
function pick(x, y, w)
    a = scale(x)
    b = norm(a)
    c = w' * w
    z = x[1] > 0 ? x : y
    s = sum(z) * w[1]
    return s + b + c[1]
end
x = rand(1000); dx = zero(x)
y = rand(1000)
w = rand(8, 8); dw = zero(w)
autodiff(ReverseWithPrimal, pick, Active, Duplicated(x, dx), Const(y), Duplicated(w, dw))
```

`minify` produces, after 43 candidates:

```julia
using Enzyme

function pick(x, y)
    z = if x[1] > 0
            x
        else
            y
        end
    return sum(z)
end

x = [0.2792555959713239]
dx = zero(x)
y = [0.821271656771825]
autodiff(Reverse, pick, Active, Duplicated(x, dx), Const(y))
```

The log of the run is informative on its own. `arg x Duplicated → Const`
being rejected means the failure needs `x` active; `remove definition 3`
being rejected in a correctness run means that definition (a wrong custom
rule, say) is the cause.

## How it works

1. **Capture.** The script's top-level expressions are evaluated one at a
   time on a worker, with `autodiff` calls rewritten to go through a hook.
   When one throws, the mode, function, return activity and the argument
   values as they were passed in are saved.
2. **Record.** The primal is run once with every `y = rhs` statement
   instrumented, so the value of every intermediate is known.
3. **Reduce.** Passes propose simpler programs; an oracle runs each one in
   a fresh module on a persistent worker and reports the failure class
   (the exception type, or `WrongGradient`). A proposal is kept only if the
   class is unchanged. The passes, in order:
   - simplify the mode and return activity
   - **truncate the suffix** of the entry function and return an
     intermediate, in binary-search order
   - **delta debugging** over statements: a chunk of statements is
     replaced at once. An assignment becomes a *placeholder* — a new
     argument of the entry function carrying its recorded value, so the
     code after it still works — and anything else is deleted. Chunk size
     halves from the whole body down to one statement.
   - eliminate dead code, drop unused arguments
   - make arguments `Const`, unbatch, shrink arrays (coupled dimensions
     together)
   - delta debugging over top-level definitions
4. **Emit.** The smallest program is written as `repro.jl`, with recorded
   values as literals (or a `.jls` file next to it when large).

Workers that segfault or hang are killed and replaced, so LLVM-level
crashes and compile hangs are just failure classes like any other.

## Limits

- Only code in the script (and files it `include`s) is reduced. Functions
  from packages stay as calls.
- The `autodiff` call must reproduce from the script's top-level
  definitions and its arguments. A closure over locals inside a function
  is reported as "does not fail in isolation" rather than guessed at.
- Keyword arguments to `autodiff` are not supported.
- The correctness check covers reverse mode with a scalar return and
  `Duplicated` floating-point array arguments, and needs a primal that does
  not mutate its arguments.
- Values of user-defined types are embedded as literals when `repr`
  round-trips; otherwise a statement producing one cannot become a
  placeholder.

## Tests

```
julia --project -e 'using Pkg; Pkg.test()'
```
