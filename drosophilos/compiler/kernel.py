"""Kernel compiler (plan: resident circuits; `docs/capacity_doom.md` §4.1): a loop body of the
IR becomes a resident pipeline spec for `lib/kernel.build_pipeline`, one cell per operation,
no fetch, no decode, no PC.

    loop_body(prog, fn, label)                -> the straight-line body of the loop at `label`
    compile_kernel(prog, body, stream, params) -> KernelSpec (cells, constants, memories)
    kernel_reference(spec, tokens, width)      -> the outputs the pipeline must produce

The loop's induction variable is the token stream: the host streams its values (the columns
of a frame) instead of the kernel counting, so the induction update is dropped from the body.
Variables the body reads but never writes are kernel parameters (levels lit at load time, the
image); constant operands become constant levels; a MOV is a renaming, not a cell; an
operation on two constants folds. Arrays read through the index word (`ADD __x idx #base;
LOADX`) become LOAD cells on a memory of their own, addressed by the index (the base folds
away); their contents come from the initialisers in the program's prologue. One OUT per
token: its source is the output cell (a MOV cell is appended if the source is not a cell).

Rejected (v0): loops inside the body, a body that writes a parameter, more
than one OUT, IN inside the body.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..isa.ir import ALU_OPS as IR_OPS, Instr, Program, _to_signed, apply as ir_apply


class NotAKernel(Exception):
    pass


@dataclass
class KernelSpec:
    cells: list  # build_pipeline's spec
    consts: dict  # name -> value
    mems: dict  # name -> (n_words, {addr: value})
    stream: str  # the variable the token stream stands for
    width: int
    out_cell: str = ""
    folded: dict = field(default_factory=dict)  # variable -> constant value at compile time
    outputs: list = field(default_factory=list)  # output cells, one value each per token
    state_cells: dict = field(default_factory=dict)  # state variable -> its carrier cell
    rams: set = field(default_factory=set)  # arrays written by the program: RAM, not ROM


def arrays_of(prog: Program) -> dict:
    """name -> (base, length) from the `name[k]` variables."""
    arrs: dict = {}
    for v, a in prog.variables.items():
        m = re.fullmatch(r"(\w+)\[(\d+)\]", v)
        if m:
            base, length = arrs.get(m.group(1), (a, 0))
            arrs[m.group(1)] = (min(base, a), max(length, int(m.group(2)) + 1))
    return arrs


def array_contents(prog: Program, fn: str = "main") -> dict:
    """name -> {index: value} from the prologue's CONST instructions on array elements."""
    out: dict = {}
    for ins in prog.functions[fn]:
        if not isinstance(ins, Instr) or ins.op != "CONST":
            continue
        m = re.fullmatch(r"(\w+)\[(\d+)\]", ins.dst or "")
        if m:
            out.setdefault(m.group(1), {})[int(m.group(2))] = ins.imm & ((1 << prog.width) - 1)
    return out


def written_arrays(prog: Program) -> set:
    """Names of the arrays some STOREX in the program writes (the index word's base before it)."""
    arrs = arrays_of(prog)
    out, base = set(), None
    for fn in prog.functions.values():
        for ins in fn:
            if not isinstance(ins, Instr):
                continue
            if ins.dst == "__x":
                base = ins.imm if ins.op in ("CONST", "ADD") and ins.imm is not None else None
            elif ins.op == "STOREX" and base is not None:
                for nm, (b0, ln) in arrs.items():
                    if b0 <= base < b0 + ln:
                        out.add(nm)
    return out


def loop_body(prog: Program, fn: str = "main", label: str | None = None) -> list:
    """The instructions between the loop's exit test (the conditional jump after `label`) and
    the back-jump to `label`, exclusive. `label` None: the function's first loop."""
    body = prog.functions[fn]
    if label is None:
        label = next((x for x in body if isinstance(x, str) and x.startswith("loop")), None)
        if label is None:
            raise NotAKernel(f"no loop in {fn}")
    if label not in body:
        raise NotAKernel(f"no label {label} in {fn}")
    start = body.index(label) + 1
    while start < len(body) and isinstance(body[start], Instr) and body[start].op not in ("JZ", "JNZ"):
        start += 1  # the exit test's temporaries
    if start >= len(body) or not isinstance(body[start], Instr):
        raise NotAKernel("loop without an exit test")
    start += 1
    end = next((i for i in range(start, len(body)) if isinstance(body[i], Instr) and body[i].op == "JMP" and body[i].target == label), None)
    if end is None:
        raise NotAKernel("loop without a back-jump")
    return body[start:end]


def compile_kernel(prog: Program, body: list, stream: str, params: dict | None = None, mems: dict | None = None,
                   state: dict | None = None, *, prefix: str = "c", stream_key: str = "input", seeds: dict | None = None,
                   into: KernelSpec | None = None, allow_no_output: bool = False, mul: str = "array") -> KernelSpec:
    """`state`: loop-carried variables and their values at power-up (the prologue's CONSTs by
    default for any variable the body reads before writing and writes). Structured `if` /
    `if-else` inside the body (a JZ/JNZ over a label, an optional JMP to an end label) become
    select cells: both arms are computed and a SEL cell picks by the condition cell's Z flag.
    Calls are inlined. Every OUT names an output cell. An `in_read()` inside the body makes
    its variable the token (fresh input per token); the induction variable is then only the
    count the host streams."""
    params = dict(params or {})
    width = prog.width
    mask = (1 << width) - 1
    arrs = arrays_of(prog)
    contents = array_contents(prog)
    if mems:
        contents.update(mems)
    # initial values come from the prologue only (before main's first loop): a CONST inside a
    # loop body is an assignment, not an initial state (review finding: [5, 10, 15] vs [25, 30, 35])
    main = prog.functions["main"]
    first_loop = next((k for k, x in enumerate(main) if isinstance(x, str) and x.startswith("loop")), len(main))
    inits = {ins.dst: ins.imm & mask for ins in main[:first_loop] if isinstance(ins, Instr) and ins.op == "CONST" and ins.dst}
    state = dict(state or {})
    spec = into or KernelSpec([], {}, {}, stream, width)
    if into is None:
        spec.outputs = []
        spec.state_cells = {}  # variable -> the cell that holds it across tokens
    n_before = len(spec.cells)
    env: dict = {stream: stream_key}  # variable -> "input" | cell name | ("const", name) | ("param", cell)
    env.update(seeds or {})
    xaddr = [None]  # what __x holds: ("const", k) or (base_value, index source)
    input_var = [None]  # the variable an in_read() inside the body assigns (the token)
    stored_in_body: set = set()  # arrays this body stores into (a later load of them has no ordering edge)
    allow_counter = [False]  # the counter's own update (i = i - 1) may read it; it is dead in the kernel
    n_cells = [0]
    written = set()
    # variables read before they are written, and written somewhere in the body: state
    reads_first, writes = set(), set()
    def scan(instrs):
        for ins in instrs:
            if isinstance(ins, str):
                continue
            if ins.op == "CALL":
                scan(prog.functions[ins.target]); continue
            for v in ins.srcs:
                if v not in writes and v != stream and v != "__x":
                    reads_first.add(v)
            if ins.op in ("JZ", "JNZ") and ins.srcs and ins.srcs[0] not in writes:
                reads_first.add(ins.srcs[0])
            if ins.dst and ins.dst != "__x":
                writes.add(ins.dst)
    scan(body)
    in_vars = set()
    def scan_in(instrs):
        for ins in instrs:
            if isinstance(ins, Instr) and ins.op == "IN":
                in_vars.add(ins.dst)
            elif isinstance(ins, Instr) and ins.op == "CALL":
                scan_in(prog.functions[ins.target])
    scan_in(body)
    for v in (reads_first & writes) - in_vars:
        if v in params and v not in state:  # a value read before the loop and updated by it: state
            state[v] = params[v] & mask
            continue
        if v not in state:
            if v not in inits:
                raise NotAKernel(f"state variable {v} has no initial value: give it in `state`")
            state[v] = inits[v]
    # a state variable's reads before its first write refer to the cell that writes it last: a
    # placeholder name resolved after the walk (feedback edge)
    state_slot = {v: f"state_{v}" for v in state}
    for v in state:
        env[v] = state_slot[v]

    def const(v):
        v &= mask
        name = f"k{v}"
        spec.consts[name] = v
        return ("const", name)

    def src_in(env_, v):
        """The source for variable v in the environment of the arm being compiled (an if-arm's
        temporaries live in its own copy of the environment: review-found closure bug)."""
        if v == stream and input_var[0] is not None and env_.get(v) == stream_key and not allow_counter[0]:
            raise NotAKernel(f"the loop counter {v} is not available in a body that reads its token with in_read()")
        if v in env_:
            return env_[v]
        if v not in prog.variables:
            raise NotAKernel(f"unknown variable {v}")
        if v not in params:
            raise NotAKernel(f"{v} is read before it is written: give it as a parameter")
        env_[v] = const(params[v])
        return env_[v]

    def is_const(s_):
        return isinstance(s_, tuple) and s_[0] == "const"

    def cval(s_):
        return spec.consts[s_[1]]

    def cell(op, a, b=None, mem=None, c=None, imm=None):
        name = f"{prefix}{n_cells[0]}_{op.lower()}"
        n_cells[0] += 1
        d = {"name": name, "op": op, "a": a, "stream": stream_key}
        if b is not None:
            d["b"] = b
        if c is not None:
            d["c"] = c
        if mem is not None:
            d["mem"] = mem
        if imm is not None:
            d["imm"] = imm
        spec.cells.append(d)
        return name

    def fold(op, a, b):
        return ir_apply(op, a, b, width)

    def walk(instrs, env):
        """Compiles a straight-line stretch with structured ifs; returns nothing, mutates env."""
        i = 0
        labels = {ins: k for k, ins in enumerate(instrs) if isinstance(ins, str)}
        while i < len(instrs):
            ins = instrs[i]
            i += 1
            if isinstance(ins, str):
                continue
            op = ins.op
            if op == "JMP" and ins.target in labels and labels[ins.target] < i:
                raise NotAKernel("a loop inside the body (a backward branch)")
            if op in ("JMP", "RET", "HALT"):
                raise NotAKernel(f"{op} inside a kernel body")
            if op == "STOREX":  # a[i] = e: a STORE cell into the array's RAM
                if xaddr[0] is None:
                    raise NotAKernel("STOREX before the index word is set")
                if xaddr[0][0] == "const":
                    addr = xaddr[0][1]
                    name = next((nm for nm, (base, ln) in arrs.items() if base <= addr < base + ln), None)
                    if name is None:
                        raise NotAKernel(f"STOREX to a scalar address {addr}")
                    a = const(addr - arrs[name][0])
                else:
                    base, a = xaddr[0]
                    name = next((nm for nm, (b0, ln) in arrs.items() if b0 == base), None)
                    if name is None:
                        raise NotAKernel(f"STOREX to base {base}, not an array")
                base, length = arrs[name]
                if any(c["op"] == "STORE" and c.get("mem") == name for c in spec.cells):
                    raise NotAKernel(f"two stores into {name}: their write ports would share the words' COPY latches (v0: one STORE cell per array)")
                spec.mems[name] = (length, dict(contents.get(name, {})), "ram")
                spec.rams.add(name)
                stored_in_body.add(name)
                data = src_in(env, ins.srcs[0])
                spec.outputs.append(cell("STORE", a, data, mem=name))  # the host sees every write land (pacing)
                continue
            if op == "IN":  # fresh input each token: the variable is the token itself
                if input_var[0] is not None:
                    raise NotAKernel("one in_read() per token")
                input_var[0] = ins.dst
                env[ins.dst] = stream_key
                continue
            if op == "CALL":
                fbody = prog.functions[ins.target]
                if fbody and isinstance(fbody[-1], Instr) and fbody[-1].op == "RET":
                    fbody = fbody[:-1]  # the closing RET; an early RET is rejected below
                walk(fbody, env)
                continue
            if op in ("JZ", "JNZ"):
                # structured if: the jump skips the then-arm to `else`; an optional `JMP end` before `else` starts an else-arm
                cond_var = ins.srcs[0]
                cond = src_in(env, cond_var)
                l_else = ins.target
                if l_else not in labels:
                    raise NotAKernel("a branch out of the body")
                j_else = labels[l_else]
                then_arm = instrs[i:j_else]
                else_arm = []
                j_end = j_else
                if j_else < i:
                    raise NotAKernel("a loop inside the body (a backward branch)")
                if then_arm and isinstance(then_arm[-1], Instr) and then_arm[-1].op == "JMP":
                    l_end = then_arm[-1].target
                    if l_end not in labels:
                        raise NotAKernel("a branch out of the body")
                    j_end = labels[l_end]
                    if j_end < j_else:
                        raise NotAKernel("a loop inside the body (a backward branch)")
                    else_arm = instrs[j_else + 1:j_end]
                    then_arm = then_arm[:-1]
                env_then, env_else = dict(env), dict(env)
                walk(then_arm, env_then)
                walk(else_arm, env_else)
                # JZ skips the then-arm when cond == 0: then-arm applies when cond != 0 (SEL's a); JNZ the reverse
                taken_nonzero = op == "JZ"
                for v in set(env_then) | set(env_else):
                    t_, e_ = env_then.get(v, env.get(v)), env_else.get(v, env.get(v))
                    if t_ == e_:
                        env[v] = t_
                        continue
                    if t_ is None or e_ is None:
                        if v.startswith("__t"):
                            continue  # an arm's temporary: dead after the arm
                        if v in params:  # first read inside one arm: the other arm has the parameter's value
                            t_ = t_ if t_ is not None else const(params[v])
                            e_ = e_ if e_ is not None else const(params[v])
                        else:
                            raise NotAKernel(f"{v} is defined on one arm only")
                    if is_const(cond):  # decided at compile time
                        env[v] = t_ if (cval(cond) != 0) == taken_nonzero else e_
                        continue
                    if not isinstance(cond, str) or cond.startswith("input"):
                        cond_cell = cell("MOV", cond, cond)  # the condition needs a cell with a Z flag
                        cond = cond_cell
                    a_, b_ = (t_, e_) if taken_nonzero else (e_, t_)
                    env[v] = cell("SEL", a_, b_, c=cond)
                    written.add(v)
                i = j_end
                continue
            if ins.dst in params and ins.dst not in state:
                raise NotAKernel(f"the body writes the parameter {ins.dst}")
            if op == "CONST":
                if ins.dst == "__x":
                    xaddr[0] = ("const", ins.imm & mask)
                else:
                    env[ins.dst] = const(ins.imm)
            elif op == "MOV":
                env[ins.dst] = src_in(env, ins.srcs[0])
            elif op in ("SHL", "SHR"):
                a = src_in(env, ins.srcs[0])
                if is_const(a):
                    env[ins.dst] = const(ir_apply(op, cval(a), ins.imm, width))
                    continue
                env[ins.dst] = cell(op, a, imm=ins.imm)
                written.add(ins.dst)
            elif op in IR_OPS:
                allow_counter[0] = ins.dst == stream
                a = src_in(env, ins.srcs[0])
                b = const(ins.imm) if ins.imm is not None else src_in(env, ins.srcs[1])
                allow_counter[0] = False
                if ins.dst == "__x":  # an address: base + index
                    if op != "ADD":
                        raise NotAKernel("only base + index addressing")
                    if is_const(a) and is_const(b):
                        xaddr[0] = ("const", (cval(a) + cval(b)) & mask)
                    elif is_const(b):
                        xaddr[0] = (cval(b), a)
                    elif is_const(a):
                        xaddr[0] = (cval(a), b)
                    else:
                        raise NotAKernel("an address with two variable parts")
                    continue
                if is_const(a) and is_const(b):
                    r = fold(op, cval(a), cval(b))
                    env[ins.dst] = const(r)
                    spec.folded[ins.dst] = r
                    continue
                env[ins.dst] = cell("MULP" if (op == "MUL" and mul == "pipelined") else op, a, b)
                written.add(ins.dst)
            elif op == "LOADX":
                if xaddr[0] is None:
                    raise NotAKernel("LOADX before the index word is set")
                if xaddr[0][0] == "const":
                    addr = xaddr[0][1]
                    name = next((nm for nm, (base, ln) in arrs.items() if base <= addr < base + ln), None)
                    if name is None:
                        raise NotAKernel(f"LOADX from a scalar address {addr}")
                    a = const(addr - arrs[name][0])
                else:
                    base, a = xaddr[0]
                    name = next((nm for nm, (b0, ln) in arrs.items() if b0 == base), None)
                    if name is None:
                        raise NotAKernel(f"LOADX from base {base}, not an array")
                base, length = arrs[name]
                if name in stored_in_body:
                    raise NotAKernel(f"{name} is read after a store to it in the same body: no ordering edge between a STORE cell and a LOAD cell (v0: read it in a later pass)")
                if name in written_arrays(prog):
                    spec.mems[name] = (length, dict(contents.get(name, {})), "ram")
                    spec.rams.add(name)
                elif name not in spec.mems:
                    spec.mems[name] = (length, dict(contents.get(name, {})))
                env[ins.dst] = cell("LOAD", a, mem=name)
                written.add(ins.dst)
            elif op == "OUT":
                s_ = const(ins.imm) if ins.imm is not None else src_in(env, ins.srcs[0])
                if not (isinstance(s_, str) and not s_.startswith("input") and not s_.startswith("state_")):
                    paced = isinstance(s_, tuple)  # a constant or a parameter has no done pulse: the token paces it
                    s_ = cell("MOV", s_, s_)  # the output must be a cell's master
                    if paced:
                        spec.cells[-1]["trigger"] = [stream_key]
                spec.outputs.append(s_)
            else:
                raise NotAKernel(op)

    walk(body, env)
    # resolve the state placeholders: a state variable's final cell is its carrier; if the body
    # left it unchanged the state has no cell (a constant), which v0 rejects
    for v, slot in state_slot.items():
        final = env.get(v)
        if final == slot:
            raise NotAKernel(f"state variable {v} is not recomputed by the body")
        if not isinstance(final, str) or final.startswith("input"):  # a constant or the token: a paced MOV carries it
            final = cell("MOV", final, final)
            if not isinstance(env[v], str):
                spec.cells[-1]["trigger"] = [stream_key]
            env[v] = final
        spec.state_cells[v] = final
        for c in spec.cells[n_before:]:
            for k in ("a", "b", "c"):
                if c.get(k) == slot:
                    c[k] = final
        for k, o in enumerate(spec.outputs):
            if o == slot:
                spec.outputs[k] = final
        for c in spec.cells:
            if c["name"] == final:
                c["init"] = state[v]
    if not spec.outputs and not allow_no_output:  # an outer body may be a bare counter: its stream only paces
        raise NotAKernel("the body emits nothing")
    spec.out_cell = spec.outputs[-1] if spec.outputs else None
    # a cell whose every cell source is a feedback edge (built at or after it) has nothing in
    # this token to request it: the input token paces it
    index = {c["name"]: k for k, c in enumerate(spec.cells)}
    for k, c in enumerate(spec.cells[n_before:], start=n_before):
        srcs = [c.get(x) for x in ("a", "b", "c") if isinstance(c.get(x), str)]
        if srcs and not any(x.startswith("input") for x in srcs) and all(index[s_] >= k for s_ in srcs):
            c["trigger"] = [stream_key]
        elif not srcs and all(isinstance(c.get(x), tuple) for x in ("a", "b", "c") if c.get(x) is not None):
            c["trigger"] = [stream_key]  # constants and params only: paced by the token
    # dead cells (the induction update nobody reads, temporaries): removed, repeatedly
    while True:
        read = set()
        for c in spec.cells:
            for k in ("a", "b", "c"):
                v = c.get(k)
                if isinstance(v, str):
                    read.add(v)
                elif isinstance(v, tuple) and v[0] == "param":
                    read.add(v[1])
        keep = [c for c in spec.cells if c["name"] in read or c["name"] in spec.outputs or c["name"] in spec.state_cells.values()]
        if len(keep) == len(spec.cells):
            break
        spec.cells[:] = keep
    used = {c[k][1] for c in spec.cells for k in ("a", "b", "c") if is_const(c.get(k))}
    spec.consts = {k: v for k, v in spec.consts.items() if k in used}  # folded bases and the induction step are gone
    return spec


def compile_program(prog: Program, params: dict | None = None, fn: str = "main", pacing: str = "host",
                    counts: dict | None = None) -> KernelSpec:
    """A function with a loop nest of depth two (a frame loop around a column loop) as one
    pipeline with two token streams: the outer loop's body without the inner loop is the
    outer kernel (stream "input:<outer var>": the tick), the inner loop is the inner kernel
    (stream "input"). What the inner body reads and the outer body writes is a parameter edge
    from the outer kernel's state cell; the outer kernel's state carriers are outputs too, so
    the host can pace the next frame's columns behind the tick (`lib/kernel.run_pipeline`).
    Values read before the loops (an `in_read()` in the prologue) come from `params`."""
    body = prog.functions[fn]
    outer = next((x for x in body if isinstance(x, str) and x.startswith("loop")), None)
    if outer is None:
        raise NotAKernel("no loop")
    ob = loop_body(prog, fn, outer)
    inner = next((x for x in ob if isinstance(x, str) and x.startswith("loop")), None)
    if inner is None:
        return compile_kernel(prog, ob, _induction_var(prog, fn, outer), params=params)
    # every inner loop of the outer body is a kernel with a stream named by its induction
    # variable (the first keeps the default stream "input"); the outer body without them is the
    # tick kernel; what the inner bodies read and the outer body writes is a parameter edge
    inners = []  # (label, body, induction var, start, end) in order
    k = 0
    while k < len(ob):
        x = ob[k]
        if isinstance(x, str) and x.startswith("loop"):
            end = next(j for j in range(k, len(ob)) if isinstance(ob[j], str) and ob[j].startswith("endloop"))
            inners.append((x, loop_body(prog, fn, x), _induction_var(prog, fn, x), k, end + 1))
            k = end + 1
        else:
            k += 1
    ovar = _induction_var(prog, fn, outer)
    ivars = {iv for _, _, iv, _, _ in inners}
    keep = [True] * len(ob)
    for _, _, _, a_, b_ in inners:
        for j in range(a_, b_):
            keep[j] = False
    outer_body = [x for x, kp in zip(ob, keep) if kp and not (isinstance(x, Instr) and x.dst in ivars)]  # the inner counters' inits go; labels stay
    okey = f"input:{ovar}"
    spec = compile_kernel(prog, outer_body, ovar, params=params, prefix="f", stream_key=okey, allow_no_output=True)
    for v, cname in spec.state_cells.items():  # the outer kernel's state carriers are outputs (the host paces on them)
        if cname not in spec.outputs:
            spec.outputs.append(cname)
    seeds = {v: ("param", cname) for v, cname in spec.state_cells.items()}
    spec.streams, spec.stream_keys = ["input", ovar], {"input": "input", ovar: okey}
    outer_state = set(spec.state_cells) | set(params or {})
    for _, ib, ivar, _, _ in inners:
        for ins in ib:
            if isinstance(ins, Instr) and ins.dst in outer_state and ins.dst != ivar:
                raise NotAKernel(f"the inner loop writes {ins.dst}, a variable the outer loop carries: the two kernels would each hold a copy (v0)")
    for idx, (_, ib, ivar, _, _) in enumerate(inners):
        key = "input" if idx == 0 else f"input:{ivar}"
        spec = compile_kernel(prog, ib, ivar, params=params, prefix=(f"c{idx}_" if idx else "c"), stream_key=key, seeds=seeds, into=spec)
        if idx:
            spec.streams.append(ivar)
            spec.stream_keys[ivar] = key
    spec.phases = None
    if pacing == "neural":
        # Stage F2, first step: a wrapping counter per inner stream (cnt = cnt == K-1 ? 0 : cnt + 1,
        # requested by the pass's output cell), the phases in program order, the tick last.
        # `counts[stream]` is the number of tokens of that stream per frame on this copy.
        phases = []
        for idx, (_, ib, ivar, _, _) in enumerate(inners):
            key = "input" if idx == 0 else f"input:{ivar}"
            kname = "input" if idx == 0 else ivar
            K = (counts or {}).get(kname)
            if K is None:
                raise NotAKernel(f"neural pacing needs the token count per frame of stream {kname}")
            out_cell = next(o for o in reversed(spec.outputs) if next(c for c in spec.cells if c["name"] == o)["stream"] == key)
            pfx = f"ph{idx}_"
            spec.consts[f"k{K - 1}"] = K - 1
            spec.consts["k1"] = 1
            spec.consts["k0"] = 0
            spec.cells.append({"name": f"{pfx}xor", "op": "XOR", "a": f"{pfx}cnt", "b": ("const", f"k{K - 1}"), "stream": key, "trigger": [out_cell]})
            spec.cells.append({"name": f"{pfx}add", "op": "ADD", "a": f"{pfx}cnt", "b": ("const", "k1"), "stream": key, "trigger": [out_cell]})
            spec.cells.append({"name": f"{pfx}cnt", "op": "SEL", "a": f"{pfx}add", "b": ("const", "k0"), "c": f"{pfx}xor", "stream": key, "init": 0})
            phases.append((kname, f"{pfx}cnt", "wrap"))
        # the tick phase ends when the tick kernel's last state carrier has landed (its done), or
        # at the token's done if the tick kernel has no cells
        tick_cells = [c["name"] for c in spec.cells if c["stream"] == okey]
        last_state = next((cn for cn in reversed(tick_cells) if cn in spec.state_cells.values()), tick_cells[-1] if tick_cells else None)
        phases.append((ovar, last_state, "each"))
        spec.phases = phases
    return spec


def _induction_var(prog: Program, fn: str, label: str) -> str:
    """The variable the loop's exit test reads (col in `while (col != 8)`)."""
    body = prog.functions[fn]
    k = body.index(label) + 1
    while k < len(body) and isinstance(body[k], Instr) and body[k].op not in ("JZ", "JNZ"):
        k += 1
    test = body[k].srcs[0]
    for x in body[body.index(label) + 1: k]:  # the test may be a temporary: XOR t col #8
        if isinstance(x, Instr) and x.dst == test and x.srcs:
            return x.srcs[0]
    return test


def kernel_reference(spec: KernelSpec, tokens: list[int]) -> list[int]:
    """What the pipeline's last output must produce for each token (see kernel_outputs)."""
    return [outs[-1] for outs in kernel_outputs(spec, tokens)]


def kernel_outputs(spec: KernelSpec, tokens: list[int]) -> list[list[int]]:
    """Per token, the value of every output cell, from the same IR semantics; state cells keep
    their value across tokens."""
    w, mask = spec.width, (1 << spec.width) - 1
    state = {c["name"]: c["init"] & mask for c in spec.cells if c.get("init") is not None}
    last = dict(state)  # every cell's last committed value (params read it)
    ram = {nm: dict(m[1]) for nm, m in spec.mems.items() if len(m) > 2 and m[2] == "ram"}
    keys = getattr(spec, "stream_keys", None) or {"input": "input"}
    outs = []
    for t in tokens:
        stream, t = (t[0], t[1]) if isinstance(t, tuple) else ("input", t)
        key = keys[stream]
        val = {key: t & mask}
        prev = dict(state)

        def get(s_):
            if isinstance(s_, tuple) and s_[0] == "const":
                return spec.consts[s_[1]]
            if isinstance(s_, tuple) and s_[0] == "param":
                return last[s_[1]]
            if s_ in val:
                return val[s_]
            return prev[s_]  # a feedback read: the previous token's value

        for c in spec.cells:
            if c.get("stream", "input") != key:
                continue
            if c["op"] == "LOAD":
                m = spec.mems[c["mem"]]
                n_words, contents = m[0], (ram[c["mem"]] if c["mem"] in ram else m[1])
                addr = get(c["a"])
                if addr >= n_words or addr not in contents:  # the read relays match nothing: the cell would stall
                    raise KeyError(f"{c['name']}: {c['mem']}[{addr}] is outside the table or unwritten (the read would stall)")
                v = contents[addr] & mask
            elif c["op"] == "STORE":
                addr, v = get(c["a"]), get(c["b"])
                if addr >= spec.mems[c["mem"]][0]:
                    raise KeyError(f"{c['name']}: {c['mem']}[{addr}] is outside the buffer")
                ram[c["mem"]][addr] = v & mask
            elif c["op"] == "MOV":
                v = get(c["b"])
            elif c["op"] == "SEL":
                v = get(c["a"]) if get(c["c"]) != 0 else get(c["b"])
            elif c["op"] in ("SHL", "SHR"):
                v = ir_apply(c["op"], get(c["a"]), c["imm"], w)
            else:
                v = ir_apply("MUL" if c["op"] == "MULP" else c["op"], get(c["a"]), get(c["b"]), w)
            val[c["name"]] = v
            last[c["name"]] = v
            if c["name"] in state:
                state[c["name"]] = v
        outs.append([val[o] for o in (getattr(spec, "outputs", None) or [spec.out_cell]) if o in val])
    return outs
