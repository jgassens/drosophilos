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

Rejected (v0): branches inside the body, calls, STOREX, a body that writes a parameter, more
than one OUT, IN inside the body.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..isa.ir import ALU_OPS as IR_OPS, Instr, Program, _to_signed


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


def loop_body(prog: Program, fn: str, label: str) -> list:
    """The instructions between the loop's exit test (the conditional jump after `label`) and
    the back-jump to `label`, exclusive."""
    body = prog.functions[fn]
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
                   state: dict | None = None) -> KernelSpec:
    """`state`: loop-carried variables and their values at power-up (the prologue's CONSTs by
    default for any variable the body reads before writing and writes). Structured `if` /
    `if-else` inside the body (a JZ/JNZ over a label, an optional JMP to an end label) become
    select cells: both arms are computed and a SEL cell picks by the condition cell's Z flag.
    Calls are inlined. Every OUT names an output cell."""
    params = dict(params or {})
    width = prog.width
    mask = (1 << width) - 1
    arrs = arrays_of(prog)
    contents = array_contents(prog)
    if mems:
        contents.update(mems)
    inits = {ins.dst: ins.imm & mask for ins in prog.functions["main"] if isinstance(ins, Instr) and ins.op == "CONST" and ins.dst}
    state = dict(state or {})
    spec = KernelSpec([], {}, {}, stream, width)
    spec.outputs = []
    spec.state_cells = {}  # variable -> the cell that holds it across tokens
    env: dict = {stream: "input"}  # variable -> "input" | cell name | ("const", name)
    xaddr = [None]  # what __x holds: ("const", k) or (base_value, index source)
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
    for v in reads_first & writes:
        if v in params:
            raise NotAKernel(f"{v} is a parameter but the body writes it")
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

    def src(v):
        if v in env:
            return env[v]
        if v not in prog.variables:
            raise NotAKernel(f"unknown variable {v}")
        if v not in params:
            raise NotAKernel(f"{v} is read before it is written: give it as a parameter")
        env[v] = const(params[v])
        return env[v]

    def is_const(s_):
        return isinstance(s_, tuple) and s_[0] == "const"

    def cval(s_):
        return spec.consts[s_[1]]

    def cell(op, a, b=None, mem=None, c=None):
        name = f"c{n_cells[0]}_{op.lower()}"
        n_cells[0] += 1
        d = {"name": name, "op": op, "a": a}
        if b is not None:
            d["b"] = b
        if c is not None:
            d["c"] = c
        if mem is not None:
            d["mem"] = mem
        spec.cells.append(d)
        return name

    def fold(op, a, b):
        if op in ("ADD", "SUB", "MUL"):
            return IR_OPS[op](_to_signed(a, width), _to_signed(b, width), width).value & mask
        return IR_OPS[op](a, b, width).value & mask

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
            if op in ("JMP", "RET", "STOREX", "IN", "HALT"):
                raise NotAKernel(f"{op} inside a kernel body")
            if op == "CALL":
                fbody = prog.functions[ins.target]
                if fbody and isinstance(fbody[-1], Instr) and fbody[-1].op == "RET":
                    fbody = fbody[:-1]  # the closing RET; an early RET is rejected below
                walk(fbody, env)
                continue
            if op in ("JZ", "JNZ"):
                # structured if: the jump skips the then-arm to `else`; an optional `JMP end` before `else` starts an else-arm
                cond_var = ins.srcs[0]
                cond = src(cond_var)
                l_else = ins.target
                if l_else not in labels:
                    raise NotAKernel("a branch out of the body")
                j_else = labels[l_else]
                then_arm = instrs[i:j_else]
                else_arm = []
                j_end = j_else
                if then_arm and isinstance(then_arm[-1], Instr) and then_arm[-1].op == "JMP":
                    l_end = then_arm[-1].target
                    if l_end not in labels:
                        raise NotAKernel("a branch out of the body")
                    j_end = labels[l_end]
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
                        raise NotAKernel(f"{v} is defined on one arm only")
                    if is_const(cond):  # decided at compile time
                        env[v] = t_ if (cval(cond) != 0) == taken_nonzero else e_
                        continue
                    if not isinstance(cond, str) or cond == "input":
                        cond_cell = cell("MOV", cond, cond)  # the condition needs a cell with a Z flag
                        cond = cond_cell
                    a_, b_ = (t_, e_) if taken_nonzero else (e_, t_)
                    env[v] = cell("SEL", a_, b_, c=cond)
                    written.add(v)
                i = j_end
                continue
            if ins.dst in params:
                raise NotAKernel(f"the body writes the parameter {ins.dst}")
            if op == "CONST":
                if ins.dst == "__x":
                    xaddr[0] = ("const", ins.imm & mask)
                else:
                    env[ins.dst] = const(ins.imm)
            elif op == "MOV":
                env[ins.dst] = src(ins.srcs[0])
            elif op in IR_OPS:
                a = src(ins.srcs[0])
                b = const(ins.imm) if ins.imm is not None else src(ins.srcs[1])
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
                if ins.dst == stream:
                    continue  # the induction update: the host streams the values
                if is_const(a) and is_const(b):
                    r = fold(op, cval(a), cval(b))
                    env[ins.dst] = const(r)
                    spec.folded[ins.dst] = r
                    continue
                env[ins.dst] = cell(op, a, b)
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
                spec.mems[name] = (length, dict(contents.get(name, {})))
                env[ins.dst] = cell("LOAD", a, mem=name)
                written.add(ins.dst)
            elif op == "OUT":
                s_ = const(ins.imm) if ins.imm is not None else src(ins.srcs[0])
                if not (isinstance(s_, str) and s_ != "input" and not s_.startswith("state_")):
                    s_ = cell("MOV", s_, s_)  # the output must be a cell's master
                spec.outputs.append(s_)
            else:
                raise NotAKernel(op)

    walk(body, env)
    # resolve the state placeholders: a state variable's final cell is its carrier; if the body
    # left it unchanged the state has no cell (a constant), which v0 rejects
    for v, slot in state_slot.items():
        final = env.get(v)
        if final == slot or not isinstance(final, str) or final == "input":
            raise NotAKernel(f"state variable {v} is not recomputed by the body")
        spec.state_cells[v] = final
        for c in spec.cells:
            for k in ("a", "b", "c"):
                if c.get(k) == slot:
                    c[k] = final
        for k, o in enumerate(spec.outputs):
            if o == slot:
                spec.outputs[k] = final
        for c in spec.cells:
            if c["name"] == final:
                c["init"] = state[v]
    if not spec.outputs:
        raise NotAKernel("the body emits nothing")
    spec.out_cell = spec.outputs[-1]
    # a cell whose every cell source is a feedback edge (built at or after it) has nothing in
    # this token to request it: the input token paces it
    index = {c["name"]: k for k, c in enumerate(spec.cells)}
    for k, c in enumerate(spec.cells):
        srcs = [c.get(x) for x in ("a", "b", "c") if isinstance(c.get(x), str)]
        if srcs and "input" not in srcs and all(index[s_] >= k for s_ in srcs):
            c["trigger"] = ["input"]
    used = {c[k][1] for c in spec.cells for k in ("a", "b", "c") if is_const(c.get(k))}
    spec.consts = {k: v for k, v in spec.consts.items() if k in used}  # folded bases and the induction step are gone
    return spec


def kernel_reference(spec: KernelSpec, tokens: list[int]) -> list[int]:
    """What the pipeline's last output must produce for each token (see kernel_outputs)."""
    return [outs[-1] for outs in kernel_outputs(spec, tokens)]


def kernel_outputs(spec: KernelSpec, tokens: list[int]) -> list[list[int]]:
    """Per token, the value of every output cell, from the same IR semantics; state cells keep
    their value across tokens."""
    w, mask = spec.width, (1 << spec.width) - 1
    state = {c["name"]: c["init"] & mask for c in spec.cells if c.get("init") is not None}
    outs = []
    for t in tokens:
        val = {"input": t & mask}
        prev = dict(state)

        def get(s_):
            if isinstance(s_, tuple):
                return spec.consts[s_[1]]
            if s_ in val:
                return val[s_]
            return prev[s_]  # a feedback read: the previous token's value

        for c in spec.cells:
            if c["op"] == "LOAD":
                n_words, contents = spec.mems[c["mem"]]
                addr = get(c["a"]) & (n_words - 1)
                if addr not in contents:
                    raise KeyError(f"{c['name']}: {c['mem']}[{addr}] is unwritten (a read would double-rail)")
                v = contents[addr] & mask
            elif c["op"] == "MOV":
                v = get(c["b"])
            elif c["op"] == "SEL":
                v = get(c["a"]) if get(c["c"]) != 0 else get(c["b"])
            else:
                a, b = get(c["a"]), get(c["b"])
                if c["op"] in ("ADD", "SUB", "MUL"):
                    v = IR_OPS[c["op"]](_to_signed(a, w), _to_signed(b, w), w).value & mask
                else:
                    v = IR_OPS[c["op"]](a, b, w).value & mask
            val[c["name"]] = v
            if c["name"] in state:
                state[c["name"]] = v
        outs.append([val[o] for o in (getattr(spec, "outputs", None) or [spec.out_cell])])
    return outs
