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


def compile_kernel(prog: Program, body: list, stream: str, params: dict | None = None, mems: dict | None = None) -> KernelSpec:
    params = dict(params or {})
    width = prog.width
    mask = (1 << width) - 1
    arrs = arrays_of(prog)
    contents = array_contents(prog)
    if mems:
        contents.update(mems)
    spec = KernelSpec([], {}, {}, stream, width)
    env: dict = {stream: "input"}  # variable -> "input" | cell name | ("const", name)
    xaddr = None  # what __x holds: ("const", k) or (base_value, index source)
    written = set()
    n_cells = 0
    used_mems = set()

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

    def is_const(s):
        return isinstance(s, tuple) and s[0] == "const"

    def cval(s):
        return spec.consts[s[1]]

    def cell(op, a, b=None, mem=None):
        nonlocal n_cells
        name = f"c{n_cells}_{op.lower()}"
        n_cells += 1
        c = {"name": name, "op": op, "a": a}
        if b is not None:
            c["b"] = b
        if mem is not None:
            c["mem"] = mem
        spec.cells.append(c)
        return name

    for ins in body:
        if isinstance(ins, str):
            raise NotAKernel("a label inside the body (a branch)")
        op = ins.op
        if op in ("JMP", "JZ", "JNZ", "CALL", "RET", "STOREX", "IN", "HALT"):
            raise NotAKernel(f"{op} inside a kernel body")
        if ins.dst in params:
            raise NotAKernel(f"the body writes the parameter {ins.dst}")
        if op == "CONST":
            if ins.dst == "__x":
                xaddr = ("const", ins.imm & mask)
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
                    xaddr = ("const", (cval(a) + cval(b)) & mask)
                elif is_const(b):
                    xaddr = (cval(b), a)
                elif is_const(a):
                    xaddr = (cval(a), b)
                else:
                    raise NotAKernel("an address with two variable parts")
                continue
            if ins.dst == stream:
                continue  # the induction update: the host streams the values
            if is_const(a) and is_const(b):
                r = IR_OPS[op](_to_signed(cval(a), width), _to_signed(cval(b), width), width).value & mask if op in ("ADD", "SUB", "MUL") \
                    else IR_OPS[op](cval(a), cval(b), width).value & mask
                env[ins.dst] = const(r)
                spec.folded[ins.dst] = r
                continue
            env[ins.dst] = cell(op, a, b)
            written.add(ins.dst)
        elif op == "LOADX":
            if xaddr is None:
                raise NotAKernel("LOADX before the index word is set")
            if xaddr[0] == "const":
                addr = xaddr[1]
                name = next((nm for nm, (base, ln) in arrs.items() if base <= addr < base + ln), None)
                if name is None:
                    raise NotAKernel(f"LOADX from a scalar address {addr}")
                a = const(addr - arrs[name][0])
            else:
                base, a = xaddr
                name = next((nm for nm, (b0, ln) in arrs.items() if b0 == base), None)
                if name is None:
                    raise NotAKernel(f"LOADX from base {base}, not an array")
            base, length = arrs[name]
            spec.mems[name] = (length, dict(contents.get(name, {})))
            env[ins.dst] = cell("LOAD", a, mem=name)
            written.add(ins.dst)
        elif op == "OUT":
            if spec.out_cell:
                raise NotAKernel("one OUT per token in v0")
            s = const(ins.imm) if ins.imm is not None else src(ins.srcs[0])
            if isinstance(s, str) and s != "input" and s == spec.cells[-1]["name"]:
                spec.out_cell = s
            else:  # the output must be the last cell's master: pass it through
                spec.out_cell = cell("MOV", s, s)
        else:
            raise NotAKernel(op)
    if not spec.out_cell:
        raise NotAKernel("the body emits nothing")
    used = {c[k][1] for c in spec.cells for k in ("a", "b") if is_const(c.get(k))}
    spec.consts = {k: v for k, v in spec.consts.items() if k in used}  # folded bases and the induction step are gone
    return spec


def kernel_reference(spec: KernelSpec, tokens: list[int]) -> list[int]:
    """What the pipeline must output for each token, from the same IR semantics."""
    w, mask = spec.width, (1 << spec.width) - 1
    outs = []
    for t in tokens:
        val = {"input": t & mask}

        def get(s):
            if isinstance(s, tuple):
                return spec.consts[s[1]]
            return val[s]

        for c in spec.cells:
            if c["op"] == "LOAD":
                n_words, contents = spec.mems[c["mem"]]
                addr = get(c["a"]) & (n_words - 1)
                if addr not in contents:
                    raise KeyError(f"{c['name']}: {c['mem']}[{addr}] is unwritten (a read would double-rail)")
                val[c["name"]] = contents[addr] & mask
            elif c["op"] == "MOV":
                val[c["name"]] = get(c["b"])
            else:
                a, b = get(c["a"]), get(c["b"])
                if c["op"] in ("ADD", "SUB", "MUL"):
                    val[c["name"]] = IR_OPS[c["op"]](_to_signed(a, w), _to_signed(b, w), w).value & mask
                else:
                    val[c["name"]] = IR_OPS[c["op"]](a, b, w).value & mask
        outs.append(val[spec.out_cell])
    return outs
