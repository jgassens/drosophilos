"""FlyISA intermediate representation (plan §Compiler): a small three-address program over
static variables, with every arithmetic op naming width, signedness and overflow policy
through isa/semantics, explicit ports, bounded calls and no recursion.

    Program: variables (name -> width, address), functions (name -> list of Instr), entry
    Instr:   op, dst, srcs, imm, target

Ops (v0, the subset the neural machine v1 executes; the interpreter executes all of them):
    CONST  dst <- imm
    MOV    dst <- src
    ADD SUB AND OR XOR   dst <- a op b   (WRAP at the variable's width; flags Z/C/V recorded)
    JMP target
    JZ  src, target       jump if src == 0
    JNZ src, target
    CALL f / RET
    LOADX dst / STOREX src  memory[X] with X the index variable __x (arrays)
    IN  dst  <- port       (the host's transduced input, read as a memory word)
    OUT src -> port        (a pixel record: written to a memory-mapped output word)
    HALT
"""

from __future__ import annotations

from dataclasses import dataclass, field

from . import semantics as sem

ALU_OPS = {"ADD": sem.add_wrap, "SUB": sem.sub_wrap, "AND": sem.and_, "OR": sem.or_, "XOR": sem.xor, "MUL": sem.mul_wrap}


@dataclass
class Instr:
    op: str
    dst: str | None = None
    srcs: tuple = ()
    imm: int | None = None
    target: str | None = None  # label or function name

    def __repr__(self) -> str:
        parts = [self.op]
        if self.dst:
            parts.append(self.dst)
        parts += list(self.srcs)
        if self.imm is not None:
            parts.append(f"#{self.imm}")
        if self.target:
            parts.append(f"->{self.target}")
        return " ".join(parts)


@dataclass
class Program:
    width: int
    variables: dict = field(default_factory=dict)  # name -> address (data word)
    functions: dict = field(default_factory=dict)  # name -> list[Instr | str label]
    entry: str = "main"
    ports: dict = field(default_factory=dict)  # port name -> address (memory-mapped)

    def canonical_state(self, memory: dict) -> list[tuple[str, int]]:
        """Declared variable order, values as unsigned width-bit words (plan: canonical serialisation)."""
        return [(v, memory.get(a, 0) & ((1 << self.width) - 1)) for v, a in sorted(self.variables.items(), key=lambda kv: kv[1])]


def _to_signed(x: int, w: int) -> int:
    x &= (1 << w) - 1
    return x - (1 << w) if x >> (w - 1) else x


def interpret(prog: Program, inputs: list[int], max_steps: int = 10000) -> dict:
    """The executable oracle: runs the IR from the entry function. `inputs` are consumed by
    IN in order. Returns the memory (address -> unsigned word), the emitted OUT words in
    order, the flags after the last ALU op, and whether it halted."""
    w = prog.width
    mask = (1 << w) - 1
    mem: dict[int, int] = {}
    outs: list[int] = []
    inputs = list(inputs)
    flags = {"z": 0, "c": 0, "v": 0}
    stack: list[tuple[str, int]] = []
    fn, pc = prog.entry, 0
    steps = 0
    halted = False

    def labels(f):
        return {ins: i for i, ins in enumerate(prog.functions[f]) if isinstance(ins, str)}

    while steps < max_steps:
        body = prog.functions[fn]
        if pc >= len(body):
            if stack:
                fn, pc = stack.pop()
                continue
            break
        ins = body[pc]
        pc += 1
        steps += 1
        if isinstance(ins, str):
            continue
        rd = lambda name: mem.get(prog.variables[name], 0)
        if ins.op == "CONST":
            mem[prog.variables[ins.dst]] = ins.imm & mask
        elif ins.op == "MOV":
            mem[prog.variables[ins.dst]] = rd(ins.srcs[0])
        elif ins.op in ALU_OPS:
            a, b = rd(ins.srcs[0]), (ins.imm & mask if ins.imm is not None else rd(ins.srcs[1]))
            if ins.op in ("ADD", "SUB"):
                r = ALU_OPS[ins.op](_to_signed(a, w), _to_signed(b, w), w)
                val = r.value & mask
                full = a + (b if ins.op == "ADD" else ((~b & mask) + 1))
                flags = {"z": int(val == 0), "c": (full >> w) & 1, "v": r.flags.ovf}
            elif ins.op == "MUL":
                val = ALU_OPS["MUL"](_to_signed(a, w), _to_signed(b, w), w).value & mask
                flags = {"z": int(val == 0), "c": 0, "v": 0}
            else:
                val = ALU_OPS[ins.op](a, b, w).value & mask
                flags = {"z": int(val == 0), "c": 0, "v": 0}
            mem[prog.variables[ins.dst]] = val
        elif ins.op == "JMP":
            pc = labels(fn)[ins.target]
        elif ins.op in ("JZ", "JNZ"):
            z = rd(ins.srcs[0]) == 0
            if (ins.op == "JZ") == z:
                pc = labels(fn)[ins.target]
        elif ins.op == "CALL":
            stack.append((fn, pc))
            fn, pc = ins.target, 0
        elif ins.op == "RET":
            fn, pc = stack.pop()
        elif ins.op == "LOADX":  # dst <- memory[X]
            xa = mem.get(prog.variables["__x"], 0)
            mem[prog.variables[ins.dst]] = mem.get(xa, 0)
        elif ins.op == "STOREX":  # memory[X] <- src
            xa = mem.get(prog.variables["__x"], 0)
            mem[xa] = rd(ins.srcs[0])
        elif ins.op == "IN":
            if not inputs:
                raise RuntimeError("IN with no input available")
            mem[prog.variables[ins.dst]] = inputs.pop(0) & mask
        elif ins.op == "OUT":
            outs.append(ins.imm & mask if ins.imm is not None else rd(ins.srcs[0]))
        elif ins.op == "HALT":
            halted = True
            break
        else:
            raise ValueError(ins.op)
    return {"memory": mem, "outs": outs, "flags": flags, "halted": halted, "steps": steps,
            "state": prog.canonical_state(mem)}
