"""Lowering FlyISA IR to the neural control machine's instruction words (lib/control.py).

The machine is an accumulator machine: every IR op becomes a short sequence
    dst <- CONST k        MOV k ; STORE dst
    dst <- MOV a          LOAD a ; STORE dst
    dst <- a op b         LOAD a ; OP b (immediate) or OPM [b] ; STORE dst
    JZ/JNZ v, L           LOAD v ; JZ/JNZ L        (LOAD commits acc = v, so Z = (v == 0))
    JMP L                 JMP L
    IN dst                LOAD [port_in] ; STORE dst
    OUT src               LOAD src ; STORE [port_out]
    LOADX/STOREX          LOADI ; STORE dst   /   LOAD src ; STOREI   (X = the __x data word)
    CALL f                the callee's body inlined (no recursion: checked by the front end)
    HALT                  JMP self
The program starts with MOV 0: the accumulator is dark at power-up and a dark operand lights
both A rails in the operand gate (a JMP/JZ/STORE runs OR 0 and would fault).
Variables and ports are data-memory addresses; the IR's CALL nesting is flattened, so the
machine program is one straight sequence with absolute jump targets.
"""

from __future__ import annotations

from ..isa.ir import Instr, Program


def lower(prog: Program) -> list[tuple]:
    words: list = []  # (op, arg) or ("__label", name)
    labels: dict[str, int] = {}
    inline_id = [0]

    def emit(op, arg=0):
        words.append((op, arg))

    def place(label):
        labels[label] = len(words)

    def body(fn: str, suffix: str):
        for ins in prog.functions[fn]:
            if isinstance(ins, str):
                place(f"{ins}{suffix}")
                continue
            op = ins.op
            addr = lambda v: prog.variables[v]
            if op == "CONST":
                emit("MOV", ins.imm & ((1 << prog.width) - 1)); emit("STORE", addr(ins.dst))
            elif op == "MOV":
                emit("LOAD", addr(ins.srcs[0])); emit("STORE", addr(ins.dst))
            elif op in ("ADD", "SUB", "AND", "OR", "XOR", "MUL"):
                emit("LOAD", addr(ins.srcs[0]))
                if ins.imm is not None:
                    emit(op, ins.imm & ((1 << prog.width) - 1))
                else:
                    emit(op + "M", addr(ins.srcs[1]))
                emit("STORE", addr(ins.dst))
            elif op in ("JZ", "JNZ"):
                emit("LOAD", addr(ins.srcs[0])); emit(op, f"{ins.target}{suffix}")
            elif op == "JMP":
                emit("JMP", f"{ins.target}{suffix}")
            elif op == "LOADX":
                emit("LOADI", 0); emit("STORE", addr(ins.dst))
            elif op == "STOREX":
                emit("LOAD", addr(ins.srcs[0])); emit("STOREI", 0)
            elif op == "IN":
                emit("LOAD", prog.ports["in"]); emit("STORE", addr(ins.dst))
            elif op == "OUT":
                if ins.imm is not None:
                    emit("MOV", ins.imm & ((1 << prog.width) - 1))
                else:
                    emit("LOAD", addr(ins.srcs[0]))
                emit("STORE", prog.ports["out"])
            elif op == "CALL":
                inline_id[0] += 1
                body(ins.target, f"{suffix}@{inline_id[0]}")
            elif op == "RET":
                pass  # the inlined body simply continues
            elif op == "HALT":
                emit("HALT", len(words))
            else:
                raise ValueError(op)

    emit("MOV", 0)  # prologue: the accumulator is dark at power-up and must be written before any OR/JZ/STORE
    body(prog.entry, "")
    out = []
    for k, (op, arg) in enumerate(words):
        if isinstance(arg, str):
            arg = labels[arg]
        out.append((op, arg))
    return out
