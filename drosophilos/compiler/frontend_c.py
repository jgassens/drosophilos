"""DrosoC front end (plan §Compiler): pycparser -> FlyISA IR for the v0 subset.

Subset accepted (everything else is rejected with a message):
    typedef'd `u8`/`u16`/`u32` scalars only (the program's one width); `static` globals
    with optional constant initialisers; functions `void f(void)` without parameters or
    return values, called by name (no recursion: checked); `int main(void)`; statements:
    assignment, compound statement, `if`/`else`, `while`, expression statements that are
    calls, `return` (from main = halt), `out_pixel(expr)`, `x = in_read()`; expressions:
    variables, integer constants, `+ - & | ^` (wrapping at the width), `!e`, `e == 0`,
    `e != 0`, `a == b`, `a != b` (lowered to XOR), `a + b` with nested subexpressions
    through temporaries.
Runtime intrinsics `in_read()` and `out_pixel()` are ports, not calls.
"""

from __future__ import annotations

from pycparser import c_ast, c_parser

from ..isa.ir import Instr, Program

WIDTHS = {"u8": 8, "u16": 16, "u32": 32, "i8": 8, "i16": 16, "i32": 32}
PRELUDE = "typedef unsigned char u8; typedef unsigned short u16; typedef unsigned int u32;\n" \
          "typedef signed char i8; typedef short i16; typedef int i32;\n" \
          "u8 in_read(void); void out_pixel(u8 v);\n"
BINOPS = {"+": "ADD", "-": "SUB", "&": "AND", "|": "OR", "^": "XOR"}


class Unsupported(Exception):
    pass


class Compiler:
    def __init__(self):
        self.prog = None
        self.tmp = 0
        self.label = 0
        self.body: list = []
        self.calls: dict[str, set[str]] = {}

    # ---------------------------------------------------------------- program
    def compile(self, source: str, width_hint: int | None = None) -> Program:
        ast = c_parser.CParser().parse(PRELUDE + source)
        globals_, funcs = [], []
        for ext in ast.ext:
            if isinstance(ext, c_ast.Decl) and isinstance(ext.type, c_ast.TypeDecl):
                globals_.append(ext)
            elif isinstance(ext, c_ast.FuncDef):
                funcs.append(ext)
            elif isinstance(ext, (c_ast.Typedef, c_ast.Decl)):
                continue
            else:
                raise Unsupported(type(ext).__name__)
        widths = set()
        for d in globals_:
            widths.add(self._width(d.type))
        width = width_hint or (widths.pop() if len(widths) == 1 else 8)
        if widths and any(w_ != width for w_ in widths):
            raise Unsupported("all variables must share one width in v0")
        self.prog = Program(width=width)
        addr = 0
        inits = []
        for d in globals_:
            self.prog.variables[d.name] = addr
            addr += 1
            if d.init is not None:
                inits.append((d.name, self._const(d.init)))
        self.prog.ports = {"in": addr, "out": addr + 1}
        self.prog.variables["__in"] = addr
        self.prog.variables["__out"] = addr + 1
        for f in funcs:
            self.body, name = [], f.decl.name
            self.calls[name] = set()
            if name == "main":
                for v, k in inits:
                    self.body.append(Instr("CONST", v, imm=k))
            self._stmt(f.body, name)
            if name == "main":
                self.body.append(Instr("HALT"))
            else:
                self.body.append(Instr("RET"))
            self.prog.functions[name] = self.body
        self._check_recursion()
        return self.prog

    # ---------------------------------------------------------------- helpers
    def _width(self, t) -> int:
        names = t.type.names if isinstance(t.type, c_ast.IdentifierType) else []
        for nm in names:
            if nm in WIDTHS:
                return WIDTHS[nm]
        raise Unsupported(f"type {names}")

    def _const(self, node) -> int:
        if isinstance(node, c_ast.Constant) and node.type == "int":
            return int(node.value, 0)
        if isinstance(node, c_ast.UnaryOp) and node.op == "-":
            return -self._const(node.expr)
        raise Unsupported("constant expected")

    def _newtmp(self) -> str:
        name = f"__t{self.tmp}"
        self.tmp += 1
        if name not in self.prog.variables:
            self.prog.variables[name] = max(self.prog.variables.values()) + 1
        return name

    def _newlabel(self, kind: str) -> str:
        self.label += 1
        return f"{kind}{self.label}"

    def _check_recursion(self):
        def reach(f, seen):
            for g in self.calls.get(f, ()):
                if g in seen:
                    raise Unsupported(f"recursion through {g}")
                reach(g, seen | {g})
        for f in self.calls:
            reach(f, {f})

    # ---------------------------------------------------------------- statements
    def _stmt(self, node, fn: str):
        if isinstance(node, c_ast.Compound):
            for item in node.block_items or []:
                self._stmt(item, fn)
        elif isinstance(node, c_ast.Assignment):
            if node.op != "=" or not isinstance(node.lvalue, c_ast.ID):
                raise Unsupported(f"assignment {node.op}")
            self._expr_into(node.rvalue, node.lvalue.name)
        elif isinstance(node, c_ast.FuncCall):
            name = node.name.name
            if name == "out_pixel":
                src = self._expr(node.args.exprs[0])
                self.body.append(Instr("OUT", srcs=(src,)))
            elif name == "in_read":
                raise Unsupported("in_read() result must be assigned")
            else:
                if node.args is not None and node.args.exprs:
                    raise Unsupported("calls take no arguments in v0")
                self.calls[fn].add(name)
                self.body.append(Instr("CALL", target=name))
        elif isinstance(node, c_ast.If):
            cond, sense = self._cond(node.cond)
            l_else, l_end = self._newlabel("else"), self._newlabel("endif")
            self.body.append(Instr("JZ" if sense else "JNZ", srcs=(cond,), target=l_else))
            self._stmt(node.iftrue, fn)
            if node.iffalse is not None:
                self.body.append(Instr("JMP", target=l_end))
            self.body.append(l_else)
            if node.iffalse is not None:
                self._stmt(node.iffalse, fn)
                self.body.append(l_end)
        elif isinstance(node, c_ast.While):
            l_top, l_end = self._newlabel("loop"), self._newlabel("endloop")
            self.body.append(l_top)
            cond, sense = self._cond(node.cond)
            self.body.append(Instr("JZ" if sense else "JNZ", srcs=(cond,), target=l_end))
            self._stmt(node.stmt, fn)
            self.body.append(Instr("JMP", target=l_top))
            self.body.append(l_end)
        elif isinstance(node, c_ast.Return):
            self.body.append(Instr("HALT" if fn == "main" else "RET"))
        elif isinstance(node, c_ast.Decl):
            raise Unsupported("local variables: declare statics at file scope in v0")
        elif isinstance(node, c_ast.EmptyStatement):
            pass
        else:
            raise Unsupported(type(node).__name__)

    def _cond(self, node) -> tuple[str, bool]:
        """Returns (variable, sense): the branch is taken when variable != 0 if sense else == 0."""
        if isinstance(node, c_ast.UnaryOp) and node.op == "!":
            v, s = self._cond(node.expr)
            return v, not s
        if isinstance(node, c_ast.BinaryOp) and node.op in ("==", "!="):
            if self._is_zero(node.right):
                v = self._expr(node.left)
                return v, node.op == "!="
            t = self._newtmp()
            a = self._expr(node.left)
            self._binop("XOR", t, a, node.right)
            return t, node.op == "!="
        return self._expr(node), True

    def _is_zero(self, node) -> bool:
        return isinstance(node, c_ast.Constant) and node.type == "int" and int(node.value, 0) == 0

    # ---------------------------------------------------------------- expressions
    def _expr(self, node) -> str:
        """Evaluates into a variable (existing, or a temporary) and returns its name."""
        if isinstance(node, c_ast.ID):
            if node.name not in self.prog.variables:
                raise Unsupported(f"unknown variable {node.name}")
            return node.name
        t = self._newtmp()
        self._expr_into(node, t)
        return t

    def _expr_into(self, node, dst: str):
        if isinstance(node, c_ast.Constant):
            self.body.append(Instr("CONST", dst, imm=self._const(node)))
        elif isinstance(node, c_ast.ID):
            self.body.append(Instr("MOV", dst, srcs=(node.name,)))
        elif isinstance(node, c_ast.FuncCall) and node.name.name == "in_read":
            self.body.append(Instr("IN", dst))
        elif isinstance(node, c_ast.BinaryOp) and node.op in BINOPS:
            a = self._expr(node.left)
            self._binop(BINOPS[node.op], dst, a, node.right)
        elif isinstance(node, c_ast.UnaryOp) and node.op == "-":
            self.body.append(Instr("CONST", dst, imm=self._const(node)))
        else:
            raise Unsupported(f"expression {type(node).__name__} {getattr(node, 'op', '')}")

    def _binop(self, op: str, dst: str, a: str, right):
        if isinstance(right, c_ast.Constant):
            self.body.append(Instr(op, dst, srcs=(a,), imm=self._const(right)))
        else:
            b = self._expr(right)
            self.body.append(Instr(op, dst, srcs=(a, b)))


def compile_c(source: str, width: int | None = None) -> Program:
    return Compiler().compile(source, width)
