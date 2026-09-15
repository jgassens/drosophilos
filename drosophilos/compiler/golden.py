"""The portable C reference for a DrosoC program (plan: portable C <-> IR interpreter <->
neural execution). The program is compiled with clang -fsanitize=undefined together with
a runtime shim: `in_read()` pops the given inputs, `out_pixel(v)` records v, and at exit
the canonical state (declared static variables in declaration order, as unsigned words)
and the outputs are printed as JSON."""

from __future__ import annotations

import json
import re
import subprocess
import tempfile
from pathlib import Path

SHIM = r"""
#include <stdio.h>
#include <stdlib.h>
typedef unsigned char u8; typedef unsigned short u16; typedef unsigned int u32;
typedef signed char i8; typedef short i16; typedef int i32;
static int __inputs[64]; static int __n_in = 0, __i_in = 0;
static int __outs[256]; static int __n_out = 0;
WORD in_read(void) { if (__i_in >= __n_in) { fprintf(stderr, "IN with no input\n"); exit(3); } return (WORD)__inputs[__i_in++]; }
void out_pixel(WORD v) { if (__n_out < 256) __outs[__n_out++] = v; }
"""  # WORD: the program's width (u8/u16/u32); a fixed u8 truncated wider ports (review finding)


def run_golden(source: str, inputs: list[int], variables: list[str], width: int) -> dict:
    mask = (1 << width) - 1
    src = SHIM.replace("WORD", {8: "u8", 16: "u16", 32: "u32"}[width]) + source.replace("int main(void)", "int __user_main(void)")
    src += "\nint main(int argc, char **argv) {\n  for (int i = 1; i < argc; i++) __inputs[__n_in++] = atoi(argv[i]);\n  __user_main();\n"
    src += '  printf("{\\"state\\": [");\n'
    for k, v in enumerate(variables):
        sep = '""' if k == 0 else '","'
        src += f'  printf("%s%u", {sep}, (unsigned)({v} & {mask}u));\n'  # a[k] spells as C too
    src += '  printf("], \\"outs\\": [");\n  for (int i = 0; i < __n_out; i++) printf("%s%d", i ? "," : "", __outs[i]);\n  printf("]}\\n");\n  return 0;\n}\n'
    with tempfile.TemporaryDirectory() as d:
        c = Path(d) / "prog.c"
        c.write_text(src)
        exe = Path(d) / "prog"
        import shutil
        compiler = next((x for x in ("clang", "gcc", "cc") if shutil.which(x)), None)
        if compiler is None:
            raise RuntimeError("no C compiler (clang/gcc/cc) on PATH for the golden reference")
        cc = subprocess.run([compiler, "-std=c11", "-O1", "-fsanitize=undefined", "-fno-sanitize-recover=all", "-o", str(exe), str(c)],
                            capture_output=True, text=True)
        if cc.returncode:
            raise RuntimeError("clang failed:\n" + cc.stderr + "\n" + src)
        out = subprocess.run([str(exe)] + [str(x) for x in inputs], capture_output=True, text=True, check=True).stdout
    res = json.loads(out)
    return {"state": list(zip(variables, res["state"])), "outs": res["outs"]}
