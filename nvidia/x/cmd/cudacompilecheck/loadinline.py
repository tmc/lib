# Prints, as JSON, the CUDA sources and nvcc flags of every
# torch.utils.cpp_extension.load_inline call in a Python file, without
# importing it. Arguments are evaluated statically: string and list
# literals, names bound by top-level assignments, and + between them.
# Anything else (an f-string, a call) is reported as an error for that call.
import ast
import json
import sys


class Unresolved(Exception):
    pass


def main():
    path = sys.argv[1]
    with open(path, encoding="utf-8") as f:
        tree = ast.parse(f.read(), path)
    top = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            top[node.targets[0].id] = node.value
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.value is not None:
            top[node.target.id] = node.value

    def ev(node, depth=0):
        if depth > 50:
            raise Unresolved("names nest too deeply")
        if isinstance(node, ast.Constant) and isinstance(node.value, (str, bool)) or (
                isinstance(node, ast.Constant) and node.value is None):
            return node.value
        if isinstance(node, ast.Name):
            if node.id not in top:
                raise Unresolved(f"line {node.lineno}: {node.id} is not a top-level assignment")
            return ev(top[node.id], depth + 1)
        if isinstance(node, (ast.List, ast.Tuple)):
            return [ev(e, depth + 1) for e in node.elts]
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            a, b = ev(node.left, depth + 1), ev(node.right, depth + 1)
            if type(a) is not type(b) or not isinstance(a, (str, list)):
                raise Unresolved(f"line {node.lineno}: + of {type(a).__name__} and {type(b).__name__}")
            return a + b
        raise Unresolved(f"line {node.lineno}: cannot evaluate {type(node).__name__} statically")

    calls = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        if not (isinstance(fn, ast.Name) and fn.id == "load_inline" or
                isinstance(fn, ast.Attribute) and fn.attr == "load_inline"):
            continue
        call = {"line": node.lineno}
        kw = {k.arg: k.value for k in node.keywords if k.arg}
        try:
            if "name" in kw:
                call["name"] = ev(kw["name"])
            elif node.args:
                call["name"] = ev(node.args[0])
            src = kw.get("cuda_sources")
            if src is None and len(node.args) > 2:
                src = node.args[2]
            if src is None:
                raise Unresolved("no cuda_sources")
            src = ev(src)
            call["cuda_sources"] = [src] if isinstance(src, str) else src
            if not all(isinstance(s, str) for s in call["cuda_sources"]):
                raise Unresolved("cuda_sources is not a string or a list of strings")
            if "extra_cuda_cflags" in kw:
                call["extra_cuda_cflags"] = ev(kw["extra_cuda_cflags"]) or []
            if "no_implicit_headers" in kw:
                call["no_implicit_headers"] = bool(ev(kw["no_implicit_headers"]))
        except Unresolved as e:
            call["error"] = str(e)
        calls.append(call)
    json.dump(calls, sys.stdout)


main()
