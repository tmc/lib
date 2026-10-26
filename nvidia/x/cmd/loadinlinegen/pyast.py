# Helper for loadinlinegen, run as: python3 - MODE FILE [NAME].
#
#   extract FILE NAME  prints the value of the top-level string assignment
#                      NAME in FILE, which is parsed, never imported.
#   fragment FILE      extracts FUNCTIONS (list of str), CPP (str), CUDA (str)
#                      and optional PY (str) from top-level assignments in FILE
#                      and prints them as JSON.
#   check FILE         parses FILE and exits 3 unless one of its string
#                      constants contains stdin exactly.
import ast
import json
import sys

mode, path = sys.argv[1], sys.argv[2]
with open(path, encoding="utf-8") as f:
    tree = ast.parse(f.read(), path)

if mode == "extract":
    name = sys.argv[3]
    for node in tree.body:
        if (isinstance(node, ast.Assign) and len(node.targets) == 1 and
                isinstance(node.targets[0], ast.Name) and node.targets[0].id == name):
            if not (isinstance(node.value, ast.Constant) and isinstance(node.value.value, str)):
                sys.exit(f"{path}:{node.lineno}: {name} is not a string literal")
            sys.stdout.write(node.value.value)
            sys.exit(0)
    sys.exit(f"{path}: no top-level assignment to {name}")

if mode == "fragment":
    assigns = {}
    lines = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            assigns[node.targets[0].id] = node.value
            lines[node.targets[0].id] = node.lineno

    def get_str(names, required=False):
        for n in names:
            if n in assigns:
                v = assigns[n]
                if not (isinstance(v, ast.Constant) and isinstance(v.value, str)):
                    sys.exit(f"{path}:{lines[n]}: {n} is not a string literal")
                return v.value
        if required:
            sys.exit(f"{path}: fragment is missing {'/'.join(names)}")
        return ""

    funcs = []
    if "FUNCTIONS" in assigns:
        v = assigns["FUNCTIONS"]
        if not isinstance(v, (ast.List, ast.Tuple)):
            sys.exit(f"{path}:{lines['FUNCTIONS']}: FUNCTIONS is not a list or tuple of strings")
        for elt in v.elts:
            if not (isinstance(elt, ast.Constant) and isinstance(elt.value, str)):
                sys.exit(f"{path}:{lines['FUNCTIONS']}: FUNCTIONS elements must be string literals")
            funcs.append(elt.value)

    out = {
        "functions": funcs,
        "cpp": get_str(["CPP", "_CPP_DECL", "CPP_SRC"], required=False),
        "cuda": get_str(["CUDA", "_CUDA_SRC", "CUDA_SRC"], required=True),
        "py": get_str(["PY", "PY_SRC"], required=False),
    }
    sys.stdout.write(json.dumps(out))
    sys.exit(0)

want = sys.stdin.read()
for node in ast.walk(tree):
    if isinstance(node, ast.Constant) and isinstance(node.value, str) and want in node.value:
        sys.exit(0)
sys.exit(3)
