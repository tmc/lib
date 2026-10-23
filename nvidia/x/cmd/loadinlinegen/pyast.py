# Helper for loadinlinegen, run as: python3 - MODE FILE [NAME].
#
#   extract FILE NAME  prints the value of the top-level string assignment
#                      NAME in FILE, which is parsed, never imported.
#   check FILE         parses FILE and exits 3 unless one of its string
#                      constants contains stdin exactly.
import ast
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
want = sys.stdin.read()
for node in ast.walk(tree):
    if isinstance(node, ast.Constant) and isinstance(node.value, str) and want in node.value:
        sys.exit(0)
sys.exit(3)
