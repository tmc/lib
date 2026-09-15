"""Generate testdata/layout_golden.json from a MoonEP checkout.

The golden values are produced by executing MoonEP's own source text, not by
transcribing it: the helpers _align_up (moonep/api.py), pad_to_granularity and
pad_dim0_for_alignment (moonep/buffer.py) are compiled from their ast nodes,
and the layout arithmetic is the prefix of _create_context's body up to and
including the meta_chunk_padded assignment, executed in a namespace where the
CUDA and torch.distributed dependencies are stubbed out. moonep/_C is a
compiled CUDA extension that cannot be built or imported on a machine without
CUDA, so `import moonep` fails; nothing in the executed prefix needs it beyond
get_vmm_granularity and get_multicast_granularity, which are the granularities
the test parameterizes over anyway.

Usage:

    python3 gen_golden.py [MOONEP_CHECKOUT] > testdata/layout_golden.json
"""

import ast
import json
import subprocess
import sys

DEFAULT_SRC = "/Users/tmc/go/src/github.com/MoonshotAI/MoonEP"

# Each case is (name, S, H, K, E, R, B, token_padding, vmm, mc). B=None means
# the MoonEP default of E/R.
CASES = [
    ("production-8rank", 4096, 7168, 8, 256, 8, None, 128, 1 << 21, 1 << 21),
    ("small-mixed-granularity", 128, 512, 2, 16, 4, 2, 4, 65536, 131072),
    ("no-token-padding", 1024, 2048, 4, 64, 8, None, 1, 1 << 21, 1 << 22),
    ("defaults", 1000, 1536, 6, 96, 8, None, 128, 1 << 21, 1 << 22),
]

KEYS = [
    "N", "NvS", "NvS_padded", "num_vblocks", "WEIGHTS_OFF", "TPE_OFF",
    "PLAN_OFF", "planning_out_elems", "TOPK0_OFF", "ORDER_OFF", "ORDER0_OFF",
    "BARRIER_OFF", "SRC_INFO_OFF", "meta_chunk_logical", "meta_chunk_padded",
    "B", "num_sms",
]


def func_node(path, name):
    for n in ast.parse(open(path).read()).body:
        if isinstance(n, ast.FunctionDef) and n.name == name:
            return n
    raise SystemExit("%s: no function %s" % (path, name))


def exec_node(node, path, ns):
    exec(compile(ast.Module(body=[node], type_ignores=[]), path, "exec"), ns)


def stub_namespace(src, R, vmm, mc):
    api, buf = src + "/moonep/api.py", src + "/moonep/buffer.py"
    ns = {}
    for path, name in ((api, "_align_up"), (buf, "pad_to_granularity"),
                       (buf, "pad_dim0_for_alignment")):
        exec_node(func_node(path, name), path, ns)

    class stub:
        pass

    torch = stub()
    torch.bfloat16, torch.float32, torch.int32 = "bf16", "f32", "i32"
    torch.cuda = stub()
    torch.cuda.current_device = lambda: 0
    torch.cuda.get_device_properties = lambda d: type(
        "props", (), {"multi_processor_count": 132})()
    dist = stub()
    dist.get_rank = lambda group=None: 0
    dist.get_world_size = lambda group=None: R

    ns.update(
        torch=torch,
        dist=dist,
        _ELEM_SIZE={"bf16": 2, "f32": 4, "i32": 4},
        get_vmm_granularity=lambda: vmm,
        get_multicast_granularity=lambda r: mc,
        _num_sms_dedup_from_env=lambda max_sms: max_sms,
    )
    return ns


def layout(src, S, H, K, E, R, B, token_padding, vmm, mc):
    api = src + "/moonep/api.py"
    ns = stub_namespace(src, R, vmm, mc)
    ns.update(S=S, H=H, K=K, E=E, num_ep_ranks=R, num_sms=None,
              token_padding=token_padding, B=B, group=None)

    body = func_node(api, "_create_context").body
    for i, st in enumerate(body):
        if isinstance(st, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "meta_chunk_padded"
                for t in st.targets):
            break
    else:
        raise SystemExit("api.py: _create_context has no meta_chunk_padded assignment")
    exec(compile(ast.Module(body=body[:i + 1], type_ignores=[]), api, "exec"), ns)
    return {k: int(ns[k]) for k in KEYS}


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_SRC
    rev = subprocess.run(["git", "-C", src, "rev-parse", "--short", "HEAD"],
                         capture_output=True, text=True, check=True).stdout.strip()
    cases = []
    for name, S, H, K, E, R, B, tp, vmm, mc in CASES:
        cases.append({
            "name": name,
            "config": {"S": S, "H": H, "K": K, "E": E, "R": R, "B": B,
                       "token_padding": tp},
            "vmm_granularity": vmm,
            "mc_granularity": mc,
            "want": layout(src, S, H, K, E, R, B, tp, vmm, mc),
        })
    json.dump({
        "source": "github.com/MoonshotAI/MoonEP",
        "revision": rev,
        "generator": "gen_golden.py",
        "cases": cases,
    }, sys.stdout, indent="\t")
    sys.stdout.write("\n")


main()
