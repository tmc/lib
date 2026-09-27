r"""NVTX ranges named by PyTorch module path, for cudaprof compare -by module.

    import torchnvtx
    torchnvtx.install(model, r"^(encoder\.layers\.\d+|head)$")
    ...
    with torchnvtx.range("optimizer"):
        opt.step()

Each module whose name matches gets a range "NAME:fwd" around its forward
pass and "NAME:bwd" around its backward pass, the names a Go trainer emits
with nvtx.RangePush, so two captures line up module by module.

Two things make the backward ranges need care under non-reentrant gradient
checkpointing (torch.utils.checkpoint with use_reentrant=False):

  - The recompute runs the forward hooks again inside backward. install
    skips forward hooks while a backward graph task is running, so the
    recompute counts toward the module's ":bwd" range, as it does in a
    trainer that recomputes by hand.
  - The full backward hook of a checkpointed module never fires, so its
    ":bwd" range cannot be closed there. Ranges of checkpointed modules
    (the checkpointed pattern) are closed instead when the next backward
    range opens or an enclosing one closes, or by close_backward after
    loss.backward(). The pattern should therefore not also select modules
    inside a checkpointed one.

Ranges are no-ops unless a profiler injects NVTX, as cudaprof capture does.
"""

import contextlib
import re
import threading

import torch

_local = threading.local()


def _stack():
    if not hasattr(_local, "stack"):
        _local.stack = []
    return _local.stack


def _in_backward():
    return torch._C._current_graph_task_id() != -1


def install(model, pattern, checkpointed=None):
    """Add NVTX ranges to the modules of model whose names match pattern.

    checkpointed, a pattern, names the modules run under non-reentrant
    gradient checkpointing; their backward ranges close as described in the
    module doc. It returns the hook handles.
    """
    push, pop = torch.cuda.nvtx.range_push, torch.cuda.nvtx.range_pop
    want = re.compile(pattern)
    ckpt = re.compile(checkpointed) if checkpointed else None
    stack = _stack

    def is_ckpt_bwd(n):
        return ckpt is not None and n.endswith(":bwd") and ckpt.match(n[: -len(":bwd")])

    def close_ckpt():
        s = stack()
        while s and is_ckpt_bwd(s[-1]):
            s.pop()
            pop()

    def open_(n):
        if n.endswith(":bwd"):
            close_ckpt()
        stack().append(n)
        push(n)

    def close():
        close_ckpt()
        stack().pop()
        pop()

    def fwd_open(n):
        if not _in_backward():
            open_(n)

    def fwd_close():
        if not _in_backward():
            close()

    handles = []
    for name, mod in model.named_modules():
        if not want.match(name):
            continue
        handles.append(mod.register_forward_pre_hook(lambda m, a, n=name: fwd_open(n + ":fwd")))
        handles.append(mod.register_forward_hook(lambda m, a, o: fwd_close()))
        handles.append(mod.register_full_backward_pre_hook(lambda m, g, n=name: open_(n + ":bwd")))
        if not (ckpt and ckpt.match(name)):
            handles.append(mod.register_full_backward_hook(lambda m, gi, go: close()))
    return handles


def close_backward():
    """Close the backward ranges of checkpointed modules still open."""
    s = _stack()
    while s and s[-1].endswith(":bwd"):
        s.pop()
        torch.cuda.nvtx.range_pop()


@contextlib.contextmanager
def range(name):
    """Hold an NVTX range named name, such as "optimizer"."""
    torch.cuda.nvtx.range_push(name)
    try:
        yield
    finally:
        torch.cuda.nvtx.range_pop()
