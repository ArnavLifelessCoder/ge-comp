"""PyTorch .pt/.pth/.bin loader.

Two independent things happen here and they must stay independent:

1. OPCODE SCAN (no execution). We open the embedded pickle(s) and walk their
   opcodes with pickletools.genops. We NEVER unpickle to inspect. Any GLOBAL /
   REDUCE / STACK_GLOBAL referencing a dangerous callable is recorded as a
   container note for the serialization detector. This is the safe way to read a
   potentially malicious pickle: parse the bytecode, do not run it.

2. TENSOR EXTRACTION. Only after the scan, and only to read weight VALUES, we
   load tensors with weights_only=True, which refuses arbitrary globals. If that
   fails we still return the container notes so the serialization detector works
   on files we deliberately refuse to fully load.
"""
from __future__ import annotations

import io
import pickletools
import zipfile
from ..tensorgraph import Tensor, TensorGraph, infer_role

# Fully-qualified callables that must never appear in a weights-only artifact.
_DANGEROUS = {
    "posix.system", "nt.system", "os.system", "os.popen", "os.execv",
    "subprocess.Popen", "subprocess.call", "subprocess.run", "subprocess.check_output",
    "builtins.exec", "builtins.eval", "builtins.__import__", "builtins.getattr",
    "builtins.compile", "runpy._run_code", "importlib.import_module",
    "pty.spawn", "socket.socket", "webbrowser.open",
}
# Dangerous by function name regardless of module (covers aliased imports).
_DANGEROUS_FUNCS = {
    "system", "popen", "exec", "eval", "Popen", "call", "run", "spawn",
    "check_output", "__import__", "_run_code", "import_module",
}


def _scan_pickle_opcodes(data: bytes, source: str) -> list[dict]:
    """Walk opcodes WITHOUT unpickling. Track a lightweight literal stack so we
    can resolve protocol-4 STACK_GLOBAL, whose module/name are pushed as two
    preceding string constants rather than carried in the opcode arg."""
    notes = []
    lit_stack = []          # recent string literals
    last_global = None
    try:
        for opcode, arg, pos in pickletools.genops(data):
            nm = opcode.name
            if nm in ("SHORT_BINUNICODE", "BINUNICODE", "BINUNICODE8",
                      "SHORT_BINSTRING", "BINSTRING", "UNICODE", "STRING"):
                lit_stack.append(arg if isinstance(arg, str) else str(arg))
                if len(lit_stack) > 8:
                    lit_stack.pop(0)
            elif nm == "GLOBAL":
                # proto <= 1: arg is "module name" (space or newline separated)
                mod_func = arg.replace(" ", ".").replace("\n", ".") if isinstance(arg, str) else str(arg)
                last_global = mod_func
                _check_global(mod_func, notes, source, pos)
            elif nm == "STACK_GLOBAL":
                # proto >= 2: module, name are the two most recent literals
                if len(lit_stack) >= 2:
                    mod, name = lit_stack[-2], lit_stack[-1]
                else:
                    mod, name = "?", (lit_stack[-1] if lit_stack else "?")
                mod_func = f"{mod}.{name}"
                last_global = mod_func
                _check_global(mod_func, notes, source, pos)
            elif nm == "REDUCE" and last_global:
                func = last_global.split(".")[-1]
                if last_global in _DANGEROUS or func in _DANGEROUS_FUNCS:
                    notes.append({"source": source, "severity": "critical", "pos": pos,
                                  "detail": f"REDUCE invokes {last_global}()"})
    except Exception as e:
        notes.append({"source": source, "severity": "info", "pos": -1,
                      "detail": f"opcode scan incomplete: {e}"})
    return notes


def _check_global(mod_func: str, notes: list, source: str, pos: int):
    func = mod_func.split(".")[-1]
    if mod_func in _DANGEROUS or func in _DANGEROUS_FUNCS:
        notes.append({"source": source, "severity": "critical", "pos": pos,
                      "detail": f"references dangerous global {mod_func}"})


def _iter_pickles(path: str):
    """Yield (source_name, pickle_bytes) for every pickle stream in the file."""
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as z:
            for nm in z.namelist():
                if nm.endswith(".pkl") or nm.endswith("data.pkl"):
                    yield nm, z.read(nm)
    else:
        with open(path, "rb") as f:
            yield "data.pkl", f.read()


def load(path: str) -> TensorGraph:
    g = TensorGraph(path=path, fmt="pytorch")

    for src, data in _iter_pickles(path):
        g.container_notes.extend(_scan_pickle_opcodes(data, src))

    # Tensor extraction, safe path only.
    try:
        import torch, warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            state = torch.load(path, map_location="cpu", weights_only=True)
        if hasattr(state, "state_dict"):
            state = state.state_dict()
        if isinstance(state, dict):
            for name, val in state.items():
                if not hasattr(val, "detach"):
                    continue
                arr = val.detach().cpu().contiguous().numpy()
                g.tensors.append(Tensor(name=str(name), dtype=str(arr.dtype),
                                        shape=tuple(arr.shape),
                                        raw=arr.tobytes(),
                                        role=infer_role(str(name), arr.shape)))
    except Exception as e:
        g.container_notes.append({"source": "loader", "severity": "info", "pos": -1,
                                  "detail": f"weights_only load failed, "
                                            f"container scanned only: {e}"})
    return g
