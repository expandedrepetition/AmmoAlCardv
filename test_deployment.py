"""Offline Hugging Face deployment-shape checks; does not download or run model weights."""
from __future__ import annotations

import ast
import os
from pathlib import Path

import app

ROOT = Path(__file__).resolve().parent


def check(name: str, condition: bool) -> None:
    if not condition:
        raise AssertionError(name)
    print(f"PASS {name}")


def has_spaces_gpu_decorator(tree: ast.AST, function_name: str) -> bool:
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) or node.name != function_name:
            continue
        for dec in node.decorator_list:
            target = dec.func if isinstance(dec, ast.Call) else dec
            if (isinstance(target, ast.Attribute) and target.attr == "GPU"
                    and isinstance(target.value, ast.Name) and target.value.id == "spaces"):
                return True
    return False


def main() -> int:
    chat_source = (ROOT / "chat_app.py").read_text(encoding="utf-8")
    chat_tree = ast.parse(chat_source)
    check("chat callback has a literal spaces.GPU decorator", has_spaces_gpu_decorator(chat_tree, "respond"))
    check("Gradio SDK selected for ZeroGPU Space", "sdk: gradio" in (ROOT / "README.md").read_text(encoding="utf-8"))
    req = (ROOT / "requirements.txt").read_text(encoding="utf-8")
    check("requirements constrain torch to currently documented ZeroGPU versions", "torch>=2.8.0,<=2.13.0" in req)

    # Restore every touched environment variable so this module can be imported by other tests.
    old_size = os.environ.get("EIM_GPU_SIZE")
    old_seconds = os.environ.get("EIM_GPU_SECONDS")
    try:
        os.environ["EIM_GPU_SIZE"] = "large"
        os.environ["EIM_GPU_SECONDS"] = "120"
        check("large allocation allows at most 120 seconds", app._gpu_duration_seconds() == 120)
        os.environ["EIM_GPU_SECONDS"] = "999"
        check("large allocation clamps oversized requests", app._gpu_duration_seconds() == 120)
        os.environ["EIM_GPU_SIZE"] = "xlarge"
        os.environ["EIM_GPU_SECONDS"] = "120"
        check("xlarge allocation clamps duration to 60 seconds", app._gpu_duration_seconds() == 60)
        os.environ["EIM_GPU_SECONDS"] = "bad"
        check("invalid duration value falls back safely", app._gpu_duration_seconds() == 60)
    finally:
        if old_size is None:
            os.environ.pop("EIM_GPU_SIZE", None)
        else:
            os.environ["EIM_GPU_SIZE"] = old_size
        if old_seconds is None:
            os.environ.pop("EIM_GPU_SECONDS", None)
        else:
            os.environ["EIM_GPU_SECONDS"] = old_seconds

    # A changed repair policy must change the result-cache fingerprint to avoid stale prompts.
    import inspect
    eim_source = inspect.getsource(__import__("eim_plus").EIMPlus.run)
    check("result-cache fingerprint includes repair rules", "_repair_rules()" in eim_source)
    check("default model is the planned FP8 Qwen checkpoint", app.Config().model_name == "Qwen/Qwen3-4B-Instruct-2507-FP8")
    print("\nDeployment checks passed; live GPU/model inference remains untested in this local environment.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
