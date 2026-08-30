"""Fail-closed runtime and CUDA-kernel verification for the pinned H100 stack."""

from __future__ import annotations

import importlib
import json
import os
import platform
import sys
from importlib.metadata import version
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agents.launcher import _resolve_c_compiler, _verify_triton_cuda_utils  # noqa: E402

EXPECTED_DISTRIBUTIONS = {
    "datasets": "4.0.0",
    "flash-attn": "2.8.0.post2",
    "pyarrow": "20.0.0",
    "transformers": "4.53.2",
    "triton": "3.3.0",
    "verl": "0.7.0",
    "vllm": "0.9.2",
}


def main() -> int:
    assert platform.python_implementation() == "CPython"
    assert sys.version_info[:3] == (3, 10, 20), platform.python_version()
    assert platform.system() == "Linux", platform.system()
    assert platform.machine() == "x86_64", platform.machine()

    c_compiler, c_compiler_version = _resolve_c_compiler()
    os.environ["CC"] = c_compiler
    _verify_triton_cuda_utils(c_compiler)

    actual_versions: dict[str, str] = {}
    for distribution, expected in EXPECTED_DISTRIBUTIONS.items():
        importlib.import_module(distribution.replace("-", "_"))
        actual = version(distribution)
        assert actual == expected, f"{distribution}: expected {expected}, got {actual}"
        actual_versions[distribution] = actual

    assert torch.__version__ == "2.7.0+cu128", torch.__version__
    assert torch.version.cuda == "12.8", torch.version.cuda
    assert torch.cuda.is_available(), "CUDA is unavailable"
    gpu_name = torch.cuda.get_device_name(0)
    assert "H100" in gpu_name, gpu_name

    flash_attn = importlib.import_module("flash_attn")
    qkv = torch.randn(1, 16, 3, 4, 64, device="cuda", dtype=torch.float16)
    output = flash_attn.flash_attn_qkvpacked_func(qkv)
    torch.cuda.synchronize()
    assert tuple(output.shape) == (1, 16, 4, 64), tuple(output.shape)
    assert bool(torch.isfinite(output).all().item()), "FlashAttention output is not finite"

    print(
        json.dumps(
            {
                "cuda_runtime": torch.version.cuda,
                "c_compiler": {
                    "path": c_compiler,
                    "version": c_compiler_version,
                },
                "distributions": actual_versions,
                "flash_attention": {
                    "dtype": str(output.dtype),
                    "finite": True,
                    "shape": list(output.shape),
                },
                "gpu": gpu_name,
                "python": platform.python_version(),
                "status": "H100_STACK_VERIFIED",
                "torch": torch.__version__,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
