"""Opt-in CPU initialization seed for the pinned critic's newly added value head.

The current Qwen3-4B critic loads under verl's CPU weight context before FSDP.
Use its configured seed and restore the caller's CPU RNG even if loading fails.
Pretrained tensors still come from the checkpoint; no weights are reset here.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import os
import re
from functools import wraps
from pathlib import Path
from typing import Any

MODE = "cpu_seed_v1"
_MARKER = "_finance_critic_initialization_mode"
HEAD_EVIDENCE_ENV = "FINANCE_SCREENING_CRITIC_INIT_EVIDENCE_DIR"


def _record_head(model: Any, *, seed: int, model_path: str, rank: int, directory: Path) -> None:
    """Hash only the tiny actual value head before FSDP; do not sample or touch RNG."""
    import torch

    head = getattr(model, "score", None)
    if head is None:
        raise RuntimeError("critic head evidence requires the pinned Qwen score head")
    identity = os.environ.get("FINANCE_SCREENING_IDENTITY_SHA256", "")
    arm = os.environ.get("FINANCE_SCREENING_ARM", "")
    if re.fullmatch(r"[0-9a-f]{64}", identity) is None or arm not in {"M1", "M2"}:
        raise RuntimeError("critic head evidence requires the run identity and M1/M2 arm")
    digest = hashlib.sha256()
    parameters = []
    for name, parameter in sorted(head.named_parameters()):
        if parameter.is_meta or parameter.numel() > 100_000:
            raise RuntimeError("critic head evidence is limited to the materialized small head")
        tensor = parameter.detach().cpu().contiguous()
        # Preserve dtype bytes (including BF16) without converting numeric values.
        payload = tensor.view(torch.uint8).numpy().tobytes()
        metadata = {"name": name, "shape": list(tensor.shape), "dtype": str(tensor.dtype)}
        digest.update(json.dumps(metadata, sort_keys=True).encode())
        digest.update(payload)
        parameters.append(
            {**metadata, "numel": tensor.numel(), "sha256": hashlib.sha256(payload).hexdigest()}
        )
    if not parameters:
        raise RuntimeError("critic head evidence has no parameters")
    record = {
        "schema_version": 1,
        "phase": "critic_head_after_hf_load_before_fsdp",
        "mode": MODE,
        "seed": seed,
        "arm": arm,
        "rank": rank,
        "pid": os.getpid(),
        "execution_identity_sha256": identity,
        "model_path": str(Path(model_path).resolve()),
        "head_sha256": digest.hexdigest(),
        "parameters": parameters,
    }
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / f"head-rank{rank}.json").open("x") as stream:
        json.dump(record, stream, sort_keys=True, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def patch_critic_initialization(critic_class: type[Any], *, mode: str) -> bool:
    """Wrap only the selected worker subclass, before its initialization call."""

    if mode != MODE:
        raise ValueError(f"unsupported critic initialization mode: {mode}")
    existing = critic_class.__dict__.get(_MARKER)
    if existing is not None:
        if existing != mode:
            raise RuntimeError("critic initialization mode already differs")
        return False
    original = critic_class._build_critic_model_optimizer
    if tuple(inspect.signature(original).parameters) != ("self", "config"):
        raise RuntimeError("pinned critic builder signature changed")

    @wraps(original)
    def build(worker: Any, config: Any) -> Any:
        seed = config.model.fsdp_config.get("seed")
        if type(seed) is not int or not 0 <= seed < 2**63:
            raise ValueError("critic.model.fsdp_config.seed must be an integer in [0, 2**63)")
        import torch

        # Unlike torch.manual_seed, this never initializes or reseeds CUDA.
        # Only the CPU RNG used to create missing checkpoint tensors is scoped.
        with torch.random.fork_rng(devices=[], enabled=True):
            torch.random.default_generator.manual_seed(seed)
            evidence_path = os.environ.get(HEAD_EVIDENCE_ENV)
            if evidence_path is None:
                return original(worker, config)
            directory = Path(evidence_path)
            if not directory.is_absolute():
                raise ValueError("critic head evidence directory must be absolute")
            # The selected worker imports this loader inside its builder. Capture
            # the real returned HF module before the builder casts/wraps it in FSDP.
            # This is scoped to the serial critic initialization in this worker.
            from verl.utils import model as model_utils

            loader = model_utils.load_valuehead_model
            captured = False

            @wraps(loader)
            def load(*args: Any, **kwargs: Any) -> Any:
                nonlocal captured
                result = loader(*args, **kwargs)
                if captured:
                    raise RuntimeError("critic builder loaded its head more than once")
                _record_head(
                    result,
                    seed=seed,
                    model_path=str(args[0]),
                    rank=getattr(worker, "rank", 0),
                    directory=directory,
                )
                captured = True
                return result

            model_utils.load_valuehead_model = load
            try:
                result = original(worker, config)
                if not captured:
                    raise RuntimeError("critic head evidence loader was not reached")
                return result
            finally:
                model_utils.load_valuehead_model = loader

    critic_class._build_critic_model_optimizer = build
    setattr(critic_class, _MARKER, mode)
    return True
