"""Narrow runtime compatibility shims for the pinned vLLM stack."""

from __future__ import annotations

import asyncio
import logging
import os
from functools import wraps
from importlib.metadata import version as package_version
from typing import Any

VLLM_DRAIN_COMPAT_ENV = "FINANCE_VLLM_DRAIN_COMPAT"
SUPPORTED_VLLM_DRAIN_COMPAT_VERSION = "0.9.2"
_DRAIN_TIMEOUT_SECONDS = 300
_DRAIN_POLL_SECONDS = 1.0

logger = logging.getLogger(__name__)


async def _wait_for_requests_to_drain(
    instance: Any,
    drain_timeout: int = _DRAIN_TIMEOUT_SECONDS,
) -> None:
    """Backport vLLM 0.10's DP-engine drain check needed before sleep."""

    engine_core = instance.engine_core
    loop = asyncio.get_running_loop()
    start_time = loop.time()
    while loop.time() - start_time < drain_timeout:
        if not engine_core.dp_engines_running():
            logger.info("vLLM engines are idle; requests have drained")
            return
        logger.info("vLLM engines are still running; waiting for requests to drain")
        await asyncio.sleep(_DRAIN_POLL_SECONDS)

    raise TimeoutError(
        f"timeout reached after {drain_timeout} seconds waiting for vLLM requests to drain"
    )


_wait_for_requests_to_drain._finance_vllm_092_drain_compat = True  # type: ignore[attr-defined]


def patch_async_llm_class(async_llm_class: type[Any]) -> bool:
    """Install the missing vLLM 0.9.2 API without replacing native methods."""

    if hasattr(async_llm_class, "wait_for_requests_to_drain"):
        return False
    async_llm_class.wait_for_requests_to_drain = _wait_for_requests_to_drain
    return True


def _empty_pinned_host_cache() -> None:
    """Return released vLLM sleep backups to the host memory controller."""

    import torch

    empty_cache = getattr(torch._C, "_host_emptyCache", None)
    if not callable(empty_cache):
        raise RuntimeError("pinned host cache cleanup is unavailable in the pinned torch build")
    empty_cache()


def patch_cumem_allocator_class(cumem_allocator_class: type[Any]) -> bool:
    """Release cached pinned sleep backups after each successful vLLM wake-up."""

    original = cumem_allocator_class.wake_up
    if getattr(original, "_finance_vllm_092_host_cache_compat", False):
        return False

    @wraps(original)
    def wake_up(instance: Any, *args: Any, **kwargs: Any) -> Any:
        result = original(instance, *args, **kwargs)
        tags = kwargs.get("tags", args[0] if args else None)
        # Sleep level 1 backs up only weights. KV-cache wake-ups recreate
        # discarded allocations and must not flush unrelated pinned buffers.
        if tags is None or "weights" in tags:
            _empty_pinned_host_cache()
        return result

    wake_up._finance_vllm_092_host_cache_compat = True  # type: ignore[attr-defined]
    cumem_allocator_class.wake_up = wake_up
    return True


def install_from_environment() -> None:
    """Install the exact-version shim only in launcher-controlled workers."""

    requested_version = os.environ.get(VLLM_DRAIN_COMPAT_ENV)
    if not requested_version:
        return
    if requested_version != SUPPORTED_VLLM_DRAIN_COMPAT_VERSION:
        raise RuntimeError(
            f"unsupported {VLLM_DRAIN_COMPAT_ENV}={requested_version}; "
            f"expected {SUPPORTED_VLLM_DRAIN_COMPAT_VERSION}"
        )
    installed_version = package_version("vllm")
    if installed_version != requested_version:
        raise RuntimeError(
            f"vLLM drain compatibility requires vllm=={requested_version}, "
            f"got {installed_version}"
        )

    from vllm.device_allocator.cumem import CuMemAllocator
    from vllm.v1.engine.async_llm import AsyncLLM

    drain_patched = patch_async_llm_class(AsyncLLM)
    host_cache_patched = patch_cumem_allocator_class(CuMemAllocator)
    if drain_patched or host_cache_patched:
        logger.info(
            "installed vLLM %s compatibility shims (request_drain=%s, host_cache=%s)",
            installed_version,
            drain_patched,
            host_cache_patched,
        )
