from __future__ import annotations

import asyncio
import sys
from types import ModuleType

import pytest

from src.agents import vllm_compat


class _State:
    def __init__(self, values: list[bool]) -> None:
        self.values = values

    def active(self) -> bool:
        if len(self.values) > 1:
            return self.values.pop(0)
        return self.values[0]


class _EngineCore:
    def __init__(self, values: list[bool]) -> None:
        self.state = _State(values)

    def dp_engines_running(self) -> bool:
        return self.state.active()


def test_drain_compat_returns_when_dp_engines_are_idle() -> None:
    class AsyncLLMFixture:
        def __init__(self) -> None:
            self.engine_core = _EngineCore([False])

    assert vllm_compat.patch_async_llm_class(AsyncLLMFixture) is True
    instance = AsyncLLMFixture()
    drain = instance.__getattribute__("wait_for_requests_to_drain")
    asyncio.run(drain(drain_timeout=1))


def test_drain_compat_waits_until_dp_engines_finish(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class AsyncLLMFixture:
        def __init__(self) -> None:
            self.engine_core = _EngineCore([True, False])

    monkeypatch.setattr(vllm_compat, "_DRAIN_POLL_SECONDS", 0)
    assert vllm_compat.patch_async_llm_class(AsyncLLMFixture) is True
    instance = AsyncLLMFixture()
    drain = instance.__getattribute__("wait_for_requests_to_drain")
    asyncio.run(drain(drain_timeout=1))


def test_drain_compat_times_out_instead_of_sleeping_with_active_requests() -> None:
    class AsyncLLMFixture:
        def __init__(self) -> None:
            self.engine_core = _EngineCore([True])

    assert vllm_compat.patch_async_llm_class(AsyncLLMFixture) is True
    instance = AsyncLLMFixture()
    drain = instance.__getattribute__("wait_for_requests_to_drain")
    with pytest.raises(TimeoutError, match="waiting for vLLM requests to drain"):
        asyncio.run(drain(drain_timeout=0))


def test_drain_compat_never_replaces_a_native_method() -> None:
    class AsyncLLMFixture:
        async def wait_for_requests_to_drain(self) -> None:
            return None

    native_method = AsyncLLMFixture.wait_for_requests_to_drain
    assert vllm_compat.patch_async_llm_class(AsyncLLMFixture) is False
    assert AsyncLLMFixture.wait_for_requests_to_drain is native_method


def test_install_is_inert_without_launcher_control(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(vllm_compat.VLLM_DRAIN_COMPAT_ENV, raising=False)

    def unexpected_version_lookup(_: str) -> str:
        raise AssertionError("package metadata must not be read outside launcher control")

    monkeypatch.setattr(vllm_compat, "package_version", unexpected_version_lookup)
    vllm_compat.install_from_environment()


def test_install_rejects_unknown_requested_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(vllm_compat.VLLM_DRAIN_COMPAT_ENV, "0.10.0")
    with pytest.raises(RuntimeError, match="unsupported FINANCE_VLLM_DRAIN_COMPAT"):
        vllm_compat.install_from_environment()


def test_install_rejects_installed_version_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(vllm_compat.VLLM_DRAIN_COMPAT_ENV, "0.9.2")
    monkeypatch.setattr(vllm_compat, "package_version", lambda _: "0.9.3")
    with pytest.raises(RuntimeError, match="requires vllm==0.9.2, got 0.9.3"):
        vllm_compat.install_from_environment()


def test_install_patches_the_exact_vllm_class(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class AsyncLLMFixture:
        pass

    modules = {
        name: ModuleType(name)
        for name in ("vllm", "vllm.v1", "vllm.v1.engine", "vllm.v1.engine.async_llm")
    }
    for name in ("vllm", "vllm.v1", "vllm.v1.engine"):
        modules[name].__path__ = []  # type: ignore[attr-defined]
    modules["vllm.v1.engine.async_llm"].AsyncLLM = AsyncLLMFixture  # type: ignore[attr-defined]
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)

    monkeypatch.setenv(vllm_compat.VLLM_DRAIN_COMPAT_ENV, "0.9.2")
    monkeypatch.setattr(vllm_compat, "package_version", lambda _: "0.9.2")
    vllm_compat.install_from_environment()

    method = AsyncLLMFixture.__dict__["wait_for_requests_to_drain"]
    assert method is vllm_compat._wait_for_requests_to_drain
    assert method.__dict__["_finance_vllm_092_drain_compat"] is True
