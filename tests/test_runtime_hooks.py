"""Interpreter-startup tests for required G1 runtime instrumentation."""

from __future__ import annotations

import importlib.util
import json
import os
import signal
import subprocess
import sys
from pathlib import Path

import pytest

RUNTIME_HOOKS = Path(__file__).resolve().parents[1] / "runtime_hooks"


def _write_fake_hooks(
    tmp_path: Path,
    *,
    update_body: str,
    compat_body: str,
    rollout_seed_body: str,
) -> None:
    agents = tmp_path / "src" / "agents"
    agents.mkdir(parents=True)
    (tmp_path / "src" / "__init__.py").write_text("", encoding="utf-8")
    (agents / "__init__.py").write_text("", encoding="utf-8")
    (agents / "update_probe.py").write_text(update_body, encoding="utf-8")
    (agents / "vllm_compat.py").write_text(compat_body, encoding="utf-8")
    (agents / "rollout_seed.py").write_text(rollout_seed_body, encoding="utf-8")


def _start_python(tmp_path: Path) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join((str(RUNTIME_HOOKS), str(tmp_path)))
    environment.pop("FINANCE_UPDATE_PROBE_DIR", None)
    environment.pop("FINANCE_VLLM_DRAIN_COMPAT", None)
    environment.pop("FINANCE_ROLLOUT_SEED_DIR", None)
    return subprocess.run(
        [sys.executable, "-c", 'print("BODY_RAN")'],
        cwd=tmp_path,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )


def test_sitecustomize_installs_all_hooks_before_user_code(tmp_path: Path) -> None:
    _write_fake_hooks(
        tmp_path,
        update_body='def install_from_environment():\n    print("UPDATE_INSTALLED")\n',
        compat_body='def install_from_environment():\n    print("COMPAT_INSTALLED")\n',
        rollout_seed_body='def install_from_environment():\n    print("SEED_INSTALLED")\n',
    )

    completed = _start_python(tmp_path)

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.splitlines() == [
        "UPDATE_INSTALLED",
        "COMPAT_INSTALLED",
        "SEED_INSTALLED",
        "BODY_RAN",
    ]


@pytest.mark.parametrize("failing_hook", ["update", "compat", "rollout_seed"])
def test_sitecustomize_hook_failure_stops_interpreter(
    tmp_path: Path,
    failing_hook: str,
) -> None:
    success = 'def install_from_environment():\n    print("HOOK_INSTALLED")\n'
    failure = (
        'def install_from_environment():\n    raise RuntimeError("instrumentation exploded")\n'
    )
    _write_fake_hooks(
        tmp_path,
        update_body=failure if failing_hook == "update" else success,
        compat_body=failure if failing_hook == "compat" else success,
        rollout_seed_body=failure if failing_hook == "rollout_seed" else success,
    )

    completed = _start_python(tmp_path)

    assert completed.returncode != 0
    assert "BODY_RAN" not in completed.stdout
    assert "finance runtime hook installation failed: instrumentation exploded" in completed.stderr


@pytest.mark.integration
def test_real_ray_worker_installs_lazy_agent_flow_patch(tmp_path: Path) -> None:
    if importlib.util.find_spec("ray") is None:
        pytest.skip("pinned Ray runtime is not installed")
    repo = Path(__file__).resolve().parents[1]
    seed_root = (tmp_path / "ray-seeds").resolve()
    script = f"""
import json
import ray

ray.init(
    address="local",
    num_cpus=1,
    include_dashboard=False,
    runtime_env={{"env_vars": {{"FINANCE_ROLLOUT_SEED_DIR": {str(seed_root)!r}}}}},
)

@ray.remote
def inspect_worker():
    import asyncio
    import sys
    from types import SimpleNamespace
    target = "agent_r1.agent_flow.agent_flow"
    imported_before_probe = target in sys.modules
    import agent_r1.agent_flow.agent_flow as agent_flow
    modified = agent_flow.AgentFlowWorker.__ray_metadata__.modified_class
    identity = {{"step": 1, "sample_index": 3, "rollout_n": 0, "validate": False}}

    class FlowFixture:
        async def run(self, _sampling_params, **kwargs):
            return kwargs["_finance_rollout_identity_v1"]

    instance = SimpleNamespace(
        config=SimpleNamespace(data={{}}),
        server_manager=object(),
        reward_loop_worker=object(),
        tokenizer=object(),
        processor=None,
        dataset_cls=object(),
    )
    agent_flow._agent_flow_registry["trading"] = {{}}
    agent_flow.hydra.utils.instantiate = lambda **_kwargs: FlowFixture()
    forwarded_identity = asyncio.run(
        modified._run_agent_flow(
            instance,
            {{}},
            identity,
            agent_name="trading",
            trace=False,
            index=3,
        )
    )
    return {{
        "imported_before_probe": imported_before_probe,
        "base_patched": bool(getattr(
            agent_flow.AgentFlowWorkerBase._run_agent_flow,
            "_finance_rollout_seed",
            False,
        )),
        "modified_patched": bool(getattr(
            modified._run_agent_flow,
            "_finance_rollout_seed",
            False,
        )),
        "forwarded_identity": forwarded_identity,
    }}

print("RAY_HOOK_RESULT=" + json.dumps(ray.get(inspect_worker.remote()), sort_keys=True))
ray.shutdown()
"""
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        (str(RUNTIME_HOOKS), str(repo), str(repo / "third_party" / "agent-r1"))
    )
    environment.pop("FINANCE_UPDATE_PROBE_DIR", None)
    environment.pop("FINANCE_VLLM_DRAIN_COMPAT", None)
    environment.pop("FINANCE_ROLLOUT_SEED_DIR", None)
    environment.pop("RAY_ADDRESS", None)
    environment.pop("RAY_NAMESPACE", None)
    environment.pop("RAY_RUNTIME_ENV_HOOK", None)
    environment.pop("RAY_JOB_CONFIG_JSON_ENV_VAR", None)
    environment["RAY_ENABLE_UV_RUN_RUNTIME_ENV"] = "0"

    process = subprocess.Popen(
        [sys.executable, "-c", script],
        cwd=repo,
        env=environment,
        start_new_session=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=120)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            stdout, stderr = process.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            stdout, stderr = process.communicate()
        pytest.fail(f"Ray hook probe timed out\nstdout:\n{stdout}\nstderr:\n{stderr}")

    assert process.returncode == 0, stdout + stderr
    result_line = next(
        line for line in stdout.splitlines() if line.startswith("RAY_HOOK_RESULT=")
    )
    assert json.loads(result_line.removeprefix("RAY_HOOK_RESULT=")) == {
        "base_patched": True,
        "forwarded_identity": {
            "rollout_n": 0,
            "sample_index": 3,
            "step": 1,
            "validate": False,
        },
        "imported_before_probe": False,
        "modified_patched": True,
    }
