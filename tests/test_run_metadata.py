"""Run metadata must identify the exact two-source data snapshot."""

import json
import subprocess
import sys
from datetime import date
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest
import yaml

import src.agents.launcher as launcher_module
import src.agents.metadata as metadata_module
from src.agents.config import load_experiment_config
from src.agents.launcher import (
    build_upstream_command,
    verify_configured_source_manifests,
    verify_configured_t2_gate,
    verify_processed_dataset_bindings,
    verify_smoke_processed_row_provenance,
)
from src.agents.local_runs import LocalRunRecorder
from src.agents.metadata import (
    RunHashes,
    composite_data_version,
    validate_composite_data_version,
    verify_validation_identity,
)
from src.agents.update_probe import summarize_update_evidence
from src.data.episodes import benchmark_series_sha256
from src.data.integrity import audit_raw_daily_manifest


def test_failed_local_run_preserves_complete_reproducibility_record(
    tmp_path: Path,
) -> None:
    recorder = LocalRunRecorder.create(
        run_root=tmp_path / "runs",
        config={
            "run": {"name": "fixture"},
            "data": {
                "version": "history-only-fixture",
                "episode_manifest": "reviewed.yaml",
                "processed": {"train_file": "train.parquet"},
                "split_source": {"train": "history"},
                "sources": {"history": {"input_manifest_sha256": "a" * 64}},
            },
        },
        command=[sys.executable, "-m", "fixture"],
        hashes=RunHashes(
            agent_r1_commit="a" * 40,
            repo_commit="b" * 40,
            data_version="history-only-fixture",
            repo_dirty=True,
        ),
        run_id="fixture-run",
    )
    recorder.record_evidence("identity", {"repo_commit": "b" * 40})
    recorder.finish(returncode=None, error=RuntimeError("preflight blocked"))

    assert {path.name for path in recorder.path.iterdir()} == {
        "artifacts",
        "checkpoints",
        "command.txt",
        "data_manifest.json",
        "environment.json",
        "events.log",
        "git_state.json",
        "metrics.jsonl",
        "provisional_command.txt",
        "resolved_config.yaml",
        "summary.json",
    }
    summary = json.loads((recorder.path / "summary.json").read_text())
    assert summary["status"] == "FAILED"
    assert summary["error"] == {
        "message": "preflight blocked",
        "type": "RuntimeError",
    }
    assert summary["preflight_evidence"]["identity"] == {"repo_commit": "b" * 40}
    assert "global_step" not in summary
    assert summary["observed_update_record_count"] == 0
    assert json.loads((recorder.path / "git_state.json").read_text())["repo_dirty"] is True


def test_optimizer_evidence_gate_requires_a_real_finite_parameter_change(
    tmp_path: Path,
) -> None:
    probe_root = tmp_path / "probe"
    probe_root.mkdir()
    records = [
        {
            "pid": 7,
            "backward_count": 1,
            "optimizer_step_count": 1,
            "global_update_count": 1,
            "grad_norm": 0.5,
            "parameter_delta_l2": 0.0,
            "parameter_delta_max": 0.0,
            "detected": False,
        },
        {
            "pid": 7,
            "backward_count": 2,
            "optimizer_step_count": 2,
            "global_update_count": 2,
            "grad_norm": 0.4,
            "parameter_delta_l2": 0.01,
            "parameter_delta_max": 0.004,
            "detected": True,
        },
    ]
    (probe_root / "optimizer-update-7.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )

    evidence = summarize_update_evidence(probe_root)

    assert evidence["detected"] is True
    assert evidence["backward_count"] == 2
    assert evidence["optimizer_step_count"] == 2
    assert evidence["global_update_count"] == 2
    representative = evidence["representative_update"]
    assert isinstance(representative, dict)
    assert representative["parameter_delta_max"] == 0.004


@pytest.mark.parametrize("violation", ["stockbench_source", "false_provenance", "final_oos_date"])
def test_smoke_processed_provenance_rejects_disguised_final_oos(
    violation: str,
) -> None:
    provenance = {
        "split_role": "smoke_shadow",
        "source_split": "valid",
        "purpose": "wiring_only",
        "scientific_use_allowed": False,
        "checkpoint_selection_allowed": False,
        "final_oos": False,
    }
    env_kwargs: dict[str, Any] = {
        "provenance": dict(provenance),
        "episode_dates": ["2024-11-14T21:00:00+00:00"],
    }
    extra_info = {"episode_id": "smoke-test-001", "split": "test", "index": 0}
    extra_info.update(provenance)
    verify_smoke_processed_row_provenance(
        data_source="history/test",
        env_kwargs=env_kwargs,
        extra_info=extra_info,
    )

    data_source = "history/test"
    if violation == "stockbench_source":
        data_source = "stockbench/test"
    elif violation == "false_provenance":
        env_kwargs["provenance"]["purpose"] = "final_evaluation"
    else:
        env_kwargs["episode_dates"] = ["2025-03-03T21:00:00+00:00"]

    with pytest.raises(RuntimeError):
        verify_smoke_processed_row_provenance(
            data_source=data_source,
            env_kwargs=env_kwargs,
            extra_info=extra_info,
        )


def test_composite_data_version_is_deterministic_and_source_named() -> None:
    assert composite_data_version(
        history_manifest_sha256=(
            "2612e707446b1c8bcd73ac6583b84558d3c4eddbaa9fc20adf7023c3c8a76c90"
        ),
        stockbench_manifest_sha256=(
            "8b48fc7d50f4ecdbc49e920eb83d72df02b9a56d4365bce7f0db6bb4a342bd1f"
        ),
    ) == (
        "composite-raw-manifest-sha256:"
        "fb674aacaa8c7f6b6992994688733da28cb9e665209af70b778da0ce5688bbb1"
    )


def test_configured_data_version_must_match_both_source_manifests() -> None:
    with pytest.raises(RuntimeError, match="data.version does not match its source manifests"):
        validate_composite_data_version(
            configured="composite-raw-manifest-sha256:" + "0" * 64,
            history_manifest_sha256="a" * 64,
            stockbench_manifest_sha256="b" * 64,
        )


def _source_config(tmp_path: Path) -> dict:
    sources: dict[str, dict[str, object]] = {}
    for name, nested_storage in (("history", False), ("stockbench", True)):
        manifest_root = tmp_path / name
        storage_root = manifest_root / "storage" if nested_storage else manifest_root
        day_root = storage_root / "parquet" / "AAA" / "day"
        day_root.mkdir(parents=True)
        (day_root / "2025-01-02.parquet").write_bytes(name.encode())
        audit = audit_raw_daily_manifest(
            manifest_root=manifest_root,
            storage_root=storage_root,
            symbols=["AAA"],
            start=date(2025, 1, 2),
            end=date(2025, 1, 2),
        )
        sources[name] = {
            "manifest_root": str(manifest_root.relative_to(tmp_path)),
            "storage_root": str(storage_root.relative_to(tmp_path)),
            "input_window": {"start": "2025-01-02", "end": "2025-01-02"},
            "input_manifest_sha256": audit.manifest_sha256,
            "files": 1,
            "sessions_per_symbol": 1,
            "version": f"{name}-version",
        }
    return {
        "data": {
            "sources": sources,
            "universe": {"symbols": ["AAA"]},
            "availability": {"exchange_calendar": "XNYS"},
        }
    }


def test_launcher_preflight_recomputes_each_configured_source_manifest(tmp_path: Path) -> None:
    config = _source_config(tmp_path)

    audits = verify_configured_source_manifests(config, repo_root=tmp_path)

    assert set(audits) == {"history", "stockbench"}
    assert audits["history"].file_count == 1


def test_launcher_preflight_rejects_data_changed_after_configuration(tmp_path: Path) -> None:
    config = _source_config(tmp_path)
    (tmp_path / "stockbench/storage/parquet/AAA/day/2025-01-02.parquet").write_bytes(b"tampered")

    with pytest.raises(RuntimeError, match="raw manifest SHA-256 mismatch"):
        verify_configured_source_manifests(config, repo_root=tmp_path)


def test_agent_loop_uses_the_explicit_episode_horizon_not_legacy_max_turns() -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_experiment_config(root / "configs/smoke_1p7b.yaml")
    command = build_upstream_command(
        config,
        RunHashes(
            agent_r1_commit="a" * 40,
            repo_commit="b" * 40,
            data_version="fixture",
            repo_dirty=False,
        ),
    )

    assert "actor_rollout_ref.rollout.agent.max_steps=32" in command
    assert "data.train_max_samples=4" in command
    assert "data.dataloader_num_workers=0" in command
    assert "actor_rollout_ref.actor.fsdp_config.param_offload=True" in command
    assert "actor_rollout_ref.actor.fsdp_config.optimizer_offload=False" in command
    assert "actor_rollout_ref.rollout.logprobs_mode=null" in command
    assert "+ray_kwargs.ray_init.include_dashboard=False" in command
    assert "+ray_kwargs.ray_init.address=local" in command
    assert (
        "+ray_kwargs.ray_init.runtime_env.env_vars.FINANCE_VLLM_DRAIN_COMPAT=0.9.2"
        in command
    )
    assert "actor_rollout_ref.rollout.calculate_log_probs=False" in command
    assert "algorithm.rollout_correction.bypass_mode=False" in command


def test_non_smoke_run_preserves_upstream_dataloader_worker_default() -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_experiment_config(root / "configs/smoke_1p7b.yaml")
    config["run"]["smoke"] = False

    command = build_upstream_command(
        config,
        RunHashes(
            agent_r1_commit="a" * 40,
            repo_commit="b" * 40,
            data_version="fixture",
            repo_dirty=False,
        ),
    )

    assert all(not argument.startswith("data.dataloader_num_workers=") for argument in command)
    assert "actor_rollout_ref.actor.fsdp_config.param_offload=True" in command
    assert "actor_rollout_ref.actor.fsdp_config.optimizer_offload=True" in command


def test_episode_cap_rejects_non_positive_or_non_integer_values() -> None:
    root = Path(__file__).resolve().parents[1]
    for invalid in (False, 0, -1, 1.5, "4"):
        config = load_experiment_config(root / "configs/smoke_1p7b.yaml")
        config["training"]["episodes"] = invalid
        with pytest.raises(ValueError, match="training.episodes must be a positive integer"):
            build_upstream_command(
                config,
                RunHashes(
                    agent_r1_commit="a" * 40,
                    repo_commit="b" * 40,
                    data_version="fixture",
                    repo_dirty=False,
                ),
            )


def test_training_environment_overrides_hostile_online_parent_values(tmp_path: Path) -> None:
    compiler = tmp_path / "x86_64-conda-linux-gnu-gcc"
    compiler.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    compiler.chmod(0o755)
    environment = launcher_module._training_environment(
        agent_path=tmp_path / "agent-r1",
        probe_root=tmp_path / "probe",
        c_compiler=str(compiler),
        parent={
            "CC": "/tmp/untrusted-compiler",
            "FINANCE_UPDATE_PROBE_DIR": "/tmp/untrusted-probe",
            "FINANCE_ROLLOUT_SEED_DIR": "/tmp/untrusted-seeds",
            "FINANCE_VLLM_DRAIN_COMPAT": "0.10.0",
            "HF_HUB_OFFLINE": "0",
            "HF_DATASETS_OFFLINE": "0",
            "RAY_ENABLE_UV_RUN_RUNTIME_ENV": "1",
            "RAY_ADDRESS": "ray://untrusted.example:10001",
            "RAY_NAMESPACE": "untrusted-namespace",
            "RAY_RUNTIME_ENV_HOOK": "untrusted.hook",
            "RAY_JOB_CONFIG_JSON_ENV_VAR": '{"runtime_env":{"working_dir":"/tmp"}}',
            "TRANSFORMERS_OFFLINE": "false",
            "PYTHONPATH": "/tmp/untrusted-python-overlay",
        },
    )

    assert environment["CC"] == str(compiler)
    assert "FINANCE_UPDATE_PROBE_DIR" not in environment
    assert "FINANCE_ROLLOUT_SEED_DIR" not in environment
    assert "FINANCE_VLLM_DRAIN_COMPAT" not in environment
    assert environment["HF_HUB_OFFLINE"] == "1"
    assert environment["HF_DATASETS_OFFLINE"] == "1"
    assert environment["RAY_ENABLE_UV_RUN_RUNTIME_ENV"] == "0"
    assert "RAY_ADDRESS" not in environment
    assert "RAY_NAMESPACE" not in environment
    assert "RAY_RUNTIME_ENV_HOOK" not in environment
    assert "RAY_JOB_CONFIG_JSON_ENV_VAR" not in environment
    assert environment["TRANSFORMERS_OFFLINE"] == "1"
    assert environment["TRITON_CACHE_DIR"] == str(tmp_path / "triton_cache")
    assert "/tmp/untrusted-python-overlay" not in environment["PYTHONPATH"]
    command = [
        "python",
        "actor_rollout_ref.model.path=Qwen/Qwen3-1.7B",
        "critic.model.path=Qwen/Qwen3-1.7B",
    ]
    command = launcher_module._finalize_model_paths(
        command,
        arm_id="A3",
        model_snapshot_path="/cache/snapshots/exact",
    )
    assert all("Qwen/Qwen3-1.7B" not in argument for argument in command)


def test_update_probe_is_injected_only_through_ray_worker_runtime_env(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_experiment_config(root / "configs/smoke_1p7b.yaml")
    probe_root = (tmp_path / "update-probe").resolve()

    command = build_upstream_command(
        config,
        RunHashes(
            agent_r1_commit="a" * 40,
            repo_commit="b" * 40,
            data_version="fixture",
            repo_dirty=False,
        ),
        update_probe_dir=probe_root,
    )

    assert (
        "+ray_kwargs.ray_init.runtime_env.env_vars.FINANCE_UPDATE_PROBE_DIR="
        f"{probe_root}"
    ) in command
    assert (
        "+ray_kwargs.ray_init.runtime_env.env_vars.FINANCE_ROLLOUT_SEED_DIR="
        f"{probe_root.parent / 'rollout_seed'}"
    ) in command


def test_update_probe_runtime_env_rejects_relative_path() -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_experiment_config(root / "configs/smoke_1p7b.yaml")

    with pytest.raises(ValueError, match="update probe directory must be absolute"):
        build_upstream_command(
            config,
            RunHashes(
                agent_r1_commit="a" * 40,
                repo_commit="b" * 40,
                data_version="fixture",
                repo_dirty=False,
            ),
            update_probe_dir="relative/probe",
        )


def test_c_compiler_resolution_finds_pinned_micromamba_toolchain(tmp_path: Path) -> None:
    compiler = tmp_path / "bin" / "x86_64-conda-linux-gnu-gcc"
    compiler.parent.mkdir()
    compiler.write_text(
        "#!/bin/sh\nprintf 'fixture gcc 15.2.0\\n'\n",
        encoding="utf-8",
    )
    compiler.chmod(0o755)

    path, version = launcher_module._resolve_c_compiler(
        parent={"PATH": ""},
        base_prefix=tmp_path,
    )

    assert path == str(compiler.resolve())
    assert version == "fixture gcc 15.2.0"


def test_c_compiler_resolution_fails_closed_when_none_is_available(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(launcher_module.shutil, "which", lambda *_args, **_kwargs: None)

    with pytest.raises(RuntimeError, match="working host C compiler is required"):
        launcher_module._resolve_c_compiler(parent={"PATH": ""}, base_prefix=tmp_path)


def test_triton_compiler_probe_uses_a_fresh_cache_and_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed_cache: Path | None = None

    def fail_probe(*_args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        nonlocal observed_cache
        environment = kwargs["env"]
        assert isinstance(environment, dict)
        observed_cache = Path(environment["TRITON_CACHE_DIR"])
        assert observed_cache.is_dir()
        assert environment["CC"] == "/fixture/gcc"
        return subprocess.CompletedProcess([], 1, stdout="", stderr="fixture compile failure")

    monkeypatch.setattr(launcher_module.subprocess, "run", fail_probe)

    with pytest.raises(RuntimeError, match="fixture compile failure"):
        launcher_module._verify_triton_cuda_utils("/fixture/gcc")
    assert observed_cache is not None
    assert not observed_cache.exists()


def test_model_resolution_requires_the_exact_local_revision(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_experiment_config(root / "configs/smoke_1p7b.yaml")
    revision = config["model"]["revision"]
    snapshot = tmp_path / "models--Qwen--Qwen3-1.7B" / "snapshots" / revision
    snapshot.mkdir(parents=True)

    model, resolved_revision, resolved_snapshot = launcher_module._resolve_exact_model_snapshot(
        config, cache_root=tmp_path
    )
    assert model == "Qwen/Qwen3-1.7B"
    assert resolved_revision == revision
    assert resolved_snapshot == snapshot.resolve()

    config["model"]["revision"] = "c" * 40
    with pytest.raises(RuntimeError, match="exact model revision is not cached locally"):
        launcher_module._resolve_exact_model_snapshot(config, cache_root=tmp_path)


def test_validation_identity_fails_closed_on_each_code_identity_risk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    expected = "a" * 40
    agent_commit = "b" * 40
    repo = tmp_path / "repo"
    agent = repo / "third_party" / "agent-r1"
    agent.mkdir(parents=True)

    def exercise(
        *,
        head: str = expected,
        tracked: str = "",
        submodule: str | None = None,
        untracked: str = "",
    ) -> None:
        def fake_git(target: Path, *arguments: str) -> str:
            if target == agent:
                if arguments == ("rev-parse", "HEAD"):
                    return agent_commit
                if arguments[0] == "status":
                    return ""
                if arguments[:2] == ("ls-files", "--others"):
                    return ""
                raise AssertionError(arguments)
            if arguments == ("rev-parse", "HEAD"):
                return head
            if arguments[0] == "status":
                return tracked
            if arguments[:2] == ("submodule", "status"):
                return submodule if submodule is not None else f" {agent_commit} agent-r1"
            if arguments[:2] == ("ls-files", "--others"):
                return f"{untracked}\0" if untracked else ""
            raise AssertionError(arguments)

        monkeypatch.setattr(metadata_module, "_git", fake_git)
        verify_validation_identity(
            repo_root=repo,
            agent_r1_path=agent,
            expected_repo_commit=expected,
            expected_agent_r1_commit=agent_commit,
        )

    failures = (
        ({"head": "c" * 40}, "validation commit mismatch"),
        ({"tracked": " M src/agents/launcher.py"}, "clean tracked working tree"),
        ({"submodule": f"+{agent_commit} agent-r1"}, "clean pinned commit"),
        ({"untracked": "src/runtime_override.py"}, "repository overlays"),
    )
    for arguments, message in failures:
        with pytest.raises(RuntimeError, match=message):
            exercise(**arguments)
    exercise()


def _run_git(repo: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo), *arguments],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _commit_all(repo: Path, message: str) -> str:
    _run_git(repo, "add", "-A")
    _run_git(
        repo,
        "-c",
        "user.name=Acceptance Test",
        "-c",
        "user.email=acceptance@example.invalid",
        "commit",
        "-m",
        message,
    )
    return _run_git(repo, "rev-parse", "HEAD")


def test_validation_identity_accepts_a_real_clean_submodule_and_rejects_overlays(
    tmp_path: Path,
) -> None:
    agent = tmp_path / "agent-source"
    agent.mkdir()
    _run_git(agent, "init", "-q")
    (agent / "agent.py").write_text("PINNED = True\n", encoding="utf-8")
    agent_commit = _commit_all(agent, "agent")

    repo = tmp_path / "repo"
    repo.mkdir()
    _run_git(repo, "init", "-q")
    (repo / "experiment.yaml").write_text("run: fixture\n", encoding="utf-8")
    (repo / ".gitignore").write_text(
        ".venv/\ndata/\nruns/\n__pycache__/\n*.py[cod]\n*.egg-info/\n"
        ".coverage\n.ignore\n"
        "configs/reviewed.yaml\n",
        encoding="utf-8",
    )
    _commit_all(repo, "root")
    _run_git(
        repo,
        "-c",
        "protocol.file.allow=always",
        "submodule",
        "add",
        "-q",
        str(agent),
        "third_party/agent-r1",
    )
    repo_commit = _commit_all(repo, "pin submodule")
    checkout = repo / "third_party" / "agent-r1"
    for path in (
        repo / ".venv/local-package.py",
        repo / "data/history.parquet",
        repo / "runs/run/summary.json",
        repo / "__pycache__/module.pyc",
        repo / "src/fixture.egg-info/PKG-INFO",
        repo / "src/fixture.egg-info/SOURCES.txt",
        repo / "src/fixture.egg-info/dependency_links.txt",
        repo / "src/fixture.egg-info/requires.txt",
        repo / "src/fixture.egg-info/top_level.txt",
        repo / ".coverage",
        repo / ".ignore",
        repo / "configs/reviewed.yaml",
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"local artifact")

    identity = verify_validation_identity(
        repo_root=repo,
        agent_r1_path=checkout,
        expected_repo_commit=repo_commit,
        expected_agent_r1_commit=agent_commit,
        allowed_untracked_paths=(repo / "configs/reviewed.yaml",),
        required_tracked_paths=(repo / "experiment.yaml",),
    )
    submodule_status = identity["submodule_status"]
    assert isinstance(submodule_status, str)
    assert submodule_status.startswith(f" {agent_commit}")
    assert identity["tracked_config_sources"] == ["experiment.yaml"]

    (repo / "src/fixture.egg-info/bootstrap.py").write_text(
        "SENTINEL = True\n", encoding="utf-8"
    )
    with pytest.raises(RuntimeError, match="repository overlays"):
        verify_validation_identity(
            repo_root=repo,
            agent_r1_path=checkout,
            expected_repo_commit=repo_commit,
            expected_agent_r1_commit=agent_commit,
        )
    (repo / "src/fixture.egg-info/bootstrap.py").unlink()

    for overlay in (
        repo / "src/fixture.egg-info/entry_points.txt",
        repo / "rogue.pyc",
    ):
        overlay.write_bytes(b"runtime-affecting overlay")
        with pytest.raises(RuntimeError, match="repository overlays"):
            verify_validation_identity(
                repo_root=repo,
                agent_r1_path=checkout,
                expected_repo_commit=repo_commit,
                expected_agent_r1_commit=agent_commit,
            )
        overlay.unlink()

    (repo / "sitecustomize.py").write_text("SENTINEL = True\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="repository overlays"):
        verify_validation_identity(
            repo_root=repo,
            agent_r1_path=checkout,
            expected_repo_commit=repo_commit,
            expected_agent_r1_commit=agent_commit,
        )
    (repo / "sitecustomize.py").unlink()

    (checkout / "untracked_overlay.py").write_text("SENTINEL = True\n", encoding="utf-8")
    with pytest.raises(
        RuntimeError,
        match="clean tracked working tree|Agent-R1 submodule|Agent-R1 overlays",
    ):
        verify_validation_identity(
            repo_root=repo,
            agent_r1_path=checkout,
            expected_repo_commit=repo_commit,
            expected_agent_r1_commit=agent_commit,
        )


@pytest.mark.parametrize(
    ("failure", "message", "command_state"),
    [
        ("missing_expected_commit", "requires --expected-commit", "NOT_BUILT"),
        ("data_version", "smoke data version must be history-only", "NOT_BUILT"),
        ("agent_commit", "Agent-R1 checkout mismatch", "NOT_BUILT"),
    ],
)
def test_non_dry_early_failures_always_write_a_machine_readable_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
    message: str,
    command_state: str,
) -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_experiment_config(root / "configs/smoke_1p7b.yaml")
    config["tracking"]["log_root"] = str(tmp_path / "runs")
    expected_commit: str | None = "a" * 40
    if failure == "missing_expected_commit":
        expected_commit = None
    elif failure == "data_version":
        config["data"]["version"] = "stale-smoke-version"
    else:
        config["framework"]["agent_r1_commit"] = "c" * 40
    monkeypatch.setattr(
        launcher_module,
        "load_experiment_config_with_sources",
        lambda _: (config, (root / "configs/smoke_1p7b.yaml",)),
    )

    with pytest.raises(RuntimeError, match=message):
        launcher_module.launch(
            "fixture.yaml",
            dry_run=False,
            preflight_only=True,
            expected_arm="A1",
            expected_commit=expected_commit,
        )

    run_dir = next((tmp_path / "runs").iterdir())
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "FAILED"
    assert summary["command_state"] == command_state
    assert message.split()[0] in summary["error"]["message"]


def test_preflight_preserves_source_evidence_when_a_later_gate_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config: dict[str, Any] = {
        "tracking": {"backend": "local_files"},
        "framework": {"agent_r1_path": "agent", "agent_r1_commit": "b" * 40},
        "data": {
            "version": "fixture",
            "episode_manifest": "reviewed.yaml",
            "processed": {
                "train_file": "train.parquet",
                "valid_file": "valid.parquet",
                "test_file": "test.parquet",
            },
            "split_source": {"train": "history"},
            "sources": {"history": {"input_manifest_sha256": "a" * 64}},
        },
    }
    for name in ("train.parquet", "valid.parquet", "test.parquet"):
        (tmp_path / name).write_bytes(b"fixture")
    recorder = LocalRunRecorder.create(
        run_root=tmp_path / "runs",
        config=config,
        command=[sys.executable, "-m", "fixture"],
        hashes=RunHashes("b" * 40, "a" * 40, "fixture", False),
        run_id="staged-failure",
        repo_root=tmp_path,
    )
    monkeypatch.setattr(launcher_module, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(
        launcher_module, "verify_validation_identity", lambda **_: {"repo_commit": "a" * 40}
    )
    monkeypatch.setattr(launcher_module, "verify_configured_t2_gate", lambda *_args, **_kw: None)
    monkeypatch.setattr(
        launcher_module, "verify_smoke_final_oos_isolation", lambda *_args, **_kw: {"ok": True}
    )
    monkeypatch.setattr(launcher_module, "_verify_reward_freeze", lambda *_args, **_kw: None)
    monkeypatch.setattr(
        launcher_module,
        "verify_gpu_runtime_and_model",
        lambda *_args, **_kw: {"model_revision": "c" * 40},
    )
    monkeypatch.setattr(
        launcher_module,
        "verify_configured_source_manifests",
        lambda *_args, **_kw: {
            "history": SimpleNamespace(
                manifest_sha256="d" * 64,
                file_count=10820,
                sessions_per_symbol=541,
            )
        },
    )

    def fail_after_source(**_: object) -> str:
        raise RuntimeError("later benchmark gate failed")

    monkeypatch.setattr(launcher_module, "verify_episode_benchmarks", fail_after_source)
    with pytest.raises(RuntimeError, match="later benchmark gate failed") as caught:
        launcher_module._verify_all_preflight_inputs(
            config,
            command=["python"],
            is_smoke=True,
            expected_commit="a" * 40,
            recorder=recorder,
        )
    recorder.finish(returncode=None, error=caught.value)

    artifact = json.loads(
        (recorder.path / "artifacts/preflight_evidence.json").read_text(encoding="utf-8")
    )
    summary = json.loads((recorder.path / "summary.json").read_text(encoding="utf-8"))
    assert artifact["source_manifests"]["history"]["file_count"] == 10820
    assert summary["preflight_evidence"]["source_manifests"] == artifact["source_manifests"]


def test_smoke_processed_train_rows_must_equal_episode_cap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _processed_config(tmp_path)
    config["run"] = {"smoke": True}
    config["training"]["episodes"] = 2
    monkeypatch.setattr(
        launcher_module,
        "load_episode_manifest",
        lambda **_: {"splits": {"train": [{}], "valid": [{}], "test": [{}]}},
    )

    with pytest.raises(RuntimeError, match="smoke train row count 1.*training.episodes=2"):
        verify_processed_dataset_bindings(config, repo_root=tmp_path)


@pytest.mark.parametrize(
    ("arm_id", "expects_critic_seed"),
    [("A1", False), ("A2", True), ("A3", True)],
)
def test_nondefault_yaml_seed_reaches_every_upstream_rng(
    arm_id: str, expects_critic_seed: bool
) -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_experiment_config(root / "configs/smoke_1p7b.yaml")
    config["run"]["seed"] = 123
    config["arm"]["id"] = arm_id

    command = build_upstream_command(
        config,
        RunHashes(
            agent_r1_commit="a" * 40,
            repo_commit="b" * 40,
            data_version="fixture",
            repo_dirty=False,
        ),
    )

    expected = {
        "data.seed=123",
        "actor_rollout_ref.actor.data_loader_seed=123",
        "actor_rollout_ref.actor.fsdp_config.seed=123",
        "+actor_rollout_ref.rollout.engine_kwargs.vllm.seed=123",
    }
    assert expected <= set(command)
    critic_seeds = {
        "critic.data_loader_seed=123",
        "critic.model.fsdp_config.seed=123",
    }
    assert critic_seeds.issubset(command) is expects_critic_seed


def test_smoke_uses_console_only_local_tracking() -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_experiment_config(root / "configs/smoke_1p7b.yaml")
    command = build_upstream_command(
        config,
        RunHashes(
            agent_r1_commit="a" * 40,
            repo_commit="b" * 40,
            data_version="fixture",
            repo_dirty=False,
        ),
    )

    assert "wandb" not in config
    assert config["tracking"]["backend"] == "local_files"
    assert 'trainer.logger=["console"]' in command
    assert all("wandb" not in argument.casefold() for argument in command)


def test_wiring_smoke_declares_the_only_approved_run_kind() -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_experiment_config(root / "configs/smoke_1p7b.yaml")

    assert config["run"]["kind"] == "wiring_smoke_1p7b"


def test_missing_t2_v2_blocks_without_falling_back_to_the_legacy_oos_probe(
    tmp_path: Path,
) -> None:
    assert not hasattr(launcher_module, "run_configured_temporal_leakage_probe")
    config: dict[str, Any] = {
        "run": {"smoke": False},
        "evaluation": {"shifted_label_probe": {"require_before_main_training": True}},
    }

    with pytest.raises(RuntimeError, match="T2 v2 configuration is required"):
        verify_configured_t2_gate(config, repo_root=tmp_path)


def test_preregistered_t2_v2_blocks_formal_training_without_reopening_stockbench(
    tmp_path: Path,
) -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_experiment_config(root / "configs/base.yaml")
    config["run"] = {"smoke": False}
    assert not hasattr(launcher_module, "run_configured_temporal_leakage_probe")

    with pytest.raises(RuntimeError, match="T2 v2 is preregistered but not implemented"):
        verify_configured_t2_gate(config, repo_root=tmp_path)


def test_null_t2_v2_cannot_reenable_the_legacy_oos_probe(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_experiment_config(root / "configs/base.yaml")
    config["run"] = {"smoke": False}
    config["evaluation"]["shifted_label_probe_v2"] = None
    config["evaluation"]["shifted_label_probe"]["require_before_main_training"] = False
    assert not hasattr(launcher_module, "run_configured_temporal_leakage_probe")

    with pytest.raises(RuntimeError, match="T2 v2 configuration is required"):
        verify_configured_t2_gate(config, repo_root=tmp_path)


def test_legacy_t2_cli_stops_before_reading_any_oos_input(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    output = tmp_path / "must-not-exist.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(root / "scripts/run_t2_probe.py"),
            "--config",
            str(tmp_path / "missing-config.yaml"),
            "--output",
            str(output),
        ],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 2
    assert "permanently disabled" in completed.stderr
    assert not output.exists()


def test_only_declared_wiring_smoke_can_bypass_the_t2_gate(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    approved = load_experiment_config(root / "configs/smoke_1p7b.yaml")
    verify_configured_t2_gate(approved, repo_root=tmp_path)

    for invalid_run in (
        {"smoke": True},
        {"smoke": False, "kind": "wiring_smoke_1p7b"},
        {"smoke": True, "kind": "unregistered_smoke"},
    ):
        invalid = load_experiment_config(root / "configs/smoke_1p7b.yaml")
        invalid["run"] = invalid_run
        with pytest.raises(RuntimeError, match="invalid T2 smoke exemption"):
            verify_configured_t2_gate(invalid, repo_root=tmp_path)


@pytest.mark.parametrize(
    ("path", "invalid_value"),
    [
        (("arm", "id"), "A2"),
        (("model", "name"), "Qwen/Qwen3-4B"),
        (("model", "debug"), "Qwen/Qwen3-4B"),
        (("model", "revision"), "main"),
        (("model", "local_files_only"), False),
        (("training", "episodes"), 5),
        (("training", "max_agent_steps"), 31),
        (("training", "train_batch_size"), 3),
        (("training", "rollout_n"), 3),
        (("training", "total_epochs"), 2),
        (("training", "learning_rate"), 1.0),
        (("training", "n_gpus_per_node"), 2),
        (("training", "max_response_length"), 100_000),
        (("arm", "name"), "generic_step"),
        (("arm", "optimizer"), "step_po"),
        (("arm", "reward_signal"), "generic_step_reward"),
        (("arm", "credit_assignment"), "generic_step"),
        (("data", "split_source", "train"), "stockbench"),
        (("data", "episode_manifest"), "configs/episode_manifest.reviewed.yaml"),
        (("data", "processed", "train_file"), "data/processed/train.parquet"),
        (("data", "processed", "valid_file"), "data/processed/valid.parquet"),
        (("data", "processed", "test_file"), "data/processed/test.parquet"),
    ],
)
def test_wiring_smoke_exemption_rejects_contract_drift(
    path: tuple[str, ...],
    invalid_value: object,
    tmp_path: Path,
) -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_experiment_config(root / "configs/smoke_1p7b.yaml")
    target: dict[str, Any] = config
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = invalid_value
    with pytest.raises(RuntimeError, match="invalid T2 smoke exemption"):
        verify_configured_t2_gate(config, repo_root=tmp_path)


def _processed_config(tmp_path: Path) -> dict:
    config = _source_config(tmp_path)
    config["data"]["sources"]["history"]["input_window"] = {
        "start": "2025-01-02",
        "end": "2025-02-04",
    }
    config["data"]["sources"]["stockbench"]["input_window"] = {
        "start": "2025-03-03",
        "end": "2025-03-04",
    }
    config["data"]["availability"].update(
        {"timezone": "America/New_York", "market_close_time": "16:00:00"}
    )
    config["data"].update(
        {
            "episode_manifest": "processed/reviewed.yaml",
            "processed": {
                "train_file": "processed/train.parquet",
                "valid_file": "processed/valid.parquet",
            },
            "split_source": {"train": "history", "valid": "history", "test": "stockbench"},
            "windows": {
                "train": {"start": "2025-01-02", "end": "2025-01-03"},
                "valid": {"start": "2025-02-03", "end": "2025-02-04"},
                "test": {
                    "start": "2025-03-03",
                    "end": "2025-03-04",
                    "indivisible_oos": True,
                },
            },
            "retrieval": {
                "news_enabled": False,
                "allowed_modalities": [
                    "market_data",
                    "pit_fundamentals",
                    "causal_technical_indicators",
                ],
            },
        }
    )
    config["execution"] = {
        "initial_cash": 1000.0,
        "commission_bps": 10.0,
        "slippage_bps": 5.0,
    }
    config["reward"] = {
        "drawdown_coefficient": 0.1,
        "turnover_coefficient": 0.2,
        "epsilon": 0.001,
        "risk_lambda": 0.5,
        "turnover_limit": 1.0,
        "position_limit": 1.0,
        "benchmark": "buy_and_hold_equal_weight_20_tickers",
        "benchmark_accounting_engine": "shared_portfolio_ledger",
        "benchmark_execution": "next_open_with_configured_costs",
    }
    config["training"] = {
        "max_agent_steps": 1,
        "reserved_information_steps_per_session": 0,
    }
    episode_days = {
        "train": ("2025-01-02", "2025-01-03"),
        "valid": ("2025-02-03", "2025-02-04"),
        "test": ("2025-03-03", "2025-03-04"),
    }
    splits = {
        split: [
            {
                "episode_id": f"{split}-001",
                "episode_dates": [f"{day}T21:00:00+00:00" for day in days],
                "episode_open_times": [f"{day}T14:30:00+00:00" for day in days],
                "benchmark_closes": [100.0, 101.0],
            }
        ]
        for split, days in episode_days.items()
    }
    manifest = {
        "version": 1,
        "data_snapshot": {
            name: {
                "input_manifest_sha256": source["input_manifest_sha256"],
                "files": source["files"],
            }
            for name, source in config["data"]["sources"].items()
        },
        "split_windows": {
            "train": {
                "source": "history",
                "first_session": "2025-01-02",
                "last_session": "2025-01-03",
            },
            "valid": {
                "source": "history",
                "first_session": "2025-02-03",
                "last_session": "2025-02-04",
            },
            "test": {
                "source": "stockbench",
                "first_session": "2025-03-03",
                "last_session": "2025-03-04",
                "indivisible_oos": True,
            },
        },
        "universe": ["AAA"],
        "benchmark": {
            "definition": "buy_and_hold_equal_weight_20_tickers",
            "accounting_engine": "shared_portfolio_ledger",
            "execution": "next_open_with_configured_costs",
            "series_sha256": benchmark_series_sha256(splits),
        },
        "modalities": {
            "news_enabled": False,
            "allowed_retrieval": config["data"]["retrieval"]["allowed_modalities"],
        },
        "splits": splits,
    }
    processed = tmp_path / "processed"
    processed.mkdir()
    (processed / "reviewed.yaml").write_text(yaml.safe_dump(manifest), encoding="utf-8")
    for split in ("train", "valid"):
        source = config["data"]["sources"]["history"]
        episode = splits[split][0]
        env_kwargs = {
            "symbols": ["AAA"],
            "manifest_root": str(tmp_path / source["manifest_root"]),
            "storage_root": str(tmp_path / source["storage_root"]),
            "data_version": source["version"],
            "data_start": "2025-01-02",
            "data_end": "2025-02-04",
            "input_manifest_sha256": source["input_manifest_sha256"],
            "input_file_count": source["files"],
            "input_sessions_per_symbol": source["sessions_per_symbol"],
            "initial_cash": 1000.0,
            "commission_bps": 10.0,
            "slippage_bps": 5.0,
            "drawdown_coefficient": 0.1,
            "turnover_coefficient": 0.2,
            "finance_epsilon": 0.001,
            "risk_lambda": 0.5,
            "turnover_limit": 1.0,
            "position_limit": 1.0,
            "market_timezone": "America/New_York",
            "exchange_calendar": "XNYS",
            "market_close_time": "16:00:00",
            "episode_dates": episode["episode_dates"],
            "episode_open_times": episode["episode_open_times"],
            "benchmark_closes": episode["benchmark_closes"],
        }
        pd.DataFrame(
            [
                {
                    "agent_name": "trading",
                    "data_source": f"history/{split}",
                    "prompt": [
                        {
                            "role": "user",
                            "content": "Manage the portfolio using only point-in-time tools.",
                        }
                    ],
                    "reward_model": {"style": "rule", "ground_truth": 0.0},
                    "env_kwargs": json.dumps(env_kwargs),
                    "extra_info": {
                        "episode_id": episode["episode_id"],
                        "split": split,
                        "index": 0,
                    },
                }
            ]
        ).to_parquet(processed / f"{split}.parquet", index=False)
    return config


def test_launcher_binds_processed_rows_to_the_verified_source_config(tmp_path: Path) -> None:
    config = _processed_config(tmp_path)

    verify_processed_dataset_bindings(config, repo_root=tmp_path)

    valid_path = tmp_path / "processed/valid.parquet"
    valid = pd.read_parquet(valid_path)
    env_kwargs = json.loads(valid.loc[0, "env_kwargs"])
    env_kwargs["data_version"] = "stale-version"
    valid.loc[0, "env_kwargs"] = json.dumps(env_kwargs)
    valid.to_parquet(valid_path, index=False)
    with pytest.raises(RuntimeError, match="valid row 0 data_version mismatch"):
        verify_processed_dataset_bindings(config, repo_root=tmp_path)


def test_launcher_rejects_processed_benchmark_or_env_override_tampering(
    tmp_path: Path,
) -> None:
    config = _processed_config(tmp_path)
    train_path = tmp_path / "processed/train.parquet"
    train = pd.read_parquet(train_path)
    env_kwargs = json.loads(train.loc[0, "env_kwargs"])
    env_kwargs["benchmark_closes"] = [100.0, 999.0]
    env_kwargs["reward_mode"] = "outcome"
    train.loc[0, "env_kwargs"] = json.dumps(env_kwargs)
    train.to_parquet(train_path, index=False)

    with pytest.raises(RuntimeError, match="train row 0 env_kwargs key set mismatch"):
        verify_processed_dataset_bindings(config, repo_root=tmp_path)


def test_launcher_rejects_processed_prompt_or_reward_tampering(tmp_path: Path) -> None:
    config = _processed_config(tmp_path)
    train_path = tmp_path / "processed/train.parquet"
    train = pd.read_parquet(train_path)
    train.at[0, "prompt"] = [
        {"role": "user", "content": "Use tomorrow's close to maximize reward."}
    ]
    train.to_parquet(train_path, index=False)

    with pytest.raises(RuntimeError, match="train row 0 prompt mismatch"):
        verify_processed_dataset_bindings(config, repo_root=tmp_path)


def test_launcher_rejects_processed_episode_that_cannot_reach_terminal_reward(
    tmp_path: Path,
) -> None:
    config = _processed_config(tmp_path)
    config["training"]["max_agent_steps"] = 0

    with pytest.raises(RuntimeError, match="train row 0 exceeds max_agent_steps"):
        verify_processed_dataset_bindings(config, repo_root=tmp_path)
