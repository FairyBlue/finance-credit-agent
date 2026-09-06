"""Build and execute the locked Agent-R1 command for one experimental arm."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Mapping
from datetime import date, datetime, time
from hashlib import sha256
from pathlib import Path
from typing import Any

import yaml

from src.data import AvailabilityPolicy, StockBenchAdapter
from src.data.episodes import (
    SMOKE_SHADOW_PROVENANCE,
    load_episode_manifest,
    verify_episode_benchmarks,
    verify_smoke_final_oos_isolation,
)
from src.data.integrity import (
    OverlapReconciliationAudit,
    RawManifestAudit,
    audit_exchange_sessions,
    verify_overlap_reconciliation,
    verify_raw_daily_manifest,
)
from src.evaluation.t2_v2 import verify_t2_v2_artifact
from src.rewards import verify_calibration_document

from .config import load_experiment_config_with_sources
from .context_budget import verify_smoke_context_budget
from .credit_evidence import summarize_step_credit_evidence
from .local_runs import LocalRunRecorder
from .metadata import (
    RunHashes,
    collect_run_hashes,
    observe_run_hashes,
    validate_composite_data_version,
    verify_validation_identity,
)
from .rollout_seed import summarize_rollout_seed_evidence
from .update_probe import summarize_update_evidence

REPO_ROOT = Path(__file__).resolve().parents[2]
FLOW_CONFIGS = {
    "A1": REPO_ROOT / "configs" / "agent_flow_a1.yaml",
    "A2": REPO_ROOT / "configs" / "agent_flow_a2.yaml",
}
A3_FLOW_CONFIGS = {
    "m1_uniform": REPO_ROOT / "configs" / "agent_flow_a3.yaml",
}
WIRING_SMOKE_KIND = "wiring_smoke_1p7b"
WIRING_SMOKE_CONFIG_SHA256 = "2ec544cb4e539cdbc2d3301b255384d3fdd770f5801f8d0ba3194081ac39d844"
A1_WIRING_SMOKE_CURRENT_CONFIG_SHA256 = (
    "9ac8b3b495ec2451082ade8d47bb9cc4066f85d1337ec9f3acc3e0b9ecb78e6a"
)
A3_M1_WIRING_SMOKE_KIND = "a3_m1_wiring_smoke_1p7b"
A3_M1_WIRING_SMOKE_CONFIG_SHA256 = (
    "ed7237bbf4769095ec56095f017e88d3396509713af149269955a024248996a2"
)
DECLARED_SMOKE_CONFIG_SHA256 = {
    WIRING_SMOKE_KIND: A1_WIRING_SMOKE_CURRENT_CONFIG_SHA256,
    A3_M1_WIRING_SMOKE_KIND: A3_M1_WIRING_SMOKE_CONFIG_SHA256,
}
QWEN3_1P7B_REVISION = "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
DATASETS_VERSION = "4.0.0"


def _resolve_c_compiler(
    *,
    parent: dict[str, str] | None = None,
    base_prefix: str | Path | None = None,
) -> tuple[str, str]:
    """Resolve and identify the host C compiler required by Triton's driver shim."""

    environment = os.environ if parent is None else parent
    prefix = Path(sys.base_prefix if base_prefix is None else base_prefix)
    candidates: list[str | Path] = []
    configured = environment.get("CC")
    if configured:
        candidates.append(configured)
    candidates.extend(
        [
            prefix / "bin" / "x86_64-conda-linux-gnu-gcc",
            prefix / "bin" / "x86_64-conda-linux-gnu-cc",
            "cc",
            "gcc",
            "clang",
            "x86_64-conda-linux-gnu-gcc",
            "x86_64-conda-linux-gnu-cc",
        ]
    )
    searched: list[str] = []
    for candidate in candidates:
        rendered = str(candidate)
        searched.append(rendered)
        resolved = (
            shutil.which(rendered, path=environment.get("PATH"))
            if os.sep not in rendered
            else rendered
        )
        if resolved is None:
            continue
        path = Path(resolved).expanduser().resolve()
        if not path.is_file() or not os.access(path, os.X_OK):
            continue
        try:
            result = subprocess.run(
                [str(path), "--version"],
                check=True,
                capture_output=True,
                text=True,
                timeout=10,
            )
        except (OSError, subprocess.SubprocessError):
            continue
        version_line = (result.stdout or result.stderr).splitlines()
        if not version_line:
            continue
        return str(path), version_line[0]
    raise RuntimeError(
        "a working host C compiler is required for Triton's cuda_utils shim; "
        "install the pinned compiler from SETUP.md or set CC to an executable compiler "
        f"(searched: {', '.join(searched)})"
    )


def _verify_triton_cuda_utils(c_compiler: str) -> None:
    """Compile/load the exact Triton CPU driver shim needed by vLLM workers."""

    environment = dict(os.environ)
    environment["CC"] = c_compiler
    with tempfile.TemporaryDirectory(prefix="finance-triton-probe-") as cache_dir:
        # An isolated cache proves the compiler/linker path instead of merely
        # loading a cuda_utils artifact produced by an earlier run.
        environment["TRITON_CACHE_DIR"] = cache_dir
        try:
            result = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    (
                        "from triton.backends.nvidia.driver import CudaUtils; "
                        "CudaUtils(); print('TRITON_CUDA_UTILS_COMPILE_OK')"
                    ),
                ],
                cwd=REPO_ROOT,
                env=environment,
                capture_output=True,
                text=True,
                timeout=120,
            )
        except subprocess.TimeoutExpired as error:
            raise RuntimeError("Triton cuda_utils compiler probe timed out") from error
    if result.returncode != 0 or "TRITON_CUDA_UTILS_COMPILE_OK" not in result.stdout:
        detail = (result.stderr or result.stdout).strip()
        raise RuntimeError(f"Triton cuda_utils compiler probe failed: {detail}")


def _required(config: dict[str, Any], *path: str) -> Any:
    value: Any = config
    for key in path:
        if not isinstance(value, dict) or key not in value:
            raise ValueError(f"missing config value: {'.'.join(path)}")
        value = value[key]
    return value


def _hydra(key: str, value: object, *, add: bool = False) -> str:
    prefix = "+" if add else ""
    if isinstance(value, bool):
        rendered = "True" if value else "False"
    elif isinstance(value, (list, dict)):
        rendered = json.dumps(value, separators=(",", ":"))
    else:
        rendered = str(value)
    return f"{prefix}{key}={rendered}"


def _positive_integer(config: dict[str, Any], *path: str) -> int:
    value = _required(config, *path)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{'.'.join(path)} must be a positive integer")
    return value


def _replace_hydra_override(command: list[str], key: str, value: object) -> list[str]:
    prefixes = (f"{key}=", f"+{key}=")
    matches = [index for index, argument in enumerate(command) if argument.startswith(prefixes)]
    if len(matches) != 1:
        raise RuntimeError(f"expected exactly one upstream override for {key}")
    rewritten = list(command)
    rewritten[matches[0]] = _hydra(key, value)
    return rewritten


def _canonical_timestamps(values: list[object], *, context: str) -> list[str]:
    rendered: list[str] = []
    for value in values:
        if not isinstance(value, str) or "TODO" in value.upper():
            raise RuntimeError(f"{context} is not frozen")
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            raise RuntimeError(f"{context} lacks timezone")
        rendered.append(parsed.isoformat())
    return rendered


def verify_configured_source_manifests(
    config: dict[str, Any], *, repo_root: str | Path
) -> dict[str, RawManifestAudit]:
    """Bind every active snapshot hash to its actual raw Parquet bytes."""

    root = Path(repo_root).expanduser().resolve()
    data = _required(config, "data")
    sources = _required(data, "sources")
    symbols = _required(data, "universe", "symbols")
    calendar = str(_required(data, "availability", "exchange_calendar"))
    if not isinstance(symbols, list) or not all(isinstance(item, str) for item in symbols):
        raise ValueError("data.universe.symbols must be a frozen ticker list")

    split_source = data.get("split_source")
    if isinstance(split_source, dict):
        active_sources = sorted(
            {str(_required(split_source, split)) for split in ("train", "valid", "test")}
        )
    else:
        active_sources = sorted(str(source) for source in sources)
    audits: dict[str, RawManifestAudit] = {}
    for source_name in active_sources:
        source = _required(sources, source_name)
        window = _required(source, "input_window")
        manifest_root = Path(str(_required(source, "manifest_root")))
        storage_root = Path(str(_required(source, "storage_root")))
        if not manifest_root.is_absolute():
            manifest_root = root / manifest_root
        if not storage_root.is_absolute():
            storage_root = root / storage_root
        if source_name == "stockbench" and "repo_commit" in source:
            observed = _required(data, "upstream_window_observation")
            session_audit = audit_exchange_sessions(
                start=date.fromisoformat(str(_required(window, "start"))),
                end=date.fromisoformat(str(_required(window, "end"))),
                exchange_calendar=calendar,
            )
            if (
                session_audit.session_count != int(_required(observed, "calendar_session_count"))
                or len(session_audit.canonical_bytes)
                != int(_required(observed, "calendar_session_list_bytes"))
                or session_audit.sha256 != str(_required(observed, "calendar_session_list_sha256"))
            ):
                raise RuntimeError("configured XNYS session-list checksum does not match")
            expected_commit = str(_required(source, "repo_commit"))
            expected_tree = str(_required(source, "storage_tree"))
            actual_commit = subprocess.run(
                ["git", "-C", str(manifest_root), "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            actual_tree = subprocess.run(
                ["git", "-C", str(manifest_root), "rev-parse", "HEAD:storage"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            dirty_storage = subprocess.run(
                [
                    "git",
                    "-C",
                    str(manifest_root),
                    "status",
                    "--porcelain",
                    "--untracked-files=no",
                    "--",
                    "storage",
                ],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            if actual_commit != expected_commit or actual_tree != expected_tree or dirty_storage:
                raise RuntimeError(
                    "StockBench checkout does not match its frozen commit/storage tree"
                )
        audits[source_name] = verify_raw_daily_manifest(
            manifest_root=manifest_root,
            storage_root=storage_root,
            symbols=symbols,
            start=date.fromisoformat(str(_required(window, "start"))),
            end=date.fromisoformat(str(_required(window, "end"))),
            expected_sha256=str(_required(source, "input_manifest_sha256")),
            expected_file_count=int(_required(source, "files")),
            expected_sessions_per_symbol=int(_required(source, "sessions_per_symbol")),
            exchange_calendar=calendar,
        )
    return audits


def verify_configured_overlap(
    config: dict[str, Any], *, repo_root: str | Path
) -> OverlapReconciliationAudit:
    """Verify the history/StockBench price basis on a pinned pre-OOS session."""

    root = Path(repo_root).expanduser().resolve()
    data = _required(config, "data")
    audit_config = _required(data, "cross_source_overlap_audit")
    session = date.fromisoformat(str(_required(audit_config, "session")))
    availability = _required(data, "availability")
    policy = AvailabilityPolicy(
        str(_required(availability, "timezone")),
        time.fromisoformat(str(_required(availability, "market_close_time"))),
        str(_required(availability, "exchange_calendar")),
    )
    sources = _required(data, "sources")

    def adapter(source_name: str) -> StockBenchAdapter:
        source = _required(sources, source_name)
        storage_root = Path(str(_required(source, "storage_root")))
        if not storage_root.is_absolute():
            storage_root = root / storage_root
        return StockBenchAdapter.from_storage(
            storage_root,
            symbols=_required(data, "universe", "symbols"),
            data_version=str(_required(source, "version")),
            availability_policy=policy,
            start_date=session,
            end_date=session,
        )

    return verify_overlap_reconciliation(
        left=adapter("history"),
        right=adapter("stockbench"),
        as_of=policy.market_available_at(session),
        symbols=_required(data, "universe", "symbols"),
        tolerance_bps=_required(audit_config, "tolerance_bps"),
    )


def verify_processed_dataset_bindings(
    config: dict[str, Any], *, repo_root: str | Path
) -> dict[str, dict[str, object]]:
    """Reject ignored processed rows that were built from stale assumptions."""

    import pandas as pd

    root = Path(repo_root).expanduser().resolve()
    data = _required(config, "data")
    sources = _required(data, "sources")
    processed = _required(data, "processed")
    manifest_path = Path(str(_required(data, "episode_manifest")))
    if not manifest_path.is_absolute():
        manifest_path = root / manifest_path
    manifest = load_episode_manifest(config=config, manifest_path=manifest_path)
    split_documents = _required(manifest, "splits")
    execution = _required(config, "execution")
    reward = _required(config, "reward")
    availability = _required(data, "availability")
    symbols = _required(data, "universe", "symbols")
    max_agent_steps = int(_required(config, "training", "max_agent_steps"))
    reserved_information_steps = int(
        _required(config, "training", "reserved_information_steps_per_session")
    )
    is_smoke = bool(config.get("run", {}).get("smoke"))
    smoke_episodes = _positive_integer(config, "training", "episodes") if is_smoke else None
    checked_splits = ("train", "valid", "test") if is_smoke else ("train", "valid")
    evidence: dict[str, dict[str, object]] = {}
    for split in checked_splits:
        source_name = str(_required(data, "split_source", split))
        source = _required(sources, source_name)
        window = _required(source, "input_window")
        manifest_root = Path(str(_required(source, "manifest_root")))
        if not manifest_root.is_absolute():
            manifest_root = root / manifest_root
        storage_root = Path(str(_required(source, "storage_root")))
        if not storage_root.is_absolute():
            storage_root = root / storage_root
        expected_common = {
            "symbols": symbols,
            "manifest_root": str(manifest_root.resolve()),
            "storage_root": str(storage_root.resolve()),
            "data_version": _required(source, "version"),
            "data_start": _required(window, "start"),
            "data_end": _required(window, "end"),
            "input_manifest_sha256": _required(source, "input_manifest_sha256"),
            "input_file_count": int(_required(source, "files")),
            "input_sessions_per_symbol": int(_required(source, "sessions_per_symbol")),
            "initial_cash": float(_required(execution, "initial_cash")),
            "commission_bps": float(_required(execution, "commission_bps")),
            "slippage_bps": float(_required(execution, "slippage_bps")),
            "drawdown_coefficient": float(_required(reward, "drawdown_coefficient")),
            "turnover_coefficient": float(_required(reward, "turnover_coefficient")),
            "finance_epsilon": float(_required(reward, "epsilon")),
            "risk_lambda": float(_required(reward, "risk_lambda")),
            "turnover_limit": float(_required(reward, "turnover_limit")),
            "position_limit": float(_required(reward, "position_limit")),
            "market_timezone": str(_required(availability, "timezone")),
            "exchange_calendar": str(_required(availability, "exchange_calendar")),
            "market_close_time": str(_required(availability, "market_close_time")),
        }
        path = root / str(_required(processed, f"{split}_file"))
        if not path.is_file():
            raise FileNotFoundError(f"missing processed Agent-R1 dataset: {path}")
        frame = pd.read_parquet(path)
        required_columns = {
            "agent_name",
            "data_source",
            "prompt",
            "reward_model",
            "env_kwargs",
            "extra_info",
        }
        missing_columns = required_columns.difference(frame.columns)
        if missing_columns:
            raise RuntimeError(
                f"{split} processed dataset is missing columns: {sorted(missing_columns)}"
            )
        if frame.empty:
            raise RuntimeError(f"{split} processed dataset must not be empty")
        if split == "train" and smoke_episodes is not None and len(frame) != smoke_episodes:
            raise RuntimeError(
                f"smoke train row count {len(frame)} does not match "
                f"training.episodes={smoke_episodes}"
            )
        episodes = _required(split_documents, split)
        if not isinstance(episodes, list):
            raise RuntimeError(f"reviewed manifest split {split} must be a list")
        if len(frame) != len(episodes):
            raise RuntimeError(f"{split} processed row count does not match reviewed manifest")
        for index, row in frame.iterrows():
            episode = episodes[index]
            if not isinstance(episode, dict):
                raise RuntimeError(f"reviewed manifest {split} row {index} must be a mapping")
            if row["agent_name"] != "trading":
                raise RuntimeError(f"{split} row {index} agent_name mismatch")
            if row["data_source"] != f"{source_name}/{split}":
                raise RuntimeError(f"{split} row {index} data_source mismatch")
            prompt = row["prompt"]
            if hasattr(prompt, "tolist"):
                prompt = prompt.tolist()
            expected_prompt = [
                {
                    "role": "user",
                    "content": "Manage the portfolio using only point-in-time tools.",
                }
            ]
            if prompt != expected_prompt:
                raise RuntimeError(f"{split} row {index} prompt mismatch")
            if row["reward_model"] != {"style": "rule", "ground_truth": 0.0}:
                raise RuntimeError(f"{split} row {index} reward_model mismatch")
            try:
                env_kwargs = json.loads(row["env_kwargs"])
            except (TypeError, json.JSONDecodeError) as error:
                raise RuntimeError(f"{split} row {index} env_kwargs is not valid JSON") from error
            if not isinstance(env_kwargs, dict):
                raise RuntimeError(f"{split} row {index} env_kwargs must be a mapping")
            if is_smoke and split == "test":
                verify_smoke_processed_row_provenance(
                    data_source=row["data_source"],
                    env_kwargs=env_kwargs,
                    extra_info=row["extra_info"],
                )
            raw_dates = episode.get("episode_dates")
            raw_opens = episode.get("episode_open_times")
            raw_benchmark = episode.get("benchmark_closes")
            if not all(
                isinstance(values, list) for values in (raw_dates, raw_opens, raw_benchmark)
            ):
                raise RuntimeError(f"reviewed manifest {split} row {index} is incomplete")
            assert isinstance(raw_dates, list)
            assert isinstance(raw_opens, list)
            assert isinstance(raw_benchmark, list)
            budgeted_steps = (len(raw_dates) - 1) * (1 + reserved_information_steps)
            if len(raw_dates) < 2 or budgeted_steps > max_agent_steps:
                raise RuntimeError(f"{split} row {index} exceeds max_agent_steps")

            expected = {
                **expected_common,
                "episode_dates": _canonical_timestamps(
                    raw_dates,
                    context=f"reviewed manifest {split} row {index} episode_dates",
                ),
                "episode_open_times": _canonical_timestamps(
                    raw_opens,
                    context=f"reviewed manifest {split} row {index} episode_open_times",
                ),
                "benchmark_closes": [float(value) for value in raw_benchmark],
            }
            if is_smoke and split == "test":
                expected["provenance"] = dict(SMOKE_SHADOW_PROVENANCE)
            if set(env_kwargs) != set(expected):
                raise RuntimeError(f"{split} row {index} env_kwargs key set mismatch")
            for key, expected_value in expected.items():
                if env_kwargs.get(key) != expected_value:
                    raise RuntimeError(f"{split} row {index} {key} mismatch")
            extra_info = row["extra_info"]
            expected_info = {
                "episode_id": episode.get("episode_id"),
                "split": split,
                "index": index,
            }
            if is_smoke and split == "test":
                expected_info.update(SMOKE_SHADOW_PROVENANCE)
            if extra_info != expected_info:
                raise RuntimeError(f"{split} row {index} extra_info mismatch")
        evidence[split] = {
            "path": str(path.resolve()),
            "rows": int(len(frame)),
            "sha256": sha256(path.read_bytes()).hexdigest(),
            "source": source_name,
        }
    return evidence


def verify_smoke_processed_row_provenance(
    *, data_source: object, env_kwargs: object, extra_info: object
) -> None:
    """Reject a renamed Final OOS parquet row by inspecting embedded provenance."""

    if data_source != "history/test":
        raise RuntimeError("smoke test processed row must be history-backed")
    if not isinstance(env_kwargs, dict) or not isinstance(extra_info, dict):
        raise RuntimeError("smoke test processed provenance must be mappings")
    if env_kwargs.get("provenance") != SMOKE_SHADOW_PROVENANCE:
        raise RuntimeError("smoke test env provenance mismatch")
    if any(extra_info.get(key) != value for key, value in SMOKE_SHADOW_PROVENANCE.items()):
        raise RuntimeError("smoke test extra_info provenance mismatch")

    def string_values(value: object) -> list[str]:
        if isinstance(value, dict):
            return [item for nested in value.values() for item in string_values(nested)]
        if isinstance(value, list):
            return [item for nested in value for item in string_values(nested)]
        return [value.casefold()] if isinstance(value, str) else []

    if any(
        "stockbench" in value or "final_oos" in value
        for value in string_values({"env_kwargs": env_kwargs, "extra_info": extra_info})
    ):
        raise RuntimeError("smoke test processed row references Final OOS provenance")
    dates = env_kwargs.get("episode_dates")
    if not isinstance(dates, list):
        raise RuntimeError("smoke test processed dates must be frozen")
    for raw_date in dates:
        session = datetime.fromisoformat(str(raw_date).replace("Z", "+00:00")).date()
        if date(2025, 3, 3) <= session <= date(2025, 6, 30):
            raise RuntimeError(f"smoke test processed row enters Final OOS: {session}")


def _is_declared_wiring_smoke(config: dict[str, Any]) -> bool:
    run = config.get("run", {})
    if not isinstance(run, dict):
        raise RuntimeError("invalid T2 smoke exemption: run must be a mapping")
    smoke = run.get("smoke", False)
    kind = run.get("kind")
    if smoke is False and kind is None:
        return False
    if smoke is not True or kind != WIRING_SMOKE_KIND:
        raise RuntimeError(
            f"invalid T2 smoke exemption: expected run.smoke=true and run.kind={WIRING_SMOKE_KIND}"
        )

    try:
        model = _required(config, "model")
        if not isinstance(model, dict):
            raise ValueError("model must be a mapping")
        selected_model = model.get("name") or model.get("train")
        contract = (
            ("run.name", _required(config, "run", "name"), "smoke-a1-qwen3-1p7b"),
            ("run.seed", _required(config, "run", "seed"), 42),
            ("arm.id", _required(config, "arm", "id"), "A1"),
            ("arm.name", _required(config, "arm", "name"), "outcome_only"),
            ("arm.optimizer", _required(config, "arm", "optimizer"), "grpo"),
            (
                "arm.reward_signal",
                _required(config, "arm", "reward_signal"),
                "episode_return",
            ),
            (
                "arm.credit_assignment",
                _required(config, "arm", "credit_assignment"),
                "outcome_only",
            ),
            ("model.name/train", selected_model, "Qwen/Qwen3-1.7B"),
            ("model.debug", model.get("debug"), "Qwen/Qwen3-1.7B"),
            ("model.revision", model.get("revision"), QWEN3_1P7B_REVISION),
            ("model.local_files_only", model.get("local_files_only"), True),
            ("training.episodes", _required(config, "training", "episodes"), 4),
            (
                "training.max_agent_steps",
                _required(config, "training", "max_agent_steps"),
                32,
            ),
            (
                "training.train_batch_size",
                _required(config, "training", "train_batch_size"),
                2,
            ),
            ("training.rollout_n", _required(config, "training", "rollout_n"), 2),
            ("training.total_epochs", _required(config, "training", "total_epochs"), 1),
            (
                "training.learning_rate",
                _required(config, "training", "learning_rate"),
                1.0e-6,
            ),
            (
                "training.n_gpus_per_node",
                _required(config, "training", "n_gpus_per_node"),
                1,
            ),
            (
                "data.episode_manifest",
                _required(config, "data", "episode_manifest"),
                "configs/episode_manifest.smoke.reviewed.yaml",
            ),
            (
                "data.processed.train_file",
                _required(config, "data", "processed", "train_file"),
                "data/processed/smoke_train.parquet",
            ),
            (
                "data.processed.valid_file",
                _required(config, "data", "processed", "valid_file"),
                "data/processed/smoke_valid.parquet",
            ),
            (
                "data.processed.test_file",
                _required(config, "data", "processed", "test_file"),
                "data/processed/smoke_test.parquet",
            ),
        )
    except ValueError as error:
        raise RuntimeError(f"invalid T2 smoke exemption: {error}") from error

    for label, actual, expected in contract:
        if actual != expected:
            raise RuntimeError(
                f"invalid T2 smoke exemption: {label} must be {expected!r}, got {actual!r}"
            )
    canonical_config = {**config, "data": dict(_required(config, "data"))}
    # The formal-only sampling policy was added after G1 completed and cannot
    # alter the historical wiring-smoke identity or execution behavior.
    canonical_config["data"].pop("formal_episode_sampling", None)
    canonical = json.dumps(
        canonical_config,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    config_sha256 = sha256(canonical).hexdigest()
    if config_sha256 != WIRING_SMOKE_CONFIG_SHA256:
        raise RuntimeError(
            "invalid T2 smoke exemption: merged config SHA-256 must be "
            f"{WIRING_SMOKE_CONFIG_SHA256}, got {config_sha256}"
        )
    return True


def verify_declared_smoke_identity(config: dict[str, Any]) -> dict[str, str]:
    """Lock every executable smoke to one reviewed merged configuration."""

    run = config.get("run")
    if not isinstance(run, dict):
        raise RuntimeError("invalid smoke identity: run must be a mapping")
    if run.get("smoke") is not True:
        raise RuntimeError("invalid smoke identity: run.smoke must be true")
    kind = run.get("kind")
    if not isinstance(kind, str) or kind not in DECLARED_SMOKE_CONFIG_SHA256:
        raise RuntimeError(f"invalid smoke identity: unregistered run.kind={kind!r}")
    canonical_config = {**config, "tracking": dict(_required(config, "tracking"))}
    # Evidence storage is relocatable and does not alter the executable smoke.
    canonical_config["tracking"].pop("log_root", None)
    canonical = json.dumps(
        canonical_config,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    observed = sha256(canonical).hexdigest()
    expected = DECLARED_SMOKE_CONFIG_SHA256[kind]
    if observed != expected:
        raise RuntimeError(
            "invalid smoke identity: merged config SHA-256 must be "
            f"{expected}, got {observed}"
        )
    return {"kind": kind, "canonical_merged_config_sha256": observed}


def verify_configured_t2_gate(
    config: dict[str, Any], *, repo_root: str | Path
) -> dict[str, object]:
    """Require a verified T2 v2 pass, retaining only the historical smoke exemption."""

    evaluation = _required(config, "evaluation")
    if not isinstance(evaluation, dict):
        raise RuntimeError("evaluation must be a mapping")
    preregistered_v2 = evaluation.get("shifted_label_probe_v2")
    if not isinstance(preregistered_v2, dict):
        raise RuntimeError(
            "T2 v2 configuration is required; formal training is blocked without "
            "falling back to the legacy StockBench probe"
        )
    source_policy = preregistered_v2.get("source_policy")
    if (
        not isinstance(source_policy, dict)
        or source_policy.get("stockbench_oos_access") != "forbidden"
    ):
        raise RuntimeError("T2 v2 must fail closed with StockBench OOS access forbidden")
    if preregistered_v2.get("implementation_status") == "passed":
        try:
            evidence = verify_t2_v2_artifact(config, repo_root=repo_root)
        except ValueError as error:
            raise RuntimeError(f"T2 v2 configuration verification failed: {error}") from error
        return {"gate": "passed_artifact", **evidence}
    if _is_declared_wiring_smoke(config):
        return {"gate": "approved_historical_wiring_smoke_exemption"}
    raise RuntimeError(
        "T2 v2 is preregistered but not implemented; formal training is blocked "
        "without reopening the legacy StockBench probe"
    )


def resolve_agent_flow_config(config: dict[str, Any]) -> Path:
    """Resolve one fail-closed Agent-R1 flow from the declared arm mechanism."""

    arm = _required(config, "arm")
    arm_id = str(_required(arm, "id"))
    if arm_id in FLOW_CONFIGS:
        return FLOW_CONFIGS[arm_id]
    if arm_id != "A3":
        raise ValueError(f"unsupported arm id: {arm_id}")

    mechanism = str(_required(arm, "mechanism"))
    if mechanism not in A3_FLOW_CONFIGS:
        raise RuntimeError(f"A3 mechanism is not executable: {mechanism!r}")
    if _required(arm, "optimizer") != "steppo":
        raise RuntimeError("A3 requires the shared StepPO optimizer")
    if _required(arm, "credit_assignment") != "finance_native":
        raise RuntimeError("A3 requires finance_native credit_assignment")
    if _required(arm, "reward_signal") != "A_fin":
        raise RuntimeError("A3 requires reward_signal=A_fin")

    credit_semantics = str(_required(arm, "credit_semantics"))
    scientific_role = str(_required(arm, "scientific_role"))
    mechanism_config = _required(config, "finance_credit", mechanism)
    if not isinstance(mechanism_config, dict):
        raise RuntimeError(f"finance_credit.{mechanism} must be a mapping")
    if mechanism_config.get("implemented") is not True:
        raise RuntimeError(f"A3 mechanism is not implemented: {mechanism!r}")
    if mechanism_config.get("credit_semantics") != credit_semantics:
        raise RuntimeError("A3 arm and mechanism credit semantics do not match")
    if mechanism_config.get("scientific_role") != scientific_role:
        raise RuntimeError("A3 arm and mechanism scientific roles do not match")
    if mechanism == "m1_uniform" and scientific_role != "formal_ablation":
        raise RuntimeError("M1 is executable only as the formal_ablation control")

    flow_path = A3_FLOW_CONFIGS[mechanism]
    document = yaml.safe_load(flow_path.read_text(encoding="utf-8"))
    if not isinstance(document, list) or len(document) != 1:
        raise RuntimeError("A3 flow config must contain exactly one flow definition")
    flow = document[0]
    if not isinstance(flow, dict):
        raise RuntimeError("A3 flow definition must be a mapping")
    expected = {
        "_target_": "src.agents.agent_flow.FinanceCreditAgentEnvLoop",
        "env_type": "trading",
        "reward_mode": "finance_trajectory",
        "credit_strategy": mechanism,
        "credit_semantics": credit_semantics,
    }
    mismatches = {
        key: {"expected": value, "actual": flow.get(key)}
        for key, value in expected.items()
        if flow.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"A3 mechanism/flow configuration mismatch: {mismatches}")
    return flow_path


def verify_a3_smoke_credit_summary(
    step_credit_evidence: Mapping[str, object],
    rollout_seed_evidence: Mapping[str, object],
) -> None:
    """Require full, non-zero train/validation M1 evidence for a successful smoke."""

    worker_files = step_credit_evidence.get("worker_files")
    if not isinstance(worker_files, int) or isinstance(worker_files, bool) or worker_files <= 0:
        raise RuntimeError(
            "A3 training exited successfully but no step-credit evidence file was recorded"
        )
    for split in ("training", "validation"):
        trajectory_count = step_credit_evidence.get(f"{split}_trajectory_count")
        if (
            not isinstance(trajectory_count, int)
            or isinstance(trajectory_count, bool)
            or trajectory_count <= 0
        ):
            raise RuntimeError(f"A3 smoke recorded no {split} step-credit trajectories")
        nonzero_count = step_credit_evidence.get(
            f"{split}_nonzero_trajectory_advantage_count"
        )
        if (
            not isinstance(nonzero_count, int)
            or isinstance(nonzero_count, bool)
            or nonzero_count <= 0
        ):
            raise RuntimeError(f"A3 smoke recorded no non-zero {split} A_fin trajectory")
    if step_credit_evidence.get(
        "trajectory_identity_sha256"
    ) != rollout_seed_evidence.get("trajectory_identity_sha256"):
        raise RuntimeError("step-credit and rollout-seed trajectory identities do not match")


def verify_a3_optimizer_update_evidence(
    update_evidence: Mapping[str, object],
) -> None:
    """Require finite parameter changes from both halves of the A3 update."""

    role_counts = update_evidence.get("detected_role_counts")
    if not isinstance(role_counts, Mapping):
        raise RuntimeError("A3 smoke optimizer evidence has no detected role counts")
    missing = [
        role
        for role in ("critic", "actor")
        if not isinstance(role_counts.get(role), int)
        or isinstance(role_counts.get(role), bool)
        or role_counts.get(role, 0) <= 0
    ]
    if missing:
        raise RuntimeError(
            "A3 smoke requires finite non-zero optimizer updates for both critic "
            f"and actor; missing: {', '.join(missing)}"
        )


def build_upstream_command(
    config: dict[str, Any],
    hashes: RunHashes,
    *,
    update_probe_dir: str | Path | None = None,
    flow_config_path: str | Path | None = None,
) -> list[str]:
    arm_id = str(_required(config, "arm", "id"))
    flow_config = (
        resolve_agent_flow_config(config)
        if flow_config_path is None
        else Path(flow_config_path).resolve()
    )
    training = _required(config, "training")
    data = _required(config, "data")
    processed = _required(data, "processed")
    model_config = _required(config, "model")
    model_name = model_config.get("name") or model_config.get("train")
    if not model_name:
        raise ValueError("model.name or model.train is required")
    is_step_po = arm_id in {"A2", "A3"}
    seed = _required(config, "run", "seed")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("run.seed must be a non-negative integer")
    run_name = config.get("run", {}).get("name", f"{arm_id.lower()}-training")
    is_smoke = bool(config.get("run", {}).get("smoke"))
    logger = ["console"]
    episodes = _positive_integer(config, "training", "episodes")
    total_training_steps = training.get("total_training_steps")
    if total_training_steps is not None and (
        isinstance(total_training_steps, bool)
        or not isinstance(total_training_steps, int)
        or total_training_steps <= 0
    ):
        raise ValueError("training.total_training_steps must be a positive integer")
    critic_param_offload = training.get("critic_param_offload", True)
    if not isinstance(critic_param_offload, bool):
        raise ValueError("training.critic_param_offload must be a boolean")
    reference_model_dtype = training.get("reference_model_dtype")
    if reference_model_dtype is not None and reference_model_dtype not in {
        "bf16",
        "fp32",
    }:
        raise ValueError("training.reference_model_dtype must be 'bf16' or 'fp32'")

    if update_probe_dir is not None:
        update_probe_dir = Path(update_probe_dir)
        if not update_probe_dir.is_absolute():
            raise ValueError("update probe directory must be absolute")

    command = [
        sys.executable,
        "-m",
        "agent_r1.trainer.main_agent_ppo",
        _hydra("algorithm.adv_estimator", "gae" if is_step_po else "grpo"),
        _hydra("data.train_files", str(REPO_ROOT / processed["train_file"])),
        _hydra("data.val_files", str(REPO_ROOT / processed["valid_file"])),
        _hydra("data.train_batch_size", training["train_batch_size"]),
        _hydra("data.train_max_samples", episodes),
        _hydra("data.max_prompt_length", training["max_prompt_length"]),
        _hydra("data.max_response_length", training["max_response_length"]),
        _hydra("data.filter_overlong_prompts", True),
        _hydra("data.truncation", "error"),
        _hydra("data.return_raw_chat", True),
        _hydra("data.seed", seed),
        # The four-row smoke does not benefit from verl's 16 persistent train/val
        # loader children, which amplify cgroup pressure during hybrid sleep/wake.
        *([_hydra("data.dataloader_num_workers", 0)] if is_smoke else []),
        _hydra("actor_rollout_ref.model.path", model_name),
        _hydra("actor_rollout_ref.actor.optim.lr", training["learning_rate"]),
        _hydra("actor_rollout_ref.actor.data_loader_seed", seed),
        _hydra("actor_rollout_ref.model.use_remove_padding", True),
        _hydra("actor_rollout_ref.model.enable_gradient_checkpointing", True),
        _hydra("actor_rollout_ref.actor.ppo_mini_batch_size", training["train_batch_size"]),
        _hydra("actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu", 1),
        _hydra("actor_rollout_ref.actor.use_kl_loss", True),
        _hydra("actor_rollout_ref.actor.kl_loss_coef", 0.001),
        _hydra("actor_rollout_ref.actor.kl_loss_type", "low_var_kl"),
        _hydra("actor_rollout_ref.actor.entropy_coeff", 0),
        _hydra("actor_rollout_ref.actor.fsdp_config.param_offload", True),
        # Keep Adam state resident for the H100 smoke to avoid host transfers and
        # memory.high reclaim; parameter offload still preserves rollout headroom.
        _hydra("actor_rollout_ref.actor.fsdp_config.optimizer_offload", not is_smoke),
        _hydra("actor_rollout_ref.actor.fsdp_config.seed", seed),
        _hydra("actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu", 1),
        _hydra("actor_rollout_ref.rollout.tensor_model_parallel_size", 1),
        _hydra("actor_rollout_ref.rollout.name", "vllm"),
        # Pinned vLLM 0.9.2 predates this CLI flag; keep decoupled actor-side
        # old-logprob recomputation explicit when omitting the rollout mode.
        _hydra("actor_rollout_ref.rollout.logprobs_mode", "null"),
        _hydra("actor_rollout_ref.rollout.calculate_log_probs", False),
        _hydra("actor_rollout_ref.rollout.gpu_memory_utilization", 0.55),
        _hydra("actor_rollout_ref.rollout.n", 1 if is_step_po else training["rollout_n"]),
        _hydra("actor_rollout_ref.rollout.engine_kwargs.vllm.seed", seed, add=True),
        _hydra(
            "actor_rollout_ref.rollout.agent.agent_flow_config_path",
            str(flow_config),
        ),
        _hydra("actor_rollout_ref.rollout.agent.default_agent_flow", "trading"),
        _hydra("actor_rollout_ref.rollout.agent.max_steps", training["max_agent_steps"]),
        _hydra("actor_rollout_ref.rollout.agent.num_workers", 1),
        _hydra("actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu", 1),
        _hydra("actor_rollout_ref.ref.fsdp_config.param_offload", True),
        *(
            [
                _hydra(
                    "actor_rollout_ref.ref.fsdp_config.model_dtype",
                    reference_model_dtype,
                )
            ]
            if reference_model_dtype is not None
            else []
        ),
        _hydra("reward_model.enable", False),
        _hydra("algorithm.use_kl_in_reward", False),
        _hydra("algorithm.rollout_correction.bypass_mode", False),
        _hydra("trainer.critic_warmup", 0),
        _hydra("trainer.logger", logger),
        _hydra("trainer.project_name", _required(config, "tracking", "project")),
        _hydra("trainer.experiment_name", run_name),
        _hydra("trainer.n_gpus_per_node", training["n_gpus_per_node"]),
        _hydra("trainer.nnodes", 1),
        _hydra("trainer.save_freq", -1 if is_smoke else 50),
        _hydra("trainer.test_freq", 1),
        _hydra("trainer.total_epochs", training["total_epochs"]),
        *(
            [_hydra("trainer.total_training_steps", total_training_steps)]
            if total_training_steps is not None
            else []
        ),
        _hydra("ray_kwargs.ray_init.address", "local", add=True),
        _hydra("ray_kwargs.ray_init.include_dashboard", False, add=True),
        # These hooks are needed only in Ray workers. Keeping them out of the
        # trainer's inherited environment prevents Ray's dashboard/runtime-env
        # agents from eagerly importing torch, verl, and vLLM during startup.
        _hydra(
            "ray_kwargs.ray_init.runtime_env.env_vars.FINANCE_VLLM_DRAIN_COMPAT",
            "0.9.2",
            add=True,
        ),
        _hydra("reproducibility.agent_r1_commit", hashes.agent_r1_commit, add=True),
        _hydra("reproducibility.repo_commit", hashes.repo_commit, add=True),
        _hydra("reproducibility.data_version", hashes.data_version, add=True),
        _hydra("reproducibility.repo_dirty", hashes.repo_dirty, add=True),
    ]
    if is_step_po:
        command.extend(
            [
                _hydra("actor_rollout_ref.actor.policy_loss.loss_mode", "gspo"),
                _hydra("critic.model.path", model_name),
                _hydra("critic.optim.lr", 1e-5),
                _hydra("critic.model.use_remove_padding", True),
                _hydra("critic.model.enable_gradient_checkpointing", True),
                _hydra("critic.data_loader_seed", seed),
                _hydra("critic.ppo_micro_batch_size_per_gpu", 1),
                _hydra(
                    "critic.model.fsdp_config.param_offload",
                    critic_param_offload,
                ),
                _hydra("critic.model.fsdp_config.optimizer_offload", True),
                _hydra("critic.model.fsdp_config.seed", seed),
            ]
        )
    else:
        command.append(_hydra("critic.enable", False))
    if update_probe_dir is not None:
        command.append(
            _hydra(
                "ray_kwargs.ray_init.runtime_env.env_vars.FINANCE_UPDATE_PROBE_DIR",
                str(update_probe_dir),
                add=True,
            )
        )
        command.append(
            _hydra(
                "ray_kwargs.ray_init.runtime_env.env_vars.FINANCE_ROLLOUT_SEED_DIR",
                str(update_probe_dir.parent / "rollout_seed"),
                add=True,
            )
        )
        if arm_id == "A3":
            command.append(
                _hydra(
                    "ray_kwargs.ray_init.runtime_env.env_vars."
                    "FINANCE_STEP_CREDIT_EVIDENCE_DIR",
                    str(update_probe_dir.parent / "step_credit"),
                    add=True,
                )
            )
    return command


def _run_with_local_capture(
    command: list[str], *, environment: dict[str, str], recorder: LocalRunRecorder
) -> int:
    process = subprocess.Popen(
        command,
        cwd=REPO_ROOT,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="")
        recorder.append_event("subprocess_output", line=line.rstrip("\n"))
    return process.wait()


def _resolve_exact_model_snapshot(
    config: dict[str, Any], *, cache_root: Path | None = None
) -> tuple[str, str, Path]:
    """Resolve only the configured immutable revision from local files."""

    selected_model = str(
        _required(config, "model", "name")
        if "name" in _required(config, "model")
        else _required(config, "model", "train")
    )
    configured_revision = _required(config, "model").get("revision")
    if (
        not isinstance(configured_revision, str)
        or len(configured_revision) != 40
        or any(character not in "0123456789abcdef" for character in configured_revision)
    ):
        raise RuntimeError("model.revision must be a full lowercase 40-character commit")
    model_path = Path(selected_model).expanduser()
    model_revision = configured_revision
    if not model_path.is_dir():
        if cache_root is None:
            cache_root = Path(
                os.environ.get(
                    "HF_HUB_CACHE",
                    str(
                        Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface")) / "hub"
                    ),
                )
            )
        cached_model = cache_root / ("models--" + selected_model.replace("/", "--"))
        snapshots = cached_model / "snapshots"
        model_snapshot_path = (snapshots / configured_revision).resolve()
        if not model_snapshot_path.is_dir():
            raise RuntimeError(
                f"exact model revision is not cached locally: {selected_model}@"
                f"{configured_revision}; "
                "pre-download it before smoke"
            )
    else:
        model_snapshot_path = model_path.resolve()
        if model_snapshot_path.name != configured_revision:
            raise RuntimeError(
                "local model snapshot directory does not match configured exact revision"
            )
    return selected_model, model_revision, model_snapshot_path


def verify_gpu_runtime_and_model(config: dict[str, Any]) -> dict[str, object]:
    """Fail closed unless the GPU stack and selected model are locally available."""

    missing = [
        module
        for module in ("datasets", "torch", "verl", "vllm", "flash_attn")
        if importlib.util.find_spec(module) is None
    ]
    if missing:
        raise RuntimeError(
            "GPU extra is incomplete; missing importable modules: " + ", ".join(missing)
        )
    import datasets

    if datasets.__version__ != DATASETS_VERSION:
        raise RuntimeError(
            f"datasets=={DATASETS_VERSION} is required, got {datasets.__version__}"
        )
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise RuntimeError("CUDA GPU is unavailable")
    device_name = str(torch.cuda.get_device_name(0))
    c_compiler, c_compiler_version = _resolve_c_compiler()
    _verify_triton_cuda_utils(c_compiler)
    selected_model, model_revision, model_snapshot_path = _resolve_exact_model_snapshot(config)
    from transformers import AutoConfig  # type: ignore[import-not-found]

    AutoConfig.from_pretrained(model_snapshot_path, local_files_only=True)
    return {
        "cuda_available": True,
        "cuda_device_count": int(torch.cuda.device_count()),
        "gpu_model": device_name,
        "c_compiler": c_compiler,
        "c_compiler_version": c_compiler_version,
        "triton_cuda_utils_compile": True,
        "datasets_version": datasets.__version__,
        "model": selected_model,
        "model_revision": model_revision,
        "model_snapshot_path": str(model_snapshot_path),
    }


def _training_environment(
    *,
    agent_path: Path,
    probe_root: Path,
    c_compiler: str,
    parent: dict[str, str] | None = None,
) -> dict[str, str]:
    environment = dict(os.environ if parent is None else parent)
    # Ray's automatic ``uv run`` hook in 2.58 cannot handle the
    # ``working_dir: None`` emitted by verl 0.7. This is a single-node run
    # whose workers already use the driver's pinned Python, so disable that
    # packaging hook instead of uploading the repository as a working dir.
    environment["RAY_ENABLE_UV_RUN_RUNTIME_ENV"] = "0"
    # Once the uv hook is disabled, Ray would otherwise honor these inherited
    # injection points. Keep the recorded training environment self-contained.
    environment.pop("RAY_RUNTIME_ENV_HOOK", None)
    environment.pop("RAY_JOB_CONFIG_JSON_ENV_VAR", None)
    environment.pop("RAY_ADDRESS", None)
    environment.pop("RAY_NAMESPACE", None)
    # Runtime instrumentation is injected explicitly through Ray's runtime_env.
    # Remove inherited values so Ray's own Python agents remain lightweight.
    environment.pop("FINANCE_UPDATE_PROBE_DIR", None)
    environment.pop("FINANCE_ROLLOUT_SEED_DIR", None)
    environment.pop("FINANCE_STEP_CREDIT_EVIDENCE_DIR", None)
    environment.pop("FINANCE_VLLM_DRAIN_COMPAT", None)
    environment["PYTHONPATH"] = os.pathsep.join(
        [
            str(REPO_ROOT / "runtime_hooks"),
            str(REPO_ROOT),
            str(agent_path),
        ]
    )
    environment["VLLM_USE_V1"] = "1"
    environment["HYDRA_FULL_ERROR"] = "1"
    environment["HF_HUB_OFFLINE"] = "1"
    environment["HF_DATASETS_OFFLINE"] = "1"
    environment["TRANSFORMERS_OFFLINE"] = "1"
    environment["CC"] = c_compiler
    environment["TRITON_CACHE_DIR"] = str(probe_root.parent / "triton_cache")
    return environment


def _verify_reward_freeze(config: dict[str, Any]) -> None:
    reward = _required(config, "reward")
    for key in (
        "epsilon",
        "risk_lambda",
        "drawdown_coefficient",
        "turnover_coefficient",
        "turnover_limit",
        "position_limit",
    ):
        value = _required(reward, key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise RuntimeError(f"reward.{key} must be frozen to a finite number")
    calibration = _required(reward, "calibration")
    table = Path(str(_required(calibration, "table")))
    if not table.is_absolute():
        table = REPO_ROOT / table
    document = yaml.safe_load(table.read_text(encoding="utf-8"))
    verify_calibration_document(document)


def _verify_all_preflight_inputs(
    config: dict[str, Any],
    *,
    command: list[str],
    is_smoke: bool,
    expected_commit: str,
    recorder: LocalRunRecorder,
    config_sources: tuple[Path, ...] = (),
) -> dict[str, object]:
    if _required(config, "tracking", "backend") != "local_files":
        raise RuntimeError("only local_files tracking is approved")
    if any("wandb" in argument.casefold() for argument in command):
        raise RuntimeError("W&B must not be enabled")
    manifest_path = Path(str(_required(config, "data", "episode_manifest")))
    if not manifest_path.is_absolute():
        manifest_path = REPO_ROOT / manifest_path
    agent_path = REPO_ROOT / str(_required(config, "framework", "agent_r1_path"))
    identity = verify_validation_identity(
        repo_root=REPO_ROOT,
        agent_r1_path=agent_path,
        expected_repo_commit=expected_commit,
        expected_agent_r1_commit=str(_required(config, "framework", "agent_r1_commit")),
        allowed_untracked_paths=(manifest_path,),
        required_tracked_paths=config_sources,
    )
    recorder.record_evidence("validation_identity", identity)
    smoke_identity = (
        verify_declared_smoke_identity(config)
        if config.get("run", {}).get("smoke") is True
        else None
    )
    recorder.record_evidence("smoke_identity", smoke_identity)
    t2_evidence = verify_configured_t2_gate(config, repo_root=REPO_ROOT)
    recorder.record_evidence("t2_gate", t2_evidence)
    oos_evidence = verify_smoke_final_oos_isolation(config) if is_smoke else None
    recorder.record_evidence("final_oos_isolation", oos_evidence)
    _verify_reward_freeze(config)
    recorder.record_evidence("reward_freeze", {"verified": True})
    runtime = verify_gpu_runtime_and_model(config)
    recorder.record_evidence("runtime_and_model", runtime)

    required_files = (
        ("train_file", "valid_file", "test_file") if is_smoke else ("train_file", "valid_file")
    )
    for path_key in required_files:
        path = REPO_ROOT / str(_required(config, "data", "processed", path_key))
        if not path.is_file():
            raise FileNotFoundError(f"missing processed Agent-R1 dataset: {path}")
    source_audits = verify_configured_source_manifests(config, repo_root=REPO_ROOT)
    source_evidence = {
        name: {
            "manifest_sha256": audit.manifest_sha256,
            "file_count": audit.file_count,
            "sessions_per_symbol": audit.sessions_per_symbol,
        }
        for name, audit in source_audits.items()
    }
    recorder.record_evidence("source_manifests", source_evidence)
    if not is_smoke:
        overlap = verify_configured_overlap(config, repo_root=REPO_ROOT)
        recorder.record_evidence(
            "source_overlap",
            {
                "as_of": overlap.as_of.isoformat(),
                "maximum_relative_difference_bps": overlap.maximum_relative_difference_bps,
            },
        )
    benchmark_sha256 = verify_episode_benchmarks(
        config=config,
        manifest_path=manifest_path,
        repo_root=REPO_ROOT,
    )
    reviewed_evidence = {
        "path": str(manifest_path.resolve()),
        "sha256": sha256(manifest_path.read_bytes()).hexdigest(),
        "benchmark_series_sha256": benchmark_sha256,
    }
    recorder.record_evidence("reviewed_episode_manifest", reviewed_evidence)
    processed_evidence = verify_processed_dataset_bindings(config, repo_root=REPO_ROOT)
    recorder.record_evidence("processed_datasets", processed_evidence)
    context_budget = (
        verify_smoke_context_budget(config, runtime=runtime, repo_root=REPO_ROOT)
        if is_smoke
        else None
    )
    recorder.record_evidence("prompt_context_budget", context_budget)
    return {
        "identity": identity,
        "runtime": runtime,
        "oos": oos_evidence,
        "source_manifests": source_evidence,
        "reviewed_episode_manifest": reviewed_evidence,
        "processed_datasets": processed_evidence,
        "context_budget": context_budget,
    }


def _prepare_upstream_launch(
    config: dict[str, Any],
    *,
    expected_arm: str | None,
    dry_run: bool,
    update_probe_dir: str | Path | None = None,
) -> tuple[str, bool, RunHashes, list[str]]:
    """Validate static launch inputs and build the still-provisional upstream command."""

    arm_id = str(_required(config, "arm", "id"))
    if expected_arm is not None and arm_id != expected_arm:
        raise ValueError(f"entrypoint {expected_arm} cannot launch config for {arm_id}")
    agent_path = REPO_ROOT / str(_required(config, "framework", "agent_r1_path"))
    data = _required(config, "data")
    sources = _required(data, "sources")
    is_smoke = bool(config.get("run", {}).get("smoke"))
    if is_smoke:
        verify_smoke_final_oos_isolation(config)
        history_hash = str(_required(sources, "history", "input_manifest_sha256"))
        data_version = str(_required(data, "version"))
        expected_version = f"smoke-history-only:raw-manifest-sha256:{history_hash}"
        if data_version != expected_version:
            raise RuntimeError(f"smoke data version must be history-only: {expected_version}")
    else:
        data_version = validate_composite_data_version(
            configured=str(_required(data, "version")),
            history_manifest_sha256=str(_required(sources, "history", "input_manifest_sha256")),
            stockbench_manifest_sha256=str(
                _required(sources, "stockbench", "input_manifest_sha256")
            ),
        )
    hashes = collect_run_hashes(
        repo_root=REPO_ROOT,
        agent_r1_path=agent_path,
        data_version=data_version,
        expected_agent_r1_commit=str(_required(config, "framework", "agent_r1_commit")),
        require_clean_repo=(
            bool(_required(config, "training", "require_clean_repo")) and not dry_run
        ),
    )
    if is_smoke:
        verify_declared_smoke_identity(config)
    return arm_id, is_smoke, hashes, build_upstream_command(
        config,
        hashes,
        update_probe_dir=update_probe_dir,
    )


def _finalize_model_paths(
    command: list[str], *, arm_id: str, model_snapshot_path: str
) -> list[str]:
    command = _replace_hydra_override(
        command, "actor_rollout_ref.model.path", model_snapshot_path
    )
    if arm_id in {"A2", "A3"}:
        command = _replace_hydra_override(command, "critic.model.path", model_snapshot_path)
    return command


def launch(
    config_path: str | Path,
    *,
    dry_run: bool,
    preflight_only: bool = False,
    expected_arm: str | None = None,
    expected_commit: str | None = None,
) -> int:
    config, config_sources = load_experiment_config_with_sources(config_path)
    arm_id = str(_required(config, "arm", "id"))
    agent_path = REPO_ROOT / str(_required(config, "framework", "agent_r1_path"))
    is_smoke = bool(config.get("run", {}).get("smoke"))
    if dry_run:
        _, _, hashes, command = _prepare_upstream_launch(
            config,
            expected_arm=expected_arm,
            dry_run=True,
        )
        print(
            json.dumps(
                {
                    "dry_run_command": command,
                    "command_state": "PROVISIONAL",
                    "run_hashes": hashes.as_config(),
                },
                indent=2,
            )
        )
        return 0
    run_root = Path(str(_required(config, "tracking", "log_root")))
    if not run_root.is_absolute():
        run_root = REPO_ROOT / run_root
    observed_hashes = observe_run_hashes(
        repo_root=REPO_ROOT,
        agent_r1_path=agent_path,
        data_version=str(_required(config, "data", "version")),
    )
    recorder = LocalRunRecorder.create(
        run_root=run_root,
        config=config,
        command=[],
        hashes=observed_hashes,
        repo_root=REPO_ROOT,
    )
    print(f"local run record: {recorder.path}")
    probe_root = recorder.path / "artifacts" / "update_probe"
    rollout_seed_root = recorder.path / "artifacts" / "rollout_seed"
    step_credit_root = recorder.path / "artifacts" / "step_credit"
    update_evidence: dict[str, object] | None = None
    rollout_seed_evidence: dict[str, object] | None = None
    step_credit_evidence: dict[str, object] | None = None
    try:
        recorder.append_event("preflight_started")
        if expected_commit is None:
            raise RuntimeError("non-dry validation requires --expected-commit <40-char SHA>")
        arm_id, is_smoke, hashes, command = _prepare_upstream_launch(
            config,
            expected_arm=expected_arm,
            dry_run=False,
            update_probe_dir=probe_root,
        )
        recorder.record_provisional_command(command)
        recorder.record_evidence("run_hashes", hashes.as_config())
        print(
            json.dumps(
                {
                    "provisional_command": command,
                    "command_state": "PROVISIONAL",
                    "run_hashes": hashes.as_config(),
                },
                indent=2,
            )
        )
        evidence = _verify_all_preflight_inputs(
            config,
            command=command,
            is_smoke=is_smoke,
            expected_commit=expected_commit,
            recorder=recorder,
            config_sources=config_sources,
        )
        runtime_evidence = evidence.get("runtime")
        if not isinstance(runtime_evidence, dict):
            raise RuntimeError("preflight runtime evidence is invalid")
        model_revision = runtime_evidence.get("model_revision")
        if not isinstance(model_revision, str) or not model_revision:
            raise RuntimeError("preflight model revision is invalid")
        model_snapshot_path = runtime_evidence.get("model_snapshot_path")
        if not isinstance(model_snapshot_path, str) or not model_snapshot_path:
            raise RuntimeError("preflight model snapshot path is invalid")
        c_compiler = runtime_evidence.get("c_compiler")
        if not isinstance(c_compiler, str) or not c_compiler:
            raise RuntimeError("preflight C compiler evidence is invalid")
        command = _finalize_model_paths(
            command,
            arm_id=arm_id,
            model_snapshot_path=model_snapshot_path,
        )
        recorder.record_command(command)
        environment = _training_environment(
            agent_path=agent_path,
            probe_root=probe_root,
            c_compiler=c_compiler,
        )
        recorder.record_environment_flags(environment)
        recorder.append_event("preflight_passed", final_oos_accessed=False if is_smoke else None)
        if preflight_only:
            print("PREFLIGHT_PASSED")
            print("No training was started.")
            print("Final OOS not accessed." if is_smoke else "Final OOS access policy verified.")
            recorder.finish(
                returncode=0,
                extra={
                    "mode": "preflight_only",
                    "final_oos_accessed": False if is_smoke else None,
                    "preflight_evidence": evidence,
                    "model_revision": model_revision,
                },
            )
            return 0
        returncode = _run_with_local_capture(command, environment=environment, recorder=recorder)
        rollout_seed_evidence = summarize_rollout_seed_evidence(
            rollout_seed_root,
            expected_base_seed=_required(config, "run", "seed"),
        )
        if returncode == 0 and rollout_seed_evidence["training_request_count"] == 0:
            raise RuntimeError(
                "training exited successfully but no deterministic training request seed "
                "evidence was recorded"
            )
        if arm_id == "A3":
            step_credit_evidence = summarize_step_credit_evidence(
                step_credit_root,
                expected_mechanism=str(_required(config, "arm", "mechanism")),
                expected_output_semantics=str(
                    _required(config, "arm", "credit_semantics")
                ),
                expected_base_seed=_required(config, "run", "seed"),
            )
            if returncode == 0:
                verify_a3_smoke_credit_summary(
                    step_credit_evidence,
                    rollout_seed_evidence,
                )
        update_evidence = summarize_update_evidence(probe_root)
        if returncode == 0 and arm_id == "A3":
            verify_a3_optimizer_update_evidence(update_evidence)
        representative = update_evidence.get("representative_update")
        if isinstance(representative, dict):
            global_update_count = update_evidence.get("global_update_count")
            if not isinstance(global_update_count, int):
                raise RuntimeError("optimizer evidence global_update_count is invalid")
            recorder.append_metric(
                global_update_count,
                optimizer_update_detected=bool(update_evidence["detected"]),
                gradient_norm=float(representative.get("grad_norm") or 0.0),
                parameter_delta_l2=float(representative.get("parameter_delta_l2") or 0.0),
                parameter_delta_max=float(representative.get("parameter_delta_max") or 0.0),
            )
        if returncode == 0 and update_evidence["detected"] is not True:
            raise RuntimeError(
                "training exited successfully but no finite non-zero parameter update was detected"
            )
        recorder.finish(
            returncode=returncode,
            extra={
                "final_oos_accessed": False if is_smoke else None,
                "optimizer_update_evidence": update_evidence,
                "rollout_seed_evidence": rollout_seed_evidence,
                "step_credit_evidence": step_credit_evidence,
                "model_revision": model_revision,
            },
        )
        return returncode
    except BaseException as error:
        recorder.finish(
            returncode=None,
            error=error,
            extra={
                "final_oos_accessed": False if is_smoke else None,
                "optimizer_update_evidence": update_evidence,
                "rollout_seed_evidence": rollout_seed_evidence,
                "step_credit_evidence": step_credit_evidence,
            },
        )
        raise


def cli(*, expected_arm: str | None = None, default_config: str | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=default_config, required=default_config is None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--expected-commit")
    args = parser.parse_args()
    if args.dry_run and args.preflight_only:
        parser.error("--dry-run and --preflight-only are mutually exclusive")
    return launch(
        args.config,
        dry_run=args.dry_run,
        preflight_only=args.preflight_only,
        expected_arm=expected_arm,
        expected_commit=args.expected_commit,
    )
