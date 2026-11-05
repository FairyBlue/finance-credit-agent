"""Explicit CPU-only storage migration for six preserved historical BF16 actors.

--check is read-only. --pack creates six lossless archives in a fresh storage
directory and proves full reconstructed shard hashes without deleting sources.
--retire revalidates every archive and every source before unlinking only the
listed safetensors. All historical summaries, receipts and metadata survive.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.agents import actor_export  # noqa: E402
from src.agents import parameter_grid_archive as archive  # noqa: E402

PROFILE = "explicit_six_historical_actor_lossless_storage_v1"
BASELINE = "runs/m2-coldstart-v3-20260914/merged_model"
BASELINE_RECEIPT = "cf3b8e5a4e2e45708943f4c71c822d41d60cfae13170e760cb5979e1df2356db"
DEFAULT = REPO / "runs/m2-action-pooling-storage-20261012-v1"
RESERVE = 8 * 1024**3
CATALOG = (
    ("comparison42_m1", "runs/m2-sft-four-round-comparison-20261004/m1", 4, ()),
    ("comparison42_m2", "runs/m2-sft-four-round-comparison-20261004/m2_eta025", 4, ()),
    (
        "comparison43_m1",
        "runs/m2-sft-four-round-comparison-seed43-20261005-recovery/m1",
        4,
        ("runs/m2-sft-four-round-comparison-seed43-20261005/m1",),
    ),
    (
        "comparison43_m2",
        "runs/m2-sft-four-round-comparison-seed43-20261005-recovery/m2_eta025",
        4,
        (),
    ),
    ("handoff42_m1", "runs/m2-low-memory-handoff-20261004/m1", 2, ()),
    ("handoff42_m2", "runs/m2-low-memory-handoff-20261004/m2_eta025", 2, ()),
)


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def safe(path):
    path = Path(path).absolute()
    require(".." not in path.parts, "parent traversal is forbidden")
    require(not any(p.is_symlink() for p in (path, *path.parents)), "symlinks are forbidden")
    return path


def read(path):
    value = json.loads(safe(path).read_text())
    require(isinstance(value, dict), "JSON object required")
    return value


def digest(path):
    return actor_export._sha(safe(path))


def now():
    return datetime.now(timezone.utc).isoformat()


def write_new(path, value):
    with safe(path).open("x") as stream:
        json.dump(value, stream, sort_keys=True, allow_nan=False, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    actor_export._fsync_directory(path.parent)


def update_owned_receipt(path, value):
    temporary = path.with_name(path.name + ".next")
    write_new(temporary, value)
    os.replace(temporary, path)
    actor_export._fsync_directory(path.parent)


def relative(path):
    return str(safe(path).relative_to(REPO))


def source_identity():
    paths = [Path(__file__).absolute()]
    paths.extend(
        REPO / "src/agents" / name
        for name in (
            "parameter_grid_archive.py",
            "actor_export.py",
            "low_memory_pilot.py",
            "training_tuning.py",
            "screening_launcher.py",
            "common_validity_penalty.py",
            "actor_update_probe.py",
            "paired_validation.py",
            "rollout_seed.py",
        )
    )
    return {str(path): digest(path) for path in paths}


def _normalized(value):
    if isinstance(value, list):
        return [_normalized(item) for item in value]
    if isinstance(value, dict):
        return {
            key: (
                {Path(p).name: h for p, h in item.items()}
                if key == "source_sha256"
                else _normalized(item)
            )
            for key, item in value.items()
        }
    return value


def _validate_evidence(cell, summary, cfg, *, handoff_cell=None):
    """Run existing CPU validators on genuine raw training and evaluation evidence."""
    from src.agents import low_memory_pilot as runner
    from src.agents import training_tuning as tuning

    coverage = cfg["training_coverage"]
    seed = cfg["seeds"][0]
    spec = tuning.CellSpec(**summary["cell"])
    profile = runner.ExperimentProfile(
        name=coverage["profile"],
        plan=cfg["plan"],
        train_numbers=tuple(int(i.rsplit("-", 1)[1]) for i in coverage["training_episode_ids"]),
        eval_numbers=tuple(int(i.rsplit("-", 1)[1]) for i in coverage["evaluation_episode_ids"]),
        evaluation_steps=tuple(coverage["evaluation_steps"]),
        rounds=spec.rounds,
        seed=seed,
        actor_learning_rate=coverage["actor_learning_rate"],
    )
    identity = summary["execution_identity_sha256"]
    actual = runner.validate_extra_evidence(cell, spec, identity, profile=profile)
    actual["critic_fit"] = tuning.validate_fit_evidence(
        cell, spec=spec, execution_identity=identity
    )
    actual["placement"] = runner.validate_placement(cell, spec, identity)
    # Imported seed43 M1 ran under its preserved original holder contract.
    actual["gpu_handoff"] = runner.validate_handoff(handoff_cell or cell, identity)
    for key, result in actual.items():
        if key == "paired_validation":
            # Old accepted summaries precede incomplete-evaluation reporting.
            # The current validator adds these fields even in strict mode.
            added = {
                "execution_status": "PASS",
                "performance_status": "COMPLETE",
                "complete_trajectories": result["trajectories"],
                "incomplete_trajectories": 0,
                "incomplete_identities": [],
            }
            for field, value in added.items():
                require(result.get(field) == value, "historical paired population is incomplete")
            result = {k: v for k, v in result.items() if k not in added}
        require(
            _normalized(result) == _normalized(summary.get(key)),
            "raw training evidence differs from historical summary: " + key,
        )
    records = runner.read_jsonl(cell / "artifacts/screening_evidence")
    expected = {
        (step, index, sibling, False): profile.train_ids[index]
        for step in range(1, spec.rounds + 1)
        for index in (((step - 1) * 2 + k) % len(profile.train_ids) for k in (0, 1))
        for sibling in range(4)
    }
    expected.update(
        {
            (step, index, sibling, True): episode
            for step in profile.evaluation_steps
            for index, episode in enumerate(profile.eval_ids)
            for sibling in range(5)
        }
    )
    observed = set()
    calibration = read(cell.parent / "prepared/manifest.json")["calibration"]["payload_sha256"]
    for row in records:
        identity_row = row.get("rollout_identity", row.get("identity", {}))
        key = tuple(identity_row.get(k) for k in ("step", "sample_index", "rollout_n", "validate"))
        score = row.get("terminal_score")
        require(
            key in expected
            and key not in observed
            and row.get("episode_id") == expected[key]
            and row.get("base_seed") == seed
            and row.get("arm") == spec.arm
            and row.get("complete", row.get("episode_complete")) is True
            and type(score) in (int, float)
            and math.isfinite(score)
            and row.get("metrics", {}).get("terminal_score") == score
            and row.get("metrics", {}).get("calibration_sha256") == calibration,
            "historical raw trajectory grid, completion or finance differs",
        )
        observed.add(key)
    require(observed == set(expected), "historical raw trajectory population is incomplete")
    from src.agents.screening_launcher import summarize_update_evidence

    optimizer = summarize_update_evidence(cell / "artifacts/update_probe")
    require(
        optimizer == summary["evidence"]["optimizer_evidence"],
        "historical optimizer raw evidence differs",
    )
    counts = dict(
        train=spec.rounds * 8, eval=len(profile.evaluation_steps) * len(profile.eval_ids) * 5
    )
    require(summary["evidence"]["trajectories"] == counts, "historical trajectory count differs")
    return dict(
        status="PASS",
        raw_trajectories=len(records),
        trajectories=counts,
        actor_updates=spec.rounds,
        critic_updates=spec.rounds,
    )


def _worker_evidence(cell, identity):
    owners = sorted((cell / "artifacts/placement/moment_scratch").glob("*/*/owner.json"))
    require(
        len(owners) == 2 and {p.parent.name for p in owners} == {"actor", "critic"},
        "both historical worker ownership manifests are required",
    )
    pids = set()
    for path in owners:
        owner = read(path)
        pid = path.parent.parent.name
        require(
            pid.isdecimal()
            and owner.get("identity") == identity
            and owner.get("role") == path.parent.name
            and bool(owner.get("owner")),
            "historical worker ownership differs",
        )
        pids.add(int(pid))
    for folder in ("actor_update", "critic_fit", "update_probe", "paired_validation", "placement"):
        for path in (cell / "artifacts" / folder).glob("*.jsonl"):
            for line in safe(path).read_text().splitlines():
                row = json.loads(line)
                pid = row.get("pid")
                if type(pid) is int and pid > 0:
                    pids.add(pid)
    require(
        not any((Path("/proc") / str(pid)).exists() for pid in pids),
        "a historical worker PID is active; sources must be preserved",
    )
    return sorted(pids)


def _small_inputs(cell):
    paths = [
        cell / name
        for name in ("summary.json", "settings.json", "resolved_config.json", "command.json")
    ]
    paths.extend(
        cell.parent / name
        for name in (
            "state.json",
            "settings.json",
            "run_contract.json",
            "prepared/manifest.json",
            "gpu_handoff_config.json",
            "model_identity.json",
        )
    )
    paths.extend((cell / "artifacts").glob("*/*.jsonl"))
    paths.extend((cell / "artifacts/placement/moment_scratch").glob("*/*/owner.json"))
    paths.extend((cell / "artifacts/gpu_handoff").glob("*.json"))
    paths.extend((cell / "artifacts/critic_initialization").glob("*.json"))
    return {relative(p): digest(p) for p in sorted(set(paths)) if p.is_file()}


def inspect(*, verify_hashes=False):
    baseline = safe(REPO / BASELINE)
    require(
        digest(baseline / "merge_receipt.json") == BASELINE_RECEIPT,
        "original common SFT receipt changed",
    )
    baseline_tensors, _ = archive._model_layout(baseline)
    baseline_config = read(baseline / "config.json")
    groups, unique = [], set()
    for name, cell_relative, step, aliases in CATALOG:
        cell = safe(REPO / cell_relative)
        summary, cfg = read(cell / "summary.json"), read(cell / "settings.json")
        identity = summary.get("execution_identity_sha256")
        require(
            summary.get("status") == "COMPLETE"
            and summary.get("returncode") == 0
            and summary.get("actual_rounds") == step
            and summary.get("cell", {}).get("rounds") == step
            and summary.get("cell", {}).get("cell_id") == cell.name
            and read(cell.parent / "state.json").get("status") == "COMPLETE",
            "only the six accepted historical cells may migrate",
        )
        model = cfg["models"]["screening"]
        require(
            model.get("source") == "local_export"
            and safe(model.get("name", "")) == baseline
            and model.get("merge_receipt_sha256") == BASELINE_RECEIPT,
            "historical actor used a different common SFT",
        )
        evidence = _validate_evidence(
            cell, summary, cfg, handoff_cell=REPO / aliases[0] if aliases else None
        )
        worker_pids = _worker_evidence(cell, identity)
        checkpoint = cell / "checkpoints" / f"global_step_{step}"
        actor_export.validate_export_artifact(
            checkpoint, identity_sha256=identity, verify_hashes=verify_hashes
        )
        hf = checkpoint / "actor/huggingface"
        receipt = read(hf.parent / actor_export.ACTOR_RECEIPT)
        target_tensors, _ = archive._model_layout(hf)
        archive._aligned_baseline_tensors(
            baseline_tensors, target_tensors, baseline_config, read(hf / "config.json")
        )
        inputs = _small_inputs(cell)
        sources = [cell]
        for alias_relative in aliases:
            alias = safe(REPO / alias_relative)
            old = read(alias / "summary.json")
            require(
                old.get("status") == "FAILED"
                and old.get("returncode") == 0
                and old.get("execution_identity_sha256") == identity
                and summary.get("retrained") is False
                and safe(summary.get("imported_from", "")) == alias
                and summary.get("original_summary_sha256") == digest(alias / "summary.json"),
                "the explicit failed/imported alias ownership differs",
            )
            alias_checkpoint = alias / "checkpoints" / f"global_step_{step}"
            actor_export.validate_export_artifact(
                alias_checkpoint, identity_sha256=identity, verify_hashes=False
            )
            require(
                read(alias_checkpoint / "actor/actor_export_receipt.json") == receipt,
                "hardlink family export receipts differ",
            )
            inputs.update(_small_inputs(alias))
            sources.append(alias)
        files = []
        for filename in sorted(receipt["hf"]["files"]):
            source = safe(hf / filename)
            if not filename.endswith(".safetensors"):
                for owned_cell in sources:
                    other = (
                        owned_cell
                        / "checkpoints"
                        / f"global_step_{step}"
                        / "actor/huggingface"
                        / filename
                    )
                    require(
                        digest(other) == receipt["hf"]["sha256"][filename], "HF metadata differs"
                    )
                    inputs[relative(other)] = digest(other)
                continue
            stat = source.stat()
            family = [
                safe(owned / "checkpoints" / f"global_step_{step}" / "actor/huggingface" / filename)
                for owned in sources
            ]
            inode = (stat.st_dev, stat.st_ino)
            require(inode not in unique, "different model families share an unexpected inode")
            unique.add(inode)
            for other in family:
                actual = other.stat()
                require(
                    other.is_file()
                    and actual.st_dev == stat.st_dev
                    and actual.st_ino == stat.st_ino
                    and actual.st_size == receipt["hf"]["files"][filename]
                    and actual.st_nlink == len(family),
                    "foreign or missing hardlink detected",
                )
            files.append(
                dict(
                    paths=[relative(p) for p in family],
                    bytes=stat.st_size,
                    sha256=receipt["hf"]["sha256"][filename],
                    device=stat.st_dev,
                    inode=stat.st_ino,
                    links=stat.st_nlink,
                    mtime_ns=stat.st_mtime_ns,
                    allocated_bytes=stat.st_blocks * 512,
                )
            )
        for owned in sources:
            cp = owned / "checkpoints" / f"global_step_{step}"
            for p in (cp / "actor/actor_export_receipt.json", cp / "actor_export_complete.json"):
                inputs[relative(p)] = digest(p)
        groups.append(
            dict(
                name=name,
                accepted_cell=cell_relative,
                aliases=list(aliases),
                identity_sha256=identity,
                policy_step=step,
                worker_pids=worker_pids,
                evidence=evidence,
                hf=relative(hf),
                files=files,
                source_sha256=inputs,
            )
        )
    return dict(
        profile=PROFILE,
        baseline=BASELINE,
        baseline_receipt_sha256=BASELINE_RECEIPT,
        groups=groups,
        unique_shards=sum(len(g["files"]) for g in groups),
        visible_shard_paths=sum(len(f["paths"]) for g in groups for f in g["files"]),
        unique_bytes=sum(f["bytes"] for g in groups for f in g["files"]),
        unique_allocated_bytes=sum(f["allocated_bytes"] for g in groups for f in g["files"]),
        historical_summaries_modified=False,
        optimizer_scratch_touched=False,
    )


def output_directory(path):
    path = safe(path)
    require(path == safe(DEFAULT), "storage output must be the fixed new stage2 storage directory")
    return path


def pack(directory):
    directory = output_directory(directory)
    require(not directory.exists(), "fresh storage directory required; no overwrite or retry")
    population = inspect(verify_hashes=True)
    directory.mkdir(parents=True, exist_ok=False)
    manifest = dict(
        profile=PROFILE,
        created_at_UTC=now(),
        population=population,
        runtime_source_sha256=source_identity(),
        automatic_retry=False,
        deletion_authorized_by_pack=False,
    )
    write_new(directory / "manifest.json", manifest)
    results = []
    for group in population["groups"]:
        maximum_pack_bytes = sum(f["bytes"] for f in group["files"]) + 16 * 1024**2
        require(
            shutil.disk_usage(directory).free >= maximum_pack_bytes + RESERVE,
            "insufficient real space for worst-case archive coexistence and reserve",
        )
        packed = directory / "archives" / group["name"]
        result = archive.pack_model(REPO / group["hf"], REPO / BASELINE, packed)
        proof = archive.validate_archive(packed)  # Full streamed reconstruction; original file SHA.
        results.append(
            dict(
                name=group["name"],
                result=result,
                proof=proof,
                archive_manifest_sha256=digest(packed / "manifest.json"),
                verification_sha256=digest(packed / "verification.json"),
            )
        )
        write_new(directory / (group["name"] + "_pack_proof.json"), results[-1])
    require(
        inspect(verify_hashes=False) == population,
        "historical source metadata changed while packing",
    )
    proof = dict(
        profile=PROFILE,
        status="PACK_VERIFIED",
        finished_at_UTC=now(),
        manifest_sha256=digest(directory / "manifest.json"),
        groups=results,
        exact_reconstructed_original_shard_hashes=True,
        source_shards_deleted=0,
        unique_source_bytes=population["unique_bytes"],
        archive_bytes=sum(r["result"]["packed_bytes"] for r in results),
    )
    write_new(directory / "pack_proof.json", proof)
    return proof


def retire(directory):
    directory = output_directory(directory)
    receipt_path = directory / "retirement_receipt.json"
    require(not receipt_path.exists(), "retirement already attempted; no overwrite or retry")
    manifest, proof = read(directory / "manifest.json"), read(directory / "pack_proof.json")
    require(
        manifest.get("profile") == PROFILE
        and proof.get("status") == "PACK_VERIFIED"
        and proof.get("manifest_sha256") == digest(directory / "manifest.json")
        and proof.get("exact_reconstructed_original_shard_hashes") is True
        and manifest.get("runtime_source_sha256") == source_identity(),
        "sealed storage proof or runtime source differs",
    )
    population = inspect(verify_hashes=True)
    require(population == manifest.get("population"), "historical population changed after packing")
    require(
        [r.get("name") for r in proof.get("groups", [])]
        == [g["name"] for g in population["groups"]],
        "archive proof population differs",
    )
    for group, entry in zip(population["groups"], proof["groups"], strict=True):
        packed = directory / "archives" / group["name"]
        require(
            read(directory / (group["name"] + "_pack_proof.json")) == entry
            and digest(packed / "manifest.json") == entry["archive_manifest_sha256"]
            and digest(packed / "verification.json") == entry["verification_sha256"]
            and archive.validate_archive(packed) == entry["proof"],
            "lossless archive proof changed",
        )
        archived = read(packed / "manifest.json")
        source_receipt = read((REPO / group["hf"]).parent / actor_export.ACTOR_RECEIPT)
        require(
            archived["original_hf"] == source_receipt["hf"],
            "archive HF identity differs from the historical export receipt",
        )
        expected = {Path(f["paths"][0]).name: (f["bytes"], f["sha256"]) for f in group["files"]}
        actual = {
            name: (row["original_bytes"], row["original_sha256"])
            for name, row in archived["shards"].items()
        }
        require(
            actual == expected and archived["baseline"]["directory"] == str(REPO / BASELINE),
            "archive does not reconstruct this exact historical actor",
        )
    # Archive validation can take time. Recheck all workers, source metadata and
    # visible inode/link ownership together immediately before the first unlink.
    require(
        source_identity() == manifest["runtime_source_sha256"]
        and inspect(verify_hashes=False) == population,
        "runtime, historical workers or source ownership changed during archive verification",
    )
    # All groups pass before the first destructive operation; preserve a complete
    # inventory and per-path progress in the new storage directory only.
    receipt = dict(
        profile=PROFILE,
        status="PREPARED",
        started_at_UTC=now(),
        manifest_sha256=digest(directory / "manifest.json"),
        pack_proof_sha256=digest(directory / "pack_proof.json"),
        unique_bytes=population["unique_bytes"],
        unique_shards=population["unique_shards"],
        planned_visible_paths=population["visible_shard_paths"],
        files=[f for g in population["groups"] for f in g["files"]],
        deleted_paths=[],
        deleted_unique_bytes=0,
        deleted_unique_shards=0,
        original_metadata_preserved=True,
        summaries_and_lineage_preserved=True,
        source_raw_moments_touched=False,
        automatic_retry=False,
    )
    write_new(receipt_path, receipt)
    try:
        for file in receipt["files"]:
            for index, rel in enumerate(file["paths"]):
                path = safe(REPO / rel)
                stat = path.stat()
                require(
                    (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_nlink)
                    == (
                        file["device"],
                        file["inode"],
                        file["bytes"],
                        file["mtime_ns"],
                        len(file["paths"]) - index,
                    ),
                    "source ownership changed before unlink",
                )
                path.unlink()
                actor_export._fsync_directory(path.parent)
                receipt["deleted_paths"].append(rel)
                if index == len(file["paths"]) - 1:
                    receipt["deleted_unique_bytes"] += file["bytes"]
                    receipt["deleted_unique_shards"] += 1
                update_owned_receipt(receipt_path, receipt)
        require(
            all(not (REPO / rel).exists() for f in receipt["files"] for rel in f["paths"]),
            "retired source reappeared",
        )
        receipt.update(
            status="COMPLETE",
            finished_at_UTC=now(),
            actual_free_bytes_after_unlink=shutil.disk_usage(directory).free,
            filesystem_reclaim_may_be_asynchronous=True,
        )
    except BaseException as exc:
        receipt.update(status="FAILED", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        update_owned_receipt(receipt_path, receipt)
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    choice = parser.add_mutually_exclusive_group(required=True)
    for mode in ("check", "pack", "retire"):
        choice.add_argument("--" + mode, action="store_true")
    parser.add_argument("--directory", type=Path, default=DEFAULT)
    args = parser.parse_args(argv)
    result = (
        inspect() if args.check else pack(args.directory) if args.pack else retire(args.directory)
    )
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
