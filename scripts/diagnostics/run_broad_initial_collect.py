#!/usr/bin/env python3
"""Start the broad Train parent batch immediately under the persistent GPU lease."""

from pathlib import Path
from types import SimpleNamespace

from run_same_prefix_online_demo import COLLECTOR, ROOT, Controller, collection, read, save

RUN = ROOT / "runs/m2-broad-rloo-order-main-20261028-v1"


class Initial(Controller):
    def __init__(self):
        config = read(RUN / "config.json")
        self.plan = config
        super().__init__(
            SimpleNamespace(
                output=RUN / "bootstrap",
                model=Path(config["source_model"]),
                collect_only=True,
                seed=43,
                episode_ids=config["train_episode_ids"],
                alpha=0.0,
                eta=0.0,
                holder_directory=None,
                stage_timeout_seconds=7200,
                coordinator_request_timeout_seconds=60,
            )
        )

    def initial(self):
        path = RUN / "stage1/initial_train"
        if not (path / "collection_summary.json").exists():
            self.command(
                "initial_train",
                COLLECTOR,
                [
                    "--experiment-config",
                    str(RUN / "config.json"),
                    "--split",
                    "train",
                    "--model",
                    self.plan["source_model"],
                    "--output",
                    str(path),
                    "--seed",
                    "43",
                    "--round-index",
                    "10",
                    "--parents-only",
                    "--parent-repetitions",
                    "4",
                ],
            )
        result = collection(
            path, Path(self.plan["source_model"]), self.plan["train_episode_ids"], 43
        )
        return path, result


def main():
    code = 1
    try:
        code = Initial().execute()
    finally:
        save(RUN / "bootstrap_completion.json", {"status": "COMPLETE" if code == 0 else "FAILED"})
    raise SystemExit(code)


if __name__ == "__main__":
    main()
