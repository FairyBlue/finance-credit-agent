#!/usr/bin/env python3
"""Generate a deterministic, human-readable GitHub change broadcast."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPORT_MARKER = "<!-- finance-credit-agent-change-report -->"
ZERO_SHA = "0" * 40


@dataclass(frozen=True)
class ChangedFile:
    status: str
    path: str
    display_path: str
    category: str


def _git(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        check=check,
        capture_output=True,
        text=True,
    )


def _load_event(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    event_path = Path(path)
    if not event_path.is_file():
        return {}
    payload = json.loads(event_path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _event_value(payload: dict[str, Any], *keys: str) -> Any:
    value: Any = payload
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def _resolve_base(base: str, head: str) -> str:
    if base and base != ZERO_SHA:
        completed = _git("rev-parse", "--verify", f"{base}^{{commit}}", check=False)
        if completed.returncode == 0:
            return base
    parent = _git("rev-parse", "--verify", f"{head}^{{commit}}", check=False)
    if parent.returncode != 0:
        raise RuntimeError("unable to resolve a base commit for the change report")
    return parent.stdout.strip()


def _diff_spec(mode: str, base: str, head: str) -> str:
    return f"{base}...{head}" if mode == "pr" else f"{base}..{head}"


def _category(path: str) -> str:
    if path in {"pyproject.toml", "uv.lock"}:
        return "依赖"
    if path.startswith("configs/"):
        return "实验配置"
    if path.startswith("src/data/") or path.startswith("scripts/prepare_"):
        return "数据规则"
    if path.startswith("src/rewards/") or path.startswith("scripts/calibrate_reward"):
        return "奖励与 credit"
    if path.startswith("tests/"):
        return "测试"
    if path.startswith(".github/"):
        return "协作自动化"
    if path.startswith("docs/") or path.endswith(".md"):
        return "文档"
    if path.startswith("src/"):
        return "代码"
    if path.startswith("scripts/"):
        return "工具脚本"
    if path.startswith("third_party/") or path == ".gitmodules":
        return "上游依赖"
    return "其他"


def _changed_files(spec: str) -> list[ChangedFile]:
    output = _git("diff", "--name-status", "-M", spec).stdout
    changed: list[ChangedFile] = []
    for line in output.splitlines():
        fields = line.split("\t")
        if len(fields) < 2:
            continue
        status = fields[0]
        path = fields[-1]
        display_path = " → ".join(fields[1:]) if status.startswith(("R", "C")) else path
        changed.append(
            ChangedFile(
                status=status,
                path=path,
                display_path=display_path,
                category=_category(path),
            )
        )
    return changed


def _clean_manual_text(value: str | None) -> str | None:
    if not value:
        return None
    without_comments = re.sub(r"<!--.*?-->", " ", value, flags=re.DOTALL)
    cleaned = " ".join(without_comments.split())
    return cleaned or None


def _markdown_section(body: str, heading: str) -> str | None:
    pattern = re.compile(rf"(?ims)^##\s+{re.escape(heading)}\s*$\n(.*?)(?=^##\s+|\Z)")
    match = pattern.search(body)
    return _clean_manual_text(match.group(1)) if match else None


def _labelled_line(body: str, label: str) -> str | None:
    pattern = re.compile(rf"(?im)^{re.escape(label)}\s*[:：]\s*(.+)$")
    match = pattern.search(body)
    return _clean_manual_text(match.group(1)) if match else None


def _manual_fields(payload: dict[str, Any], head: str) -> dict[str, str]:
    pr_body = _event_value(payload, "pull_request", "body")
    body = pr_body if isinstance(pr_body, str) else ""
    commit_body = _git("log", "-1", "--format=%B", head).stdout
    result: dict[str, str] = {}
    for heading in ("目的", "影响", "下一步"):
        value = _markdown_section(body, heading) or _labelled_line(commit_body, heading)
        result[heading] = value or "未填写（请在 PR 或提交说明中补充）"
    return result


def _impact_lines(files: list[ChangedFile]) -> list[str]:
    paths = {item.path for item in files}
    categories = {item.category for item in files}
    lines: list[str] = []
    if any(path.startswith("src/data/") or path.startswith("configs/episode_") for path in paths):
        lines.append("触及数据或 episode 边界，需要复核数据泄漏和数据版本。")
    if any(path.startswith("src/rewards/") or "reward" in path for path in paths):
        lines.append("触及 reward/credit，需要复核冻结奖励协议和验收证据。")
    if "实验配置" in categories:
        lines.append("触及实验配置，后续运行必须绑定新的 commit。")
    if categories & {"代码", "依赖", "上游依赖", "工具脚本"}:
        lines.append("包含可能影响执行结果的实现或依赖，合并前需要查看完整 diff。")
    if categories <= {"文档", "协作自动化", "测试"}:
        lines.append("未检测到训练代码、实验配置、数据规则或依赖改动。")
    return lines or ["未命中已知实验风险路径；仍需由维护者查看完整 diff。"]


def _escape_table(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def _repository_url(payload: dict[str, Any]) -> str | None:
    html_url = _event_value(payload, "repository", "html_url")
    if isinstance(html_url, str) and html_url:
        return html_url.rstrip("/")
    repository = os.environ.get("GITHUB_REPOSITORY")
    server = os.environ.get("GITHUB_SERVER_URL", "https://github.com").rstrip("/")
    return f"{server}/{repository}" if repository else None


def _report_link(
    *, payload: dict[str, Any], repo_url: str | None, mode: str, base: str, head: str
) -> str:
    pr_url = _event_value(payload, "pull_request", "html_url")
    if mode == "pr" and isinstance(pr_url, str) and pr_url:
        return f"{pr_url}/files"
    if repo_url:
        return f"{repo_url}/compare/{base}...{head}"
    return ""


def generate_report(*, mode: str, base: str, head: str, payload: dict[str, Any]) -> str:
    resolved_base = _resolve_base(base, head)
    spec = _diff_spec(mode, resolved_base, head)
    whitespace = _git("diff", "--check", spec, check=False)
    if whitespace.returncode != 0:
        details = whitespace.stdout.strip() or whitespace.stderr.strip()
        raise RuntimeError(f"git diff --check failed:\n{details}")

    files = _changed_files(spec)
    categories = sorted({item.category for item in files}) or ["无文件变更"]
    manual = _manual_fields(payload, head)
    repo_url = _repository_url(payload)
    diff_url = _report_link(
        payload=payload,
        repo_url=repo_url,
        mode=mode,
        base=resolved_base,
        head=head,
    )
    actor = _event_value(payload, "sender", "login") or os.environ.get("GITHUB_ACTOR") or "未知"
    branch = (
        _event_value(payload, "pull_request", "head", "ref")
        or _event_value(payload, "ref")
        or os.environ.get("GITHUB_REF_NAME")
        or "未知"
    )
    title = _event_value(payload, "pull_request", "title") or _event_value(
        payload, "head_commit", "message"
    )
    title_text = str(title).splitlines()[0] if title else "未提供标题"
    shortstat = _git("diff", "--shortstat", spec).stdout.strip() or "0 files changed"
    commits = _git("log", "--format=%h\t%an\t%s", f"{resolved_base}..{head}").stdout
    checks_url = f"{repo_url}/commit/{head}/checks" if repo_url else ""
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    lines = [
        REPORT_MARKER,
        "## 自动变更播报",
        "",
        "> 文件差异由 Git 自动整理；目的、影响和下一步来自 PR 或提交说明。",
        "",
        "| 字段 | 内容 |",
        "| --- | --- |",
        f"| 修改人 | `{_escape_table(str(actor))}` |",
        f"| 时间 | `{generated_at}` |",
        f"| 标题 | {_escape_table(title_text)} |",
        f"| 分支 | `{_escape_table(str(branch))}` |",
        f"| 比较范围 | `{resolved_base[:12]}..{head[:12]}` |",
        f"| 规模 | {_escape_table(shortstat)} |",
        f"| 类型 | {', '.join(f'`{item}`' for item in categories)} |",
        "",
        "### 变更说明",
        "",
        f"- **目的：** {manual['目的']}",
        f"- **影响：** {manual['影响']}",
        f"- **下一步：** {manual['下一步']}",
        "",
        "### 自动影响提示",
        "",
    ]
    lines.extend(f"- {line}" for line in _impact_lines(files))
    lines.extend(["", f"### 改动文件（{len(files)}）", ""])
    if files:
        lines.extend(
            f"- `{item.status}` `{item.display_path}` — {item.category}" for item in files[:100]
        )
        if len(files) > 100:
            lines.append(f"- 其余 {len(files) - 100} 个文件请查看完整 diff。")
    else:
        lines.append("- 没有检测到文件变化。")

    lines.extend(["", "### 提交记录", ""])
    commit_lines = [line for line in commits.splitlines() if line.strip()]
    lines.extend(f"- `{line}`" for line in commit_lines[:50])
    if len(commit_lines) > 50:
        lines.append(f"- 其余 {len(commit_lines) - 50} 个提交请查看完整记录。")
    if not commit_lines:
        lines.append("- 没有检测到新提交。")

    lines.extend(
        [
            "",
            "### 自动验证",
            "",
            "- `git diff --check`：`PASSED`",
            "- 变更报告生成：`PASSED`",
            "- 项目测试和其他检查："
            + (f"[查看 GitHub Checks]({checks_url})" if checks_url else "请查看对应提交的 Checks"),
        ]
    )
    if diff_url:
        lines.extend(["", f"[查看完整改动]({diff_url})"])
    return "\n".join(lines) + "\n"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("pr", "push"), required=True)
    parser.add_argument("--base", required=True)
    parser.add_argument("--head", required=True)
    parser.add_argument("--event-path", default=os.environ.get("GITHUB_EVENT_PATH"))
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    payload = _load_event(args.event_path)
    report = generate_report(
        mode=args.mode,
        base=args.base,
        head=args.head,
        payload=payload,
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report, encoding="utf-8")
    else:
        print(report, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
