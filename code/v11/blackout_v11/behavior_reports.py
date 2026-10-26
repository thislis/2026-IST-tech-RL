"""Bounded trace retention and observation-based behavior reports for one study."""
import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import re
import shutil
import tempfile
import time
import numpy as np
from .io import ROOT, atomic_json, file_hash, read_json

SCHEMA = "blackout.v11.behavior.v1"


def _episodes(study_root):
    return Path(study_root).resolve() / "behavior" / "episodes"


def _inside(path, root):
    return not path.is_symlink() and path.resolve().is_relative_to(root.resolve())


def _bytes(path):
    total = 0
    for directory, _, files in os.walk(path, followlinks=False):
        for name in files:
            try:
                total += (Path(directory) / name).lstat().st_size
            except FileNotFoundError:
                pass
    return total


def enforce_retention(study_root, max_bytes=8 * 1024**3):
    """Prune oldest CLOSED traces under this study; summaries are never removed.

    An index is written only on close, including interrupted/capped episodes.
    Active traces without an index remain untouched even if they exceed budget.
    """
    if max_bytes < 0:
        raise ValueError("trace budget must be nonnegative")
    root = _episodes(study_root)
    report = dict(max_bytes=int(max_bytes), before_bytes=0, after_bytes=0,
                  freed_bytes=0, pruned=[], errors=[], active_or_unprunable_bytes=0)
    if not root.exists():
        return dict(report, over_budget=False)
    if root.is_symlink() or not _inside(root, Path(study_root).resolve()):
        raise ValueError("behavior episodes root must remain inside its study")
    with (root.parent / ".retention.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        candidates = []
        for episode in sorted(root.iterdir()):
            trace = episode / "trace"
            if not episode.is_dir() or not _inside(episode, root) or not trace.is_dir() or not _inside(trace, episode):
                continue
            size = _bytes(trace)
            report["before_bytes"] += size
            index, summary = trace / "index.json", episode / "summary.json"
            if not index.is_file() or not summary.is_file() or index.is_symlink() or summary.is_symlink():
                report["active_or_unprunable_bytes"] += size
                continue
            try:
                row, closed = read_json(summary), read_json(index)
                if row.get("schema") != SCHEMA or not closed.get("schema", "").startswith("blackout.league.trace."):
                    raise ValueError("unrecognized behavior/trace schema")
                candidates.append((index.stat().st_mtime, episode.name, trace, summary, size, row))
            except (OSError, ValueError) as exc:
                report["errors"].append(dict(episode=episode.name, error=str(exc)))
                report["active_or_unprunable_bytes"] += size
        remaining = report["before_bytes"]
        for _, name, trace, summary, size, row in sorted(candidates):
            if remaining <= max_bytes:
                break
            retention = dict(state="pruning", reason="study_trace_budget", pruned_at=time.time(),
                             removed_bytes=size, max_bytes=int(max_bytes), trace_path=str(trace))
            try:
                atomic_json(summary, dict(row, retention=retention))
                shutil.rmtree(trace)
                retention["state"] = "pruned"
                atomic_json(summary, dict(row, trace_pruned=True, retention=retention))
                remaining -= size
                report["freed_bytes"] += size
                report["pruned"].append(name)
            except OSError as exc:
                report["errors"].append(dict(episode=name, error=str(exc)))
                # A partial deletion is disclosed instead of pretending retention succeeded.
                retention.update(state="failed", error=str(exc))
                atomic_json(summary, dict(row, retention=retention))
                remaining -= size - (_bytes(trace) if trace.exists() else 0)
        report.update(after_bytes=remaining, over_budget=remaining > max_bytes, updated_at=time.time())
        atomic_json(root.parent / "retention.json", report)
    return report


def _add(destination, source):
    for category, counts in source.items():
        target = destination.setdefault(category, {})
        for key, count in counts.items():
            if isinstance(count, (int, float)) and math.isfinite(count):
                target[key] = target.get(key, 0) + count


def _rates(categories):
    result = {}
    for category, counts in sorted(categories.items()):
        row = dict(counts)
        units = counts.get("unit_steps", 0)
        observed = counts.get("movement_observed_steps", 0)
        row["rates"] = {key: (counts.get(key, 0) / units if units else None)
                        for key in ("stop", "turn", "command_move")}
        row["rates"]["observed_move"] = counts.get("observed_move", 0) / observed if observed else None
        row["events_per_1000_observed_unit_steps"] = {
            key: counts.get(key, 0) * 1000 / observed if observed else None
            for key in ("pickup_proxy", "delivery_proxy", "respawn_proxy")}
        result[category] = row
    return result


def _labels(metadata):
    source = str(metadata.get("task_source", metadata.get("attempt_source", "")))
    arm = metadata.get("arm")
    if not arm:
        match = re.search(r"(?:^|/)([PRGFD])_s\d+(?:/|$)", source)
        arm = match.group(1) if match else "unknown"
    if metadata.get("teacher"):
        stage = "teacher"
    elif "/pilot/" in source:
        stage = "pilot"
    elif "/lifecycle/" in source:
        stage = "lifecycle"
    elif metadata.get("split"):
        stage = "evaluation:" + str(metadata["split"])
    elif "/runs/" in source:
        stage = "training"
    else:
        stage = metadata.get("kind", "unknown")
    return dict(arm=str(arm), opponent=str(metadata.get("opponent_id", "unknown")),
                side=str(metadata.get("side", "unknown")), stage=stage)


def aggregate(study_root):
    study = Path(study_root).resolve()
    root = _episodes(study)
    rows, errors = [], []
    if root.exists() and not root.is_symlink():
        for path in sorted(root.glob("*/summary.json")):
            if not _inside(path.parent, root) or not _inside(path, path.parent):
                continue
            try:
                row = read_json(path)
                if row.get("schema") != SCHEMA:
                    raise ValueError("unsupported behavior summary schema")
                rows.append((path, row))
            except (OSError, ValueError) as exc:
                errors.append(dict(path=str(path), error=str(exc)))
    probes = [(path, row) for path, row in rows if row.get("metadata", {}).get("kind") == "probe"]
    rows = [(path, row) for path, row in rows if row.get("metadata", {}).get("kind") != "probe"]
    totals = {"side0": {}, "side1": {}}
    controlled, groups = {}, {key: {} for key in ("arm", "opponent", "side", "stage")}
    coverage = dict(recorded_episodes=len(rows), complete_episodes=0, incomplete_episodes=0,
                    error_episodes=0, recorded_env_steps=0, retained_traces=0, trace_pruned=0,
                    missing_or_open_traces=0, capped_traces=0, controlled_side_unknown=0,
                    reused_episodes_without_behavior=0, teacher_forced_team_steps=0,
                    calibration_probes_excluded=len(probes),
                    calibration_probe_env_steps_excluded=sum(int(row.get("steps", 0)) for _, row in probes))
    entries, truths = [], set()
    for path, row in rows:
        meta = row.get("metadata", {})
        labels = _labels(meta)
        steps = int(row.get("steps", 0))
        complete = bool(row.get("complete"))
        coverage["complete_episodes" if complete else "incomplete_episodes"] += 1
        coverage["error_episodes"] += int(bool(row.get("error")))
        coverage["recorded_env_steps"] += steps
        coverage["teacher_forced_team_steps"] += int(row.get("teacher_forced_team_steps", 0))
        trace = path.parent / "trace"
        if row.get("trace_pruned"):
            coverage["trace_pruned"] += 1
        elif trace.is_dir() and not trace.is_symlink() and (trace / "index.json").is_file():
            coverage["retained_traces"] += 1
        else:
            coverage["missing_or_open_traces"] += 1
        coverage["capped_traces"] += int(bool(row.get("trace", {}).get("capped")))
        truths.add(row.get("league", {}).get("event_truth", "observation proxies"))
        situations = row.get("situations", {})
        for side in totals:
            _add(totals[side], situations.get(side, {}))
        selected = situations.get("side" + labels["side"], {})
        if labels["side"] not in ("0", "1"):
            coverage["controlled_side_unknown"] += 1
        else:
            _add(controlled, selected)
        for dimension, label in labels.items():
            group = groups[dimension].setdefault(label, dict(episodes=0, env_steps=0, situations={}))
            group["episodes"] += 1
            group["env_steps"] += steps
            _add(group["situations"], selected)
        entries.append(dict(episode=path.parent.name, summary_path=str(path), steps=steps,
                            complete=complete, error=row.get("error"), trace_pruned=bool(row.get("trace_pruned")),
                            **labels))
    receipt = study / "continuation_import.json"
    if receipt.exists():
        imported = read_json(receipt)
        if imported.get("behavior_history_available") is False:
            coverage["reused_episodes_without_behavior"] = int(imported.get("total_reused_games", 0))
    denominator = coverage["recorded_episodes"] + coverage["reused_episodes_without_behavior"]
    coverage["recording_fraction_of_recorded_and_reused"] = len(rows) / denominator if denominator else None
    for dimension in groups.values():
        for group in dimension.values():
            group["situations"] = _rates(group["situations"])
    return dict(schema="blackout.v11.behavior.report.v1", generated_at=time.time(), study=str(study),
                coverage=coverage, event_truth=sorted(truths), errors=errors,
                situations={side: _rates(categories) for side, categories in totals.items()},
                controlled_team_situations=_rates(controlled), groups=groups, episodes=entries,
                interpretation="Situation categories overlap; never sum their denominators across categories. "
                    "Grouped metrics refer to the controlled team only. Proxy counts are not official game events.")


def _percent(value):
    return "—" if value is None else f"{100*value:.1f}%"


def _table(categories):
    lines = ["| 상황 | 유닛 step | 정지 | 방향 변경 | 이동 관측 | 획득 | 반납 | 부활 |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for name, row in categories.items():
        if name.startswith("joint:"):
            continue  # Full joint categories remain in the JSON report.
        rates = row["rates"]
        lines.append(f"| {name} | {row.get('unit_steps', 0):,} | {_percent(rates['stop'])} | "
                     f"{_percent(rates['turn'])} | {_percent(rates['observed_move'])} | "
                     f"{row.get('pickup_proxy', 0):,} | {row.get('delivery_proxy', 0):,} | {row.get('respawn_proxy', 0):,} |")
    return lines


def markdown(report):
    c = report["coverage"]
    lines = ["# v11 행동 기록 보고서", "", f"실험: `{Path(report['study']).name}`", "",
             f"기록 {c['recorded_episodes']:,}경기 · {c['recorded_env_steps']:,} env step · "
             f"완료 {c['complete_episodes']:,} · 미완료 {c['incomplete_episodes']:,} · 오류 {c['error_episodes']:,}", "",
             f"상세 trace 보존 {c['retained_traces']:,} · 용량 정리 {c['trace_pruned']:,} · "
             f"경기별 용량 제한 {c['capped_traces']:,} · 없거나 아직 열림 {c['missing_or_open_traces']:,}", "",
             f"과거 재사용 경기 중 행동 기록이 없는 경기는 **{c['reused_episodes_without_behavior']:,}건**입니다. "
             "이 경기들을 행동 통계에 포함하지 않았습니다.", "",
             f"자원 측정용 짧은 probe **{c['calibration_probes_excluded']:,}건**도 경기·행동 통계에서 제외했습니다.", "",
             "획득·반납·부활은 관측 변화에 따른 추정입니다. 정지·방향 변경률은 유닛 step, "
             "이동률은 다음 위치를 관측한 유닛 step을 분모로 사용합니다. 상황 항목은 서로 겹칩니다.", "",
             "## 제어 팀의 상황별 행동", "", *_table(report["controlled_team_situations"])]
    for dimension, heading in (("arm", "구성"), ("opponent", "상대"), ("side", "진영"), ("stage", "단계")):
        lines += ["", "## " + heading + "별 기록", ""]
        for name, group in sorted(report["groups"][dimension].items()):
            lines += [f"### {name}", "", f"{group['episodes']:,}경기 · {group['env_steps']:,} env step", "",
                      *_table(group["situations"]), ""]
    if report["errors"]:
        lines += ["", "읽지 못한 summary: " + str(len(report["errors"])) + "개. JSON의 errors를 확인하세요."]
    return "\n".join(lines) + "\n"


def write_report(study_root):
    study = Path(study_root).resolve()
    report = aggregate(study)
    json_path, md_path = study / "behavior_report.json", study / "behavior_report.md"
    atomic_json(json_path, report)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf8", dir=study, prefix=".behavior-report-", delete=False) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(markdown(report)); stream.flush(); os.fsync(stream.fileno())
            os.replace(temporary, md_path)
        finally:
            temporary.unlink(missing_ok=True)
    return dict(json=str(json_path), markdown=str(md_path), coverage=report["coverage"])


def inspect_episode(study_root, episode_id, step=None):
    if not re.fullmatch(r"[A-Za-z0-9_-]+", episode_id):
        raise ValueError("invalid episode identifier")
    root = _episodes(study_root)
    matches = [p for p in root.glob(episode_id + "*") if p.is_dir() and _inside(p, root)]
    exact = root / episode_id
    episode = exact if exact in matches else matches[0] if len(matches) == 1 else None
    if episode is None:
        raise ValueError("episode not found or prefix is ambiguous")
    summary = read_json(episode / "summary.json")
    if step is None:
        return summary
    if step < 0:
        raise ValueError("step must be nonnegative")
    trace = episode / "trace"
    if summary.get("trace_pruned"):
        raise ValueError("trace was pruned by retention; the episode summary is still available")
    if not _inside(trace, episode) or not (trace / "index.json").is_file():
        raise ValueError("trace is missing or still open")
    index = read_json(trace / "index.json")
    chunk = next((item for item in index["chunks"] if item["first"] <= step < item["first"] + item["count"]), None)
    if chunk is None:
        raise ValueError("step was not retained (outside episode or trace byte limit)")
    path = trace / chunk["path"]
    if not _inside(path, trace) or file_hash(path) != chunk["sha256"]:
        raise ValueError("trace chunk path or checksum mismatch")
    from agent_league.trace import load_chunk
    data = load_chunk(path)
    indices = np.flatnonzero(data["step"] == step)
    if len(indices) != 1:
        raise ValueError("trace step is missing or duplicated")
    i = int(indices[0])
    def plain(value):
        if isinstance(value, np.ndarray): return plain(value.tolist())
        if isinstance(value, np.generic): return plain(value.item())
        if isinstance(value, list): return [plain(v) for v in value]
        if isinstance(value, float) and not math.isfinite(value): return None
        return value
    return dict(episode=episode.name, metadata=summary.get("metadata", {}), step=step,
                transition={key: plain(value[i]) for key, value in data.items()},
                event_truth=summary.get("league", {}).get("event_truth", "observation proxies"))


def main():
    parser = argparse.ArgumentParser(description="v11 행동 요약 보고서와 저장된 step 조회")
    parser.add_argument("--name", required=True)
    parser.add_argument("--episode", help="episode hash 또는 구별 가능한 prefix")
    parser.add_argument("--step", type=int)
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.name):
        parser.error("invalid study name")
    if args.step is not None and args.episode is None:
        parser.error("--step requires --episode")
    study = ROOT / "logs/v11" / args.name
    if not study.is_dir():
        parser.error("study does not exist")
    result = inspect_episode(study, args.episode, args.step) if args.episode else write_report(study)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
