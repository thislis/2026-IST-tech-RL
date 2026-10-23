"""Build the offline, self-contained analysis of the completed 21-agent league.

No Unity launch, inference, training, or modification of original league results.
Run: .venv/bin/python code/run.py code/shared/tools/build_league_analysis.py
"""
from __future__ import annotations

import argparse
import base64
from collections import Counter, defaultdict
from datetime import datetime
import hashlib
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image

from agent_league.trace import load_chunk
from project_paths import project_root


ROOT = project_root()
DEFAULT_RUN = ROOT / "logs/shared/agent_league/version_best_ties_league"
NAMES = {
    "scripted-collector": "수집형", "scripted-default": "기본 FSM",
    "scripted-defender": "방어형", "scripted-noop": "정지",
    "scripted-raider": "침투형", "scripted-random": "무작위",
    "scripted-rush": "돌격형", "scripted-special": "특수 아이템",
    "scripted-weak": "확장 추격 FSM",
    "v1-d4c8e00e": "v1", "v2-80375a72": "v2", "v3-80375a72": "v3",
    "v4-29794421": "v4", "v5-9c6b5f28": "v5", "v6-5c651487": "v6",
    "v7-residual_graph-23262e36b5": "v7 Rewired",
    "v7-residual_graph-27d97f725e": "v7 FlyWire",
    "v8-v8_comp-c1": "v8 C1", "v8-v8_comp-flat": "v8 Flat",
    "v9-B": "v9 B", "v10-C1": "v10 C1",
}
TOP = {"v4-29794421", "v5-9c6b5f28", "v6-5c651487",
       "v7-residual_graph-23262e36b5", "v7-residual_graph-27d97f725e"}
METRICS = ["stop_fraction", "same_action_fraction", "turn_fraction", "team_spread",
           "moving_entity_fraction", "near_base_fraction", "pickup_proxy",
           "delivery_proxy", "teleport_proxy"]
DESCRIPTIONS = {
    "scripted-collector": "배터리 수집·귀환을 수행하는 BFS 규칙 정책. 팀 행동 동기화와 정지 비중이 높다.",
    "scripted-default": "3 worker + guard + carrier, 추격 반경 5의 상태 기반 규칙 정책.",
    "scripted-defender": "일부 유닛에 방어 목표를 주는 BFS 정책. 수집형보다 승점률이 낮았다.",
    "scripted-noop": "항상 (0, 0)을 출력하는 대조군. 41무이며 승리는 없다.",
    "scripted-raider": "빈손일 때 적 창고를 목표로 하는 정책. 행동 집중과 정지 비중이 높다.",
    "scripted-random": "각 유닛이 독립적인 연속 무작위 방향을 출력. 학습 정책의 최소 기준선.",
    "scripted-rush": "클래스별 중앙 집결·적 유닛 추적. 반납 없이도 raw reward 판정에서 많은 승리를 얻었다.",
    "scripted-special": "기본 FSM에 특수 아이템 사용을 켠 정책. 기본 FSM과의 맞대결은 7승 3패.",
    "scripted-weak": "ID는 weak지만 추격 반경 12의 FSM이다. 이번 리그에서는 규칙 기반 중 1위.",
    "v1-d4c8e00e": "신경망 직접 행동 정책. 움직임 관측 비율이 낮고 배터리 반납이 드물다.",
    "v2-80375a72": "직접 행동 정책. 초기 버전 중 승점률은 높지만 반납 횟수는 매우 낮다.",
    "v3-80375a72": "교사·커리큘럼 학습 계열. 정지와 큰 방향 전환이 모두 잦다.",
    "v4-29794421": "3 worker + 2 guard의 planner에 신경망 방향 보정을 결합. 공동 1위.",
    "v5-9c6b5f28": "planner 상대 방향 보정, step당 최대 한 유닛 override. 공동 4위.",
    "v6-5c651487": "planner 상태·이력에 기반한 공동 잔차 행동 선택. 공동 1위.",
    "v7-residual_graph-23262e36b5": "재배선 그래프 잔차 정책 + v6 방식 planner. 공동 1위.",
    "v7-residual_graph-27d97f725e": "FlyWire 그래프 잔차 정책 + v6 방식 planner. 공동 4위.",
    "v8-v8_comp-c1": "두 tensor 입력만 사용하는 결정론적 gate 정책. 팀 행동이 거의 완전히 동기화된다.",
    "v8-v8_comp-flat": "두 tensor 입력만 사용하는 결정론적 flat 정책. 남동 방향 출력이 지배적이다.",
    "v9-B": "두 tensor 입력의 attention 정책. 독립 확률 행동으로 방향 다양성은 있으나 반납이 적다.",
    "v10-C1": "두 tensor 입력의 map-aware critic 학습 계열 actor. 이번 참가 제출형 중 1위.",
}


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def group(mid):
    return ("scripted" if mid.startswith("scripted-") else
            "planner" if mid in TOP else
            "submission" if mid.startswith(("v8-", "v9-", "v10-")) else "early")


def mean(values):
    return float(np.mean(values)) if values else None


def summarize(records, model):
    items = []
    for rec in records:
        t, r = rec["task"], rec["result"]
        if model not in (t["a"], t["b"]):
            continue
        side = t["side"] if model == t["a"] else 1-t["side"]
        opp = t["b"] if model == t["a"] else t["a"]
        items.append((rec, side, opp, .5 if r["winner"] is None else float(r["winner"] == model)))
    counts = Counter(x[3] for x in items)
    st = dict(n=len(items), wins=counts[1.0], draws=counts[.5], losses=counts[0.0])
    st["win_rate"] = st["wins"]/st["n"]
    st["score_rate"] = mean([x[3] for x in items])
    st["side_score_rates"] = [mean([x[3] for x in items if x[1] == s]) for s in (0, 1)]
    st["round_rates"] = [mean([x[3] for x in items if x[0]["task"]["repeat"] == k]) for k in range(5)]
    st["vs_groups"] = {g: mean([x[3] for x in items if group(x[2]) == g])
                       for g in ("scripted", "planner", "submission", "early")}
    st["behavior"] = {k: mean([x[0]["result"]["behavior"]["sides"][x[1]][k] for x in items])
                      for k in METRICS}
    h = np.sum([x[0]["result"]["behavior"]["sides"][x[1]]["action_histogram"] for x in items], axis=0)
    st["action_histogram"] = h.tolist()
    st["action_rates"] = (h/h.sum()).tolist()
    st["step_weighted_stop"] = float(h[0]/h.sum())
    steps = sum(x[0]["result"]["episode_steps"] for x in items)
    for key in ("pickup_proxy", "delivery_proxy"):
        st[key+"_per_1000"] = sum(x[0]["result"]["behavior"]["sides"][x[1]][key] for x in items)/steps*1000
    delta = [x[0]["result"]["side_rewards"][x[1]]-x[0]["result"]["side_rewards"][1-x[1]] for x in items]
    st["mean_reward_delta"], st["median_reward_delta"] = mean(delta), float(np.median(delta))
    st["reward_delta_quantiles"] = np.percentile(delta, [0, 25, 50, 75, 100]).tolist()
    st["median_steps"] = float(np.median([x[0]["result"]["episode_steps"] for x in items]))
    st["mean_last_visible_score"] = mean([x[0]["result"]["behavior"]["timeline"][-1]["visible_score"][x[1]] for x in items])
    st["apparent_score_rate"] = mean([
        .5 if abs((scores := x[0]["result"]["behavior"]["timeline"][-1]["visible_score"])[0]-scores[1]) < 1e-7
        else float(scores[x[1]] > scores[1-x[1]]) for x in items])
    return st


def pair_summary(records, a, b):
    rows = [r for r in records if {r["task"]["a"], r["task"]["b"]} == {a, b}]
    rows.sort(key=lambda r: (r["task"]["repeat"], r["task"]["side"]))
    a_side = lambda r: r["task"]["side"] if r["task"]["a"] == a else 1-r["task"]["side"]
    score = lambda r: .5 if r["result"]["winner"] is None else float(r["result"]["winner"] == a)
    return dict(a=a, b=b, n=len(rows), wins=sum(r["result"]["winner"] == a for r in rows),
                draws=sum(r["result"]["winner"] is None for r in rows),
                losses=sum(r["result"]["winner"] == b for r in rows),
                rate=mean([score(r) for r in rows]),
                sides=[mean([score(r) for r in rows if a_side(r) == s]) for s in (0, 1)],
                rounds=[mean([score(r) for r in rows if r["task"]["repeat"] == k]) for k in range(5)],
                matches=[dict(id=r["id"], repeat=r["task"]["repeat"], side=a_side(r),
                              winner=r["result"]["winner"], rewards=r["result"]["side_rewards"],
                              steps=r["result"]["episode_steps"]) for r in rows])


PALETTE = np.array([
    [243, 245, 242], [40, 54, 62], [38, 116, 224], [231, 92, 75],
    [65, 148, 246], [252, 130, 112], [246, 197, 65], [103, 203, 151],
    [166, 135, 219], [43, 172, 165], [225, 138, 191]], dtype=np.uint8)


def png(graphic):
    pixels = PALETTE[graphic.argmax(-1)]
    im = Image.fromarray(pixels).resize((384, 384), Image.Resampling.NEAREST)
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return "data:image/png;base64,"+base64.b64encode(buf.getvalue()).decode()


def case_data(record, title, why, selection):
    t, r = record["task"], record["result"]
    trace = Path(record["directory"])/"trace"
    idx = read(trace/"index.json")
    frames = []
    for snap in idx["snapshots"]:
        step = snap["step"]
        terminal_frame = step == r["episode_steps"]
        action_step = step-1 if terminal_frame else step
        entry = next(c for c in idx["chunks"] if c["first"] <= action_step < c["first"]+c["count"])
        data = load_chunk(trace/entry["path"])
        j = action_step-entry["first"]
        with np.load(trace/snap["path"], allow_pickle=False) as f:
            graphic = f["unit_0"]
            image = png(graphic)
        # At the terminal snapshot the listed action is the transition INTO it.
        assert int(data["step"][j]) == action_step
        if terminal_frame:
            assert bool(data["after_present"][j])
        vector = data["after"][j] if terminal_frame else data["before"][j]
        frames.append(dict(step=step, action_step=action_step,
                           phase="after" if terminal_frame else "before",
                           image=image, actions=data["actions"][j].tolist(),
                           rewards=data["rewards"][j].tolist(), cumulative=data["cumulative_reward"][j].tolist(),
                           visible=vector[0, 93:95].tolist(),
                           remaining=float(vector[0, 95])))
    names = [None, None]
    names[t["side"]], names[1-t["side"]] = t["a"], t["b"]
    return dict(id=t["id"], title=title, why=why, selection=selection, sides=names,
                winner=r["winner"], rewards=r["side_rewards"], steps=r["episode_steps"],
                timeline=r["behavior"]["timeline"], frames=frames)


def build(run):
    manifest, summary = read(run/"manifest.json"), read(run/"summary.json")
    records = [read(f) for f in sorted((run/"cells").glob("*.json"))]
    specs = manifest["roster"]["models"]
    ids = [r["id"] for r in specs]
    assert set(ids) == set(NAMES), "This authored report is scoped to the 21-agent league."
    assert len(records) == 2100 and len({r["id"] for r in records}) == 2100
    assert all(r["valid"] and r["result"]["natural_terminal"] and not r["result"]["probe"] for r in records)
    assert {r["id"] for r in records} == {t["id"] for t in manifest["tasks"]}
    models = []
    for spec in specs:
        st = summarize(records, spec["id"])
        assert st["n"] == 200
        old = next(r for r in summary["ranking"] if r["id"] == spec["id"])
        assert all(abs(st[k]-old[k]) < 1e-10 for k in ("wins", "draws", "losses", "score_rate"))
        st.update(id=spec["id"], name=NAMES[spec["id"]], group=group(spec["id"]), kind=spec["kind"],
                  description=DESCRIPTIONS[spec["id"]], seed=spec["seed"], training_steps=spec["step"],
                  checkpoint=spec["path"], contract=spec["contract"], selection_reason=spec["selection_reason"])
        models.append(st)
    models.sort(key=lambda x: (-x["score_rate"], x["id"]))
    prev, rank = None, 0
    for i, row in enumerate(models):
        if row["score_rate"] != prev:
            rank = i+1
        row["rank"], prev = rank, row["score_rate"]
    pairs = [pair_summary(records, a, b) for i, a in enumerate(ids) for b in ids[i+1:]]
    assert len(pairs) == 210 and all(p["n"] == 10 for p in pairs)
    assert all(Counter((r["repeat"], r["side"]) for r in p["matches"]) ==
               Counter((i, s) for i in range(5) for s in range(2)) for p in pairs)

    print("2100 results reconciled. Auditing initial snapshots and sampled latency.", flush=True)
    hashes = {k: Counter() for k in ("all", "walls", "bases", "batteries")}
    initial_per_pair = defaultdict(list)
    latency = defaultdict(list)
    disagreement, both_tie, reward_tie_score_diff = [], 0, 0
    terminal_scores = {}
    preterminal_opposite, score_winner_changed = 0, 0
    source_keys = [
        "code/shared/agent_league/trace.py", "code/shared/agent_league/match.py",
        "code/shared/agent_league/adapters.py", "code/shared/blackout_rl/policy.py",
        "code/shared/blackout_rl/scripted_fsm.py", "code/shared/blackout_rl/observation.py",
        "code/v6/blackout_rl/mappo_v6.py", "code/v10/blackout_v10/opponents.py",
        "code/v8/blackout_rl/v8/competition/policy.py",
    ]
    source_audit = {k: sha(ROOT/k) == manifest["sources"].get(k) for k in source_keys}
    for rec in records:
        t, r = rec["task"], rec["result"]
        trace = Path(rec["directory"])/"trace"
        idx = read(trace/"index.json")
        assert idx["complete"] and idx["steps"] == r["episode_steps"]
        initial = idx["snapshots"][0]
        assert sha(trace/initial["path"]) == initial["sha256"]
        with np.load(trace/initial["path"], allow_pickle=False) as f:
            g = f["unit_0"]
            for k, c in [("all", list(range(11))), ("walls", [1]), ("bases", [2, 3]), ("batteries", [6])]:
                hashes[k][hashlib.sha256(g[:, :, c].tobytes()).hexdigest()] += 1
        initial_per_pair[(t["a"], t["b"], t["repeat"])].append(r["initial_observation_sha256"])
        # Sparse, deterministic temporal sampling; do not claim full-run latency percentiles.
        cs = idx["chunks"]
        for ci in sorted({0, len(cs)//2, len(cs)-1}):
            with np.load(trace/cs[ci]["path"], allow_pickle=False) as f:
                x = f["inference_seconds"]*1000
            latency[t["a"]].append(x[:, t["side"]])
            latency[t["b"]].append(x[:, 1-t["side"]])
        # Use the actual post-terminal observation when available, rather than
        # silently treating the last pre-action timeline sample as a final score.
        terminal_path = trace/cs[-1]["path"]
        assert sha(terminal_path) == cs[-1]["sha256"]
        with np.load(terminal_path, allow_pickle=False) as f:
            assert bool(f["after_present"][-1]), "Missing final observation: revise analysis scope."
            if str(f["observation_layout"]) == "chain" and str(f["observation_encoding"]) == "xor_uint32_le":
                bits = np.bitwise_xor.reduce(f["observations_xor"][:, 0, 93:95], axis=0)
                sc = bits.astype("<u4", copy=False).view("<f4").astype(float).tolist()
            else:
                sc = load_chunk(terminal_path)["after"][-1, 0, 93:95].astype(float).tolist()
        terminal_scores[rec["id"]] = sc
        apparent = None if abs(sc[0]-sc[1]) < 1e-7 else int(sc[1] > sc[0])
        prev = r["behavior"]["timeline"][-1]["visible_score"]
        previous_winner = None if abs(prev[0]-prev[1]) < 1e-7 else int(prev[1] > prev[0])
        preterminal_opposite += previous_winner is not None and r["winner_side"] is not None and previous_winner != r["winner_side"]
        score_winner_changed += previous_winner != apparent
        if apparent is not None and r["winner_side"] is not None and apparent != r["winner_side"]:
            disagreement.append(rec)
        if apparent is None and r["winner_side"] is not None:
            both_tie += 1
        if apparent is not None and r["winner_side"] is None:
            reward_tie_score_diff += 1
    for m in models:
        values = np.concatenate(latency[m["id"]])
        m["latency_ms"] = dict(n=len(values), p50=float(np.median(values)),
                              p95=float(np.percentile(values, 95)), mean=float(values.mean()))
        score_rates, final_values = [], []
        for rec in records:
            t = rec["task"]
            if m["id"] not in (t["a"], t["b"]):
                continue
            side = t["side"] if m["id"] == t["a"] else 1-t["side"]
            sc = terminal_scores[rec["id"]]
            final_values.append(sc[side])
            score_rates.append(.5 if abs(sc[0]-sc[1]) < 1e-7 else float(sc[side] > sc[1-side]))
        m["terminal_score_rate"] = mean(score_rates)
        m["mean_terminal_visible_score"] = mean(final_values)
    audit = dict(valid_games=len(records), pairs=len(pairs), total_steps=sum(r["result"]["episode_steps"] for r in records),
                 model_count=len(models), samples_per_model=200, failures=summary["failed_cells"],
                 initial_full_hashes=len({r["result"]["initial_observation_sha256"] for r in records}),
                 map_diversity={k: dict(unique=len(v), max_repeats=max(v.values())) for k, v in hashes.items()},
                 equal_side_pair_initials=sum(len(v) == 2 and v[0] == v[1] for v in initial_per_pair.values()),
                 paired_initials=len(initial_per_pair), source_audit=source_audit,
                 reward_winners=dict(Counter(str(r["result"]["winner_side"]) for r in records)),
                 terminal_observations=len(terminal_scores),
                 preterminal_score_opposite=preterminal_opposite,
                 score_winner_changed_on_final_transition=score_winner_changed,
                 apparent_score_opposite=len(disagreement), visible_tie_reward_decisive=both_tie,
                 reward_tie_visible_decisive=reward_tie_score_diff,
                 discordant_with_rush=sum("scripted-rush" in (r["task"]["a"], r["task"]["b"]) for r in disagreement),
                 full_length_games=sum(r["result"]["episode_steps"] == 21004 for r in records),
                 trace_bytes=sum(r["result"].get("trace_bytes", 0) for r in records),
                 status=read(run/"status.json"), manifest_sha256=sha(run/"manifest.json"),
                 result_files_sha256=hashlib.sha256("".join(sha(run/"cells"/(r["id"]+".json")) for r in records).encode()).hexdigest())
    assert all(source_audit.values()), "Source drift: revise interpretation against archived code before publishing."

    def choose(a, b, mode="median", winner=None):
        rows = [r for r in records if {r["task"]["a"], r["task"]["b"]} == {a, b}
                and (winner is None or r["result"]["winner"] == winner)]
        rows.sort(key=lambda r: (r["result"]["episode_steps"], r["id"]))
        if mode == "max_reward":
            return max(rows, key=lambda r: abs(r["result"]["side_rewards"][0]-r["result"]["side_rewards"][1]))
        return rows[len(rows)//2]
    cases = [
        case_data(choose("v6-5c651487", "v10-C1"), "목표 수행의 격차 · v6 대 v10",
                  "v6는 이 대진 10경기를 모두 이겼다. 실제 맵과 행동, 반납으로 연결되는 관측 점수 흐름을 함께 비교한다.",
                  "해당 10경기를 step 수로 정렬한 상위 중앙값 1경기"),
        case_data(choose("v4-29794421", "v6-5c651487"), "상위권의 접전 · v4 대 v6",
                  "전체 맞대결은 5승 5패다. 이 한 경기의 승자를 최강 모델로 일반화하지 않는다.",
                  "해당 10경기를 step 수로 정렬한 상위 중앙값 1경기"),
        case_data(choose("scripted-rush", "scripted-collector", "max_reward"), "보상과 점수의 차이 · 돌격 대 수집",
                  "누적 reward와 배터리 관측 점수가 서로 다른 성공을 측정할 수 있음을 보여주는 사례다.",
                  "해당 대진에서 누적 reward 차이의 절댓값이 가장 큰 경기: 의도적으로 극단 사례"),
        case_data(choose("v8-v8_comp-c1", "scripted-noop"), "행동 동기화 · v8 C1 대 정지",
                  "한 방향 출력이 팀 전체에 반복되는지 확인한다. 정지 상대를 포함한 단일 사례이며 원인 진단은 추가 실험이 필요하다.",
                  "해당 10경기를 step 수로 정렬한 상위 중앙값 1경기"),
        case_data(choose("v9-B", "v10-C1"), "직접 비교 · v9 대 v10",
                  "두 모델의 전체 승점률 차이는 0.5%p지만 직접 대결은 5승 5패다. 우열을 확정할 수 없다.",
                  "해당 10경기를 step 수로 정렬한 상위 중앙값 1경기"),
        case_data(choose("v5-9c6b5f28", "scripted-weak", winner="scripted-weak"), "실패를 살펴보기 · v5의 하위 상대 패배",
                  "v5는 상위 5개 밖의 160경기 중 이 한 경기만 졌다. 좁은 실패 조건을 찾기 위한 분석 사례다.",
                  "확장 추격 FSM이 승리한 유일한 경기"),
    ]
    return dict(schema="league.analysis.v1", run=run.name, generated_at=datetime.now().astimezone().isoformat(),
                models=models, pairs=pairs, audit=audit, cases=cases,
                cycles=read(run/"observed_cycles.json"))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", type=Path, default=DEFAULT_RUN)
    ap.add_argument("--render-only", action="store_true", help="Use already audited analysis_data.json.")
    args = ap.parse_args()
    run = args.run.resolve()
    data_path = run/"analysis_data.json"
    data = read(data_path) if args.render_only else build(run)
    text = json.dumps(data, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    if not args.render_only:
        data_path.write_text(text+"\n")
    template = Path(__file__).with_name("league_analysis_template.html").read_text()
    # Safe inline JSON, including future labels containing HTML-significant characters.
    encoded = text.replace("&", "\\u0026").replace("<", "\\u003c").replace(">", "\\u003e")
    assert template.count("__ANALYSIS_DATA__") == 1
    (run/"analysis.html").write_text(template.replace("__ANALYSIS_DATA__", encoded))
    print(run/"analysis.html")
    print(json.dumps(data["audit"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
