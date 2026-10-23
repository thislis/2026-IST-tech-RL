"""Offline, deterministic Korean Markdown reports from committed matches only."""
import csv
import gzip
import html
import itertools
from pathlib import Path
import numpy as np
from .common import read,atomic


def write(path,text):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(text,encoding='utf8');tmp.replace(path)


def heatmap(matrix,labels,path):
    n=len(labels);cell=30;left=390;top=390;w=left+cell*n+20;h=top+cell*n+20
    rows=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}"><rect width="100%" height="100%" fill="white"/>']
    for i,label in enumerate(labels):
        label=html.escape(label)
        rows.append(f'<text x="{left-8}" y="{top+i*cell+20}" text-anchor="end" font-size="10">{label}</text>')
        rows.append(f'<text transform="translate({left+i*cell+20},{top-8}) rotate(-65)" font-size="10">{label}</text>')
        for j,v in enumerate(matrix[i]):
            color='#eeeeee' if v is None else f'rgb({int(245-120*v)},{int(235-30*v)},{int(235-150*v)})'
            rows.append(f'<rect x="{left+j*cell}" y="{top+i*cell}" width="29" height="29" fill="{color}"/>')
            if v is not None:rows.append(f'<text x="{left+j*cell+14}" y="{top+i*cell+18}" text-anchor="middle" font-size="9">{v:.2f}</text>')
    write(path,'\n'.join(rows+['</svg>']))


def match_report(path,record):
    task=record['task'];r=record['result'];timeline=r['behavior']['timeline'];trace=Path(record['directory'])/'trace'
    events=[]
    if (trace/'events.jsonl.gz').exists():
        import json
        with gzip.open(trace/'events.jsonl.gz','rt') as f:
            for line in f:
                e=json.loads(line)
                if e['kind']!='policy_decision':events.append(e)
    # Keep narrative bounded; all events remain in the compressed log.
    picks=[0,len(timeline)//2,len(timeline)-1] if timeline else []
    text=[f"# {task['a']} vs {task['b']}",f"\n반복 {task['repeat']+1}, A의 진영 {task['side']} (0=Blue, 1=Red). 승자: {r['winner'] or '무승부'}.",
          f"\n{r['episode_steps']:,} step, {r['wall_seconds']:.1f}초. Blue/Red 누적 reward: {r['side_rewards']}.",
          '\n## 경기 흐름','\n| 구간 | Step | 남은 시간 관측 | Blue/Red 누적 reward | 관측 점수 |','|---|---:|---:|---|---|']
    for phase,i in zip(('초반','중반','후반'),picks):
        p=timeline[i];text.append(f"| {phase} | {p['step']} | {p['time']:.5f} | {p['reward']} | {p['visible_score']} |")
    text+=['\n## 주요 사건','\n추정 사건은 관측 proxy입니다. unit ID와 entity slot을 같은 개체로 단정하지 않습니다.']
    event_indices=sorted(set(np.linspace(0,len(events)-1,min(30,len(events)),dtype=int).tolist())) if events else []
    for i in event_indices:
        e=events[i];step=e['step'];chunk=step//task['config']['chunk_steps']*task['config']['chunk_steps']
        rel=__import__('os').path.relpath(trace/f'steps-{chunk:06d}.npz',path.parent)
        text.append(f"- [step {step}]({rel}): `{e}`")
    rel=__import__('os').path.relpath(trace,path.parent)
    text += [f'\n[전체 trace 인덱스]({rel}/index.json) · [전체 이벤트]({rel}/events.jsonl.gz)',
        f"\n상세 조회: `bash code/shared/run_agent_league.sh --name {path.parents[1].name} --trace {task['id']} --step 100`"]
    write(path,'\n'.join(text)+'\n')


def generate(directory,detailed=True):
    directory=Path(directory);manifest=read(directory/'manifest.json');models=manifest['roster']['models'];ids=[m['id'] for m in models]
    records=[read(p) for p in sorted((directory/'cells').glob('*.json'))]
    rows=[r for r in records if r['result']['natural_terminal'] and not r['result'].get('probe')]
    expected=len(manifest['tasks']);pairs={};stats={m:dict(n=0,w=0,d=0,reward=[],sides=[[],[]],behavior=[],examples=[]) for m in ids}
    for record in rows:
        t=record['task'];r=record['result'];pairs.setdefault((t['a'],t['b']),[]).append(record)
        for model,side in [(t['a'],t['side']),(t['b'],1-t['side'])]:
            st=stats[model];score=.5 if r['winner'] is None else float(r['winner']==model)
            st['n']+=1;st['w']+=score==1;st['d']+=score==.5;st['sides'][side].append(score)
            st['reward'].append(r['side_rewards'][side]-r['side_rewards'][1-side]);st['behavior'].append(r['behavior']['sides'][side]);st['examples'].append(t['id'])
        if detailed:match_report(directory/'matches'/(t['id']+'.md'),record)
    matrix=[[None]*len(ids) for _ in ids];ranking=[]
    for (a,b),rs in pairs.items():
        i,j=ids.index(a),ids.index(b)
        win=sum(x['result']['winner']==a for x in rs);draw=sum(x['result']['winner'] is None for x in rs)
        matrix[i][j]=(win+.5*draw)/len(rs);matrix[j][i]=1-matrix[i][j]
        if detailed:
            txt=[f'# {a} vs {b}',f'\n완료 {len(rs)}/{manifest["config"]["rounds"]*2}경기. A 승점률 {matrix[i][j]:.1%}.',
                 '\n| 반복 | A Blue 결과 | A Red 결과 |','|---|---|---|']
            for rep in range(manifest['config']['rounds']):
                values=[]
                for side in (0,1):
                    row=next((x for x in rs if x['task']['repeat']==rep and x['task']['side']==side),None)
                    values.append('미완료' if row is None else f"[{row['result']['winner'] or '무승부'}](../matches/{row['task']['id']}.md)")
                txt.append(f'| {rep+1} | {values[0]} | {values[1]} |')
            set_rates=[]
            for rep in range(manifest['config']['rounds']):
                rr=[x for x in rs if x['task']['repeat']==rep]
                if len(rr)==2:set_rates.append(sum(.5 if x['result']['winner'] is None else float(x['result']['winner']==a) for x in rr)/2)
            if set_rates:txt.append(f'\n완료된 진영 교대 세트별 A 승점률: {set_rates}; 범위 {min(set_rates):.1%}–{max(set_rates):.1%}.')
            txt+=['\n진영 교대 세트의 탐색적 비교입니다. 새 맵 일반화나 통계적으로 확정된 우위를 뜻하지 않습니다.']
            write(directory/'pairs'/f'{a}__{b}.md','\n'.join(txt))
    for i,m in enumerate(ids):
        st=stats[m];rates=[x for x in matrix[i] if x is not None]
        ranking.append(dict(id=m,n=st['n'],wins=st['w'],draws=st['d'],losses=st['n']-st['w']-st['d'],
            win_rate=st['w']/st['n'] if st['n'] else None,score_rate=float(np.mean(rates)) if rates else None,
            mean_reward_delta=float(np.mean(st['reward'])) if st['n'] else None,
            side_score_rates=[float(np.mean(x)) if x else None for x in st['sides']]))
        if detailed:
            spec=next(x for x in models if x['id']==m)
            txt=[f'# {m}',f'\n{spec["label"]}; 계약: `{spec["contract"]}`; step {spec["step"]:,}; seed {spec["seed"]}.',f'\n대표 선택: {spec["selection_reason"]}.',
                '\n## 관측된 행동 특징','\n행동은 실제 출력, 공간·이벤트 지표는 관측 proxy입니다. 정책 의도를 확정하지 않습니다.']
            for key in ('stop_fraction','turn_fraction','same_action_fraction','moving_entity_fraction','team_spread','near_base_fraction','pickup_proxy','delivery_proxy','teleport_proxy','backtrack_proxy_fraction'):
                vals=[x[key] for x in st['behavior']];txt.append(f'- {key}: 경기 평균 {np.mean(vals):.4f}' if vals else f'- {key}: 자료 없음')
            if st['behavior']:
                hist=np.sum([x['action_histogram'] for x in st['behavior']],axis=0).tolist();txt.append(f'- 실제 행동 방향 집계(정지, E, NE, N, NW, W, SW, S, SE): {hist}')
                atomic(directory/'models'/f'{m}-occupancy.json',np.sum([x['occupancy'] for x in st['behavior']],axis=0).tolist())
            if st['behavior']:
                means={k:float(np.mean([b[k] for b in st['behavior']])) for k in ('stop_fraction','same_action_fraction','near_base_fraction','pickup_proxy','delivery_proxy','moving_entity_fraction')}
                txt.append(f"\n실제 출력에서 정지 행동은 {means['stop_fraction']:.1%}, 팀원 전원이 같은 행동을 낸 step은 {means['same_action_fraction']:.1%}였습니다.")
                txt.append(f"관측상 창고 근처 아군 비중은 평균 {means['near_base_fraction']:.1%}이며, 경기당 획득 추정 {means['pickup_proxy']:.1f}회·반납 추정 {means['delivery_proxy']:.1f}회였습니다. 이는 행동·공간 통계이며 의도한 전략의 판정은 아닙니다.")
                observed=[(v,ids[j]) for j,v in enumerate(matrix[i]) if v is not None]
                if observed:
                    high=max(observed);low=min(observed)
                    txt.append(f'상대별 관측 승점률은 {high[1]} 상대 {high[0]:.1%}로 가장 높고, {low[1]} 상대 {low[0]:.1%}로 가장 낮았습니다. 아래 대진별 자료에서 표본과 진영 차이를 확인할 수 있습니다.')
            txt+=['\n## 상대별 성적','\n| 상대 | 승점률 |','|---|---:|']
            for j,v in enumerate(matrix[i]):
                if v is not None:
                    a,b=sorted([m,ids[j]]);txt.append(f'| [{ids[j]}](../pairs/{a}__{b}.md) | {v:.1%} |')
            txt+=['\n## 근거 경기']+[f'- [{g}](../matches/{g}.md)' for g in st['examples'][:5]]
            write(directory/'models'/f'{m}.md','\n'.join(txt))
    ranking.sort(key=lambda x:(-(x['score_rate'] if x['score_rate'] is not None else -1),x['id']))
    previous=None;rank=0
    for i,row in enumerate(ranking):
        if row['score_rate']!=previous or i==0:rank=i+1
        row['rank']=rank;previous=row['score_rate']
    status=read(directory/'status.json') if (directory/'status.json').exists() else {}
    summary=dict(complete=len(rows)==expected,completed=len(rows),expected=expected,ranking=ranking,ids=ids,score_matrix=matrix,
                 failed_cells=len(list((directory/'failures').glob('*.json'))),status=status)
    atomic(directory/'summary.json',summary)
    with (directory/'ranking.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(ranking[0]) if ranking else ['id']);w.writeheader();w.writerows(ranking)
    cycles=[]
    for a,b,c in itertools.combinations(range(len(ids)),3):
        for x,y,z in [(a,b,c),(a,c,b)]:
            if all(matrix[u][v] is not None and matrix[u][v]>.5 for u,v in [(x,y),(y,z),(z,x)]):cycles.append([ids[x],ids[y],ids[z]])
    atomic(directory/'observed_cycles.json',cycles)
    heatmap(matrix,ids,directory/'figures/score_matrix.svg')
    win_matrix=[[None]*len(ids) for _ in ids]
    for (a,b),rs in pairs.items():
        i,j=ids.index(a),ids.index(b)
        win_matrix[i][j]=sum(x['result']['winner']==a for x in rs)/len(rs)
        win_matrix[j][i]=sum(x['result']['winner']==b for x in rs)/len(rs)
    atomic(directory/'win_rate_matrix.json',dict(ids=ids,matrix=win_matrix))
    heatmap(win_matrix,ids,directory/'figures/win_rate_matrix.svg')
    scope=manifest.get('experiment_kind','league')
    title='전체 Agent 풀리그' if scope=='league' else '소규모 실행 검증 — 전체 리그 아님'
    text=['# '+title,f'\n상태: {"완료" if summary["complete"] else "진행 중/미완료 — 잠정 순위"}. {len(rows):,}/{expected:,}경기, 실패 셀 {summary["failed_cells"]}.',
          f'\n실행 상태: `{status.get("state","prepared")}`. 예상 잔여 시간(초): {status.get("eta_seconds","측정 전")}.',
          '\n## 순위','\n상대별 승점률 평균으로 정렬합니다. 승점률=(승+0.5×무)/경기 수. 동률은 공동 순위입니다.',
          '\n| 순위 | 모델 | 경기 | 승/무/패 | 승률 | 상대 평균 승점률 | 평균 reward 차 |','|---:|---|---:|---|---:|---:|---:|']
    for r in ranking:
        fmt=lambda x:'—' if x is None else f'{x:.1%}'
        delta='—' if r['mean_reward_delta'] is None else f"{r['mean_reward_delta']:.3f}"
        text.append(f"| {r['rank']} | [{r['id']}](models/{r['id']}.md) | {r['n']} | {r['wins']}/{r['draws']}/{r['losses']} | {fmt(r['win_rate'])} | {fmt(r['score_rate'])} | {delta} |")
    text+=['\n## 상대 관계','\n![행 모델의 열 상대에 대한 승점률](figures/score_matrix.svg)\n\n[순수 승률 행렬](figures/win_rate_matrix.svg) · [승률 JSON](win_rate_matrix.json)',f'\n관측된 3자 순환 관계: {len(cycles)}개. [전체 목록](observed_cycles.json).']
    text += ['- '+' → '.join(c+[c[0]]) for c in cycles[:20]]
    text+=['\n## 해석 범위','\n승패는 제공 runner 누적 raw reward 기준입니다. 관측 점수와 공식 대회 결과는 동일하다고 가정하지 않습니다. 초기 관측 해시를 기록하지만 seed가 서로 다른 맵을 생성한다고 검증하지 않았습니다.',
           '\nlegacy ID/planner/state 모델과 두 tensor 제출 모델이 함께 참가합니다. 입력 계약과 추론 동작을 보존한 탐색적 비교이며, 소수 반복으로 확정적 전략 우위를 주장하지 않습니다.',
           '\n## 참가·제외 기록',f'\n[고정 manifest](manifest.json) · [실패 기록](failures/) · [자원 측정](benchmark.json)',
           '\n| 후보 | 제외 사유 |','|---|---|']
    for e in manifest['roster']['excluded']:text.append(f"| {e['path']} | {e['reason']} |")
    write(directory/'report.md','\n'.join(text)+'\n')
    return summary
