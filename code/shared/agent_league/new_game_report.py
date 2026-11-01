"""Score-primary reports for the isolated November league.

Shared trace, behavior, pair, and chart generation is retained. This wrapper
adds the score evidence and updates every result criterion and launcher link.
"""
import csv
from pathlib import Path

from .common import read
from .report import generate as shared_generate, write
from blackout_v12.io import atomic_json as atomic


def generate(directory, detailed=True):
    from .new_game import completed_rows
    directory = Path(directory)
    manifest = read(directory / 'manifest.json')
    if manifest.get('schema') != 'blackout.league.new_game.v1':
        raise ValueError('New-game reports require a new-game manifest')
    records = completed_rows(directory, manifest)
    summary = shared_generate(directory, detailed=detailed)
    deltas = {model['id']: [] for model in manifest['roster']['models']}
    for record in records:
        task, result = record['task'], record['result']
        for model, side in ((task['a'], task['side']), (task['b'], 1-task['side'])):
            deltas[model].append(result['side_scores'][side] - result['side_scores'][1-side])
    for row in summary['ranking']:
        values = deltas[row['id']]
        row['mean_score_delta'] = sum(values)/len(values) if values else None
    summary['primary_objective'] = 'terminal_team_scores_v1'
    summary['evaluation_audit']['primary_score_games'] = len(records)
    summary['evaluation_audit']['score_reward_winner_disagreements'] = sum(
        record['result'].get('winner_criteria_agree') is False for record in records)
    summary['game_source_commit'] = manifest['inputs']['new_game_runtime']['unity_commit']
    summary['api_source_commit'] = manifest['inputs']['new_game_runtime']['api_commit']
    summary['game_source_proof'] = manifest['inputs'].get('build', {}).get('source_proof', 'user_attestation')
    atomic(directory / 'summary.json', summary)
    with (directory / 'ranking.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary['ranking'][0])
                                if summary['ranking'] else ['id'])
        writer.writeheader()
        writer.writerows(summary['ranking'])

    report_path = directory / 'report.md'
    text = report_path.read_text(encoding='utf-8')
    text = text.replace('# 전체 Agent 풀리그', '# 최신 게임 전체 Agent 풀리그', 1)
    text = text.replace(
        '승패는 제공 runner 누적 raw reward 기준입니다. 관측 점수와 공식 대회 결과는 동일하다고 가정하지 않습니다. 초기 관측 해시를 기록하지만 seed가 서로 다른 맵을 생성한다고 검증하지 않았습니다.',
        '승패는 자연 종료 시 모든 에이전트의 infos에서 일치한 최종 score_0/score_1 기준입니다. '
        '동점은 무승부입니다. 누적 raw reward와 API infos.winner는 별도 진단으로 보존합니다. '
        '초기 맵 해시와 진영 교대 맵 일치를 기록하며, 서로 다른 seed가 서로 다른 맵을 만든다고 가정하지 않습니다.')
    text = text.replace('공식 terminal winner 확보', '최종 점수 승패 확보')
    text = text.replace('누적 보상 승패와 불일치', '최종 점수와 누적 보상 승패 불일치')
    table_start = text.index('| 순위 |')
    table_end = text.index('\n\n', table_start)
    table = ['| 순위 | 모델 | 경기 | 승/무/패 | 승률 | 상대 평균 승점률 | 평균 최종 점수 차 | 보조 평균 reward 차 |',
             '|---:|---|---:|---|---:|---:|---:|---:|']
    for row in summary['ranking']:
        percentage = lambda value: '—' if value is None else f'{value:.1%}'
        number = lambda value: '—' if value is None else f'{value:.3f}'
        table.append(f"| {row['rank']} | [{row['id']}](models/{row['id']}.md) | {row['n']} | "
                     f"{row['wins']}/{row['draws']}/{row['losses']} | {percentage(row['win_rate'])} | "
                     f"{percentage(row['score_rate'])} | {number(row['mean_score_delta'])} | "
                     f"{number(row['mean_reward_delta'])} |")
    text = text[:table_start] + '\n'.join(table) + text[table_end:]
    config = manifest['config']
    text += (f"\n등록된 게임 source: `{summary['game_source_commit']}` "
             '(사용자 제공 출처 확인; 컴파일 결과의 source 일치를 암호학적으로 증명하지 않음). '
             f"API source: `{summary['api_source_commit']}`.\n"
             f"\n동시에 최대 {config['workers']}경기입니다. slot 0은 {config['visible_time_scale']:g}배속 표시 창, "
             f"나머지는 {config['time_scale']:g}배속 백그라운드 렌더링입니다. "
             '각 시도의 실제 slot·표시 모드·배속은 execution.json과 결과에 보존합니다.\n')
    write(report_path, text)
    if detailed:
        for record in records:
            task, result = record['task'], record['result']
            path = directory / 'matches' / (task['id'] + '.md')
            text = path.read_text(encoding='utf-8').replace('run_league.ps1', 'run_league_new_game.ps1')
            marker = '\n## 경기 흐름'
            evidence = (f"\n최종 Blue/Red 점수: **{result['side_scores']}**. "
                        f"최종 점수 승자: **{result['winner'] or '무승부'}**.\n"
                        f"\n보조 누적 reward 승자: {result.get('reward_winner') or '무승부'}. "
                        f"점수·보상 승패 일치: {result.get('winner_criteria_agree')}. "
                        f"API infos.winner 진영: {result.get('legacy_info_winner_side')}.\n")
            text = text.replace(marker, evidence + marker, 1)
            write(path, text)
    return summary
