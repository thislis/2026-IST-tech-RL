# 외부 조사 출발 자료: 확인한 원문·공식 문서

조회일: 2026-09-27 KST. 아래는 서지·공식 소개/원문 HTML에서 주제와 관련성을 확인한 탐색 시작점이다. 모든 본문·보충자료·공식 코드를 재현한 목록이나 최신 연구의 완전한 목록이 아니다. 실제 채택 판단은 외부 조사자가 정정·버전·대조 실험·코드·라이선스를 확인하고, 내부 증거와 연결한 뒤 수행한다.

기본 논문과 2026-09-27까지의 후속 자료를 함께 검색한다. 최종 출처는 논문 원문/출판사/저자 코드/공식 문서를 우선한다. 검색 결과 요약·블로그·게임 시연은 발견용 자료로만 쓰고 근거 원문을 추적한다.

## S01. Gymnasium: Handling Time Limits

공식 문서; 조회 2026-09-27

원문/공식 주소: https://gymnasium.farama.org/tutorials/gymnasium_basics/handling_time_limits/

관련 요청: E01, E07

확인할 지점: 게임 내재 시간제한과 외부 truncation, bootstrap의 구분. 프로젝트 설치 버전에 적용되는 부분은 별도 대조.

## S02. Unity ML-Agents: Getting started with the LLAPI

공식 문서; 조회 2026-09-27

원문/공식 주소: https://unity-technologies.github.io/ml-agents/Python-LLAPI/

관련 요청: E01, E08

확인할 지점: DecisionSteps/TerminalSteps의 observation/reward/agent_id/interrupted. 대회 loader와 점수 전달 계약까지 보장하지 않음.

## S03. Residual Reinforcement Learning for Robot Control

Johannink et al.; ICRA 2019; preprint 2018

원문/공식 주소: https://arxiv.org/abs/1812.03201

관련 요청: E02

확인할 지점: 고전 제어와 학습 residual의 결합. 로봇 연속 additive 제어를 방향 정규화 게임에 그대로 옮기지 말 것.

## S04. A Closer Look at Invalid Action Masking in Policy Gradient Algorithms

Huang & Ontañón; FLAIRS 2022; preprint 2020

원문/공식 주소: https://doi.org/10.32473/flairs.v35i.130584

관련 요청: E02

확인할 지점: mask 적용·학습/평가 조건 비교. arXiv 2006.14171의 최종 수정판과 저자 코드도 대조.

## S05. Proximal Policy Optimization Algorithms

Schulman et al.; 2017

원문/공식 주소: https://arxiv.org/abs/1707.06347

관련 요청: E02, E03

확인할 지점: behavior policy와 likelihood ratio, clipped surrogate. gate·mask·강제개입의 정확한 joint probability를 정의할 기준.

## S06. The Option-Critic Architecture

Bacon, Harb & Precup; AAAI 2017

원문/공식 주소: https://ojs.aaai.org/index.php/AAAI/article/view/10916

관련 요청: E03

확인할 지점: 옵션 내부 정책, 옵션 선택, 종료조건. action repeat와 고수준 제어의 차이를 읽을 출발점.

## S07. High-Dimensional Continuous Control Using Generalized Advantage Estimation

Schulman et al.; preprint 2015; arXiv 수정 이력 확인

원문/공식 주소: https://arxiv.org/abs/1506.02438

관련 요청: E03

확인할 지점: GAE의 bias/variance와 value bootstrap. 추정 시간척도를 학습 가능 시간의 절대 상한으로 해석하지 말 것.

## S08. The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games

Yu et al.; NeurIPS 2022; preprint 2021

원문/공식 주소: https://arxiv.org/abs/2103.01955

관련 요청: E03, E06

확인할 지점: MAPPO의 구현·하이퍼파라미터·critic 입력 ablation과 저자 on-policy 구현. 본 프로젝트의 team joint actor와 차이 확인.

## S09. Counterfactual Multi-Agent Policy Gradients

Foerster et al.; AAAI 2018

원문/공식 주소: https://ojs.aaai.org/index.php/AAAI/article/view/11794

관련 요청: E03

확인할 지점: 팀 보상에서 individual action의 기여도 baseline. COMA의 decentralized actor를 현재 joint residual과 구별.

## S10. Recurrent Model-Free RL Can Be a Strong Baseline for Many POMDPs

Ni, Eysenbach & Salakhutdinov; ICML 2022

원문/공식 주소: https://proceedings.mlr.press/v162/ni22a.html

관련 요청: E04

확인할 지점: 기억 기반 baseline의 입력·구조·훈련 설정. 공식 inference가 받지 못하는 reward/metadata는 제외.

## S11. A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning

Ross, Gordon & Bagnell; AISTATS 2011

원문/공식 주소: https://proceedings.mlr.press/v15/ross11a.html

관련 요청: E04

확인할 지점: DAgger, learner-induced distribution과 순차 오류. v3에서 이미 사용한 방법이므로 기존 실패와의 구체적 차이를 요구.

## S12. Transforming a head direction signal into a goal-oriented steering command

Nature 2024; 2025-03-11 Author Correction 표시

원문/공식 주소: https://www.nature.com/articles/s41586-024-07039-2

관련 요청: E04, E05

확인할 지점: heading·goal의 좌표변환과 PFL/steering. 정정과 보충자료를 반영. 신경생물학 결과를 게임 성능 증거로 읽지 말 것.

## S13. Connectome-constrained networks predict neural activity across the fly visual system

Nature 2024

원문/공식 주소: https://doi.org/10.1038/s41586-024-07939-3

관련 요청: E05

확인할 지점: connectome 제약과 task optimization의 분리, cell-type parameterization, random model/구조 ablation. 저자 구현·데이터 확인.

## S14. A Drosophila computational brain model reveals sensorimotor processing

Nature 2024

원문/공식 주소: https://doi.org/10.1038/s41586-024-07763-9

관련 요청: E05

확인할 지점: connectivity/transmitter 기반 모델의 감각·출력·개입 검증과 가정. feeding/grooming 검증을 범용 게임 능력으로 일반화하지 말 것.

## S15. Prioritized Level Replay

Jiang, Grefenstette & Rocktäschel; ICML 2021

원문/공식 주소: https://proceedings.mlr.press/v139/jiang21b.html

관련 요청: E06

확인할 지점: 미래 방문할 level의 학습 가능성 기반 선택. stale PPO transition 재사용과 구분.

## S16. Deep Reinforcement Learning at the Edge of the Statistical Precipice

Agarwal et al.; NeurIPS 2021

원문/공식 주소: https://proceedings.neurips.cc/paper_files/paper/2021/hash/f514cec81cb148559cf475e7426eed5e-Abstract.html

관련 요청: E07

확인할 지점: 소수 학습 run의 통계 불확실성, 구간/aggregate metric. 저자 rliable 도구가 우리 crossed seed×map 구조에 자동 정답인 것은 아님.

## S17. Implementation Matters in Deep Policy Gradients: A Case Study on PPO and TRPO

Engstrom et al.; 2020

원문/공식 주소: https://arxiv.org/abs/2005.12729

관련 요청: E03, E07

확인할 지점: 코드-level 선택과 성능 귀속. 버전/튜닝/정규화 차이와 알고리즘 기여를 분리.

## S18. PyTorch Performance Tuning Guide

공식 문서; 조회 2026-09-27

원문/공식 주소: https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide

관련 요청: E08

확인할 지점: CPU/GPU/스레드·전송/동기화 비용 측정의 출발점. 현재 설치 버전/장치에서 재검증하고 무조건 GPU·AMP·업그레이드 처방 금지.

## 문헌별 근거 카드

ID / E·F·D ID:
원문 제목·저자·정식 출판연도·preprint연도:
원문 URL·DOI·버전·정정:
읽은 절·수식·표·그림:
저자 주장:
직접 확인한 근거:
원래 task·observation·action·reward·training budget·hardware:
baseline·ablation·training seeds·불확실성:
코드 URL·commit·실행 확인·라이선스:
BlackOut와 동일한 조건:
BlackOut에서 다른 조건:
제안하는 최소 변경:
기각 가능한 내부 증거:
필요한 신규 계측:
비용(측정/추정):
보류/실험/채택 판단:
