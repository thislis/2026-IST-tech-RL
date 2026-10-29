"""Bounded synthetic calibration. Never starts Unity or persists trained weights."""
from concurrent.futures import ProcessPoolExecutor
import copy
import multiprocessing
import statistics
import time
import torch
from .io import atomic_json, digest
from .runtime import hardware_identity, budget, configure_worker


def fixture():
    """Small public-observation fixture for short CPU checks; no environment."""
    import numpy as np
    vector = np.zeros((5,96), np.float32)
    entities = vector[0,:90].reshape(10,9)
    entities[:5,:2] = np.array([[3,18],[4,18],[5,18],[4,17],[5,17]], np.float32)/24
    entities[5:,:2] = np.array([[20,3],[19,3],[18,3],[19,4],[18,4]], np.float32)/24
    entities[:5,2], entities[5:,2], entities[:,3] = 1,-1,1
    vector[:,:90] = entities.reshape(90)
    vector[:,90], vector[:,95] = 1,1
    graphic = np.zeros((5,11,96,96), np.float32)
    graphic[:,0] = 1
    for channel, points in [(2,[(1,21),(2,21),(7,15)]),(3,[(21,2),(20,2)]),
                            (6,[(7,18),(9,16),(3,12),(10,9),(18,12)])]:
        for x,y in points:
            row = (23-y)*4
            graphic[:,:,row:row+4,x*4:x*4+4] = 0
            graphic[:,channel,row:row+4,x*4:x*4+4] = 1
    return vector, graphic


def _sync(device):
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


def _times(function, device, repeats):
    for _ in range(2):
        function()
    _sync(device)
    values = []
    for _ in range(repeats):
        begin = time.perf_counter()
        function()
        _sync(device)
        values.append(1000 * (time.perf_counter() - begin))
    values.sort()
    return dict(p50_ms=statistics.median(values), p95_ms=values[min(len(values)-1, int(.95*len(values)))],
                repetitions=repeats, total_ms=sum(values))


def inference_probe(device, threads, repeats):
    from .model import TaskScorer
    from .assignment import map_assignment
    torch.set_num_threads(threads)
    scorer = TaskScorer().to(device).eval()
    features = torch.randn(5, 24, 64)
    prior = torch.randn(5, 24)
    legal = torch.ones(5, 24, dtype=torch.bool)
    capacities = torch.ones(24, dtype=torch.long)
    @torch.no_grad()
    def operation():
        # Includes host transfers; planner and exact decoding stay on the CPU.
        scores = scorer(features.to(device), prior.to(device)).cpu()
        return map_assignment(scores, capacities, legal)
    return dict(device=device, threads=threads, valid=True, scope="120-edge scorer + transfers + CPU MAP DP",
                **_times(operation, device, repeats))


def learner_probe(device, threads, batch, repeats):
    from .model import TaskScorer
    from .critic import Critic
    from .assignment import log_partition
    torch.set_num_threads(threads)
    scorer, critic = TaskScorer().to(device), Critic().to(device)
    parameters = [*scorer.parameters(), *critic.parameters()]
    optimizer = torch.optim.Adam(parameters, lr=1e-4)
    features = torch.randn(batch, 5, 24, 64)
    priors = torch.randn(batch, 5, 24)
    legal = torch.ones(batch, 5, 24, dtype=torch.bool)
    caps = torch.ones(batch, 24, dtype=torch.long)
    def operation():
        # Represents the exact structured likelihood's expensive forward/backward.
        x, p = features.to(device), priors.to(device)
        scores = scorer(x, p)
        partition = log_partition(scores, caps.to(device), legal.to(device))
        loss = (partition - scores[..., :5].diagonal(dim1=-2, dim2=-1).sum(-1)).mean()
        loss = loss + critic(x.mean((1, 2))).square().mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    timing = _times(operation, device, repeats)
    return dict(device=device, threads=threads, batch_events=batch, valid=True,
                events_per_second=batch / (timing["p50_ms"] / 1000),
                scope="synthetic exact-DP gradient + MLP critic + host transfers", **timing)


def policy_probe(config, repeats=6):
    """Full public CPU boundary, reported separately from scorer/DP timings."""
    from .policy import MyPolicy
    torch.set_num_threads(1)
    policy = MyPolicy(config, arm='F').eval()
    vector, graphic = fixture()
    @torch.inference_mode()
    def operation(event=False, cold=False):
        if cold:
            policy.reset()
        elif event:
            policy.runtime.last_event_step = policy.runtime.step_count-10000
        vector[:,95] -= .02/420
        result = policy.step(vector,graphic,deterministic=True)
        if bool(result['event']) != bool(event or cold):
            raise AssertionError('Synthetic timing fixture event classification changed')
    with torch.inference_mode():
        policy.step(vector,graphic,deterministic=True)
    return dict(scope='public decode + FSM + navigation + scorer + assignment; synthetic fixed map',
                no_event=_times(operation,'cpu',repeats),
                decision_cached=_times(lambda:operation(event=True),'cpu',repeats),
                reset_cold=_times(lambda:operation(cold=True),'cpu',repeats),
                measured_game_throughput=False)


_BARRIER = None


def _pool_initialize(barrier):
    global _BARRIER
    configure_worker()
    _BARRIER = barrier


def _cpu_worker(seconds):
    from .model import TaskScorer
    from .assignment import map_assignment
    import numpy as np
    model = TaskScorer().eval()
    x = torch.ones(5, 24, 64)
    legal, caps = torch.ones(5, 24, dtype=torch.bool), torch.ones(24, dtype=torch.long)
    model(x)
    _BARRIER.wait(timeout=60)
    count, started = 0, time.perf_counter()
    with torch.no_grad():
        while time.perf_counter() - started < seconds:
            grid = np.full((24, 24), 1000., dtype=np.float32)
            grid[12, 12] = 0
            # A short 24x24 min-plus stencil, not a claim about game throughput.
            for _ in range(24):
                grid[1:] = np.minimum(grid[1:], grid[:-1] + 1)
                grid[:-1] = np.minimum(grid[:-1], grid[1:] + 1)
                grid[:, 1:] = np.minimum(grid[:, 1:], grid[:, :-1] + 1)
                grid[:, :-1] = np.minimum(grid[:, :-1], grid[:, 1:] + 1)
            map_assignment(model(x), caps, legal)
            count += 1
    return dict(events=count, seconds=time.perf_counter() - started)


def worker_probe(config, limits):
    cap = limits["selected_workers"]
    counts = sorted(set(min(count,cap) for count in (1,2,4,6,8,cap)))
    context = multiprocessing.get_context("spawn")
    reports = []
    for count in counts:
        barrier = context.Barrier(count)
        begin = time.perf_counter()
        with ProcessPoolExecutor(max_workers=count, mp_context=context,
                                 initializer=_pool_initialize, initargs=(barrier,)) as pool:
            futures = [pool.submit(_cpu_worker, config.get("benchmark_worker_seconds", .4)) for _ in range(count)]
            rows = [future.result(timeout=75) for future in futures]
        throughput = sum(r["events"] for r in rows) / max(r["seconds"] for r in rows)
        reports.append(dict(workers=count, events_per_second=throughput,
                            startup_and_probe_seconds=time.perf_counter()-begin, valid=True))
    best = max(row["events_per_second"] for row in reports)
    selected = min(row["workers"] for row in reports if row["events_per_second"] >= .95*best)
    return dict(selected_workers=selected, probes=reports,
                measured_game_throughput=False,
                selection="fastest synthetic aggregate; within 5% prefer fewer workers")


def calibrate(directory, config, progress=lambda **_: None):
    hardware = hardware_identity()
    limits = budget(config, hardware)
    devices = ["cpu"] + (["cuda"] if hardware["cuda_available"] else [])
    if hardware["mps_available"]:
        devices.append("mps")
    requested = config.get("device", "auto")
    if requested != "auto" and requested not in devices:
        raise RuntimeError(f"Requested learner device {requested} is unavailable: {hardware}")
    repeats = int(config.get("benchmark_repeats", 8))
    inference, learners = [], []
    with torch.random.fork_rng():
        torch.manual_seed(20261026)
        for device in devices:
            for threads in (config.get('benchmark_cpu_threads',[1,2]) if device == "cpu" else [1]):
                progress(phase="synthetic_inference_probe", device=device, threads=threads)
                inference.append(inference_probe(device, threads, repeats))
                if requested not in ("auto", device):
                    continue
                for batch in config.get("benchmark_batches", [64, 256]):
                    progress(phase="synthetic_learner_probe", device=device, threads=threads, batch=batch)
                    try:
                        learners.append(learner_probe(device, threads, batch, max(3, repeats//2)))
                    except RuntimeError as exc:
                        learners.append(dict(device=device, threads=threads, batch_events=batch, valid=False, error=str(exc)))
                        if device.startswith("cuda"):
                            torch.cuda.empty_cache()
        progress(phase="synthetic_parallel_probe")
        workers = worker_probe(config, limits)
        progress(phase='synthetic_public_policy_probe')
        public_policy = policy_probe(config)
    valid = [row for row in learners if row["valid"]]
    if not valid:
        raise RuntimeError("No valid synthetic learner probe")
    best = max(valid, key=lambda r:r["events_per_second"])
    actor = min(inference, key=lambda r:r["p50_ms"])
    # Per-team GPU actors would contend with Unity rendering and the learner.
    # A GPU must beat the CPU by 20% including transfers before it is selected.
    cpu_actor = min((r for r in inference if r["device"] == "cpu"), key=lambda r:r["p50_ms"])
    if actor["device"] != "cpu" and actor["p50_ms"] > .8*cpu_actor["p50_ms"]:
        actor = cpu_actor
    parallel = min(limits["parallel_runs"], workers["selected_workers"])
    workers_per_run = max(1, workers['selected_workers']//parallel)
    profile = dict(schema="v12.runtime.1", hardware=hardware, measured_at=time.time(),
                   synthetic_only=True, unity_started=False, persisted_training_steps=0,
                   inference=inference, learner=learners, workers=workers, public_policy=public_policy,
                   selected=dict(actor_device="cpu", measured_fastest_actor_device=actor["device"],
                                 planner_device="cpu", actor_threads=1,
                                 learner_device=best["device"], learner_threads=best["threads"],
                                 minibatch_events=best["batch_events"],
                                 workers=workers['selected_workers'], parallel_runs=parallel,
                                 workers_per_run=workers_per_run,
                                 measured_worker_cap=workers['selected_workers']),
                   limitations=["Synthetic worker scaling excludes Unity rendering, IPC and opponent cost.",
                                "CPU policy/planner avoids per-worker GPU contexts; CUDA reserved for rendering and batched learning.",
                                "Full study records measured episode throughput without silently changing registered worker count."])
    profile["signature"] = digest(dict(hardware=hardware, config=config))
    atomic_json(directory / "runtime_profile.json", profile)
    torch.set_num_threads(1)
    return profile


def apply_profile(config, profile):
    resolved = copy.deepcopy(config)
    selected = profile["selected"]
    resolved.update(selected)
    resolved["device"] = selected["learner_device"]
    resolved["ppo"]["minibatch_events"] = selected["minibatch_events"]
    return resolved
