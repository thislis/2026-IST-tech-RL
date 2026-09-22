"""Bounded collection benchmark. Never updates a policy or writes production runs."""
import argparse
import cProfile
import hashlib
import io
import json
from pathlib import Path
import pstats
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', default='configs/v7/main_study/v7_1_rewired.yaml')
    p.add_argument('--steps', type=int, default=128)
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--engine', action='store_true')
    p.add_argument('--profile', action='store_true')
    p.add_argument('--fast', choices=('queue', 'native', 'both'))
    p.add_argument('--mps-brain', action='store_true')
    p.add_argument('--barrier', type=Path)
    p.add_argument('--seconds', type=float)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    if a.fast:
        from scripts.connectome_fast_runtime import install
        install(native=a.fast in ('native','both'), queues=a.fast in ('queue','both'))
    import torch
    from blackout_rl.v7_registry import load_config, check_config, make_model
    from blackout_rl.v7_training import Collector
    from blackout_rl.env import ContractBlackOutEnv
    torch.set_num_threads(a.threads)
    if a.mps_brain:
        from scripts.connectome_metal import install
        install()
    cfg = load_config(a.config)
    graph, splits = check_config(cfg)
    torch.manual_seed(11)
    model = make_model(cfg, graph)
    env = ContractBlackOutEnv(env_path=str(ROOT/cfg['environment']['build']), background=True,
                             no_graphics=False, time_scale=cfg['environment']['time_scale'])
    if a.engine:
        env._engine_channel.set_configuration_parameters(width=128, height=128, target_frame_rate=-1)
    try:
        with tempfile.TemporaryDirectory(prefix='connectome-benchmark-') as tmp:
            collector = Collector(env, model, graph, cfg, splits, 11, Path(tmp))
            collector.collect(16)
            if a.barrier:
                import os
                a.barrier.with_name(a.barrier.name+f'.ready.{os.getpid()}').touch()
                deadline = time.monotonic()+120
                while not a.barrier.exists():
                    if time.monotonic()>deadline:raise TimeoutError('benchmark barrier')
                    time.sleep(.05)
            profiler = cProfile.Profile()
            start = time.perf_counter()
            if a.profile:
                profiler.enable()
            measured = 0
            while True:
                count = 64 if a.seconds else a.steps
                batch, _ = collector.collect(count)
                measured += count
                if a.seconds is None or time.perf_counter()-start>=a.seconds:break
            if a.profile:
                profiler.disable()
            elapsed = time.perf_counter() - start
            result = dict(config=a.config, steps=measured, seconds=elapsed, steps_per_second=measured/elapsed,
                          threads=a.threads, engine=a.engine, profile=a.profile, fast=a.fast,
                          mps_brain=a.mps_brain, training_updates=0,
                          batch_sha256={k:hashlib.sha256(v.numpy().tobytes()).hexdigest()
                                        for k,v in batch.items() if isinstance(v,torch.Tensor)})
            if a.profile:
                stream = io.StringIO()
                pstats.Stats(profiler, stream=stream).sort_stats('cumulative').print_stats(35)
                result['profile_text'] = stream.getvalue()
            output = Path(a.output)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(result, indent=2)+'\n')
            print(json.dumps(result, indent=2))
    finally:
        env.close()


if __name__ == '__main__':
    main()
