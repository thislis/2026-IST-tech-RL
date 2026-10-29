"""One explicit experiment job; imported checks never execute main()."""
import argparse
from pathlib import Path
import traceback
from .io import atomic_json, read_json


def execute(task):
    kind, cfg = task["kind"], task["config"]
    if kind == "train":
        from .training import train
        return train(task["output"], cfg, task["steps"], task["seed"], task["arm"],
                     initial_checkpoint=task.get("initial_checkpoint"), workers=task["workers"],
                     learner_device=cfg["learner_device"])
    if kind == "data":
        from .demonstrations import collect
        return collect(task["output"], dict(cfg, split="data"), task["steps"], task["seed"], task["workers"])
    if kind == "evaluate":
        from .evaluation import evaluate
        return evaluate(task["output"], dict(cfg, decoder=task.get("decoder", "map")),
                        task.get("checkpoint"), task["arm"], task["opponents"], task["pairs"],
                        task["split"], workers=task["workers"])
    raise ValueError("Unknown job kind: " + kind)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    args = parser.parse_args()
    path = Path(args.task)
    try:
        atomic_json(path.with_name("job_result.json"), execute(read_json(path)))
    except BaseException as exc:
        atomic_json(path.with_name("job_error.json"), dict(error=str(exc), traceback=traceback.format_exc()))
        raise


if __name__ == "__main__":
    main()
