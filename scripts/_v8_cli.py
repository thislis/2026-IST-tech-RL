"""The old research CLI is retired; the supplied environment is unmodified."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

def setup(config_path):
    from blackout_rl.v8.provided_environment import reject_research_environment
    reject_research_environment()

def make_env(config,run_id):
    from blackout_rl.v8.provided_environment import reject_research_environment
    reject_research_environment()
