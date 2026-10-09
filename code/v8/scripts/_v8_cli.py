"""The old research CLI is retired; the supplied environment is unmodified."""

# Locate shared modules from the physical versioned layout.
from pathlib import Path as _LayoutPath
import sys as _layout_sys
_layout_root = next(p for p in _LayoutPath(__file__).resolve().parents
                    if (p / "code/shared/project_paths.py").is_file())
_layout_sys.path.insert(0, str(_layout_root / "code/shared"))
from project_paths import activate as _activate_layout
_activate_layout()
from project_paths import project_root, project_path

from pathlib import Path
import sys
ROOT=project_root()
sys.path.insert(0,str(ROOT))

def setup(config_path):
    from blackout_rl.v8.provided_environment import reject_research_environment
    reject_research_environment()

def make_env(config,run_id):
    from blackout_rl.v8.provided_environment import reject_research_environment
    reject_research_environment()
