from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'shared'))
from project_paths import activate
activate()

if __name__=='__main__':
    from blackout_v13.migrate_evaluation import main
    main()
