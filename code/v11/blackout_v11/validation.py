"""Small contract/algorithm checks; never launches Unity or a training study."""
import io
from pathlib import Path
import unittest


def validate_algorithms():
    tests=Path(__file__).parents[1]/'tests/v11'
    suite=unittest.defaultTestLoader.discover(str(tests),pattern='test_*.py')
    output=io.StringIO()
    result=unittest.TextTestRunner(stream=output,verbosity=1).run(suite)
    if not result.wasSuccessful():
        raise AssertionError('v11 essential checks failed:\n'+output.getvalue())
    return dict(passed=True,tests=result.testsRun,skipped=len(result.skipped),unity_started=False)
