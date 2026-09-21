# This file is intentionally empty.
#
# Its presence at the project root tells pytest that this directory is
# the rootdir, which makes pytest add it to sys.path automatically.
# Without it, running plain `pytest` (rather than `python -m pytest`)
# can fail with:
#     ModuleNotFoundError: No module named 'app'
# because `app/` wouldn't otherwise be importable from tests/test_all.py.
