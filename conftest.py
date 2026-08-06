"""Shared test setup: point everything at a temporary SQLite database.

This must run before `app.*` is imported anywhere, so pytest loads it first.
"""

import os
import tempfile

_tmp = tempfile.mkdtemp(prefix="docchat-test-")
os.environ["DOCCHAT_DB_PATH"] = os.path.join(_tmp, "test.db")
os.environ["DOCCHAT_SEED_DEMO_USER"] = "false"
os.environ["DOCCHAT_SECRET_KEY"] = "test-secret-key-that-is-long-enough-1234567890"
