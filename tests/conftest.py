import os
import tempfile

# Set BEFORE any `app.*` import (config reads the environment at import time). Windows are shrunk so the drift
# tests run on ~100 documents instead of ~300; the statistics and decision rules are unchanged.
_tmp = tempfile.mkdtemp()
os.environ.update({
    "PIPELINE_DB": os.path.join(_tmp, "test.db"), "DATA_DIR": os.path.join(_tmp, "data"),
    "EXTRACTOR": "simulated", "EMBEDDED_WORKERS": "0", "SIM_LATENCY_MS": "0",
    "BACKOFF_BASE_S": "0.01", "BACKOFF_CAP_S": "0.05", "MAX_ATTEMPTS": "3", "CONFIG_ERROR_PAUSE_AFTER": "3",
    "BASELINE_N": "30", "WINDOW_N": "30", "MIN_SAMPLES": "15", "DRIFT_EVAL_EVERY": "10",
})
for k in ("API_KEY", "ALERT_WEBHOOK_URL", "GROK_API_KEY", "CROSS_CHECK"):
    os.environ.pop(k, None)

import uuid

import pytest

from app import db

db.init()


@pytest.fixture(autouse=True)
def clean_queue():
    """The queue is global across namespaces and drain loops wait for queued+running to reach zero, so a job
    left behind by one test would stall or pollute another. Namespaces isolate everything else."""
    def wipe():
        with db.tx() as c:
            c.execute("DELETE FROM jobs")
            c.execute("DELETE FROM kv WHERE key LIKE 'queue_%'")
    wipe()
    yield
    wipe()


@pytest.fixture()
def ns():
    return f"t-{uuid.uuid4().hex[:8]}"
