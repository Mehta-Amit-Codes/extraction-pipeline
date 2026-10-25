"""Central config. Every threshold here is a starting point to be tuned on YOUR documents."""
import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

ROOT = Path(__file__).resolve().parent.parent


def _f(name: str, default: str) -> float:
    return float(os.environ.get(name, default))


def _i(name: str, default: str) -> int:
    return int(os.environ.get(name, default))


# --- Storage ---------------------------------------------------------------------------
DB_PATH = os.environ.get("PIPELINE_DB", str(ROOT / "pipeline.db"))
DATA_DIR = Path(os.environ.get("DATA_DIR", ROOT / "data"))

# --- Untrusted-input limits ------------------------------------------------------------------
MAX_UPLOAD_MB = _i("MAX_UPLOAD_MB", "20")
MAX_PAGES = _i("MAX_PAGES", "3")
MAX_PIXELS = _i("MAX_PIXELS", "50000000")
MAX_SIDE_PX = _i("MAX_SIDE_PX", "1600")      # downscale before sending to the model: bounds cost and latency

# --- Extraction --------------------------------------------------------------------------
EXTRACTOR = os.environ.get("EXTRACTOR", "simulated")             # "grok" for the real vision model
EXTRACTION_MODEL = os.environ.get("EXTRACTION_MODEL", "grok-4")
CROSSCHECK_MODEL = os.environ.get("CROSSCHECK_MODEL", "")        # blank = same model, different prompt
GROK_BASE_URL = os.environ.get("GROK_BASE_URL", "https://api.x.ai/v1")
CROSS_CHECK = os.environ.get("CROSS_CHECK", "always")            # always | selective | never
SELECTIVE_CONF = _f("SELECTIVE_CONF", "0.95")                    # selective: pass 2 if any pass-1 field is below this
REVIEW_THRESHOLD = _f("REVIEW_THRESHOLD", "0.85")                # any field below this -> human review
DISPUTE_PENALTY = _f("DISPUTE_PENALTY", "0.5")                   # confidence multiplier when the two passes disagree
ARITH_TOLERANCE = _f("ARITH_TOLERANCE", "0.011")                 # money mismatch tolerated (rounding)
# Illustrative rates per 1M tokens: update to current published pricing before trusting cost figures.
COST_PER_M_INPUT = _f("COST_PER_M_INPUT", "3.0")
COST_PER_M_OUTPUT = _f("COST_PER_M_OUTPUT", "15.0")

# --- Active learning ------------------------------------------------------------------------
FEWSHOT_K = _i("FEWSHOT_K", "2")
FEWSHOT_MAX_DIST = _i("FEWSHOT_MAX_DIST", "14")                  # only show examples whose layout is this close (of 64 bits)
AUDIT_SAMPLE_RATE = _f("AUDIT_SAMPLE_RATE", "0.03")              # share of auto-approved docs sent to humans anyway

# --- Queue / retries ----------------------------------------------------------------------
MAX_ATTEMPTS = _i("MAX_ATTEMPTS", "5")
BACKOFF_BASE_S = _f("BACKOFF_BASE_S", "2")
BACKOFF_CAP_S = _f("BACKOFF_CAP_S", "300")
# Must exceed worst-case processing: 2 passes x (1 call + 1 repair call) x 90s API timeout = 360s. A worker that
# outlives its lease has its result discarded (fenced), which wastes the model spend, so keep real headroom.
LEASE_S = _i("LEASE_S", "600")
RECLAIM_EVERY_S = _f("RECLAIM_EVERY_S", "30")                    # how often workers look for jobs whose worker died

# --- Worker -----------------------------------------------------------------------------
EMBEDDED_WORKERS = _i("EMBEDDED_WORKERS", "2")                   # worker threads inside the API process (0 = run `python -m app.cli worker`)
RATE_LIMIT_RPS = _f("RATE_LIMIT_RPS", "0")                       # model calls/sec across all threads (0 = unlimited)
CONFIG_ERROR_PAUSE_AFTER = _i("CONFIG_ERROR_PAUSE_AFTER", "3")   # consecutive auth/config errors before pausing the queue
SIM_LATENCY_MS = _i("SIM_LATENCY_MS", "0")

# --- Drift detection ------------------------------------------------------------------------
BASELINE_N = _i("BASELINE_N", "60")                              # first N extractions form the baseline window
WINDOW_N = _i("WINDOW_N", "60")                                  # most recent N form the current window
MIN_SAMPLES = _i("MIN_SAMPLES", "30")                            # never test with fewer than this per window
DRIFT_ALPHA = _f("DRIFT_ALPHA", "0.01")                          # strict: many metrics are tested repeatedly
MIN_EFFECT_PROP = _f("MIN_EFFECT_PROP", "0.10")                  # a shift must be statistically AND practically significant
MIN_EFFECT_CONF = _f("MIN_EFFECT_CONF", "0.05")
DRIFT_EVAL_EVERY = _i("DRIFT_EVAL_EVERY", "20")                  # evaluate after this many new extractions
LAYOUT_MIN_T = _i("LAYOUT_MIN_T", "6")                           # floor for the adaptive "novel layout" distance

ALERT_WEBHOOK_URL = os.environ.get("ALERT_WEBHOOK_URL", "")      # Slack-style incoming webhook (optional)
API_KEY = os.environ.get("API_KEY", "")                          # if set, every request needs X-API-Key
