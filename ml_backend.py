"""
ml_backend.py -- forces this project to be PyTorch-only wherever it
touches Hugging Face `transformers` (directly, or transitively through
`sentence_transformers` / `bertopic`), and reports on that choice without
ever importing TensorFlow itself.

Why this exists: this project's ENTIRE ML stack (NLLB via
AutoModelForSeq2SeqLM, the nlptown/BART sentiment models, BERTopic's
SentenceTransformer embeddings) runs on PyTorch. There is no TensorFlow
requirement anywhere in Step 2 or later stages. But `transformers` will
still probe for and import TensorFlow at ITS OWN import time unless told
not to -- and on at least one real reviewer machine (an Anaconda base
environment with an old TensorFlow build compiled for AVX instructions the
CPU doesn't have), that probe aborts the entire Python process outright
("TensorFlow library was compiled to use AVX instructions, but these
aren't available on your machine" -> `zsh: abort`), well before NLLB, the
sentiment models, or BERTopic ever get a chance to run. This is an
environment/import-isolation problem, not an NLLB or methodology problem
-- the fix is to never let `transformers` reach for TensorFlow in the
first place.

USAGE: `import ml_backend` as the FIRST import in any module that goes on
to import `transformers` (or `sentence_transformers` / `bertopic`, which
import it transitively) -- before `torch`, before `transformers`, before
anything else. The `os.environ.setdefault(...)` calls below only have an
effect if they run before `transformers` reads these variables at ITS
import time; importing this module after the fact is a no-op. Every
module in this project that eventually reaches `transformers`
(`preprocess_v2.py`, `sentiment_analysis.py`, `topic_modeling.py`,
`visualization.py`) does this import first, so the guard is in effect
regardless of which one Python happens to import first (e.g. via
`terminal.py`'s lazy getters).

`setdefault`, not plain assignment: an operator who has deliberately set
one of these themselves (e.g. to actually test a TensorFlow/Flax backend)
is not silently overridden.

VERIFIED against the project's actual pinned `transformers==4.33.2`
(reading `transformers/utils/import_utils.py` directly, not assuming):
with `USE_TF` set to anything outside {"1","ON","YES","TRUE","AUTO"} and
`USE_TORCH` set to anything in that set, `transformers` takes the `else`
branch of its TensorFlow-detection block and never even calls
`importlib.util.find_spec("tensorflow")` -- so it cannot discover, import,
or crash on a broken TensorFlow install, regardless of what's sitting in
the environment. `USE_TF`/`USE_TORCH`/`USE_FLAX` are the three variables
that actually do this work in that version. `TRANSFORMERS_NO_TF` is NOT a
variable `transformers==4.33.2` reads anywhere (confirmed by searching its
source) -- it is set anyway, harmlessly, as a defensive no-op in case a
different library or transformers version does honor it; the real
guarantee comes from the three variables above.
"""
import os

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")
os.environ.setdefault("USE_TORCH", "1")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")  # harmless no-op safety net -- see VERIFIED note above


def get_ml_backend_info() -> dict:
    """Reports the ML backend actually in effect, as reproducibility
    metadata for the Step 2 report / preflight output.

    Deliberately does NOT `import tensorflow` to check whether it's
    installed -- doing so would risk triggering the exact AVX-incompatible-
    build abort this module exists to prevent, just to report on it.
    `importlib.util.find_spec` locates a module (reads its location from
    sys.path/import machinery) WITHOUT executing any of its code, so it is
    safe to use even when a broken TensorFlow build sits in the
    environment.
    """
    import importlib.util

    try:
        import torch
        pytorch_available = True
        pytorch_version = torch.__version__
    except Exception:
        pytorch_available = False
        pytorch_version = None

    tensorflow_installed = importlib.util.find_spec("tensorflow") is not None

    return {
        "ml_backend": "pytorch",
        "pytorch_available": pytorch_available,
        "pytorch_version": pytorch_version,
        "tensorflow_installed": tensorflow_installed,
        "tensorflow_used": False,
        "env": {
            "USE_TF": os.environ.get("USE_TF"),
            "USE_FLAX": os.environ.get("USE_FLAX"),
            "USE_TORCH": os.environ.get("USE_TORCH"),
            "TRANSFORMERS_NO_TF": os.environ.get("TRANSFORMERS_NO_TF"),
        },
    }
