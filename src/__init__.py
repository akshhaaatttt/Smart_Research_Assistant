"""Smart Research Assistant - RAG package.

Importing the ``src`` package applies the HuggingFace/TensorFlow environment fix
as an early side effect (see :mod:`src.utils.env`). Because virtually every entry
point imports something under ``src`` before touching ``transformers`` /
``sentence_transformers`` / ``langchain``, this guarantees the PyTorch-only code
path is selected process-wide without each module having to remember to do it.
"""

from src.utils import env as _env  # noqa: F401  (import for side effects)

__all__ = ["__version__"]
__version__ = "0.1.0"
