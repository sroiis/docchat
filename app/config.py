"""Tiny config layer, read from environment variables with sane defaults.

Kept trivially small on purpose — it's the one place to look for knobs.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    # Directory of .md/.txt files auto-indexed on startup.
    docs_dir: str = os.getenv("DOCCHAT_DOCS_DIR", "sample_docs")
    # Default number of chunks returned per question.
    default_k: int = int(os.getenv("DOCCHAT_DEFAULT_K", "4"))


settings = Settings()
