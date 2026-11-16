"""LanceDB filter expression for one qasper paper.

doc_id values in these corpora are identifiers, not free text. Anything
outside the safe alphabet is rejected instead of being interpolated.
"""

from __future__ import annotations

import re

_SAFE = re.compile(r"[A-Za-z0-9_.:\-]+")


def lance_where(doc_id: str) -> str:
    if not isinstance(doc_id, str) or not _SAFE.fullmatch(doc_id):
        raise ValueError("doc_id is not a safe identifier")
    return f"doc_id = '{doc_id}'"
