"""The canonical DSL identity: defaults filled, ``null``-valued optional fields dropped, keys
sorted, sha256."""

from __future__ import annotations

import hashlib
import json

from seikan.dsl.schema import Thesis as DslThesis
from seikan.types import (
    DslDocument,
)


def _canonical_payload(dsl: DslDocument) -> str:
    """The exact byte string :func:`canonical_dsl_hash` digests — the *normalized* thesis DSL.

    Normalizing through the ``Thesis`` model fills every default, and the dump drops every
    optional field left at ``None`` (``exclude_none``), so an omitted optional field and its
    explicit ``null`` spelling — and a document written before the field existed — all
    canonicalize to one payload. Keys are sorted and separators compact, so key order and
    whitespace never move the identity. Raises (pydantic ``ValidationError``) on an invalid DSL.
    """
    normalized = DslThesis.model_validate(dsl).model_dump(mode="json", exclude_none=True)
    # allow_nan=False is a backstop: the schema already rejects non-finite numbers, and a hash
    # over the invalid-JSON tokens `NaN`/`Infinity` would name an identity no strict parser
    # could reconstruct.
    return json.dumps(normalized, sort_keys=True, separators=(",", ":"), allow_nan=False)


def canonical_dsl_hash(dsl: DslDocument) -> str:
    """sha256 over the *normalized* thesis DSL (defaults filled, ``null``-valued optional fields
    dropped, keys sorted).

    Two spellings of the same rules share one identity, and — because a ``None``-defaulted
    optional field contributes nothing until it is set — adding such a field to a model, or a
    new node type, moves no existing hash. A non-``None`` default still moves every hash.
    """
    return hashlib.sha256(_canonical_payload(dsl).encode("utf-8")).hexdigest()
