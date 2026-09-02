"""The canonical DSL identity: defaults filled, keys sorted, sha256."""

from __future__ import annotations

import hashlib
import json

from seikan.dsl.schema import Thesis as DslThesis
from seikan.types import (
    DslDocument,
)


def canonical_dsl_hash(dsl: DslDocument) -> str:
    """sha256 over the *normalized* thesis DSL (defaults filled, keys sorted).

    Normalizing through the ``Thesis`` model makes the hash insensitive to key order / omitted
    defaults, so two spellings of the same rules share one identity. Raises (pydantic
    ``ValidationError``) on an invalid DSL.
    """
    normalized = DslThesis.model_validate(dsl).model_dump(mode="json")
    # allow_nan=False is a backstop: the schema already rejects non-finite numbers, and a hash
    # over the invalid-JSON tokens `NaN`/`Infinity` would name an identity no strict parser
    # could reconstruct.
    payload = json.dumps(normalized, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
