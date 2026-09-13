"""The thesis DSL package — import the models from :mod:`seikan.dsl.schema`."""

# The Series and Condition vocabularies are mutually recursive and are rebuilt together in
# ``conditions`` (see its bottom). Importing it here means no consumer can ever reach a
# half-built ``seikan.dsl.nodes`` model, whichever submodule it imports first.
from seikan.dsl import conditions as _conditions  # noqa: F401
