"""The turtle command's own refusal classes."""

from __future__ import annotations


class TurtleRequestError(ValueError):
    """A request the turtle simulation cannot honor AS GIVEN — a thesis whose direction is not
    long, a thesis declaring no OHLCV target, a missing benchmark binding. The CLI maps it to the
    exit-3 ``usage`` envelope: nothing about the data can make such a request answerable."""
