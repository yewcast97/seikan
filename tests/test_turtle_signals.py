"""The thesis → entry-cell translation stays bit-identical to the engine's own listing."""

from __future__ import annotations

import numpy as np
import pytest

from seikan.api import list_entries
from seikan.dsl.schema import Thesis
from seikan.turtle import TurtleRequestError, entry_cells
from tests._helpers import load
from tests._turtle_helpers import first_true_above, thesis_doc, worked_example_rows, write_bars


def test_one_combo_one_target_fires_on_the_signal_bar(tmp_path):
    px = write_bars(tmp_path / "px.csv", worked_example_rows())
    thesis = Thesis.model_validate(thesis_doc(["PX"], first_true_above(50.0)))
    md = load(thesis, {"PX": str(px)})
    axes, cells = entry_cells(thesis, md)
    assert axes == [] and len(cells) == 1
    cell = cells[0]
    assert cell.cell_id == "entry" and cell.params == {}
    fired = cell.fired["PX"]
    assert fired.dtype == bool and fired.shape == (31,)
    assert np.flatnonzero(fired).tolist() == [24]


def test_swept_axes_expand_to_one_cell_per_combo_in_listing_order(tmp_path):
    rows = worked_example_rows()
    px = write_bars(tmp_path / "px.csv", rows)
    doc = thesis_doc(
        ["PX"],
        {
            "type": "threshold",
            "left": {"type": "field", "column": "close"},
            "op": ">",
            "right": {
                "type": "rolling_agg",
                "window": {"axis": "N"},
                "agg": "mean",
                "input": {"type": "field"},
            },
        },
    )
    doc["axes"] = {"N": [3, 5]}
    thesis = Thesis.model_validate(doc)
    md = load(thesis, {"PX": str(px)})
    axes, cells = entry_cells(thesis, md)
    listing = list_entries(thesis, md)
    assert axes == ["N"]
    assert (
        [c.cell_id for c in cells]
        == list(listing.entry_flags.columns)
        == ["entry[N=3]", "entry[N=5]"]
    )
    assert [c.params for c in cells] == [{"N": 3}, {"N": 5}]
    for cell in cells:
        expected = listing.entry_flags[cell.cell_id].to_numpy().astype(bool)
        assert np.array_equal(cell.fired["PX"], expected)


def test_several_targets_strip_the_target_suffix(tmp_path):
    rows = worked_example_rows()
    a = write_bars(tmp_path / "a.csv", rows)
    b = write_bars(tmp_path / "b.csv", [(o + 10, h + 10, lo + 10, c + 10) for o, h, lo, c in rows])
    thesis = Thesis.model_validate(thesis_doc(["A", "B"], first_true_above(50.0)))
    md = load(thesis, {"A": str(a), "B": str(b)})
    _axes, cells = entry_cells(thesis, md)
    assert len(cells) == 1 and cells[0].cell_id == "entry"
    assert set(cells[0].fired) == {"A", "B"}
    assert np.flatnonzero(cells[0].fired["A"]).tolist() == [24]
    # B sits above 50 from bar 0: first_true needs a false→true edge, so it never fires.
    assert np.flatnonzero(cells[0].fired["B"]).tolist() == []


def test_a_final_bar_firing_is_present(tmp_path):
    rows = [*worked_example_rows()[:24], (49.5, 51.0, 49.0, 50.0)]
    px = write_bars(tmp_path / "px.csv", rows)
    thesis = Thesis.model_validate(thesis_doc(["PX"], first_true_above(50.0)))
    md = load(thesis, {"PX": str(px)})
    _axes, cells = entry_cells(thesis, md)
    assert cells[0].fired["PX"][-1]


def test_a_short_thesis_refuses(tmp_path):
    px = write_bars(tmp_path / "px.csv", worked_example_rows())
    thesis = Thesis.model_validate(
        thesis_doc(["PX"], first_true_above(50.0), direction="shortonly")
    )
    md = load(thesis, {"PX": str(px)})
    with pytest.raises(TurtleRequestError, match="shortonly"):
        entry_cells(thesis, md)
