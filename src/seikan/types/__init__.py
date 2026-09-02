"""The shapes the engine's dicts actually have — the pipeline contract, stated once.

Every number this package computes travels between its modules as a plain dict: the runner
assembles the per-cell panel, ``analysis.stats`` builds the blocks that hang off it, the gate
READS those blocks by key, and ``serialize``/``cli`` emit the result as one JSON document. That
contract is real and load-bearing — a renamed key silently deletes a check — so it is written
down here as ``TypedDict`` declarations rather than left implicit in a hundred dict literals.

A LEAF module by design, exactly like :mod:`seikan.constants`: it imports nothing from the rest
of the package (stdlib typing machinery only at runtime), so any module — ``analysis.stats``,
``compiler.runner``, ``gate``, ``serialize``, ``cli`` — can import it without a cycle.

Three reading rules, so an annotation here is never mistaken for a claim the engine does not make:

- **The unions state the EMITTED contract.** These shapes are validated at the emission seam
  (``emitted.py``) over the ``json_safe``'d document, where every non-finite float has become
  ``null`` — so a field that can be NaN in memory is typed ``T | None``, and the same TypedDict
  still types the in-memory builder literals (a ``float`` write is compatible with
  ``float | None``). Whether a given ``None`` means "declined to compute" (the bootstrap's
  interval bounds, a refusal block's value, a ``reason``-paired field) or "computed as
  undefined" (an empty pool's NaN) is stated in each class's own docstring — the type no longer
  distinguishes them, the prose does.
- **These describe the emitted dicts** the runner produces, the gate reads, and the seam
  validates.
- **Optional means optional.** ``NotRequired`` marks a key that is genuinely absent in some legal
  runs — ``pooled`` on a conjunction cell, ``rows_written`` on the report's own output entry, a
  document section another subcommand does not emit — never a key that merely might be missing
  from drifted input. The gate treats a missing REQUIRED key as drifted input and refuses; that
  refusal is its job, not this module's.

One module per report section behind this facade: ``scalars``, ``cells``, ``pools``, ``run``,
``reliability``, ``data_report``, ``describe``, ``documents`` and ``gate_section``.
"""

from seikan.types.cells import (
    CellOutcomeCoverage,
    CellPooledPanel,
    CellSignalCoverage,
    CellTargetPanel,
    DeclaredCell,
    SummaryCell,
)
from seikan.types.data_report import (
    DataIssue,
    DataIssueCode,
    DataIssueExtras,
    DataReport,
    FileReportEntry,
    IssueExample,
    JoinInfo,
)
from seikan.types.describe import (
    ChangeBlock,
    DescribeResult,
    ExtremumPoint,
    FileProfile,
    FullSampleBlock,
    LastBarBlock,
    MissingnessBlock,
    RangeDistance,
    RangePositionBlock,
    RefusalStub,
    SeriesProfile,
    VolumeBlock,
    VolumeWindow,
)
from seikan.types.documents import (
    DataDigest,
    EmittedDocument,
    ErrorEnvelope,
    OutputEntry,
    ReportIdentity,
    SerializedResult,
    ThresholdDoc,
    ThresholdProvenance,
    ThresholdsSnapshot,
    ValidationRecord,
)
from seikan.types.gate_section import (
    GateCellDict,
    GateCheckDict,
    GateSection,
)
from seikan.types.pools import (
    BenchmarkRegressionBlock,
    BenchmarkRegressionReason,
    BucketMonotonicity,
    BucketRecord,
    CellBucketPanels,
    ConcentrationBlock,
    EpisodeBootstrapCI,
    EpisodeLedgerBlock,
    EpisodeLedgerEntry,
    EpisodeProfileBlock,
    EpisodeStatsBlock,
    FeatureAssociation,
    FeatureBucketPanel,
    MaeQuantiles,
    MemberShareBlock,
    MfeQuantiles,
    PoolMoments,
    PoolQuantiles,
    SubperiodCounts,
    SubperiodEntry,
    TimingBlock,
)
from seikan.types.reliability import (
    DegradationSlopeReason,
    PboBlock,
    ReliabilityCell,
    ReliabilityRead,
    ReliabilitySummary,
)
from seikan.types.run import (
    BarSpacing,
    BaselineEntry,
    BaselinePool,
    BaselineStats,
    CrossBreadthEntry,
    OutcomeStamp,
    RotationStamp,
    RunSummary,
    SourceAvailability,
    SourceCoverage,
)
from seikan.types.scalars import (
    EXCLUSION_REASONS,
    EXIT_REASONS,
    OUTCOME_KINDS,
    TARGET_MODES,
    BenchmarkMode,
    CellKey,
    ComboKey,
    Direction,
    DslDocument,
    EntryRow,
    EvidenceBasis,
    ExclusionReason,
    ExitReason,
    JsonValue,
    OutcomeKind,
    OutcomeUnits,
    ParamValue,
    TargetMode,
    TargetShape,
)

__all__ = [
    "EXCLUSION_REASONS",
    "EXIT_REASONS",
    "OUTCOME_KINDS",
    "TARGET_MODES",
    "BarSpacing",
    "BaselineEntry",
    "BaselinePool",
    "BaselineStats",
    "BenchmarkMode",
    "BenchmarkRegressionBlock",
    "BenchmarkRegressionReason",
    "BucketMonotonicity",
    "BucketRecord",
    "CellBucketPanels",
    "CellKey",
    "CellOutcomeCoverage",
    "CellPooledPanel",
    "CellSignalCoverage",
    "CellTargetPanel",
    "ChangeBlock",
    "ComboKey",
    "ConcentrationBlock",
    "CrossBreadthEntry",
    "DataDigest",
    "DataIssue",
    "DataIssueCode",
    "DataIssueExtras",
    "DataReport",
    "DeclaredCell",
    "DegradationSlopeReason",
    "DescribeResult",
    "Direction",
    "DslDocument",
    "EmittedDocument",
    "EntryRow",
    "EpisodeBootstrapCI",
    "EpisodeLedgerBlock",
    "EpisodeLedgerEntry",
    "EpisodeProfileBlock",
    "EpisodeStatsBlock",
    "ErrorEnvelope",
    "EvidenceBasis",
    "ExclusionReason",
    "ExitReason",
    "ExtremumPoint",
    "FeatureAssociation",
    "FeatureBucketPanel",
    "FileProfile",
    "FileReportEntry",
    "FullSampleBlock",
    "GateCellDict",
    "GateCheckDict",
    "GateSection",
    "IssueExample",
    "JoinInfo",
    "JsonValue",
    "LastBarBlock",
    "MaeQuantiles",
    "MemberShareBlock",
    "MfeQuantiles",
    "MissingnessBlock",
    "OutcomeKind",
    "OutcomeStamp",
    "OutcomeUnits",
    "OutputEntry",
    "ParamValue",
    "PboBlock",
    "PoolMoments",
    "PoolQuantiles",
    "RangeDistance",
    "RangePositionBlock",
    "RefusalStub",
    "ReliabilityCell",
    "ReliabilityRead",
    "ReliabilitySummary",
    "ReportIdentity",
    "RotationStamp",
    "RunSummary",
    "SerializedResult",
    "SeriesProfile",
    "SourceAvailability",
    "SourceCoverage",
    "SubperiodCounts",
    "SubperiodEntry",
    "SummaryCell",
    "TargetMode",
    "TargetShape",
    "ThresholdDoc",
    "ThresholdProvenance",
    "ThresholdsSnapshot",
    "TimingBlock",
    "ValidationRecord",
    "VolumeBlock",
    "VolumeWindow",
]
