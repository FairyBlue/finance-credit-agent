"""Point-in-time StockBench data access.

`StockBenchAdapter.get_observations(as_of)` is the adapter's sole observation read
interface.  Callers never receive the unfiltered backing collection.
"""

from .calendar import validate_episode_session_boundaries
from .history import (
    HistoryManifest,
    HistoryWindow,
    StooqDailyBarProvider,
    YFinanceDailyBarProvider,
    materialize_daily_history,
)
from .integrity import (
    OverlapReconciliationAudit,
    RawManifestAudit,
    SessionCalendarAudit,
    audit_exchange_sessions,
    audit_raw_daily_manifest,
    verify_overlap_reconciliation,
    verify_raw_daily_manifest,
)
from .stockbench import (
    AvailabilityPolicy,
    MarketObservation,
    ObservationSnapshot,
    StockBenchAdapter,
)
from .transforms import (
    FullSampleFitError,
    TimeWindow,
    TrainWindowStandardizer,
    numeric_feature_bytes,
)

__all__ = [
    "AvailabilityPolicy",
    "HistoryManifest",
    "HistoryWindow",
    "MarketObservation",
    "ObservationSnapshot",
    "OverlapReconciliationAudit",
    "RawManifestAudit",
    "SessionCalendarAudit",
    "StockBenchAdapter",
    "StooqDailyBarProvider",
    "FullSampleFitError",
    "TimeWindow",
    "TrainWindowStandardizer",
    "audit_raw_daily_manifest",
    "audit_exchange_sessions",
    "numeric_feature_bytes",
    "YFinanceDailyBarProvider",
    "materialize_daily_history",
    "verify_overlap_reconciliation",
    "verify_raw_daily_manifest",
    "validate_episode_session_boundaries",
]
