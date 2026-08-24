"""Point-in-time StockBench data access.

`StockBenchAdapter.get_observations(as_of)` is the adapter's sole observation read
interface.  Callers never receive the unfiltered backing collection.
"""

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
    "MarketObservation",
    "ObservationSnapshot",
    "StockBenchAdapter",
    "FullSampleFitError",
    "TimeWindow",
    "TrainWindowStandardizer",
    "numeric_feature_bytes",
]
