"""Shared exchange-session boundary validation for manifests and environments."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timezone

import exchange_calendars as xcals


def validate_episode_session_boundaries(
    *,
    session_closes: Sequence[datetime],
    session_opens: Sequence[datetime],
    exchange_calendar: str,
) -> None:
    """Require each timestamp to equal the exchange's exact open/close instant."""

    if len(session_closes) != len(session_opens):
        raise ValueError("episode session opens and closes must align one-to-one")
    calendar = xcals.get_calendar(exchange_calendar)
    for close_at, open_at in zip(session_closes, session_opens, strict=True):
        if close_at.tzinfo is None or close_at.utcoffset() is None:
            raise ValueError("episode close must be timezone-aware")
        if open_at.tzinfo is None or open_at.utcoffset() is None:
            raise ValueError("episode open must be timezone-aware")
        session = close_at.astimezone(calendar.tz).date().isoformat()
        if not calendar.is_session(session):
            raise ValueError(f"{session} is not an {exchange_calendar} session")
        expected_close = calendar.session_close(session).to_pydatetime().astimezone(timezone.utc)
        expected_open = calendar.session_open(session).to_pydatetime().astimezone(timezone.utc)
        if close_at.astimezone(timezone.utc) != expected_close:
            raise ValueError(
                f"episode close {close_at.isoformat()} must equal {exchange_calendar} "
                f"session close {expected_close.isoformat()}"
            )
        if open_at.astimezone(timezone.utc) != expected_open:
            raise ValueError(
                f"episode open {open_at.isoformat()} must equal {exchange_calendar} "
                f"session open {expected_open.isoformat()}"
            )
