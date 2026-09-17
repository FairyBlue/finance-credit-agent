"""Agent-R1 plugin integration exercised with deterministic mock actions."""

import asyncio
import json
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd
import pytest
from agent_r1.env import ToolEnv
from agent_r1.env.base import Action

import src.env.trading as trading_module
from src.data import MarketObservation, StockBenchAdapter, audit_raw_daily_manifest
from src.data.stockbench import dataframe_content_sha256
from src.env.trading import RetrieveArgs, TradingEnv, _cached_storage_adapter

UTC = timezone.utc


def _bar(day: int, *, open_price: float, close_price: float) -> MarketObservation:
    available_at = datetime(2025, 1, day, 21, 0, tzinfo=UTC)
    return MarketObservation(
        observation_id=f"bar-{day}",
        kind="market",
        symbol="AAA",
        event_at=available_at,
        available_at=available_at,
        payload={
            "date": f"2025-01-{day:02d}",
            "open": open_price,
            "close": close_price,
        },
    )


def _call(name: str, arguments: str = "{}") -> Action:
    return Action(text=f'<tool_call>{{"name":"{name}","arguments":{arguments}}}</tool_call>')


def _write_daily_bar(root: Path, session: date, close: float) -> None:
    day_root = root / "parquet" / "AAA" / "day"
    day_root.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(
        {
            "date": [session],
            "open": [close],
            "high": [close],
            "low": [close],
            "close": [close],
            "volume": [1000.0],
        }
    )
    path = day_root / f"{session.isoformat()}.parquet"
    frame.to_parquet(path, index=False)
    path.with_suffix(".parquet.sha256").write_text(
        dataframe_content_sha256(
            frame,
            sort_by=[column for column in frame.columns if column != "date"],
        ),
        encoding="utf-8",
    )


def test_rollout_envs_share_one_immutable_storage_adapter(tmp_path: Path) -> None:
    _write_daily_bar(tmp_path, date(2025, 1, 2), 101.0)
    audit = audit_raw_daily_manifest(
        manifest_root=tmp_path,
        storage_root=tmp_path,
        symbols=["AAA"],
        start=date(2025, 1, 2),
        end=date(2025, 1, 2),
    )
    _cached_storage_adapter.cache_clear()
    kwargs = {
        "manifest_root": str(tmp_path),
        "storage_root": str(tmp_path),
        "symbols": ("AAA",),
        "data_version": "fixture-cache",
        "data_start": "2025-01-02",
        "data_end": "2025-01-02",
        "input_manifest_sha256": audit.manifest_sha256,
        "input_file_count": audit.file_count,
        "input_sessions_per_symbol": 1,
        "exchange_calendar": "XNYS",
        "market_timezone": "America/New_York",
        "market_close_time": "16:00:00",
    }

    first = _cached_storage_adapter(**kwargs)
    second = _cached_storage_adapter(**kwargs)

    assert first is second
    _cached_storage_adapter.cache_clear()


def test_worker_binds_ingestion_to_bytes_verified_by_raw_manifest(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _write_daily_bar(tmp_path, date(2025, 1, 2), 101.0)
    frozen = audit_raw_daily_manifest(
        manifest_root=tmp_path,
        storage_root=tmp_path,
        symbols=["AAA"],
        start=date(2025, 1, 2),
        end=date(2025, 1, 2),
    )

    real_verify = trading_module.verify_raw_daily_manifest

    def verify_then_replace(**kwargs: object) -> object:
        audit = real_verify(**kwargs)  # type: ignore[arg-type]
        # Simulate replacement after the worker aggregate audit but before
        # dataframe ingestion, including a matching replacement sidecar.
        _write_daily_bar(tmp_path, date(2025, 1, 2), 999.0)
        return audit

    monkeypatch.setattr(trading_module, "verify_raw_daily_manifest", verify_then_replace)
    _cached_storage_adapter.cache_clear()
    with pytest.raises(RuntimeError, match="raw daily-bar SHA-256 mismatch"):
        _cached_storage_adapter(
            manifest_root=str(tmp_path),
            storage_root=str(tmp_path),
            symbols=("AAA",),
            data_version="fixture-toctou",
            data_start="2025-01-02",
            data_end="2025-01-02",
            input_manifest_sha256=frozen.manifest_sha256,
            input_file_count=frozen.file_count,
            input_sessions_per_symbol=1,
            exchange_calendar="XNYS",
            market_timezone="America/New_York",
            market_close_time="16:00:00",
        )
    _cached_storage_adapter.cache_clear()


def test_trading_env_passes_the_frozen_source_window_to_storage_ingestion(
    tmp_path: Path,
) -> None:
    _write_daily_bar(tmp_path, date(2025, 1, 2), 100.0)
    _write_daily_bar(tmp_path, date(2025, 1, 3), 101.0)
    _write_daily_bar(tmp_path, date(2025, 1, 4), 999.0)  # Saturday, outside the frozen input.
    audit = audit_raw_daily_manifest(
        manifest_root=tmp_path,
        storage_root=tmp_path,
        symbols=["AAA"],
        start=date(2025, 1, 2),
        end=date(2025, 1, 3),
    )

    env = TradingEnv(
        episode_dates=["2025-01-02T21:00:00+00:00", "2025-01-03T21:00:00+00:00"],
        episode_open_times=["2025-01-02T14:30:00+00:00", "2025-01-03T14:30:00+00:00"],
        symbols=["AAA"],
        initial_cash=1000.0,
        commission_bps=10.0,
        slippage_bps=5.0,
        drawdown_coefficient=0.0,
        turnover_coefficient=0.0,
        benchmark_closes=[100.0, 101.0],
        manifest_root=tmp_path,
        storage_root=tmp_path,
        data_version="windowed-fixture",
        data_start="2025-01-02",
        data_end="2025-01-03",
        input_manifest_sha256=audit.manifest_sha256,
        input_file_count=audit.file_count,
        input_sessions_per_symbol=2,
        market_timezone="America/New_York",
        market_close_time="16:00:00",
    )

    assert "2025-01-02" in env.reset().messages[-1]["content"]


def test_mock_agent_runs_multiple_episodes_with_next_open_fills_and_safe_errors() -> None:
    dates = [datetime(2025, 1, day, 21, 0, tzinfo=UTC) for day in (2, 3, 6)]
    open_times = [datetime(2025, 1, day, 14, 30, tzinfo=UTC) for day in (2, 3, 6)]
    future_fundamental = MarketObservation(
        observation_id="future-fundamental",
        kind="fundamental",
        symbol="AAA",
        event_at=dates[2],
        available_at=dates[2],
        payload={"text": "future surprise"},
    )
    adapter = StockBenchAdapter(
        [
            _bar(2, open_price=100.0, close_price=105.0),
            _bar(3, open_price=110.0, close_price=112.0),
            _bar(6, open_price=115.0, close_price=114.0),
            future_fundamental,
        ],
        data_version="fixture-v1",
    )
    env = TradingEnv(
        adapter=adapter,
        episode_dates=dates,
        episode_open_times=open_times,
        symbols=["AAA"],
        initial_cash=1000.0,
        commission_bps=10.0,
        slippage_bps=5.0,
        drawdown_coefficient=0.0,
        turnover_coefficient=0.0,
        benchmark_closes=[100.0, 100.0, 100.0],
    )
    assert isinstance(env, ToolEnv)

    async def first_episode() -> None:
        initial = env.reset()
        assert "2025-01-02" in initial.messages[-1]["content"]
        assert len(initial.messages) == 2
        assert "observation_id" not in initial.messages[-1]["content"]

        retrieved, _, done, _ = await env.step(
            _call("retrieve", '{"query":"future surprise","limit":3}')
        )
        assert not done
        assert '"returned":0' in retrieved.messages[-1]["content"]
        assert '"kind":"fundamental"' not in retrieved.messages[-1]["content"]
        assert len(retrieved.messages) == 2

        _, _, done, info = await env.step(_call("buy", '{"symbol":"AAA","target_weight":0.5}'))
        assert not done
        assert info["fills"][0]["raw_open_price"] == 110.0
        assert info["fills"][0]["executed_at"] == open_times[1]
        assert env.ledger.positions["AAA"] > 0.0
        assert len(env._messages) == 2

        before_error_date = env.current_date
        error_observation, reward, done, info = await env.step(
            _call("buy", '{"symbol":"AAA","target_weight":-1}')
        )
        assert reward == 0.0
        assert not done
        assert info["invalid_action"] is True
        assert env.current_date == before_error_date
        assert "Error:" in error_observation.messages[-1]["content"]
        assert len(error_observation.messages) == 2

        _, _, done, _ = await env.step(_call("hold"))
        assert done

    async def second_episode() -> None:
        env.reset()
        _, _, done, _ = await env.step(_call("hold"))
        assert not done
        _, _, done, _ = await env.step(_call("abstain", '{"reason":"insufficient signal"}'))
        assert done

    asyncio.run(first_episode())
    asyncio.run(second_episode())


@pytest.mark.parametrize(
    ("action", "target", "gap_price", "should_skip"),
    [
        ("buy", 0.55, 200.0, True),
        ("sell", 0.45, 50.0, True),
        ("buy", 0.55, 80.0, False),
        ("sell", 0.45, 120.0, False),
    ],
)
def test_directional_target_gap_advances_without_an_opposite_side_fill(
    action: str, target: float, gap_price: float, should_skip: bool,
) -> None:
    days = (2, 3, 6, 7)
    prices = (100.0, 100.0, gap_price, 100.0)
    env = TradingEnv(
        adapter=StockBenchAdapter(
            [
                _bar(day, open_price=price, close_price=price)
                for day, price in zip(days, prices, strict=True)
            ],
            data_version="directional-gap-fixture",
        ),
        episode_dates=[datetime(2025, 1, day, 21, tzinfo=UTC) for day in days],
        episode_open_times=[datetime(2025, 1, day, 14, 30, tzinfo=UTC) for day in days],
        symbols=["AAA"], initial_cash=1000.0, commission_bps=10.0, slippage_bps=5.0,
        drawdown_coefficient=0.0, turnover_coefficient=0.0,
        benchmark_closes=[100.0] * len(days),
    )

    async def episode() -> None:
        env.reset()
        await env.step(_call("buy", '{"symbol":"AAA","target_weight":0.5}'))
        cash_before = env.ledger.cash
        positions_before = dict(env.ledger.positions)
        fills_before = env.ledger.fills

        observation, _, done, info = await env.step(
            _call(action, json.dumps({"symbol": "AAA", "target_weight": target}))
        )

        assert env.current_date == datetime(2025, 1, 6, 21, tzinfo=UTC)
        assert not done
        assert info["invalid_action"] is False
        assert env.ledger.pending_orders == ()
        if should_skip:
            assert info["fills"] == []
            assert info["cost_fraction"] == 0
            assert info["turnover"] == 0
            assert env.ledger.fills == fills_before
            assert env.ledger.cash == cash_before
            assert dict(env.ledger.positions) == positions_before
            assert info["equity"] == pytest.approx(
                cash_before + positions_before["AAA"] * gap_price
            )
            skip, = info["skipped_orders"]
            assert skip["reason"] == "opposite_direction_at_next_open"
            assert skip["requested_side"] == action
            assert skip["reason"] in observation.messages[-1]["content"]
        else:
            fill, = info["fills"]
            assert fill["side"] == action
            assert info["cost_fraction"] > 0
            assert info["skipped_orders"] == []

        observation, _, done, info = await env.step(_call("hold"))
        assert done
        assert info["fills"] == []
        assert info["skipped_orders"] == []
        assert "opposite_direction_at_next_open" not in observation.messages[-1]["content"]

    asyncio.run(episode())


def test_g1_history_and_retrieve_output_stay_bounded_across_eight_advances() -> None:
    session_days = (2, 3, 6, 7, 8, 10, 13, 14, 15)
    dates = [datetime(2025, 1, day, 21, 0, tzinfo=UTC) for day in session_days]
    opens = [datetime(2025, 1, day, 14, 30, tzinfo=UTC) for day in session_days]
    oversized = MarketObservation(
        observation_id="oversized-technical",
        kind="technical",
        symbol="AAA",
        event_at=dates[0],
        available_at=dates[0],
        payload={"text": "oversized " + "x" * 20_000, "nested": {"secret": "y" * 20_000}},
    )
    env = TradingEnv(
        adapter=StockBenchAdapter(
            [
                *[
                    _bar(day, open_price=100.0 + index, close_price=101.0 + index)
                    for index, day in enumerate(session_days)
                ],
                oversized,
            ],
            data_version="bounded-context-fixture",
        ),
        episode_dates=dates,
        episode_open_times=opens,
        symbols=["AAA"],
        initial_cash=1000.0,
        commission_bps=0.0,
        slippage_bps=0.0,
        drawdown_coefficient=0.0,
        turnover_coefficient=0.0,
        benchmark_closes=[100.0] * len(dates),
    )

    async def episode() -> None:
        observation = env.reset()
        assert len(observation.messages) == 2
        for index in range(8):
            for _ in range(3):
                retrieved, _, done, _ = await env.step(
                    _call(
                        "retrieve",
                        json.dumps(
                            {
                                "query": "💹" * 200,
                                "limit": TradingEnv.MAX_RETRIEVE_RESULTS,
                            },
                            ensure_ascii=False,
                        ),
                    )
                )
                assert not done
                assert len(retrieved.messages) == 2
                assert "\\ud83d" not in retrieved.messages[-1]["content"]
                assert "x" * 1000 not in retrieved.messages[-1]["content"]
                event_arguments = env._recent_events[-1]["arguments"]
                assert isinstance(event_arguments, dict)
                query_summary = event_arguments["query"]
                assert isinstance(query_summary, str)
                assert query_summary.isascii()
                assert len(query_summary) <= TradingEnv.MAX_EVENT_TEXT_CHARS

            observation, _, done, _ = await env.step(_call("hold"))
            assert done is (index == 7)
            assert len(observation.messages) == 2
            assert "observation_id" not in observation.messages[-1]["content"]

        payload = env._retrieve(
            RetrieveArgs(query="oversized", limit=TradingEnv.MAX_RETRIEVE_RESULTS)
        )
        encoded = json.dumps(
            payload,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
        assert len(encoded) <= TradingEnv.MAX_RETRIEVE_BYTES
        hidden_payload = env._retrieve(
            RetrieveArgs(query="secret", limit=TradingEnv.MAX_RETRIEVE_RESULTS)
        )
        assert hidden_payload["returned"] == 0

    asyncio.run(episode())


def test_finance_trajectory_mode_emits_a_fin_only_on_the_final_step() -> None:
    dates = [datetime(2025, 1, day, 21, 0, tzinfo=UTC) for day in (2, 3, 6)]
    opens = [datetime(2025, 1, day, 14, 30, tzinfo=UTC) for day in (2, 3, 6)]
    adapter = StockBenchAdapter(
        [
            _bar(2, open_price=100.0, close_price=100.0),
            _bar(3, open_price=100.0, close_price=100.0),
            _bar(6, open_price=100.0, close_price=100.0),
        ],
        data_version="fixture-v1",
    )
    env = TradingEnv(
        adapter=adapter,
        episode_dates=dates,
        episode_open_times=opens,
        symbols=["AAA"],
        initial_cash=1000.0,
        commission_bps=10.0,
        slippage_bps=5.0,
        drawdown_coefficient=0.0,
        turnover_coefficient=0.0,
        benchmark_closes=[100.0, 101.0, 102.0],
        reward_mode="finance_trajectory",
        finance_epsilon=0.01,
        risk_lambda=1.0,
        turnover_limit=1.0,
        position_limit=1.0,
    )

    async def episode() -> None:
        env.reset()
        _, first_reward, first_done, _ = await env.step(_call("hold"))
        _, final_reward, final_done, info = await env.step(_call("hold"))
        assert first_reward == 0.0
        assert not first_done
        assert final_done
        assert final_reward == pytest.approx(-2.0)
        assert info["trajectory_reward"] == pytest.approx(-2.0)
        assert env.episode_complete is True
        assert env.reward_mode == "finance_trajectory"
        trace = env.credit_step_trace
        assert [record.action for record in trace] == ["hold", "hold"]
        assert [record.decision_session_index for record in trace] == [0, 1]
        assert [record.result_session_index for record in trace] == [1, 2]
        assert [record.reward for record in trace] == pytest.approx([0.0, -2.0])
        assert [record.done for record in trace] == [False, True]
        assert all(record.invalid_action is False for record in trace)

    asyncio.run(episode())


def test_finance_risk_uses_maximum_daily_bilateral_turnover_not_episode_sum() -> None:
    dates = [datetime(2025, 1, day, 21, 0, tzinfo=UTC) for day in (2, 3, 6, 7)]
    opens = [datetime(2025, 1, day, 14, 30, tzinfo=UTC) for day in (2, 3, 6, 7)]
    adapter = StockBenchAdapter(
        [_bar(day, open_price=100.0, close_price=100.0) for day in (2, 3, 6, 7)],
        data_version="daily-turnover-fixture-v1",
    )
    env = TradingEnv(
        adapter=adapter,
        episode_dates=dates,
        episode_open_times=opens,
        symbols=["AAA"],
        initial_cash=1000.0,
        commission_bps=0.0,
        slippage_bps=0.0,
        drawdown_coefficient=0.0,
        turnover_coefficient=0.0,
        benchmark_closes=[100.0] * 4,
        reward_mode="finance_trajectory",
        finance_epsilon=0.01,
        risk_lambda=1.0,
        turnover_limit=0.8,
        position_limit=1.0,
    )

    async def episode() -> None:
        env.reset()
        await env.step(_call("buy", '{"symbol":"AAA","target_weight":0.6}'))
        await env.step(_call("sell", '{"symbol":"AAA","target_weight":0.0}'))
        _, final_reward, done, _ = await env.step(_call("hold"))
        assert done
        assert final_reward == pytest.approx(0.0)

    asyncio.run(episode())


def test_missing_current_session_bar_cannot_reuse_a_stale_price() -> None:
    dates = [datetime(2025, 1, day, 21, 0, tzinfo=UTC) for day in (2, 3)]
    opens = [datetime(2025, 1, day, 14, 30, tzinfo=UTC) for day in (2, 3)]
    env = TradingEnv(
        adapter=StockBenchAdapter(
            [_bar(2, open_price=100.0, close_price=101.0)],
            data_version="missing-current-session-fixture",
        ),
        episode_dates=dates,
        episode_open_times=opens,
        symbols=["AAA"],
        initial_cash=1000.0,
        commission_bps=10.0,
        slippage_bps=5.0,
        drawdown_coefficient=0.0,
        turnover_coefficient=0.0,
        benchmark_closes=[100.0, 101.0],
    )

    async def episode() -> None:
        env.reset()
        with pytest.raises(RuntimeError, match="current-session .* prices"):
            await env.step(_call("hold"))
        assert env.current_date == dates[0]

    asyncio.run(episode())


def test_episode_timestamps_must_equal_the_exchange_session_boundaries() -> None:
    with pytest.raises(ValueError, match="must equal XNYS session close"):
        TradingEnv(
            adapter=StockBenchAdapter(
                [
                    _bar(2, open_price=100.0, close_price=101.0),
                    _bar(3, open_price=101.0, close_price=102.0),
                ],
                data_version="wrong-boundary-fixture",
            ),
            episode_dates=[
                "2025-01-02T20:59:00+00:00",
                "2025-01-03T20:59:00+00:00",
            ],
            episode_open_times=[
                "2025-01-02T14:30:00+00:00",
                "2025-01-03T14:30:00+00:00",
            ],
            symbols=["AAA"],
            initial_cash=1000.0,
            commission_bps=10.0,
            slippage_bps=5.0,
            drawdown_coefficient=0.0,
            turnover_coefficient=0.0,
            benchmark_closes=[100.0, 101.0],
        )
