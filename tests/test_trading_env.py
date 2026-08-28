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
from src.agents.prompt_budget import _measure_episode
from src.data import MarketObservation, StockBenchAdapter, audit_raw_daily_manifest
from src.data.stockbench import dataframe_content_sha256
from src.env.trading import TradingEnv, _cached_storage_adapter

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

        retrieved, _, done, _ = await env.step(
            _call("retrieve", '{"query":"future surprise","limit":10}')
        )
        assert not done
        assert "future surprise" not in retrieved.messages[-1]["content"]

        _, _, done, info = await env.step(_call("buy", '{"symbol":"AAA","target_weight":0.5}'))
        assert not done
        assert info["fills"][0]["raw_open_price"] == 110.0
        assert info["fills"][0]["executed_at"] == open_times[1]
        assert env.ledger.positions["AAA"] > 0.0

        before_error_date = env.current_date
        error_observation, reward, done, info = await env.step(
            _call("buy", '{"symbol":"AAA","target_weight":-1}')
        )
        assert reward == 0.0
        assert not done
        assert info["invalid_action"] is True
        assert env.current_date == before_error_date
        assert "Error:" in error_observation.messages[-1]["content"]

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


def test_smoke_prompt_history_is_bounded_across_retrieve_and_full_episode() -> None:
    # 2025-01-09 was an XNYS closure for the national day of mourning.
    days = (2, 3, 6, 7, 8, 10, 13, 14, 15)
    dates = [datetime(2025, 1, day, 21, 0, tzinfo=UTC) for day in days]
    opens = [datetime(2025, 1, day, 14, 30, tzinfo=UTC) for day in days]
    symbols = [f"S{index:02d}" for index in range(20)]
    observations = [
        MarketObservation(
            observation_id=f"{symbol}-{day}",
            kind="market",
            symbol=symbol,
            event_at=session,
            available_at=session,
            payload={
                "date": session.date().isoformat(),
                "open": 100.0 + index,
                "high": 101.0 + index,
                "low": 99.0 + index,
                "close": 100.5 + index,
                "volume": 1_000_000.0 + index,
            },
        )
        for day, session in zip(days, dates, strict=True)
        for index, symbol in enumerate(symbols)
    ]
    observations.extend(
        MarketObservation(
            observation_id=f"signal-{index}",
            kind="fundamental",
            symbol=symbols[index % len(symbols)],
            event_at=dates[0],
            available_at=dates[0],
            payload={"headline": f"visible signal {index}", "value": index},
        )
        for index in range(15)
    )
    env = TradingEnv(
        adapter=StockBenchAdapter(observations, data_version="bounded-prompt-fixture"),
        episode_dates=dates,
        episode_open_times=opens,
        symbols=symbols,
        initial_cash=1000.0,
        commission_bps=10.0,
        slippage_bps=5.0,
        drawdown_coefficient=0.0,
        turnover_coefficient=0.0,
        benchmark_closes=[100.0] * len(dates),
    )
    raw_prompt: list[dict[str, object]] = [
        {
            "role": "user",
            "content": "Manage the portfolio using only point-in-time tools.",
        }
    ]

    class ApproximateTokenizer:
        def apply_chat_template(
            self,
            conversation: object,
            *,
            tools: object,
            add_generation_prompt: bool,
            tokenize: bool,
        ) -> list[int]:
            assert add_generation_prompt is True
            assert tokenize is True
            rendered = json.dumps(
                {"conversation": conversation, "tools": tools},
                default=str,
                sort_keys=True,
            )
            return [0] * ((len(rendered) + 3) // 4)

    initial = env.reset(raw_prompt=raw_prompt)
    assert len(initial.messages) == len(raw_prompt)
    initial_content = initial.messages[-1]["content"]
    current_state = json.loads(initial_content.split("Current observation:\n", 1)[1])
    assert [record["symbol"] for record in current_state["market"]] == symbols
    assert len(current_state["signals"]) == 10

    async def assert_message_structure_is_constant() -> None:
        retrieved, _, done, _ = await env.step(_call("retrieve", '{"query":":","limit":10}'))
        assert not done
        assert len(retrieved.messages) == len(raw_prompt)
        advanced, _, done, _ = await env.step(_call("hold"))
        assert not done
        assert len(advanced.messages) == len(raw_prompt)

    asyncio.run(assert_message_structure_is_constant())

    measurements = asyncio.run(
        _measure_episode(
            tokenizer=ApproximateTokenizer(),
            env=env,
            raw_prompt=raw_prompt,
        )
    )

    assert len(measurements) == 32
    assert measurements[0]["stage"] == "hold_session_0_state"
    assert measurements[1]["stage"] == "hold_session_0_after_max_retrieve"
    assert measurements[-1]["stage"] == "positions_session_7_after_max_retrieve"
    assert max(int(item["prompt_tokens"]) for item in measurements) < 4096


def test_retrieve_only_matches_evidence_present_in_compact_result() -> None:
    dates = [datetime(2025, 1, day, 21, 0, tzinfo=UTC) for day in (2, 3)]
    opens = [datetime(2025, 1, day, 14, 30, tzinfo=UTC) for day in (2, 3)]
    discarded_payload = {f"a{index:02d}": f"value-{index}" for index in range(12)}
    discarded_payload["zzz"] = "hidden needle"
    signal = MarketObservation(
        observation_id="large-signal",
        kind="fundamental",
        symbol="AAA",
        event_at=dates[0],
        available_at=dates[0],
        payload=discarded_payload,
    )
    env = TradingEnv(
        adapter=StockBenchAdapter(
            [
                _bar(2, open_price=100.0, close_price=100.0),
                _bar(3, open_price=100.0, close_price=100.0),
                signal,
            ],
            data_version="retrieve-visible-evidence-fixture",
        ),
        episode_dates=dates,
        episode_open_times=opens,
        symbols=["AAA"],
        initial_cash=1000.0,
        commission_bps=0.0,
        slippage_bps=0.0,
        drawdown_coefficient=0.0,
        turnover_coefficient=0.0,
        benchmark_closes=[100.0, 100.0],
    )

    async def retrieve() -> None:
        env.reset()
        hidden, _, _, _ = await env.step(
            _call("retrieve", '{"query":"hidden needle","limit":10}')
        )
        hidden_text = hidden.messages[-1]["content"]
        hidden_state = json.loads(hidden_text.split("Current observation:\n", 1)[-1])
        assert hidden_state["last_tool"]["result"]["results"] == []

        visible, _, _, _ = await env.step(
            _call("retrieve", '{"query":"value-0","limit":10}')
        )
        visible_text = visible.messages[-1]["content"]
        visible_state = json.loads(visible_text.split("Current observation:\n", 1)[-1])
        results = visible_state["last_tool"]["result"]["results"]
        assert len(results) == 1
        assert "value-0" in json.dumps(results[0], sort_keys=True)

    asyncio.run(retrieve())


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

    asyncio.run(episode())


def test_finance_risk_uses_maximum_daily_bilateral_turnover_not_episode_sum() -> None:
    dates = [datetime(2025, 1, day, 21, 0, tzinfo=UTC) for day in (2, 3, 6, 7)]
    opens = [datetime(2025, 1, day, 14, 30, tzinfo=UTC) for day in (2, 3, 6, 7)]
    adapter = StockBenchAdapter(
        [
            _bar(day, open_price=100.0, close_price=100.0)
            for day in (2, 3, 6, 7)
        ],
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
