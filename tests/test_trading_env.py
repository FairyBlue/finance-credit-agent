"""Agent-R1 plugin integration exercised with deterministic mock actions."""

import asyncio
from datetime import datetime, timezone

import pytest
from agent_r1.env import ToolEnv
from agent_r1.env.base import Action

from src.data import MarketObservation, StockBenchAdapter
from src.env.trading import TradingEnv

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


def test_mock_agent_runs_multiple_episodes_with_next_open_fills_and_safe_errors() -> None:
    dates = [datetime(2025, 1, day, 21, 0, tzinfo=UTC) for day in (2, 3, 6)]
    open_times = [datetime(2025, 1, day, 14, 30, tzinfo=UTC) for day in (2, 3, 6)]
    future_news = MarketObservation(
        observation_id="future-news",
        kind="news",
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
            future_news,
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

        _, _, done, info = await env.step(
            _call("buy", '{"symbol":"AAA","target_weight":0.5}')
        )
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
