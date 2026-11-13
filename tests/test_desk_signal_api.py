# -*- coding: utf-8 -*-
"""Desk Signal fetch boundary, REST history contract, and chat prefix."""

import asyncio
import json
from datetime import datetime, time, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api.app import create_app
from api.v1.endpoints import agent as agent_endpoint
from src.agent.tools.desk_signal import LOOKBACK_CALENDAR_DAYS, evaluate_desk_signal
from src.agent.tools.desk_signal_rules import SHANGHAI
from src.config import Config
from src.services.agent_chat_session_service import AgentChatSessionService
from src.storage import DatabaseManager
from tests.test_desk_signal_rules import _weekdays


@pytest.fixture(autouse=True)
def isolated_api_database(tmp_path: Path):
    DatabaseManager.reset_instance()
    Config.reset_instance()
    DatabaseManager(db_url=f"sqlite:///{tmp_path / 'desk.db'}")
    yield
    DatabaseManager.reset_instance()
    Config.reset_instance()


def _history_frame():
    return pd.DataFrame([
        {
            "date": "2026-04-24",
            "open": 1.0,
            "high": 2.0,
            "low": 0.5,
            "close": 1.5,
            "volume": 100.0,
            "amount": 1000.0,
            "pct_chg": 1.0,
        }
    ])


def test_rest_history_accepts_365_rejects_500_and_hides_adjustment(tmp_path: Path):
    manager = MagicMock()
    manager.get_daily_data.return_value = (_history_frame(), "EfinanceFetcher")
    manager.get_stock_name.return_value = "贵州茅台"

    with patch("api.middlewares.auth.is_auth_enabled", return_value=False), \
         patch("data_provider.base.DataFetcherManager", return_value=manager), \
         patch("src.storage.DatabaseManager.get_data_range") as read_db, \
         patch("src.storage.DatabaseManager.save_daily_data") as write_db:
        client = TestClient(create_app(static_dir=tmp_path / "static"))
        accepted = client.get("/api/v1/stocks/600519/history", params={"days": 365})
        rejected = client.get("/api/v1/stocks/600519/history", params={"days": 500})

    assert accepted.status_code == 200
    assert rejected.status_code == 422
    payload = accepted.json()
    assert set(payload) == {"stock_code", "stock_name", "period", "data"}
    blob = json.dumps(payload)
    for key in ("adjustment", "timezone", "source"):
        assert key not in payload
        assert key not in blob
    read_db.assert_not_called()
    write_db.assert_not_called()
    manager.get_daily_data.assert_called_once()


def test_tool_fetches_explicit_window_and_ignores_partial_cache():
    bars = _weekdays(130)
    frame = pd.DataFrame(bars)
    frame["date"] = frame["date"].astype(str)
    manager = MagicMock()
    manager.get_daily_data.return_value = (frame, "AkshareFetcher")
    db = MagicMock()
    db.get_data_range.return_value = bars[:30]
    as_of = datetime.combine(bars[-1]["date"], time(16, 0), tzinfo=SHANGHAI)

    with patch("src.agent.tools.desk_signal._get_manager", return_value=manager), \
         patch("src.storage.get_db", return_value=db), \
         patch("src.services.history_loader.load_history_df", side_effect=AssertionError("cache")):
        result = evaluate_desk_signal("600519", as_of=as_of)

    start = (as_of.date() - timedelta(days=LOOKBACK_CALENDAR_DAYS)).isoformat()
    manager.get_daily_data.assert_called_once_with(
        "600519",
        start_date=start,
        end_date=as_of.date().isoformat(),
        days=LOOKBACK_CALENDAR_DAYS,
    )
    db.get_data_range.assert_not_called()
    db.save_daily_data.assert_not_called()
    assert result["status"] == "confirmed"
    assert result["source"] == "AkshareFetcher"
    assert result["adjustment"] == "qfq"


def test_registered_tool_has_only_the_stock_code_parameter():
    from src.agent.factory import reset_tool_registry, get_tool_registry

    reset_tool_registry()
    try:
        tool = get_tool_registry(Config()).get("evaluate_desk_signal")
    finally:
        reset_tool_registry()
    assert [parameter.name for parameter in tool.parameters] == ["stock_code"]
    assert tool.parameters[0].required is True


def _litellm_config():
    return SimpleNamespace(
        agent_backend="auto",
        is_agent_available=lambda: True,
        report_language="zh",
    )


def _immediate_to_thread(func, /, *args, **kwargs):
    return func(*args, **kwargs)


def _executor(run):
    from src.agent.chat_executor import AgentChatExecutor

    backend = MagicMock()
    backend.backend_id = "litellm"
    backend.runtime_owns_loop = False
    backend.run.side_effect = run
    return MagicMock(wraps=AgentChatExecutor(
        backend=backend, config=_litellm_config(), context_llm_adapter=None,
    ))


async def _collect(request):
    response = await agent_endpoint.agent_chat_stream(
        request,
        session_service=AgentChatSessionService(),
    )
    return [
        json.loads(chunk.removeprefix("data: ").strip())
        async for chunk in response.body_iterator
    ]


def _confirmed_payload():
    return {
        "evaluation": "desk_signal",
        "status": "confirmed",
        "stock_code": "600519",
        "source": "EfinanceFetcher",
        "adjustment": "qfq",
        "timezone": "Asia/Shanghai",
        "coverage": {"start": "2024-01-02", "end": "2024-07-01", "completed_bars": 130},
        "signal_date": "2024-07-01",
        "entry_gate": {
            "status": "pass",
            "breakout": True,
            "volume": True,
            "rsi14": 61.5,
            "rsi_above_50": True,
        },
        "volatility_block": {"status": "not_blocked", "atr14": 1.25, "percentile_90": 1.25},
        "context": {"ema8": 60.0, "ema21": 59.0, "ema50": 58.0, "ema200_warmup": "omitted"},
        "warmup_notes": ["EMA200 omitted because fewer than 200 completed bars"],
        "calendar_gaps": [],
        "constraints": [
            "A-share T+1: this result is not an order.",
            "Limit-up/down tradability and halt identity have no independently verified data.",
        ],
    }


def test_bare_code_runs_tool_before_model_and_prefixes_sse_content():
    from src.agent.agent_backend import AgentRunResult

    order = []
    payload = _confirmed_payload()

    def _evaluate(stock_code, **_kwargs):
        order.append(("tool", stock_code))
        return payload

    def _run(_request):
        order.append("model")
        return AgentRunResult(
            success=True,
            final_answer="模型计算 RSI14=99.9",
            backend="litellm",
            total_steps=2,
            diagnostics={"provider": "mock"},
        )

    executor = _executor(_run)
    with patch("api.v1.endpoints.agent.asyncio.to_thread", side_effect=_immediate_to_thread), \
         patch("api.v1.endpoints.agent.get_config", return_value=_litellm_config()), \
         patch("api.v1.endpoints.agent._build_executor", return_value=executor), \
         patch("src.agent.tools.desk_signal.evaluate_desk_signal", side_effect=_evaluate):
        events = asyncio.run(_collect(agent_endpoint.ChatRequest(
            message="600519",
            session_id="desk-bare",
            skills=["desk_signal"],
        )))

    done = events[-1]
    assert [event["type"] for event in events] == ["accepted", "done"]
    assert order == [("tool", "600519"), "model"]
    assert done["total_steps"] == 2
    assert done["success"] is True
    assert done["content"].startswith("【Desk Signal 权威结果】状态：confirmed")
    assert done["content"].index("61.5") < done["content"].index("99.9")
    assert done["desk_signal"] == payload


def test_ambiguous_sentence_clarifies_before_the_tool_loop():
    called = {"tool": 0, "model": 0}

    def _evaluate(*_args, **_kwargs):
        called["tool"] += 1
        raise AssertionError("tool loop started")

    def _run(_request):
        called["model"] += 1
        raise AssertionError("model loop started")

    executor = _executor(_run)
    message = (
        "请先从宏观流动性、行业政策和公司质地三个层面讨论研究范围，"
        "先不要绑定任何具体证券，我想确认这次讨论的对象是否已经足够明确。"
    )
    with patch("api.v1.endpoints.agent.asyncio.to_thread", side_effect=_immediate_to_thread), \
         patch("api.v1.endpoints.agent.get_config", return_value=_litellm_config()), \
         patch("api.v1.endpoints.agent._build_executor", return_value=executor), \
         patch("src.agent.tools.desk_signal.evaluate_desk_signal", side_effect=_evaluate):
        events = asyncio.run(_collect(agent_endpoint.ChatRequest(
            message=message,
            session_id="desk-ambiguous",
            skills=["desk_signal"],
        )))

    done = events[-1]
    assert [event["type"] for event in events] == ["accepted", "done"]
    assert done["total_steps"] == 0
    assert "请明确提供" in done["content"]
    assert "desk_signal" not in done
    assert called == {"tool": 0, "model": 0}
    executor.backend.run.assert_not_called()


def test_failed_evaluation_drops_model_indicator_numbers():
    from src.agent.agent_backend import AgentRunResult

    def _evaluate(*_args, **_kwargs):
        raise RuntimeError("feed unavailable")

    def _run(_request):
        return AgentRunResult(
            success=True,
            final_answer="模型计算 RSI14=99.9 ATR14=3.14",
            backend="litellm",
            total_steps=2,
            diagnostics={"provider": "mock"},
        )

    executor = _executor(_run)
    with patch("api.v1.endpoints.agent.asyncio.to_thread", side_effect=_immediate_to_thread), \
         patch("api.v1.endpoints.agent.get_config", return_value=_litellm_config()), \
         patch("api.v1.endpoints.agent._build_executor", return_value=executor), \
         patch("src.agent.tools.desk_signal.evaluate_desk_signal", side_effect=_evaluate):
        events = asyncio.run(_collect(agent_endpoint.ChatRequest(
            message="600519",
            session_id="desk-failed",
            skills=["desk_signal"],
        )))

    done = events[-1]
    assert done["content"].startswith("【Desk Signal 权威结果】状态：unconfirmed")
    assert "99.9" not in done["content"]
    assert "3.14" not in done["content"]
    assert "rsi14" not in done["desk_signal"]
    assert done["desk_signal"]["status"] == "unconfirmed"
    assert done["desk_signal"]["volatility_block"]["status"] == "unconfirmed"
