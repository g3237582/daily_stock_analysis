# -*- coding: utf-8 -*-
"""Read-only Desk Signal tool.

Fetches an explicit A-share window into memory. It does not call
``load_history_df`` and does not write ``stock_daily``.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any, Optional

from src.agent.tools.desk_signal_rules import CONSTRAINTS, evaluate_bars, shanghai_now
from src.agent.tools.execution import check_tool_execution
from src.agent.tools.registry import ToolDefinition, ToolParameter, ToolPolicy

logger = logging.getLogger(__name__)

DESK_SIGNAL_SKILL_ID = "desk_signal"
LOOKBACK_CALENDAR_DAYS = 1100

_READ_ONLY_POLICY = ToolPolicy.declared(
    read_only=True,
    side_effects=["network_read"],
    permissions=["market_data:read"],
    scope_dimensions=["stock"],
    supported_asset_types=("stock",),
)


def is_complete_evaluation(result: Any) -> bool:
    """A successful confirmed/unconfirmed payload produced by this tool."""
    if not isinstance(result, dict) or result.get("evaluation") != "desk_signal":
        return False
    if result.get("status") not in {"confirmed", "unconfirmed"}:
        return False
    if result.get("timezone") != "Asia/Shanghai":
        return False
    gate = result.get("entry_gate")
    block = result.get("volatility_block")
    if not isinstance(gate, dict) or not isinstance(block, dict):
        return False
    if result["status"] == "unconfirmed" and (
        gate.get("status") != "unconfirmed" or block.get("status") != "unconfirmed"
    ):
        return False
    return isinstance(result.get("constraints"), list)


def incomplete_evaluation(stock_code: Optional[str] = None) -> dict:
    """Unconfirmed shell with no estimated indicator values."""
    return {
        "status": "unconfirmed",
        "stock_code": stock_code,
        "source": None,
        "adjustment": "unknown",
        "timezone": "Asia/Shanghai",
        "coverage": {"start": None, "end": None, "completed_bars": 0},
        "signal_date": None,
        "entry_gate": {"status": "unconfirmed"},
        "volatility_block": {"status": "unconfirmed"},
        "warmup_notes": [],
        "calendar_gaps": [],
        "constraints": list(CONSTRAINTS),
        "unconfirmed_reason": "evaluation did not complete",
    }


def unconfirmed_symbol_clarification(language: str) -> str:
    if language == "en":
        return (
            "The discussion target is not confirmed. "
            "Please provide the complete stock or index code."
        )
    return "请明确提供本次讨论的股票或指数代码，或确认是否切换。"


def _format_number(value: Any) -> str:
    return format(float(value), ".6g")


def format_desk_signal_paragraph(result: dict) -> str:
    """Authoritative text prefix. The portable UI only renders ``content``."""
    coverage = result.get("coverage") or {}
    parts = [
        f"【Desk Signal 权威结果】状态：{result.get('status')}。",
        f"时区：{result.get('timezone')}。数据源：{result.get('source') or 'unknown'}。复权：{result.get('adjustment') or 'unknown'}。",
    ]
    if coverage.get("start") and coverage.get("end"):
        parts.append(
            f"覆盖：{coverage['start']} 至 {coverage['end']}，"
            f"完成K线 {coverage.get('completed_bars')} 根。"
        )
    if result.get("signal_date"):
        parts.append(f"信号日：{result['signal_date']}。")
    if result.get("status") == "confirmed":
        gate = result.get("entry_gate") or {}
        block = result.get("volatility_block") or {}
        parts.append(f"入场门：{gate.get('status')}。波动阻挡：{block.get('status')}。")
        parts.append(
            f"Wilder RSI14：{_format_number(gate.get('rsi14'))}。"
            f"信号 ATR14：{_format_number(block.get('atr14'))}。"
            f"前序 ATR14 第90位：{_format_number(block.get('percentile_90'))}。"
        )
        context = result.get("context") or {}
        ema_bits = [
            f"EMA{period}={_format_number(context[f'ema{period}'])}"
            for period in (8, 21, 50, 200)
            if context.get(f"ema{period}") is not None
        ]
        if ema_bits:
            parts.append("均线上下文（不决定入场）：" + "，".join(ema_bits) + "。")
    else:
        parts.append(f"原因：{result.get('unconfirmed_reason') or '评估未确认'}。不输出估计指标。")
    notes = result.get("warmup_notes") or []
    if notes:
        parts.append("预热：" + "；".join(notes) + "。")
    gaps = result.get("calendar_gaps") or []
    if gaps:
        parts.append(f"日历缺口 {len(gaps)} 处，未补齐。")
    constraints = result.get("constraints") or []
    if constraints:
        parts.append("约束：" + "；".join(constraints))
    return "".join(parts)


def compose_desk_signal_reply(model_content: str, evaluation: Any) -> tuple[str, dict]:
    """Prefix a complete evaluation. An incomplete run drops model indicator text."""
    if is_complete_evaluation(evaluation):
        paragraph = format_desk_signal_paragraph(evaluation)
        body = (model_content or "").strip()
        content = paragraph if not body else f"{paragraph}\n\n{body}"
        return content, evaluation
    payload = incomplete_evaluation()
    return format_desk_signal_paragraph(payload), payload


def _is_a_share_stock(stock_code: str) -> bool:
    from src.services.stock_list_parser import ParseStatus, parse_analysis_target

    target = parse_analysis_target(stock_code)
    return target.asset_type == ParseStatus.STOCK and target.exchange in {"SH", "SZ", "BJ"}


def _get_manager():
    from src.agent.tools.data_tools import _get_fetcher_manager

    return _get_fetcher_manager()


def fetch_desk_bars(stock_code: str, as_of: datetime):
    """Fetch one explicit forward window. Callers must not persist the frame."""
    local = shanghai_now(as_of)
    start = local.date() - timedelta(days=LOOKBACK_CALENDAR_DAYS)
    end = local.date()
    manager = _get_manager()
    return manager.get_daily_data(
        stock_code,
        start_date=start.isoformat(),
        end_date=end.isoformat(),
        days=LOOKBACK_CALENDAR_DAYS,
    )


def evaluate_desk_signal(
    stock_code: str,
    *,
    as_of: Optional[datetime] = None,
    bars: Any = None,
    source: Optional[str] = None,
) -> dict:
    """Evaluate ``stock_code``. Pass ``bars`` only from tests; the tool fetches."""
    local = shanghai_now(as_of)
    code = str(stock_code or "").strip()
    if not _is_a_share_stock(code):
        return evaluate_bars([], stock_code=code, source=source, as_of=local) | {
            "unconfirmed_reason": "A-share only",
            "coverage": {"start": None, "end": None, "completed_bars": 0},
        }
    if bars is None:
        try:
            bars, source = fetch_desk_bars(code, local)
        except Exception as exc:
            logger.warning("Desk Signal fetch failed for %s: %s", code, exc)
            bars, source = [], None
    return evaluate_bars(bars, stock_code=code, source=source, as_of=local)


def _handle_evaluate_desk_signal(stock_code: str) -> dict:
    check_tool_execution()
    return evaluate_desk_signal(stock_code)


evaluate_desk_signal_tool = ToolDefinition(
    name="evaluate_desk_signal",
    description=(
        "Read-only A-share Desk Signal evaluation from an explicit forward-adjusted "
        "daily window. Returns confirmed or unconfirmed, the entry gate, and the "
        "volatility block. Does not read or write stock_daily and does not submit an order."
    ),
    parameters=[
        ToolParameter(
            name="stock_code",
            type="string",
            description="A-share stock code, for example 600519.",
        ),
    ],
    handler=_handle_evaluate_desk_signal,
    category="data",
    policy=_READ_ONLY_POLICY,
)
