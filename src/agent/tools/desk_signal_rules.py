# -*- coding: utf-8 -*-
"""Deterministic Desk Signal math. No network, database, or order fields."""

from __future__ import annotations

import math
from datetime import date, datetime, time, timedelta
from typing import Any, Optional, Sequence
from zoneinfo import ZoneInfo

SHANGHAI = ZoneInfo("Asia/Shanghai")
SESSION_CLOSE = time(15, 0)
MIN_COMPLETED_BARS = 115
PRIOR_WINDOW = 20
WILDER_PERIOD = 14
PRIOR_ATR_COUNT = 100
PERCENTILE_RANK = 90
EMA_PERIODS = (8, 21, 50, 200)
EMA200_WARM_BARS = 500
VOLUME_MULTIPLE = 1.5

CONSTRAINTS = (
    "A-share T+1: this result is not an order.",
    "Limit-up/down tradability and halt identity have no independently verified data.",
)

# Confirmation follows the fetcher implementation, not a guessed adjust flag.
QFQ_SOURCES = {
    "EfinanceFetcher": "qfq",
    "AkshareFetcher": "qfq",
    "BaostockFetcher": "qfq",
    "TencentFetcher": "qfq",
}
UNCONFIRMED_ADJUSTMENTS = {
    "PytdxFetcher": "none",
    "LongbridgeFetcher": "none",
    "YfinanceFetcher": "auto_adjust",
}

RSI_UNDEFINED = object()


def adjustment_for_source(source: Optional[str]) -> tuple[str, bool]:
    """Return ``(adjustment, forward_adjusted)`` for the hit fetcher name."""
    name = str(source or "")
    if name in QFQ_SOURCES:
        return QFQ_SOURCES[name], True
    if name in UNCONFIRMED_ADJUSTMENTS:
        return UNCONFIRMED_ADJUSTMENTS[name], False
    return "unknown", False


def ema_last(closes: Sequence[float], period: int) -> Optional[float]:
    """Seed with the first ``period`` closes, then ``alpha = 2 / (period + 1)``.

    Returns ``None`` when fewer than ``period`` closes exist. The seed is not
    replaced by ``min_periods=1`` or by a shorter average.
    """
    if period < 1 or len(closes) < period:
        return None
    alpha = 2.0 / (period + 1.0)
    value = sum(closes[:period]) / period
    for price in closes[period:]:
        value = alpha * price + (1.0 - alpha) * value
    return value


def _wilder_series(values: Sequence[float], period: int) -> list[float]:
    """Simple average of the first ``period`` values, then ``(prev*(n-1)+x)/n``."""
    if period < 1 or len(values) < period:
        return []
    current = sum(values[:period]) / period
    series = [current]
    for value in values[period:]:
        current = (current * (period - 1) + value) / period
        series.append(current)
    return series


def wilder_rsi(closes: Sequence[float], period: int = WILDER_PERIOD):
    """Wilder RSI. ``RSI_UNDEFINED`` when both average gain and loss are zero.

    A short sample returns ``None``. Missing values stay missing; nothing is
    filled with 50.
    """
    if len(closes) < period + 1:
        return None
    changes = [closes[index] - closes[index - 1] for index in range(1, len(closes))]
    gains = [change if change > 0 else 0.0 for change in changes]
    losses = [-change if change < 0 else 0.0 for change in changes]
    avg_gain = _wilder_series(gains, period)[-1]
    avg_loss = _wilder_series(losses, period)[-1]
    if avg_gain == 0 and avg_loss == 0:
        return RSI_UNDEFINED
    if avg_loss == 0 and avg_gain > 0:
        return 100.0
    return 100.0 - (100.0 / (1.0 + (avg_gain / avg_loss)))


def true_ranges(bars: Sequence[dict]) -> list[float]:
    """True range starts at the second bar."""
    ranges = []
    for index in range(1, len(bars)):
        high = bars[index]["high"]
        low = bars[index]["low"]
        prev_close = bars[index - 1]["close"]
        ranges.append(max(high - low, abs(high - prev_close), abs(low - prev_close)))
    return ranges


def wilder_atr_series(bars: Sequence[dict], period: int = WILDER_PERIOD) -> list[float]:
    """Wilder ATR aligned so the last value belongs to the last bar."""
    return _wilder_series(true_ranges(bars), period)


def percentile_90(values: Sequence[float]) -> float:
    """1-indexed nearest rank with no interpolation. Rank 90 of a 100-sample."""
    if len(values) != PRIOR_ATR_COUNT:
        raise ValueError("percentile window requires 100 ATR values")
    ordered = sorted(values)
    return ordered[PERCENTILE_RANK - 1]


def volatility_block(prior_atrs: Sequence[float], signal_atr: float) -> dict:
    """Block only when the signal ATR is strictly above the prior 90th rank."""
    rank = percentile_90(prior_atrs)
    return {
        "status": "blocked" if signal_atr > rank else "not_blocked",
        "atr14": signal_atr,
        "percentile_90": rank,
    }


def shanghai_now(as_of: Optional[datetime] = None) -> datetime:
    if as_of is None:
        return datetime.now(SHANGHAI)
    if as_of.tzinfo is None:
        return as_of.replace(tzinfo=SHANGHAI)
    return as_of.astimezone(SHANGHAI)


def _finite(value: Any) -> Optional[float]:
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _as_date(value: Any) -> Optional[date]:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if value is None:
        return None
    text = str(value).strip()
    if len(text) < 10:
        return None
    try:
        return datetime.strptime(text[:10], "%Y-%m-%d").date()
    except ValueError:
        return None


def _records(bars: Any) -> list[dict]:
    if bars is None:
        return []
    if hasattr(bars, "to_dict"):
        return list(bars.to_dict(orient="records"))
    return [dict(item) for item in list(bars)]


def parse_bars(bars: Any) -> tuple[list[dict], Optional[str]]:
    """Parse bars without sorting or filling. Preserve the source order."""
    parsed = []
    for raw in _records(bars):
        bar_date = _as_date(raw.get("date"))
        if bar_date is None:
            return [], "unreadable bar date"
        values = {}
        for field in ("open", "high", "low", "close", "volume"):
            number = _finite(raw.get(field))
            if number is None:
                return [], "non-finite price or volume"
            values[field] = number
        parsed.append({"date": bar_date, **values})
    for prev, curr in zip(parsed, parsed[1:]):
        if curr["date"] <= prev["date"]:
            return [], "dates are not strictly increasing"
    if any(bar["date"].weekday() >= 5 for bar in parsed):
        return [], "filled non-trading calendar day"
    return parsed, None


def calendar_gaps(bars: Sequence[dict]) -> list[dict]:
    """Report missing weekdays. Halt gaps stay empty; they are not filled."""
    gaps = []
    for prev, curr in zip(bars, bars[1:]):
        missing = 0
        day = prev["date"] + timedelta(days=1)
        while day < curr["date"]:
            if day.weekday() < 5:
                missing += 1
            day += timedelta(days=1)
        if missing:
            gaps.append({
                "after": prev["date"].isoformat(),
                "before": curr["date"].isoformat(),
                "missing_weekdays": missing,
            })
    return gaps


def _coverage(bars: Sequence[dict]) -> dict:
    if not bars:
        return {"start": None, "end": None, "completed_bars": 0}
    return {
        "start": bars[0]["date"].isoformat(),
        "end": bars[-1]["date"].isoformat(),
        "completed_bars": len(bars),
    }


def _base_result(
    *,
    stock_code: str,
    source: Optional[str],
    adjustment: str,
    bars: Sequence[dict],
    reason: str,
    signal_date: Optional[str] = None,
    gaps: Optional[list[dict]] = None,
    warmup_notes: Optional[list[str]] = None,
) -> dict:
    return {
        "evaluation": "desk_signal",
        "status": "unconfirmed",
        "stock_code": stock_code,
        "source": source,
        "adjustment": adjustment,
        "timezone": "Asia/Shanghai",
        "coverage": _coverage(bars),
        "signal_date": signal_date,
        "entry_gate": {"status": "unconfirmed"},
        "volatility_block": {"status": "unconfirmed"},
        "warmup_notes": list(warmup_notes or []),
        "calendar_gaps": list(gaps or []),
        "constraints": list(CONSTRAINTS),
        "unconfirmed_reason": reason,
    }


def truncate_signal_bars(bars: Sequence[dict], as_of: datetime) -> tuple[list[dict], Optional[str]]:
    """Drop an incomplete A-share session. A later-than-today bar is unconfirmed."""
    local = shanghai_now(as_of)
    if any(bar["date"] > local.date() for bar in bars):
        return [], "future bar"
    selected = list(bars)
    if selected and selected[-1]["date"] == local.date() and local.time() < SESSION_CLOSE:
        selected = selected[:-1]
    if not selected or not math.isfinite(selected[-1]["close"]):
        return [], "signal close cannot be confirmed"
    return selected, None


def _ema_context(closes: Sequence[float]) -> tuple[dict, list[str]]:
    context = {}
    notes = []
    for period in EMA_PERIODS:
        if period == 200 and len(closes) < 200:
            notes.append("EMA200 omitted because fewer than 200 completed bars")
            continue
        value = ema_last(closes, period)
        if value is None:
            continue
        context[f"ema{period}"] = value
    if len(closes) >= 200:
        if len(closes) < EMA200_WARM_BARS:
            notes.append("EMA200 warmup is insufficient")
            context["ema200_warmup"] = "insufficient"
        else:
            notes.append("EMA200 warmup is sufficient")
            context["ema200_warmup"] = "sufficient"
    else:
        context["ema200_warmup"] = "omitted"
    return context, notes


def evaluate_bars(
    bars: Any,
    *,
    stock_code: str,
    source: Optional[str],
    as_of: Optional[datetime] = None,
) -> dict:
    """Evaluate one in-memory bar window. Indicator math runs only after the gates pass."""
    local = shanghai_now(as_of)
    adjustment, forward_adjusted = adjustment_for_source(source)
    parsed, parse_error = parse_bars(bars)
    if parse_error:
        return _base_result(
            stock_code=stock_code,
            source=source,
            adjustment=adjustment,
            bars=parsed,
            reason=parse_error,
        )
    if not parsed:
        return _base_result(
            stock_code=stock_code,
            source=source,
            adjustment=adjustment,
            bars=[],
            reason="daily bars unavailable",
        )
    selected, signal_error = truncate_signal_bars(parsed, local)
    if signal_error:
        shown = parsed if signal_error == "future bar" else selected
        return _base_result(
            stock_code=stock_code,
            source=source,
            adjustment=adjustment,
            bars=shown,
            reason=signal_error,
        )
    gaps = calendar_gaps(selected)
    signal_date = selected[-1]["date"].isoformat()
    if not forward_adjusted:
        return _base_result(
            stock_code=stock_code,
            source=source,
            adjustment=adjustment,
            bars=selected,
            reason="adjustment is not confirmed forward-adjusted",
            signal_date=signal_date,
            gaps=gaps,
        )
    if len(selected) < MIN_COMPLETED_BARS:
        return _base_result(
            stock_code=stock_code,
            source=source,
            adjustment=adjustment,
            bars=selected,
            reason="percentile window is undefined",
            signal_date=signal_date,
            gaps=gaps,
        )

    closes = [bar["close"] for bar in selected]
    rsi = wilder_rsi(closes, WILDER_PERIOD)
    if rsi is None or rsi is RSI_UNDEFINED:
        return _base_result(
            stock_code=stock_code,
            source=source,
            adjustment=adjustment,
            bars=selected,
            reason="Wilder RSI14 is undefined because average gain and average loss are both zero",
            signal_date=signal_date,
            gaps=gaps,
        )

    prior = selected[-(PRIOR_WINDOW + 1):-1]
    signal = selected[-1]
    breakout = signal["close"] > max(bar["high"] for bar in prior)
    mean_volume = sum(bar["volume"] for bar in prior) / PRIOR_WINDOW
    volume_ok = signal["volume"] >= VOLUME_MULTIPLE * mean_volume
    rsi_ok = rsi > 50
    atrs = wilder_atr_series(selected, WILDER_PERIOD)
    prior_atrs = atrs[:-1][-PRIOR_ATR_COUNT:]
    if len(prior_atrs) != PRIOR_ATR_COUNT:
        return _base_result(
            stock_code=stock_code,
            source=source,
            adjustment=adjustment,
            bars=selected,
            reason="percentile window is undefined",
            signal_date=signal_date,
            gaps=gaps,
        )
    block = volatility_block(prior_atrs, atrs[-1])
    context, notes = _ema_context(closes)
    return {
        "evaluation": "desk_signal",
        "status": "confirmed",
        "stock_code": stock_code,
        "source": source,
        "adjustment": adjustment,
        "timezone": "Asia/Shanghai",
        "coverage": _coverage(selected),
        "signal_date": signal_date,
        "entry_gate": {
            "status": "pass" if breakout and volume_ok and rsi_ok else "fail",
            "breakout": breakout,
            "volume": volume_ok,
            "rsi14": rsi,
            "rsi_above_50": rsi_ok,
        },
        "volatility_block": block,
        "context": context,
        "warmup_notes": notes,
        "calendar_gaps": gaps,
        "constraints": list(CONSTRAINTS),
    }
