# -*- coding: utf-8 -*-
"""Offline Desk Signal rules. Fixed series only; no quotes and no database."""

from datetime import date, datetime, time, timedelta

import pandas as pd
import pytest

from src.agent.tools.desk_signal_rules import (
    SHANGHAI,
    ema_last,
    evaluate_bars,
    volatility_block,
    wilder_atr_series,
    wilder_rsi,
)

FORBIDDEN = ("sell", "exit", "stop", "target", "sizing", "pivot")


def _weekdays(count: int, start: date = date(2024, 1, 2)):
    bars = []
    day = start
    while len(bars) < count:
        if day.weekday() < 5:
            close = 100.0 + len(bars) * 0.1
            bars.append({
                "date": day,
                "open": close - 0.05,
                "high": close,
                "low": close - 1.0,
                "close": close,
                "volume": 1000.0,
            })
        day += timedelta(days=1)
    bars[-1]["volume"] = 2000.0
    return bars


def _as_of(bars, clock: time = time(15, 0)):
    return datetime.combine(bars[-1]["date"], clock, tzinfo=SHANGHAI)


def _evaluate(bars, source="EfinanceFetcher", clock=time(15, 0)):
    return evaluate_bars(bars, stock_code="600519", source=source, as_of=_as_of(bars, clock))


def _assert_no_order_fields(payload) -> None:
    def walk(value):
        if isinstance(value, dict):
            for key, item in value.items():
                assert str(key).lower() not in FORBIDDEN
                walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)
        elif isinstance(value, str):
            lowered = value.lower()
            for word in FORBIDDEN:
                assert word not in lowered.split()

    walk(payload)


def _assert_no_indicator_estimates(result) -> None:
    assert result["status"] == "unconfirmed"
    assert result["entry_gate"] == {"status": "unconfirmed"}
    assert result["volatility_block"] == {"status": "unconfirmed"}
    blob = str(result)
    for key in ("rsi14", "atr14", "percentile_90", "ema8", "ema21", "ema50", "ema200"):
        assert key not in result
        assert key not in blob
    _assert_no_order_fields(result)


def test_ema_diverges_from_sma_after_seed_and_is_empty_before_period():
    closes = [1.0] * 8 + [10.0, 1.0]
    seeded = ema_last(closes[:8], 8)
    assert seeded == pytest.approx(sum(closes[:8]) / 8)
    ema8 = ema_last(closes, 8)
    sma8 = sum(closes[-8:]) / 8
    assert ema8 is not None
    assert ema8 != pytest.approx(sma8)
    assert ema_last(closes[:7], 8) is None
    assert ema_last(list(range(199)), 200) is None


def test_wilder_rsi_and_atr_diverge_from_ewm_fill_50():
    closes = [10, 11, 10.5, 12, 11.5, 13, 12.5, 14, 13.5, 15, 14.2, 16, 15.4, 17, 16.1, 18, 17.2]
    ours = wilder_rsi(closes, 14)
    changes = pd.Series(closes).diff()
    gain = changes.clip(lower=0)
    loss = -changes.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / 14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / 14, adjust=False).mean()
    ewm_rsi = (100 - (100 / (1 + avg_gain / avg_loss))).fillna(50)
    assert isinstance(ours, float)
    assert ours != pytest.approx(50)
    assert float(ewm_rsi.iloc[-1]) != pytest.approx(ours)

    bars = []
    for close in closes:
        bars.append({
            "date": date(2024, 1, 2),
            "open": close,
            "high": close + 1.2,
            "low": close - 0.4,
            "close": close,
            "volume": 1000.0,
        })
    atrs = wilder_atr_series(bars, 14)
    frame = pd.DataFrame(bars)
    previous_close = frame["close"].shift(1)
    true_range = pd.concat([
        frame["high"] - frame["low"],
        (frame["high"] - previous_close).abs(),
        (frame["low"] - previous_close).abs(),
    ], axis=1).max(axis=1)
    ewm_atr = true_range.ewm(alpha=1 / 14, adjust=False).mean().fillna(50)
    assert atrs[-1] != pytest.approx(float(ewm_atr.iloc[-1]))


def test_dual_zero_rsi_is_unconfirmed_and_not_filled_with_50():
    bars = _weekdays(130)
    for bar in bars:
        bar["open"] = bar["high"] = bar["low"] = bar["close"] = 20.0
    result = _evaluate(bars)
    assert "both zero" in result["unconfirmed_reason"]
    _assert_no_indicator_estimates(result)
    assert "50" not in str(result["entry_gate"])


def test_percentile_excludes_signal_atr_and_equal_does_not_block():
    prior = [float(index) for index in range(1, 101)]
    equal = volatility_block(prior, 90.0)
    above = volatility_block(prior, 90.5)
    assert equal["status"] == "not_blocked"
    assert equal["percentile_90"] == 90.0
    assert above["status"] == "blocked"

    bars = _weekdays(130)
    bars[-1]["high"] = bars[-1]["close"] + 40
    bars[-1]["low"] = bars[-1]["close"] - 40
    seen = {}
    real = volatility_block

    def _capture(prior_atrs, signal_atr):
        seen["prior"] = list(prior_atrs)
        seen["signal"] = signal_atr
        return real(prior_atrs, signal_atr)

    from src.agent.tools import desk_signal_rules
    original = desk_signal_rules.volatility_block
    desk_signal_rules.volatility_block = _capture
    try:
        result = _evaluate(bars)
    finally:
        desk_signal_rules.volatility_block = original

    series = wilder_atr_series(bars, 14)
    assert seen["prior"] == series[:-1][-100:]
    assert seen["signal"] == pytest.approx(series[-1])
    assert seen["signal"] not in seen["prior"]
    assert result["volatility_block"]["status"] == "blocked"
    assert result["status"] == "confirmed"


def test_constant_range_equal_to_percentile_does_not_block_and_entry_passes():
    result = _evaluate(_weekdays(130))
    assert result["status"] == "confirmed"
    assert result["adjustment"] == "qfq"
    assert result["timezone"] == "Asia/Shanghai"
    assert result["entry_gate"]["status"] == "pass"
    assert result["entry_gate"]["rsi14"] == pytest.approx(100.0)
    assert result["volatility_block"]["status"] == "not_blocked"
    assert result["volatility_block"]["atr14"] == pytest.approx(result["volatility_block"]["percentile_90"])
    assert result["context"]["ema200_warmup"] == "omitted"
    assert "ema200" not in result["context"]
    assert "EMA200 omitted" in " ".join(result["warmup_notes"])
    _assert_no_order_fields(result)


@pytest.mark.parametrize("source,adjustment", [
    ("PytdxFetcher", "none"),
    ("LongbridgeFetcher", "none"),
    ("YfinanceFetcher", "auto_adjust"),
    ("TushareFetcher", "unknown"),
])
def test_non_forward_adjusted_sources_are_unconfirmed(source, adjustment):
    result = _evaluate(_weekdays(130), source=source)
    assert result["adjustment"] == adjustment
    assert "forward-adjusted" in result["unconfirmed_reason"]
    _assert_no_indicator_estimates(result)


def test_future_bar_filled_gap_and_non_finite_are_unconfirmed():
    future = _weekdays(130)
    future.append({
        "date": future[-1]["date"] + timedelta(days=1),
        "open": 1, "high": 2, "low": 0.5, "close": 1.5, "volume": 10,
    })
    if future[-1]["date"].weekday() >= 5:
        future[-1]["date"] = future[-1]["date"] + timedelta(days=2)
    future_result = evaluate_bars(
        future, stock_code="600519", source="EfinanceFetcher", as_of=_as_of(future[:-1]),
    )
    assert future_result["unconfirmed_reason"] == "future bar"
    _assert_no_indicator_estimates(future_result)

    filled = _weekdays(130)
    friday_index = next(index for index, bar in enumerate(filled) if bar["date"].weekday() == 4)
    saturday = filled[friday_index]["date"] + timedelta(days=1)
    filled.insert(friday_index + 1, dict(filled[friday_index], date=saturday))
    filled_result = _evaluate(filled)
    assert filled_result["unconfirmed_reason"] == "filled non-trading calendar day"
    _assert_no_indicator_estimates(filled_result)

    dirty = _weekdays(130)
    dirty[12]["close"] = float("nan")
    dirty_result = _evaluate(dirty)
    assert dirty_result["unconfirmed_reason"] == "non-finite price or volume"
    _assert_no_indicator_estimates(dirty_result)

    reversed_bars = list(reversed(_weekdays(130)))
    assert _evaluate(reversed_bars)["unconfirmed_reason"] == "dates are not strictly increasing"


def test_halt_gap_is_reported_and_not_filled():
    bars = _weekdays(130)
    removed = bars.pop(40)
    result = _evaluate(bars)
    assert result["status"] == "confirmed"
    assert result["coverage"]["completed_bars"] == 129
    assert any(gap["missing_weekdays"] >= 1 for gap in result["calendar_gaps"])
    assert removed["date"].isoformat() not in {
        result["coverage"]["start"], result["coverage"]["end"], result["signal_date"],
    }


def test_short_windows_do_not_emit_estimated_indicators(monkeypatch):
    def _forbidden(*_args, **_kwargs):
        raise AssertionError("indicator estimated from a short window")

    monkeypatch.setattr("src.agent.tools.desk_signal_rules.wilder_rsi", _forbidden)
    monkeypatch.setattr("src.agent.tools.desk_signal_rules.ema_last", _forbidden)
    monkeypatch.setattr("src.agent.tools.desk_signal_rules.wilder_atr_series", _forbidden)

    for count in (37, 114):
        result = _evaluate(_weekdays(count))
        assert result["unconfirmed_reason"] == "percentile window is undefined"
        assert result["volatility_block"]["status"] == "unconfirmed"
        _assert_no_indicator_estimates(result)


def test_incomplete_session_is_not_the_signal_bar():
    bars = _weekdays(130)
    before_close = _evaluate(bars, clock=time(14, 59))
    assert before_close["signal_date"] == bars[-2]["date"].isoformat()
    assert before_close["coverage"]["end"] == bars[-2]["date"].isoformat()
    after_close = _evaluate(bars, clock=time(15, 0))
    assert after_close["signal_date"] == bars[-1]["date"].isoformat()
