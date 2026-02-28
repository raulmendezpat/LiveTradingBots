#!/usr/bin/env python3
# Regime + Volatility Targeting Bot (1H, BOTH) - v3 (Aggressive w/ Guardrails)
# Goal: keep higher trade frequency than v1, but avoid v2 blow-ups.
# Notes:
# - Exposure reduced vs v2
# - Vol-target lowered and max leverage capped
# - Regime filters still permissive, but MR entries less "always on"
# - Trailing stop slightly looser vs v2 to reduce churn

params = {
    "name": "run_btc_regime_1h_v3",

    "exchange": "bitget",
    "symbol": "BTC/USDT:USDT",
    "timeframe": "1h",
    "margin_mode": "isolated",

    # EXPOSURE (guardrails)
    "position_size_percentage": 30,     # notional BEFORE leverage
    "use_longs": True,
    "use_shorts": True,

    # Leverage / vol targeting (guardrails)
    "base_leverage": 2.2,
    "target_vol_annual": 1.2,         # annualized
    "vol_lookback_hours": 168,
    "min_leverage": 1.0,
    "max_leverage": 6.0,

    # Regime detection (moderately permissive)
    "ema_fast": 50,
    "ema_slow": 200,
    "adx_period": 14,
    "adx_trend_min": 18,
    "adx_chop_max": 25,

    # Trend breakout (more signals than v1, less churn than v2)
    "donchian_period": 12,
    "atr_period": 14,
    "trail_atr_mult": 1.9,
    "breakout_confirm": 0.0005,                # 0.05% buffer to reduce noise breakouts
    "trend_leverage_boost": 1.35,

    # Mean reversion (still active, but less trigger-happy than v2)
    "bb_period": 20,
    "bb_std": 2.2,                              # wider than 2.0 => fewer low-edge touches
    "rsi_period": 14,
    "rsi_long_max": 38,
    "rsi_short_min": 62,
    "mr_stop_atr_mult": 1.25,
    "mr_leverage_boost": 1.0,

    # Fees (override in CLI if needed)
    "maker_fee": 0.0002,
    "taker_fee": 0.0006,

    # Safety
    "cooldown_bars_after_exit": 2,              # reduce rapid churn
}
