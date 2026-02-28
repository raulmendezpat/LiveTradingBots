#!/usr/bin/env python3
# Regime Strategy v1.3 (1H, BTC) - REAL knobs (this backtester does NOT use ema_sep_*)
# Goal: more trades than v1, while keeping DD reasonable.
# Changes vs v1:
# - EMA faster/smaller window for direction (34/144)
# - ADX gates relaxed so both regimes fire more often
# - BB std tightened so bands are touched more often
# - RSI thresholds slightly relaxed

params = {
    "name": "run_btc_regime_1h_v1_3",

    "exchange": "bitget",
    "symbol": "BTC/USDT:USDT",
    "timeframe": "1h",
    "margin_mode": "isolated",

    # Risk & sizing
    "position_size_percentage": 22,
    "use_longs": True,
    "use_shorts": True,

    # Volatility targeting (annualized)
    "base_leverage": 2.0,
    "target_vol_annual": 1.10,
    "vol_lookback_hours": 168,
    "min_leverage": 1.0,
    "max_leverage": 5.0,

    # Regime detection + direction filter
    "ema_fast": 34,
    "ema_slow": 144,
    "adx_period": 14,
    "adx_trend_min": 18,      # was 22
    "adx_chop_max": 24,       # was 18 (allow MR more often)

    # Trend strategy (breakout)
    "donchian_period": 18,
    "atr_period": 14,
    "trail_atr_mult": 2.1,
    "breakout_confirm": 0.0,
    "trend_leverage_boost": 1.15,

    # Mean reversion strategy (tighter bands => more touches)
    "bb_period": 20,
    "bb_std": 1.85,           # was 2.0 in v1; slightly tighter for more entries
    "rsi_period": 14,
    "rsi_long_max": 36,       # was 32
    "rsi_short_min": 64,      # was 68
    "mr_stop_atr_mult": 1.35,
    "mr_leverage_boost": 0.95,

    "cooldown_bars_after_exit": 1,

    "maker_fee": 0.0002,
    "taker_fee": 0.0006,
}
