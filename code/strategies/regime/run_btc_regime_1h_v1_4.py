#!/usr/bin/env python3
# Regime Strategy v1.4 (1H, BTC) - trend entries at MARKET CLOSE (fixes missed-touch issue)

params = {
    "name": "run_btc_regime_1h_v1_4",
    "exchange": "bitget",
    "symbol": "BTC/USDT:USDT",
    "timeframe": "1h",
    "margin_mode": "isolated",

    "position_size_percentage": 20,
    "use_longs": True,
    "use_shorts": True,

    # Vol targeting
    "base_leverage": 2.0,
    "target_vol_annual": 1.05,
    "vol_lookback_hours": 168,
    "min_leverage": 1.0,
    "max_leverage": 4.0,

    # Regime detection
    "ema_fast": 34,
    "ema_slow": 144,
    "adx_period": 14,
    "adx_trend_min": 18,
    "adx_chop_max": 24,

    # Trend (breakout) - NEW: market-at-close entries
    "trend_entry_mode": "close",   # NEW
    "donchian_period": 18,
    "atr_period": 14,
    "trail_atr_mult": 2.2,
    "breakout_confirm": 0.0,
    "trend_leverage_boost": 1.10,

    # Mean reversion
    "bb_period": 20,
    "bb_std": 1.90,
    "rsi_period": 14,
    "rsi_long_max": 36,
    "rsi_short_min": 64,
    "mr_stop_atr_mult": 1.35,
    "mr_leverage_boost": 0.95,

    "cooldown_bars_after_exit": 1,

    "maker_fee": 0.0002,
    "taker_fee": 0.0006,
}
