#!/usr/bin/env python3
# Regime Strategy v1.2 (1H, BTC) - based on v1/v1.1, tuned to trade more and reduce fee-churn

params = {
    "name": "run_btc_regime_1h_v1_2",
    "exchange": "bitget",
    "symbol": "BTC/USDT:USDT",
    "timeframe": "1h",
    "margin_mode": "isolated",

    # Exposure
    "position_size_percentage": 14,   # slightly lower than v1.1 to keep DD stable
    "base_leverage": 2.0,
    "use_longs": True,
    "use_shorts": True,

    # Vol targeting inputs (kept compatible with backtester)
    "vol_lookback_hours": 168,
    "target_vol_annual": 1.10,        # mild
    "min_leverage": 1.0,
    "max_leverage": 3.0,

    # Regime detection (more permissive than v1.1 -> more signals)
    "ema_fast": 34,
    "ema_slow": 144,
    "ema_sep_trend": 0.0014,
    "ema_sep_chop": 0.0010,

    # Trend leg (easier breakouts)
    "donchian_period": 18,
    "breakout_confirm": 0.00035,
    "atr_period": 14,
    "adx_period": 14,
    "adx_trend_min": 15,
    "trail_atr_mult": 2.1,
    "trend_tp_atr_mult": 2.8,
    "max_hold_bars": 72,

    # Mean-reversion leg (kept but slightly stricter RSI to avoid middling entries)
    "bb_period": 20,
    "bb_std": 2.6,
    "rsi_period": 14,
    "rsi_long_max": 30,
    "rsi_short_min": 70,
    "mr_stop_atr_mult": 1.7,
    "mr_tp_mid": True,

    # Filters to reduce fee-churn but not kill trading
    "min_atr_pct": 0.00025,
    "cooldown_bars_after_exit": 1,

    # Risk circuit breakers
    "dd_pause_pct": 0.12,
    "dd_pause_bars": 72,
    "loss_streak_pause_n": 3,
    "loss_streak_pause_bars": 24,

    # Fees
    "maker_fee": 0.0002,
    "taker_fee": 0.0006,
}
