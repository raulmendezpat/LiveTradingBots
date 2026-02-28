#!/usr/bin/env python3
# Regime Strategy v4b (1H, BOTH) - tuned to trade while keeping guardrails

params = {
    "name": "run_btc_regime_1h_v4b",
    "exchange": "bitget",
    "symbol": "BTC/USDT:USDT",
    "timeframe": "1h",
    "margin_mode": "isolated",

    # Exposure / leverage (guarded)
    "position_size_percentage": 18,
    "base_leverage": 2.0,
    "target_vol_annual": 1.1,
    "vol_lookback_hours": 168,
    "min_leverage": 1.0,
    "max_leverage": 5.0,

    # 1H trend/chop proxy thresholds (EMA separation)
    "ema_fast": 50,
    "ema_slow": 200,
    "ema_sep_trend": 0.0022,
    "ema_sep_chop": 0.002,

    # HTF trend filter (4H)
    "htf_ema_fast": 50,
    "htf_ema_slow": 200,
    "htf_neutral_band": 0.0035,

    # Trend breakout
    "donchian_period": 16,
    "atr_period": 14,
    "trail_atr_mult": 2.1,
    "breakout_confirm": 0.0006,

    # Mean reversion
    "bb_period": 20,
    "bb_std": 2.35,
    "rsi_period": 14,
    "rsi_long_max": 34,
    "rsi_short_min": 66,
    "mr_stop_atr_mult": 1.7,

    # Circuit breakers
    "dd_pause_pct": 0.14,
    "dd_pause_bars": 72,
    "loss_streak_pause_n": 3,
    "loss_streak_pause_bars": 24,

    "cooldown_bars_after_exit": 3,

    "use_longs": True,
    "use_shorts": True,

    "maker_fee": 0.0002,
    "taker_fee": 0.0006,
}
