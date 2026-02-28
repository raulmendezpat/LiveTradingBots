#!/usr/bin/env python3
# Regime Strategy v4 (1H, BOTH) - HTF filter + circuit breakers

params = {
    "name": "run_sol_regime_1h_v4",
    "exchange": "bitget",
    "symbol": "SOL/USDT:USDT",
    "timeframe": "1h",
    "margin_mode": "isolated",

    # Exposure / leverage
    "position_size_percentage": 12,
    "base_leverage": 1.8,
    "target_vol_annual": 1.0,
    "vol_lookback_hours": 168,
    "min_leverage": 1.0,
    "max_leverage": 3.5,

    # 1H trend/chop proxy thresholds (EMA separation)
    "ema_fast": 45,
    "ema_slow": 180,
    "ema_sep_trend": 0.0032,
    "ema_sep_chop": 0.0018,

    # HTF trend filter (4H)
    "htf_ema_fast": 50,
    "htf_ema_slow": 200,
    "htf_neutral_band": 0.002,

    # Trend breakout
    "donchian_period": 22,
    "atr_period": 14,
    "trail_atr_mult": 2.7,
    "breakout_confirm": 0.0018,

    # Mean reversion
    "bb_period": 20,
    "bb_std": 2.7,
    "rsi_period": 14,
    "rsi_long_max": 31,
    "rsi_short_min": 69,
    "mr_stop_atr_mult": 1.9,

    # Circuit breakers
    "dd_pause_pct": 0.14,
    "dd_pause_bars": 96,
    "loss_streak_pause_n": 3,
    "loss_streak_pause_bars": 36,

    "cooldown_bars_after_exit": 4,

    "use_longs": True,
    "use_shorts": True,

    "maker_fee": 0.0002,
    "taker_fee": 0.0006,
}
