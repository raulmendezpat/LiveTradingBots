#!/usr/bin/env python3
# Regime Strategy v4b (1H, BOTH) - tuned to trade while keeping guardrails

params = {
    "name": "run_sol_regime_1h_v4b",
    "exchange": "bitget",
    "symbol": "SOL/USDT:USDT",
    "timeframe": "1h",
    "margin_mode": "isolated",

    # Exposure / leverage (guarded)
    "position_size_percentage": 16,
    "base_leverage": 1.8,
    "target_vol_annual": 1.05,
    "vol_lookback_hours": 168,
    "min_leverage": 1.0,
    "max_leverage": 4.5,

    # 1H trend/chop proxy thresholds (EMA separation)
    "ema_fast": 45,
    "ema_slow": 180,
    "ema_sep_trend": 0.0026,
    "ema_sep_chop": 0.0022,

    # HTF trend filter (4H)
    "htf_ema_fast": 50,
    "htf_ema_slow": 200,
    "htf_neutral_band": 0.004,

    # Trend breakout
    "donchian_period": 16,
    "atr_period": 14,
    "trail_atr_mult": 2.3,
    "breakout_confirm": 0.0007,

    # Mean reversion
    "bb_period": 20,
    "bb_std": 2.4,
    "rsi_period": 14,
    "rsi_long_max": 35,
    "rsi_short_min": 65,
    "mr_stop_atr_mult": 1.8,

    # Circuit breakers
    "dd_pause_pct": 0.16,
    "dd_pause_bars": 96,
    "loss_streak_pause_n": 3,
    "loss_streak_pause_bars": 36,

    "cooldown_bars_after_exit": 3,

    "use_longs": True,
    "use_shorts": True,

    "maker_fee": 0.0002,
    "taker_fee": 0.0006,
}
