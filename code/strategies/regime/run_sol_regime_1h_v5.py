#!/usr/bin/env python3
# Regime Strategy v5 (1H, BOTH) - PnL-focused with guardrails

params = {
    "name": "run_sol_regime_1h_v5",
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
    "max_leverage": 3.8,

    # Filters to avoid fee/noise churn
    "min_atr_pct": 0.001,
    "min_edge_pct": 0.0025,

    # 1H trend/chop proxy thresholds (EMA separation)
    "ema_fast": 45,
    "ema_slow": 180,
    "ema_sep_trend": 0.0028,
    "ema_sep_chop": 0.0021,

    # HTF trend filter (4H)
    "htf_ema_fast": 50,
    "htf_ema_slow": 200,
    "htf_neutral_band": 0.0035,

    # Trend breakout
    "donchian_period": 18,
    "atr_period": 14,
    "trail_atr_mult": 2.4,
    "breakout_confirm": 0.0011,
    "trend_tp_atr_mult": 3.0,
    "max_hold_bars": 84,

    # Mean reversion
    "bb_period": 20,
    "bb_std": 2.7,
    "rsi_period": 14,
    "rsi_long_max": 31,
    "rsi_short_min": 69,
    "mr_stop_atr_mult": 1.7,

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
