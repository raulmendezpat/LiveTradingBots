#!/usr/bin/env python3
# Regime Strategy v6 (1H, BTC, trend-only) - aim: improve PF by avoiding MR churn

params = {
    "name": "run_btc_regime_1h_v6",
    "exchange": "bitget",
    "symbol": "BTC/USDT:USDT",
    "timeframe": "1h",
    "margin_mode": "isolated",

    # Enable/disable legs
    "enable_trend": True,
    "enable_mr": False,

    # Exposure / leverage
    "position_size_percentage": 14,
    "base_leverage": 2.0,
    "target_vol_annual": 1.05,
    "vol_lookback_hours": 168,
    "min_leverage": 1.0,
    "max_leverage": 4.0,

    # Filters to avoid fee/noise churn
    "min_atr_pct": 0.0007,
    "min_edge_pct": 0.0020,

    # 1H trend/chop proxy thresholds (EMA separation)
    "ema_fast": 50,
    "ema_slow": 200,
    "ema_sep_trend": 0.0026,
    "ema_sep_chop": 0.0020,

    # HTF trend filter (4H)
    "htf_ema_fast": 50,
    "htf_ema_slow": 200,
    "htf_neutral_band": 0.0030,

    # Trend breakout
    "donchian_period": 20,
    "atr_period": 14,
    "trail_atr_mult": 2.1,
    "breakout_confirm": 0.0012,   # stricter than v5 to reduce false breaks
    "trend_tp_atr_mult": 3.5,     # take profit earlier to avoid giveback
    "max_hold_bars": 72,          # time stop

    # MR params (unused when enable_mr=False)
    "bb_period": 20,
    "bb_std": 2.6,
    "rsi_period": 14,
    "rsi_long_max": 30,
    "rsi_short_min": 70,
    "mr_stop_atr_mult": 1.6,

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
