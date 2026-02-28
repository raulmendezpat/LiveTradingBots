#!/usr/bin/env python3
# Regime Strategy v5 (1H, BOTH) - PnL-focused with guardrails

params = {
    "name": "run_btc_regime_1h_v5",
    "exchange": "bitget",
    "symbol": "BTC/USDT:USDT",
    "timeframe": "1h",
    "margin_mode": "isolated",

    # Exposure / leverage
    "position_size_percentage": 14,
    "base_leverage": 2.0,
    "target_vol_annual": 1.05,
    "vol_lookback_hours": 168,
    "min_leverage": 1.0,
    "max_leverage": 4.0,

    # Filters to avoid fee/noise churn
    "min_atr_pct": 0.0007,
    "min_edge_pct": 0.002,

    # 1H trend/chop proxy thresholds (EMA separation)
    "ema_fast": 50,
    "ema_slow": 200,
    "ema_sep_trend": 0.0024,
    "ema_sep_chop": 0.0019,

    # HTF trend filter (4H)
    "htf_ema_fast": 50,
    "htf_ema_slow": 200,
    "htf_neutral_band": 0.003,

    # Trend breakout
    "donchian_period": 18,
    "atr_period": 14,
    "trail_atr_mult": 2.2,
    "breakout_confirm": 0.001,
    "trend_tp_atr_mult": 3.2,
    "max_hold_bars": 72,

    # Mean reversion
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
