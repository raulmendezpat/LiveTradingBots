#!/usr/bin/env python3
# Regime Strategy v4 (1H, BOTH) - HTF filter + circuit breakers

params = {
    "name": "run_btc_regime_1h_v4",
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

    # 1H trend/chop proxy thresholds (EMA separation)
    "ema_fast": 50,
    "ema_slow": 200,
    "ema_sep_trend": 0.0028,
    "ema_sep_chop": 0.0016,

    # HTF trend filter (4H)
    "htf_ema_fast": 50,
    "htf_ema_slow": 200,
    "htf_neutral_band": 0.0015,

    # Trend breakout
    "donchian_period": 24,
    "atr_period": 14,
    "trail_atr_mult": 2.4,
    "breakout_confirm": 0.0015,

    # Mean reversion
    "bb_period": 20,
    "bb_std": 2.6,
    "rsi_period": 14,
    "rsi_long_max": 30,
    "rsi_short_min": 70,
    "mr_stop_atr_mult": 1.8,

    # Circuit breakers
    "dd_pause_pct": 0.12,
    "dd_pause_bars": 72,
    "loss_streak_pause_n": 3,
    "loss_streak_pause_bars": 24,

    "cooldown_bars_after_exit": 4,

    "use_longs": True,
    "use_shorts": True,

    "maker_fee": 0.0002,
    "taker_fee": 0.0006,
}
