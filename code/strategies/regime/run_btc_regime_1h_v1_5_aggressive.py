
# Aggressive BTC Regime v1.5
# Short-only configuration with improved RR and reduced exposure

params = {

    # --- Symbol & timeframe ---
    "symbol": "BTC/USDT",
    "timeframe": "1h",

    # --- Position & leverage (reduced exposure) ---
    "position_size_percentage": 15,
    "leverage": 1,

    # --- Directional bias ---
    "use_longs": False,
    "use_shorts": True,

    # --- Aggressive Risk/Reward ---
    "stop_loss_pct": 0.007,      # 0.7%
    "take_profit_pct": 0.018,    # 1.8%

    # --- Required by backtest_regime.py ---
    "ema_fast": 20,
    "ema_slow": 50,
    "atr_period": 14,
    "donchian_period": 20,
    "bb_period": 20,
    "bb_std": 2.0,
    "rsi_period": 14,
    "rsi_long_max": 35,
    "rsi_short_min": 65,
    "trail_atr_mult": 2.5,
    "mr_stop_atr_mult": 2.0,
    "adx_period": 14,
    "adx_trend_min": 18,
    "adx_chop_max": 16,
    "breakout_confirm": 0.0,
    "cooldown_bars_after_exit": 6,

    # --- Fees (compatible with backtester) ---
    "maker_fee": 0.0002,
    "taker_fee": 0.0006,

    # --- Optional volatility settings ---
    "base_leverage": 1.0,
    "target_vol_annual": 1.0,
    "min_leverage": 1.0,
    "max_leverage": 3.0,
    "trend_leverage_boost": 1.0,
    "mr_leverage_boost": 1.0,
    "vol_lookback_hours": 168,
}
