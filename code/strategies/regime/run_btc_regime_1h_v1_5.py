# -*- coding: utf-8 -*-
"""
Regime strategy – BTC 1h – v1.5

BTC v1.4 was slightly profitable but with tiny edge (PF ~1.09).
v1.5 focuses on robustness & cost-awareness:
1) Add volatility floor (min ATR%) to avoid low-edge churn.
2) Gap protection to avoid rare large adverse moves.
3) Cooldown after stop-loss.
4) Slightly better RR (TP a bit larger than SL).

This is a params-only file intended to plug into your existing backtest_regime.py like v1.4.
"""

BOT_NAME = "run_btc_regime_1h_v1_5"

params = {
    "symbol": "BTC/USDT:USDT",
    "timeframe": "1h",
    "margin_mode": "isolated",
    "leverage": 2,

    "balance_fraction": 1.0,
    "position_size_percentage": 25,

    "regime_lookback": 200,
    "trend_strength_period": 14,
    "trend_strength_min": 17,

    # Slight RR improvement vs v1.4
    "stop_loss_pct": 0.010,             # 1.0%
    "take_profit_pct": 0.015,           # 1.5%

    "use_trailing_stop": True,
    "trailing_stop_pct": 0.008,

    "enable_gap_filter": True,
    "max_entry_gap_pct": 0.015,         # BTC is less jumpy than SOL on 1h

    "enable_min_atr_filter": True,
    "atr_period": 14,
    "min_atr_pct": 0.004,               # 0.4%

    "enable_cooldown_after_sl": True,
    "cooldown_bars_after_sl": 4,

    "use_longs": True,
    "use_shorts": True,

    "taker_fee_pct": 0.0006,
    "maker_fee_pct": 0.0002,
    "slippage_pct": 0.0003,

    # --- Backtester-required keys (added for backtest_regime.py compatibility) ---
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

    # Normalize fees (backtester reads maker_fee/taker_fee)
    "maker_fee": 0.0002,
    "taker_fee": 0.0006,
}
