# rtai/config.py
"""
RTAI Configuration - Parametri centralizzati per oscillatori RSI-like
===================================================================

Configurazione unificata per evitare magic numbers sparsi nel codice.
Tutti gli oscillatori usano questi parametri di default.
"""

# Parametri oscillatori RSI-like
OSC = {
    "hl_sigma": 30,        # Half-life per varianza EW (30 bar = 30 min)
    "alpha_fast": 0.35,    # EWMA veloce (reattività)
    "alpha_slow": 0.12,    # EWMA lenta (stabilità)
    "scale": 1.5,          # Scala tanh per mapping 0-100
}

# Soglie con isteresi
THRESH = {
    "enter": 80,           # Soglia ingresso upper
    "exit": 75,            # Soglia uscita upper (isteresi)
    "enter_low": 20,       # Soglia ingresso lower
    "exit_low": 25,        # Soglia uscita lower (isteresi)
}

# Soglie speciali per indicatori specifici
THRESH_SPECIAL = {
    "vpin": {"enter": 85, "exit": 80, "enter_low": 15, "exit_low": 20},
    "liq": {"enter": 85, "exit": 80, "enter_low": 15, "exit_low": 20},
    "cvd": {"enter": 85, "exit": 80, "enter_low": 15, "exit_low": 20},
    "basis": {"enter": 75, "exit": 70, "enter_low": 25, "exit_low": 30},
}

# Cooldown per segnali (secondi)
COOLDOWN_SEC = 60  # 1 minuto default

# Event accumulator windows (secondi)
EVENT_WINDOWS = {
    "liquidation": 5.0,    # Cluster liquidazioni in 5s
    "big_orders": 3.0,     # Cluster big orders in 3s
}

# Parametri per indicatori specifici
INDICATORS = {
    "ofi": {
        **OSC,
        "scale": 1.7,  # Leggermente più reattivo
    },
    "vpin": {
        **OSC,
        "alpha_fast": 0.25,  # Meno reattivo
        "scale": 1.3,
    },
    "kyle": {
        **OSC,
        "alpha_fast": 0.25,
        "scale": 1.4,
    },
    "lpi": {
        **OSC,
        "scale": 1.8,  # Più sensibile per liquidazioni
    },
    "cvd": {
        **OSC,
        "scale": 1.6,
    },
    "basis": {
        **OSC,
        "alpha_fast": 0.15,  # Molto più lento
        "alpha_slow": 0.08,
        "scale": 1.2,
    }
}

# Big orders threshold (USD)
BIG_ORDER_THRESHOLD = 100000.0

# Plotting configuration
PLOT = {
    "dpi": 120,
    "figsize": (16, 12),
    "rsi_colors": {
        "line": "#2E86AB",
        "upper_band": "#A23B72",
        "lower_band": "#F18F01",
        "extreme_up": "#C73E1D",
        "extreme_down": "#1B998B",
    }
}
