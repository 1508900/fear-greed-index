import streamlit as st

FRED_API_KEY = st.secrets["FRED_API_KEY"]

CONFIGS = {
    "sp500": {
        "name":       "S&P 500 - US Fear & Greed",
        "flag":       "US",
        "index":      "^GSPC",
        "volatility": "^VIX",
        "bond_yield": "^TNX",
        "hy_spread":  "BAMLH0A0HYM2",
        "ma_period":  125,
        "lookback":   504,
    },
    "eurostoxx": {
        "name":       "EuroStoxx 50 - EU Fear & Greed",
        "flag":       "EU",
        "index":      "^STOXX50E",
        "volatility": "^V2TX",
        "bond_yield": "^FBTP",
        "hy_spread":  "BAMLHE00EHYIOAS",
        "ma_period":  125,
        "lookback":   504,
    },
}
