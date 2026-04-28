import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import yfinance as yf
import requests
from config import CONFIGS, FRED_API_KEY
from datetime import datetime

st.set_page_config(page_title="Fear & Greed Index", page_icon="U+1F4CA", layout="wide")

st.markdown("""
<style>
.block-container { padding-top: 1.5rem; }
.stButton > button {
    background: #3949ab; color: white; border: none;
    padding: .6rem 2rem; border-radius: 8px; font-size: 1rem; font-weight: 600;
}
.stButton > button:hover { background: #5c6bc0; }
</style>
""", unsafe_allow_html=True)

def get_label(score):
    if score <= 25:   return ("Extreme Fear", "#c62828")
    elif score <= 45: return ("Fear", "#ef6c00")
    elif score <= 55: return ("Neutral", "#f9a825")
    elif score <= 75: return ("Greed", "#558b2f")
    else:             return ("Extreme Greed", "#1b5e20")

def normalize(series, lookback, invert=False):
    clean = series.dropna()
    if len(clean) < 20:
        return 50.0
    window = clean.iloc[-lookback:]
    current = float(clean.iloc[-1])
    score = float((window < current).sum() / len(window) * 100)
    result = 100.0 - score if invert else score
    return round(max(0.0, min(100.0, result)), 1)

@st.cache_data(ttl=600, show_spinner=False)
def get_yf(ticker):
    try:
        data = yf.download(ticker, period="2y", progress=False, auto_adjust=True)
        col = "Close"
        if col not in data.columns:
            col = data.columns[0]
        s = data[col].dropna()
        if isinstance(s, pd.DataFrame):
            s = s.iloc[:, 0]
        return s
    except Exception:
        return pd.Series(dtype=float)

@st.cache_data(ttl=600, show_spinner=False)
def get_fred(series_id):
    try:
        start = (pd.Timestamp.today() - pd.DateOffset(years=3)).date()
        url = (f"https://api.stlouisfed.org/fred/series/observations"
               f"?series_id={series_id}&api_key={FRED_API_KEY}"
               f"&file_type=json&observation_start={start}")
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        obs = r.json().get("observations", [])
        df = pd.DataFrame(obs)
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df.index = pd.to_datetime(df["date"])
        return df["value"].dropna()
    except Exception:
        return pd.Series(dtype=float)

def compute(cfg):
    lb = cfg["lookback"]
    prices = get_yf(cfg["index"])
    vix    = get_yf(cfg["volatility"])
    bond   = get_yf(cfg["bond_yield"])
    hy     = get_fred(cfg["hy_spread"])

    if len(prices) < 30:
        dummy = {k: 50.0 for k in ["Momentum","Strength (RSI)","Breadth","Junk Bond","Volatility","Safe Haven"]}
        return {"score": 50.0, "label": "Neutral", "color": "#f9a825",
                "components": dummy, "name": cfg["name"], "flag": cfg["flag"]}

    ma = prices.rolling(cfg["ma_period"]).mean()
    momentum = (prices / ma - 1) * 100

    delta = prices.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rsi   = 100 - (100 / (1 + gain / (loss + 1e-9)))

    up      = (prices.diff() > 0).rolling(20).sum()
    down    = (prices.diff() < 0).rolling(20).sum()
    breadth = (up / (down + 1)) * 50

    eq_ret  = prices.pct_change(20) * 100
    if len(bond) > 20:
        bd_chg  = bond.reindex(eq_ret.index, method="ffill").diff(20)
        safehav = eq_ret - bd_chg
    else:
        safehav = eq_ret

    components = {
        "Momentum":       normalize(momentum, lb),
        "Strength (RSI)": normalize(rsi,      lb),
        "Breadth":        normalize(breadth,  lb),
        "Junk Bond":      normalize(hy,       lb, invert=True) if len(hy) > 0 else 50.0,
        "Volatility":     normalize(vix,      lb, invert=True) if len(vix) > 0 else 50.0,
        "Safe Haven":     normalize(safehav,  lb),
    }
    total = round(sum(components.values()) / len(components), 1)
    lbl, color = get_label(total)
    return {"score": total, "label": lbl, "color": color,
            "components": components, "name": cfg["name"], "flag": cfg["flag"]}

def make_gauge(result):
    score = result["score"]
    color = result["color"]
    lbl   = result["label"]
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number={"font": {"size": 60, "color": color}},
        title={"text": f"{result['flag']} {result['name']}<br><span style='font-size:1.2em;color:{color}'>{lbl}</span>",
               "font": {"size": 15}},
        gauge={
            "axis": {"range": [0, 100], "tickvals": [0,25,45,55,75,100], "tickfont": {"size":10}},
            "bar": {"color": color, "thickness": 0.22},
            "bgcolor": "#1e2130", "borderwidth": 0,
            "steps": [
                {"range": [0,  25],  "color": "#3b0000"},
                {"range": [25, 45],  "color": "#3b1a00"},
                {"range": [45, 55],  "color": "#3b3000"},
                {"range": [55, 75],  "color": "#153000"},
                {"range": [75, 100], "color": "#001a00"},
            ],
            "threshold": {"line": {"color": "white", "width": 3},
                          "thickness": 0.75, "value": score},
        }
    ))
    fig.update_layout(height=290, margin=dict(l=20,r=20,t=70,b=10),
                      paper_bgcolor="#0e1117", font_color="white")
    return fig

def make_bars(result):
    names  = list(result["components"].keys())
    values = list(result["components"].values())
    colors = [get_label(v)[1] for v in values]
    fig = go.Figure(go.Bar(
        x=values, y=names, orientation="h",
        marker_color=colors,
        text=[f"{v:.0f}" for v in values],
        textposition="outside"
    ))
    fig.add_vline(x=50, line_dash="dash", line_color="#555", line_width=1)
    fig.update_layout(
        xaxis=dict(range=[0,118], showgrid=False, zeroline=False),
        yaxis=dict(autorange="reversed"),
        height=260, margin=dict(l=10,r=50,t=15,b=15),
        paper_bgcolor="#0e1117", plot_bgcolor="#1e2130", font_color="white"
    )
    return fig

def make_radar(result):
    cats = list(result["components"].keys())
    vals = list(result["components"].values())
    cats_c = cats + [cats[0]]
    vals_c = vals + [vals[0]]
    color = result["color"]
    fig = go.Figure(go.Scatterpolar(
        r=vals_c, theta=cats_c, fill="toself",
        fillcolor="rgba(100,100,100,0.15)",
        line=dict(color=color, width=2)
    ))
    fig.update_layout(
        polar=dict(
            bgcolor="#1e2130",
            radialaxis=dict(range=[0,100], tickfont=dict(size=9), gridcolor="#444"),
            angularaxis=dict(tickfont=dict(size=10), gridcolor="#444")
        ),
        showlegend=False, height=300,
        margin=dict(l=50,r=50,t=20,b=20),
        paper_bgcolor="#0e1117", font_color="white"
    )
    return fig

# ── UI ────────────────────────────────────────────────────────────────────────
st.title("Fear & Greed Index")
st.caption("S&P 500 (USA) vs EuroStoxx 50 (Europa) - Inspirado en CNN Money")

col_btn, col_ts = st.columns([1, 3])
with col_btn:
    refresh = st.button("Actualizar datos")
with col_ts:
    st.markdown(
        f"<small style='color:#888'>Actualizado: {datetime.now().strftime('%d/%m/%Y %H:%M')}</small>",
        unsafe_allow_html=True
    )

if refresh:
    st.cache_data.clear()

with st.spinner("Descargando datos de mercado..."):
    results = {}
    for key, cfg in CONFIGS.items():
        results[key] = compute(cfg)

keys = list(results.keys())
r1 = results[keys[0]]
r2 = results[keys[1]]

c1, c2 = st.columns(2)
with c1:
    st.plotly_chart(make_gauge(r1), use_container_width=True)
with c2:
    st.plotly_chart(make_gauge(r2), use_container_width=True)

st.divider()
st.subheader("Detalle por componente")
c3, c4 = st.columns(2)

for col, r in zip([c3, c4], [r1, r2]):
    with col:
        st.markdown(f"**{r['flag']} {r['name']}**")
        t1, t2 = st.tabs(["Barras", "Radar"])
        with t1:
            st.plotly_chart(make_bars(r), use_container_width=True)
        with t2:
            st.plotly_chart(make_radar(r), use_container_width=True)


st.divider()
st.subheader("Que mide cada componente")

COMPONENT_DESC = {
    "Momentum":       "Compara el precio actual del indice con su media movil de 125 dias. Si cotiza por encima, hay euforia; si cae por debajo, hay miedo.",
    "Strength (RSI)": "RSI de 14 dias. Mide si el mercado esta sobrecomprado (codicia) o sobrevendido (miedo).",
    "Breadth":        "Proporcion de dias alcistas frente a bajistas en los ultimos 20 dias. Mas dias verdes = mayor codicia.",
    "Junk Bond":      "Diferencial de tipos entre bonos de alto riesgo y bonos seguros (FRED/ICE BofA). Un spread alto indica miedo e incertidumbre.",
    "Volatility":     "VIX (S&P) o VSTOXX (EuroStoxx). A mayor volatilidad, mayor miedo en el mercado.",
    "Safe Haven":     "Retorno relativo de acciones frente a bonos del tesoro. Si los inversores huyen a bonos, hay miedo.",
}

cols_desc = st.columns(3)
items = list(COMPONENT_DESC.items())
for i, (name, desc) in enumerate(items):
    with cols_desc[i % 3]:
        label, color = get_label(50)  # color neutro para el titulo
        st.markdown(
            f"**{name}**  \n{desc}",
            help=desc
        )

st.divider()
st.subheader("Comparativa USA vs Europa")

comp_rows = []
for k in r1["components"]:
    s1 = r1["components"][k]
    s2 = r2["components"].get(k, 50.0)
    comp_rows.append({
        "Componente": k,
        r1["name"]: s1,
        r2["name"]: s2,
        "Diferencia": round(s1 - s2, 1)
    })
comp_rows.append({
    "Componente": "TOTAL",
    r1["name"]: r1["score"],
    r2["name"]: r2["score"],
    "Diferencia": round(r1["score"] - r2["score"], 1)
})

df_cmp = pd.DataFrame(comp_rows)
st.dataframe(df_cmp, use_container_width=True, hide_index=True)

st.divider()
st.caption("Fuentes: Yahoo Finance / FRED ICE BofA. Cache 10 min. Pulsa Actualizar para forzar recarga.")
