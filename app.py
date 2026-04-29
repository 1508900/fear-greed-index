import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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
    if score <= 25:
        return ("Extreme Fear", "#c62828")
    elif score <= 45:
        return ("Fear", "#ef6c00")
    elif score <= 55:
        return ("Neutral", "#f9a825")
    elif score <= 75:
        return ("Greed", "#558b2f")
    else:
        return ("Extreme Greed", "#1b5e20")

def normalize(series, lookback, invert=False):
        clean = series.dropna()
        if len(clean) < 20:
                    return 50.0
                window = clean.iloc[-lookback:]
    current = float(clean.iloc[-1])
    score = float((window < current).sum() / len(window) * 100)
    result = 100.0 - score if invert else score
    return round(max(0.0, min(100.0, result)), 1)

def normalize_rolling(series, lookback, invert=False):
        """Vectorized rolling percentile rank (0-100 scale)."""
    clean = series.dropna()
    if len(clean) < lookback + 5:
                return pd.Series(dtype=float)
            # Use rolling rank: for each point i, compute percentile vs previous `lookback` points
    def pct_rank(x):
                return float((x[:-1] < x[-1]).sum() / (len(x) - 1) * 100)
            rolled = clean.rolling(window=lookback + 1).apply(pct_rank, raw=True)
    rolled = rolled.dropna()
    if invert:
                rolled = 100.0 - rolled
            rolled = rolled.clip(0, 100).round(1)
    return rolled

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
        vix = get_yf(cfg["volatility"])
        bond = get_yf(cfg["bond_yield"])
        hy = get_fred(cfg["hy_spread"])
    
    if len(prices) < 30:
                dummy = {k: 50.0 for k in ["Momentum","Strength (RSI)","Breadth","Junk Bond","Volatility","Safe Haven"]}
                return {"score": 50.0, "label": "Neutral", "color": "#f9a825",
                                        "components": dummy, "name": cfg["name"], "flag": cfg["flag"],
                                        "series": {}, "fg_history": pd.Series(dtype=float), "prices": prices}
        
    ma = prices.rolling(cfg["ma_period"]).mean()
    momentum = (prices / ma - 1) * 100

    delta = prices.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rsi = 100 - (100 / (1 + gain / (loss + 1e-9)))

    up = (prices.diff() > 0).rolling(20).sum()
    down = (prices.diff() < 0).rolling(20).sum()
    breadth = (up / (down + 1)) * 50

    eq_ret = prices.pct_change(20) * 100
    if len(bond) > 20:
                bd_chg = bond.reindex(eq_ret.index, method="ffill").diff(20)
                safehav = eq_ret - bd_chg
    else:
                safehav = eq_ret
        
    # Historical lookback for charts: use 126 trading days (~6 months) for speed
    hist_lb = 126

    components = {
                "Momentum": normalize(momentum, lb),
                "Strength (RSI)": normalize(rsi, lb),
                "Breadth": normalize(breadth, lb),
                "Junk Bond": normalize(hy, lb, invert=True) if len(hy) > 0 else 50.0,
                "Volatility": normalize(vix, lb, invert=True) if len(vix) > 0 else 50.0,
                "Safe Haven": normalize(safehav, lb),
    }

    # Compute vectorized rolling normalized series
    hy_aligned = hy.reindex(momentum.index, method="ffill") if len(hy) > 20 else pd.Series(dtype=float)
    vix_aligned = vix.reindex(momentum.index, method="ffill") if len(vix) > 20 else pd.Series(dtype=float)

    series = {
                "Momentum": normalize_rolling(momentum, hist_lb),
                "Strength (RSI)": normalize_rolling(rsi, hist_lb),
                "Breadth": normalize_rolling(breadth, hist_lb),
                "Junk Bond": normalize_rolling(hy_aligned, hist_lb, invert=True) if len(hy_aligned) > hist_lb else pd.Series(dtype=float),
                "Volatility": normalize_rolling(vix_aligned, hist_lb, invert=True) if len(vix_aligned) > hist_lb else pd.Series(dtype=float),
                "Safe Haven": normalize_rolling(safehav, hist_lb),
    }

    # Compute rolling Fear & Greed Index (average of all components)
    valid_series = [s for s in series.values() if len(s) > 0]
    if valid_series:
                combined = pd.concat(valid_series, axis=1).dropna()
                fg_history = combined.mean(axis=1).round(1)
                # Keep last 252 trading days
        fg_history = fg_history.iloc[-252:] if len(fg_history) > 252 else fg_history
else:
        fg_history = pd.Series(dtype=float)
    
    # Trim series to last 252 points
    series = {k: (v.iloc[-252:] if len(v) > 252 else v) for k, v in series.items()}

    total = round(sum(components.values()) / len(components), 1)
    lbl, color = get_label(total)
    return {"score": total, "label": lbl, "color": color,
                        "components": components, "name": cfg["name"], "flag": cfg["flag"],
                        "series": series, "fg_history": fg_history, "prices": prices}

def make_gauge(result):
        score = result["score"]
        color = result["color"]
        lbl = result["label"]
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
                                                        {"range": [0, 25], "color": "#3b0000"},
                                                        {"range": [25, 45], "color": "#3b1a00"},
                                                        {"range": [45, 55], "color": "#3b3000"},
                                                        {"range": [55, 75], "color": "#153000"},
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
        names = list(result["components"].keys())
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
    
def make_component_history(result):
        """Create a 3x2 subplot with the historical evolution of each indicator."""
        series = result["series"]
        valid = [(name, s) for name, s in series.items() if len(s) > 5]
        if not valid:
                    return None
            
        n = len(valid)
        cols = 2
        rows = (n + 1) // cols
    
    fig = make_subplots(
                rows=rows, cols=cols,
                subplot_titles=[name for name, _ in valid],
                vertical_spacing=0.12,
                horizontal_spacing=0.08
    )

    for idx, (name, s) in enumerate(valid):
                row = idx // cols + 1
                col = idx % cols + 1
                score_color = get_label(float(s.iloc[-1]))[1] if len(s) > 0 else "#f9a825"
                fig.add_trace(
                                go.Scatter(
                                                    x=s.index,
                                                    y=s.values,
                                                    mode="lines",
                                                    line=dict(color=score_color, width=1.5),
                                                    fill="tozeroy",
                                                    fillcolor=score_color + "26",
                                                    name=name,
                                                    showlegend=False,
                                                    hovertemplate="%{x|%d/%m/%Y}<br>" + name + ": %{y:.0f}<extra></extra>"
                                ),
                                row=row, col=col
                )
                fig.add_hline(y=50, line_dash="dash", line_color="#555", line_width=1, row=row, col=col)
                fig.update_yaxes(range=[0, 100], row=row, col=col)
        
    fig.update_layout(
                height=120 * rows + 80,
                margin=dict(l=10, r=10, t=40, b=10),
                paper_bgcolor="#0e1117",
                plot_bgcolor="#1e2130",
                font_color="white",
                font_size=11
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="#333")
    return fig

def make_fg_vs_index(r1, r2, period_days=365):
    """Gráfico dual-eje: Fear & Greed (izq) vs precio del índice (der)."""
    fg1 = r1.get("fg_history", pd.Series(dtype=float))
    fg2 = r2.get("fg_history", pd.Series(dtype=float))
    p1  = r1.get("prices", pd.Series(dtype=float))
    p2  = r2.get("prices", pd.Series(dtype=float))

    if len(fg1) == 0 and len(fg2) == 0:
        return None

    def trim(s, days):
        if len(s) == 0:
            return s
        cutoff = s.index[-1] - pd.Timedelta(days=days)
        return s[s.index >= cutoff]

    fg1 = trim(fg1, period_days)
    fg2 = trim(fg2, period_days)
    p1  = trim(p1,  period_days)
    p2  = trim(p2,  period_days)

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=False,
        vertical_spacing=0.10,
        subplot_titles=[
            f"{r1.get('flag','')} {r1.get('name','US')} — Fear & Greed vs Precio",
            f"{r2.get('flag','')} {r2.get('name','EU')} — Fear & Greed vs Precio",
        ],
        specs=[[{"secondary_y": True}], [{"secondary_y": True}]],
    )

    FG_US   = "#0085CA"
    FG_EU   = "#37BBF4"
    PR_CLR  = "#C8DAE2"
    FEAR_Z  = "rgba(235,86,86,0.12)"
    GREED_Z = "rgba(54,123,53,0.12)"

    def add_panel(row, fg, price, fg_color):
        if len(fg) == 0:
            return
        fig.add_trace(go.Scatter(
            x=fg.index, y=[40]*len(fg), fill=None,
            mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"
        ), row=row, col=1, secondary_y=False)
        fig.add_trace(go.Scatter(
            x=fg.index, y=[0]*len(fg), fill="tonexty",
            mode="lines", line=dict(width=0), fillcolor=FEAR_Z,
            showlegend=False, hoverinfo="skip"
        ), row=row, col=1, secondary_y=False)
        fig.add_trace(go.Scatter(
            x=fg.index, y=[60]*len(fg), fill=None,
            mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"
        ), row=row, col=1, secondary_y=False)
        fig.add_trace(go.Scatter(
            x=fg.index, y=[100]*len(fg), fill="tonexty",
            mode="lines", line=dict(width=0), fillcolor=GREED_Z,
            showlegend=False, hoverinfo="skip"
        ), row=row, col=1, secondary_y=False)
        fig.add_trace(go.Scatter(
            x=fg.index, y=fg.values,
            mode="lines", name="Fear & Greed",
            line=dict(color=fg_color, width=2.5),
            hovertemplate="%{x|%d %b %Y}<br><b>F&G: %{y:.0f}</b><extra></extra>",
        ), row=row, col=1, secondary_y=False)
        if len(price) > 0:
            common = fg.index.intersection(price.index)
            if len(common) > 0:
                pr = price.reindex(common)
                fig.add_trace(go.Scatter(
                    x=pr.index, y=pr.values,
                    mode="lines", name="Precio índice",
                    line=dict(color=PR_CLR, width=1.8, dash="dot"),
                    opacity=0.85,
                    hovertemplate="%{x|%d %b %Y}<br><b>Precio: %{y:,.0f}</b><extra></extra>",
                ), row=row, col=1, secondary_y=True)

    add_panel(1, fg1, p1, FG_US)
    add_panel(2, fg2, p2, FG_EU)

    for row in [1, 2]:
        for val in [25, 50, 75]:
            fig.add_hline(y=val, line_dash="dot",
                          line_color="rgba(200,218,226,0.3)", line_width=1,
                          row=row, col=1)

    fig.update_layout(
        height=700,
        plot_bgcolor="#0A3A50",
        paper_bgcolor="#062D3F",
        font=dict(color="#F3F3F3", size=12),
        showlegend=False,
        hovermode="x unified",
        margin=dict(l=10, r=10, t=50, b=10),
    )
    for row in [1, 2]:
        fg_c = FG_US if row == 1 else FG_EU
        fig.update_yaxes(range=[0, 100], title_text="Fear & Greed",
                         gridcolor="rgba(200,218,226,0.1)", zeroline=False,
                         tickfont=dict(color=fg_c), title_font=dict(color=fg_c),
                         secondary_y=False, row=row, col=1)
        fig.update_yaxes(title_text="Precio",
                         gridcolor="rgba(0,0,0,0)", zeroline=False,
                         tickfont=dict(color=PR_CLR), title_font=dict(color=PR_CLR),
                         secondary_y=True, row=row, col=1)
        fig.update_xaxes(gridcolor="rgba(200,218,226,0.08)", zeroline=False,
                         tickfont=dict(color="#C8DAE2"), row=row, col=1)
    return fig

elif len(prices1) > 0:
            p1 = prices1.iloc[-252:] if len(prices1) > 252 else prices1
            p2 = pd.Series(dtype=float)
    else:
                p1 = pd.Series(dtype=float)
                p2 = prices2.iloc[-252:] if len(prices2) > 252 else prices2
        
    if len(p1) > 0:
                p1_norm = (p1 / p1.iloc[0] * 100).round(2)
                fig.add_trace(go.Scatter(
                                x=p1_norm.index, y=p1_norm.values,
                                mode="lines", name="S&P 500",
                                line=dict(color="#4fc3f7", width=2),
                                hovertemplate="%{x|%d/%m/%Y}<br>S&P 500: %{y:.1f}<extra></extra>"
                ), row=2, col=1)
        
    if len(p2) > 0:
                p2_norm = (p2 / p2.iloc[0] * 100).round(2)
                fig.add_trace(go.Scatter(
                                x=p2_norm.index, y=p2_norm.values,
                                mode="lines", name="EuroStoxx 50",
                                line=dict(color="#ffb74d", width=2),
                                hovertemplate="%{x|%d/%m/%Y}<br>EuroStoxx 50: %{y:.1f}<extra></extra>"
                ), row=2, col=1)
        
    fig.add_hline(y=100, line_dash="dash", line_color="#666", line_width=1, row=2, col=1)

    fig.update_layout(
                height=520,
                margin=dict(l=10, r=10, t=50, b=10),
                paper_bgcolor="#0e1117",
                plot_bgcolor="#1e2130",
                font_color="white",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                hovermode="x unified"
    )
    fig.update_yaxes(range=[0, 100], row=1, col=1, gridcolor="#333", title_text="Score")
    fig.update_yaxes(gridcolor="#333", row=2, col=1, title_text="Base 100")
    fig.update_xaxes(showgrid=False)
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
                                    
            # ── NUEVA SECCION: Evolucion de cada indicador ────────────────────────────────
st.divider()
st.subheader("Evolucion historica de cada indicador")
st.caption("Ultimos 12 meses aprox. Score normalizado 0-100 (50 = neutral)")

tab_us, tab_eu = st.tabs([f"US - {r1['name']}", f"EU - {r2['name']}"])

with tab_us:
        fig_hist1 = make_component_history(r1)
        if fig_hist1:
                    st.plotly_chart(fig_hist1, use_container_width=True)
        else:
                    st.info("No hay suficientes datos historicos para el indice US.")
            
    with tab_eu:
            fig_hist2 = make_component_history(r2)
            if fig_hist2:
                        st.plotly_chart(fig_hist2, use_container_width=True)
            else:
                        st.info("No hay suficientes datos historicos para el indice EU.")
                
        # ── NUEVA SECCION: F&G vs Indices bursatiles ─────────────────────────────────
st.divider()
st.subheader("Fear & Greed Index vs Índices bursátiles")
st.caption("Doble eje: Fear & Greed (izquierda, línea sólida) · Precio del índice (derecha, línea punteada).")

_opts = {"6 meses": 180, "1 año": 365, "2 años": 730}
_sel  = st.radio("Período:", options=list(_opts.keys()), index=1,
                  horizontal=True, label_visibility="collapsed")

fig_vs = make_fg_vs_index(r1, r2, period_days=_opts[_sel])
if fig_vs:
    st.plotly_chart(fig_vs, use_container_width=True)
else:
    st.info("No hay suficientes datos para mostrar la evolución histórica.")
    
st.divider()
st.subheader("Que mide cada componente")

COMPONENT_DESC = {
        "Momentum": "Compara el precio actual del indice con su media movil de 125 dias. Si cotiza por encima, hay euforia; si cae por debajo, hay miedo.",
        "Strength (RSI)": "RSI de 14 dias. Mide si el mercado esta sobrecomprado (codicia) o sobrevendido (miedo).",
        "Breadth": "Proporcion de dias alcistas frente a bajistas en los ultimos 20 dias. Mas dias verdes = mayor codicia.",
        "Junk Bond": "Diferencial de tipos entre bonos de alto riesgo y bonos seguros (FRED/ICE BofA). Un spread alto indica miedo e incertidumbre.",
        "Volatility": "VIX (S&P) o VSTOXX (EuroStoxx). A mayor volatilidad, mayor miedo en el mercado.",
        "Safe Haven": "Retorno relativo de acciones frente a bonos del tesoro. Si los inversores huyen a bonos, hay miedo.",
}

cols_desc = st.columns(3)
items = list(COMPONENT_DESC.items())
for i, (name, desc) in enumerate(items):
        with cols_desc[i % 3]:
                    label, color = get_label(50)
                    st.markdown(
                                    f"**{name}** \n{desc}",
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
st.caption("Fuentes: Yahoo Finance / FRED ICE BofA. Cache 10 min. Pulsa Actualizar para forzar recarga.")import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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
        if score <= 25: return ("Extreme Fear", "#c62828")
elif score <= 45: return ("Fear", "#ef6c00")
elif score <= 55: return ("Neutral", "#f9a825")
elif score <= 75: return ("Greed", "#558b2f")
else: return ("Extreme Greed", "#1b5e20")

def normalize(series, lookback, invert=False):
        clean = series.dropna()
        if len(clean) < 20:
                    return 50.0
                window = clean.iloc[-lookback:]
    current = float(clean.iloc[-1])
    score = float((window < current).sum() / len(window) * 100)
    result = 100.0 - score if invert else score
    return round(max(0.0, min(100.0, result)), 1)

def normalize_series(series, lookback, invert=False):
        """Compute rolling normalized score for each point in the series."""
    clean = series.dropna()
    if len(clean) < 20:
                return pd.Series(dtype=float)
            result = []
    dates = []
    for i in range(lookback, len(clean)):
                window = clean.iloc[i-lookback:i]
                current = float(clean.iloc[i])
                score = float((window < current).sum() / len(window) * 100)
                val = 100.0 - score if invert else score
                result.append(round(max(0.0, min(100.0, val)), 1))
                dates.append(clean.index[i])
            return pd.Series(result, index=dates)

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
    vix = get_yf(cfg["volatility"])
    bond = get_yf(cfg["bond_yield"])
    hy = get_fred(cfg["hy_spread"])

    if len(prices) < 30:
                dummy = {k: 50.0 for k in ["Momentum","Strength (RSI)","Breadth","Junk Bond","Volatility","Safe Haven"]}
        return {"score": 50.0, "label": "Neutral", "color": "#f9a825",
                                "components": dummy, "name": cfg["name"], "flag": cfg["flag"],
                                "series": {}, "prices": prices}

    ma = prices.rolling(cfg["ma_period"]).mean()
    momentum = (prices / ma - 1) * 100

    delta = prices.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rsi = 100 - (100 / (1 + gain / (loss + 1e-9)))

    up = (prices.diff() > 0).rolling(20).sum()
    down = (prices.diff() < 0).rolling(20).sum()
    breadth = (up / (down + 1)) * 50

    eq_ret = prices.pct_change(20) * 100
    if len(bond) > 20:
                bd_chg = bond.reindex(eq_ret.index, method="ffill").diff(20)
                safehav = eq_ret - bd_chg
else:
        safehav = eq_ret

    components = {
                "Momentum": normalize(momentum, lb),
                "Strength (RSI)": normalize(rsi, lb),
                "Breadth": normalize(breadth, lb),
                "Junk Bond": normalize(hy, lb, invert=True) if len(hy) > 0 else 50.0,
                "Volatility": normalize(vix, lb, invert=True) if len(vix) > 0 else 50.0,
                "Safe Haven": normalize(safehav, lb),
    }

    # Compute historical series for each component (last 252 trading days)
    hist_lb = min(lb, 252)
    series = {
                "Momentum": normalize_series(momentum, hist_lb),
                "Strength (RSI)": normalize_series(rsi, hist_lb),
                "Breadth": normalize_series(breadth, hist_lb),
                "Junk Bond": normalize_series(hy.reindex(momentum.index, method="ffill").dropna(), hist_lb, invert=True) if len(hy) > 20 else pd.Series(dtype=float),
                "Volatility": normalize_series(vix.reindex(momentum.index, method="ffill").dropna(), hist_lb, invert=True) if len(vix) > 20 else pd.Series(dtype=float),
                "Safe Haven": normalize_series(safehav, hist_lb),
    }

    # Compute rolling Fear & Greed Index (average of all 6 components)
    valid_series = [s for s in series.values() if len(s) > 0]
    if valid_series:
                combined = pd.concat(valid_series, axis=1).dropna()
                fg_history = combined.mean(axis=1).round(1)
else:
        fg_history = pd.Series(dtype=float)

    total = round(sum(components.values()) / len(components), 1)
    lbl, color = get_label(total)
    return {"score": total, "label": lbl, "color": color,
                        "components": components, "name": cfg["name"], "flag": cfg["flag"],
                        "series": series, "fg_history": fg_history, "prices": prices}

def make_gauge(result):
        score = result["score"]
    color = result["color"]
    lbl = result["label"]
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
                                                    {"range": [0, 25], "color": "#3b0000"},
                                                    {"range": [25, 45], "color": "#3b1a00"},
                                                    {"range": [45, 55], "color": "#3b3000"},
                                                    {"range": [55, 75], "color": "#153000"},
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
        names = list(result["components"].keys())
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

def make_component_history(result):
        """Create a subplot with the historical evolution of each indicator."""
    series = result["series"]
    comp_names = list(series.keys())
    valid = [(name, s) for name, s in series.items() if len(s) > 0]
    if not valid:
                return None

    n = len(valid)
    cols = 2
    rows = (n + 1) // cols

    fig = make_subplots(
                rows=rows, cols=cols,
                subplot_titles=[name for name, _ in valid],
                vertical_spacing=0.12,
                horizontal_spacing=0.08
    )

    for idx, (name, s) in enumerate(valid):
                row = idx // cols + 1
                col = idx % cols + 1
                # Last 252 trading days
                s_plot = s.iloc[-252:] if len(s) > 252 else s
        colors_line = [get_label(v)[1] for v in s_plot.values]
        # Use a single scatter with color gradient approximation
        score_color = get_label(float(s_plot.iloc[-1]))[1] if len(s_plot) > 0 else "#f9a825"
        fig.add_trace(
                        go.Scatter(
                                            x=s_plot.index,
                                            y=s_plot.values,
                                            mode="lines",
                                            line=dict(color=score_color, width=1.5),
                                            fill="tozeroy",
                                            fillcolor=score_color.replace(")", ", 0.15)").replace("rgb", "rgba") if "rgb" in score_color else score_color + "26",
                                            name=name,
                                            showlegend=False,
                                            hovertemplate="%{x|%d/%m/%Y}<br>" + name + ": %{y:.0f}<extra></extra>"
                        ),
                        row=row, col=col
        )
        # Add reference line at 50
        fig.add_hline(y=50, line_dash="dash", line_color="#555", line_width=1, row=row, col=col)
        fig.update_yaxes(range=[0, 100], row=row, col=col)

    fig.update_layout(
                height=120 * rows + 60,
                margin=dict(l=10, r=10, t=40, b=10),
                paper_bgcolor="#0e1117",
                plot_bgcolor="#1e2130",
                font_color="white",
                font_size=11
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="#333")
    return fig

st.subheader("Fear & Greed Index vs Índices bursátiles")
st.caption("Doble eje: Fear & Greed (izquierda, línea sólida) · Precio del índice (derecha, línea punteada). Sin descargas adicionales — usa los datos ya cargados.")

_opts = {"6 meses": 180, "1 año": 365, "2 años": 730}
_sel  = st.radio("Período:", options=list(_opts.keys()), index=1,
                  horizontal=True, label_visibility="collapsed")

fig_vs = make_fg_vs_index(r1, r2, period_days=_opts[_sel])
if fig_vs:
    st.plotly_chart(fig_vs, use_container_width=True)
else:
    st.info("No hay suficientes datos para mostrar la evolución histórica.")st.title("Fear & Greed Index")
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

                                    # ── NUEVA SECCION: Evolucion de cada indicador ────────────────────────────────
                                    st.divider()
st.subheader("Evolucion historica de cada indicador")
st.caption("Ultimos 12 meses aprox. Score normalizado 0-100 (50 = neutral)")

tab_us, tab_eu = st.tabs([f"US {r1['flag']} - {r1['name']}", f"EU {r2['flag']} - {r2['name']}"])

with tab_us:
        fig_hist1 = make_component_history(r1)
    if fig_hist1:
                st.plotly_chart(fig_hist1, use_container_width=True)
else:
        st.info("No hay suficientes datos historicos para mostrar la evolucion.")

with tab_eu:
        fig_hist2 = make_component_history(r2)
    if fig_hist2:
                st.plotly_chart(fig_hist2, use_container_width=True)
else:
        st.info("No hay suficientes datos historicos para mostrar la evolucion.")

# ── NUEVA SECCION: F&G vs Indices bursatiles ─────────────────────────────────
st.divider()
st.subheader("Fear & Greed Index vs Indices bursatiles")
st.caption("Comparativa entre el indicador de sentimiento y la evolucion del mercado (ultimos 12 meses)")

fig_vs = make_fg_vs_index(r1, r2)
if fig_vs:
        st.plotly_chart(fig_vs, use_container_width=True)
else:
    st.info("No hay suficientes datos para mostrar la comparativa.")

st.divider()
st.subheader("Que mide cada componente")

COMPONENT_DESC = {
        "Momentum": "Compara el precio actual del indice con su media movil de 125 dias. Si cotiza por encima, hay euforia; si cae por debajo, hay miedo.",
        "Strength (RSI)": "RSI de 14 dias. Mide si el mercado esta sobrecomprado (codicia) o sobrevendido (miedo).",
        "Breadth": "Proporcion de dias alcistas frente a bajistas en los ultimos 20 dias. Mas dias verdes = mayor codicia.",
        "Junk Bond": "Diferencial de tipos entre bonos de alto riesgo y bonos seguros (FRED/ICE BofA). Un spread alto indica miedo e incertidumbre.",
        "Volatility": "VIX (S&P) o VSTOXX (EuroStoxx). A mayor volatilidad, mayor miedo en el mercado.",
        "Safe Haven": "Retorno relativo de acciones frente a bonos del tesoro. Si los inversores huyen a bonos, hay miedo.",
}

cols_desc = st.columns(3)
items = list(COMPONENT_DESC.items())
for i, (name, desc) in enumerate(items):
        with cols_desc[i % 3]:
                    label, color = get_label(50)
        st.markdown(
                        f"**{name}** \n{desc}",
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
