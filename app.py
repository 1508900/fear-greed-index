    st.cache_data.clear()


with st.spinner("Descargando datos de mercado..."):
    results = {key: compute(cfg) for key, cfg in CONFIGS.items()}


keys = list(results.keys())
r1, r2 = results[keys[0]], results[keys[1]]


c1, c2 = st.columns(2)
with c1: st.plotly_chart(make_gauge(r1), use_container_width=True)
with c2: st.plotly_chart(make_gauge(r2), use_container_width=True)


st.divider()
st.subheader("🔍 Detalle por componente")
c3, c4 = st.columns(2)
for col, r in zip([c3, c4], [r1, r2]):
    with col:
        st.markdown(f"**{r['flag']} {r['name']}**")
        t1, t2 = st.tabs(["📊 Barras", "🕸️ Radar"])
        with t1: st.plotly_chart(make_bars(r), use_container_width=True)
        with t2: st.plotly_chart(make_radar(r), use_container_width=True)


st.divider()
st.subheader("⚖️ Comparativa USA vs Europa")
df_cmp = make_compare(r1, r2)


def color_score(val):
    if isinstance(val, (int, float)): return f"color: {get_label(val)[1]}; font-weight:600"
    return ""
def color_diff(val):
    if isinstance(val, (int, float)):
        if val > 5:  return "color: #558b2f; font-weight:600"
        if val < -5: return "color: #c62828; font-weight:600"
    return ""


st.dataframe(
    df_cmp.style
        .applymap(color_score, subset=[f"{r1['flag']} {r1['name']}", f"{r2['flag']} {r2['name']}"])
        .applymap(color_diff,  subset=["Dif (US-EU)"]),
    use_container_width=True, hide_index=True)


st.divider()
st.caption("Fuentes: Yahoo Finance · FRED/ICE BofA. Cache 10 min. Pulsa Actualizar para forzar recarga.")

