import math
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf


# =========================================================
# CONFIG
# =========================================================
st.set_page_config(
    page_title="Payoff Lab",
    page_icon="📈",
    layout="wide",
)

st.title("Payoff Lab")
st.caption("Visualizador introductorio de payoffs al vencimiento")


# =========================================================
# DATA HELPERS
# =========================================================
@st.cache_data(ttl=300, show_spinner=False)
def get_market_data(ticker: str, period: str = "6mo"):
    try:
        data = yf.download(
            ticker,
            period=period,
            interval="1d",
            auto_adjust=False,
            progress=False,
            threads=False,
        )

        if data is None or data.empty:
            return None, None, None, None

        if isinstance(data.columns, pd.MultiIndex):
            if ("Close", ticker) in data.columns:
                close = data[("Close", ticker)].dropna()
            else:
                close = data.xs("Close", axis=1, level=0).iloc[:, 0].dropna()
        else:
            close = data["Close"].dropna()

        if close.empty:
            return None, None, None, None

        spot = float(close.iloc[-1])
        prev = float(close.iloc[-2]) if len(close) >= 2 else spot
        chg = spot - prev
        chg_pct = (chg / prev * 100.0) if prev != 0 else 0.0

        hist = close.reset_index()
        hist.columns = ["Date", "Close"]

        return spot, chg, chg_pct, hist

    except Exception:
        return None, None, None, None


# =========================================================
# PAYOFF HELPERS
# =========================================================
def forward_price_no_dividends(S: float, r: float, T: float) -> float:
    return S * math.exp(r * T)


def payoff_single(S, instrument, side, K, qty=1.0):
    if instrument == "Call":
        raw = np.maximum(S - K, 0.0)
    elif instrument == "Put":
        raw = np.maximum(K - S, 0.0)
    elif instrument == "Forward":
        raw = S - K
    elif instrument == "Stock":
        raw = S
    else:
        raw = np.zeros_like(S, dtype=float)

    sign = 1.0 if side == "Long" else -1.0
    return sign * raw * qty


def portfolio_payoff(S, legs):
    total_payoff = np.zeros_like(S, dtype=float)
    leg_payoffs = []

    for leg in legs:
        payoff_i = payoff_single(
            S=S,
            instrument=leg["instrument"],
            side=leg["side"],
            K=float(leg["K"]),
            qty=float(leg["qty"]),
        )
        total_payoff += payoff_i
        leg_payoffs.append(payoff_i)

    return total_payoff, leg_payoffs


def build_price_grid_from_view(spot, view_mode="Medio", n=801):
    mapping = {
        "Corto": 0.20,
        "Medio": 0.40,
        "Amplio": 0.75,
        "Muy amplio": 1.25,
    }
    width = mapping.get(view_mode, 0.40)
    s_min = max(0.0, spot * (1 - width))
    s_max = max(s_min + 1e-8, spot * (1 + width))
    return np.linspace(s_min, s_max, n), s_min, s_max


def clamp(x, xmin, xmax):
    return min(max(float(x), float(xmin)), float(xmax))


def make_leg(instrument, side, K, T, qty):
    return {
        "instrument": instrument,
        "side": side,
        "K": float(K),
        "T": float(T),
        "qty": float(qty),
    }


def load_template(name, spot, r, T):
    if name == "Long Call":
        return [make_leg("Call", "Long", round(spot, 2), T, 1.0)]
    if name == "Long Put":
        return [make_leg("Put", "Long", round(spot, 2), T, 1.0)]
    if name == "Long Forward":
        fwd = forward_price_no_dividends(spot, r, T)
        return [make_leg("Forward", "Long", round(fwd, 2), T, 1.0)]
    if name == "Protective Put":
        return [
            make_leg("Stock", "Long", 0.0, T, 1.0),
            make_leg("Put", "Long", round(spot, 2), T, 1.0),
        ]
    if name == "Covered Call":
        return [
            make_leg("Stock", "Long", 0.0, T, 1.0),
            make_leg("Call", "Short", round(1.05 * spot, 2), T, 1.0),
        ]
    if name == "Bull Call Spread":
        return [
            make_leg("Call", "Long", round(0.95 * spot, 2), T, 1.0),
            make_leg("Call", "Short", round(1.05 * spot, 2), T, 1.0),
        ]
    if name == "Bear Put Spread":
        return [
            make_leg("Put", "Long", round(1.05 * spot, 2), T, 1.0),
            make_leg("Put", "Short", round(0.95 * spot, 2), T, 1.0),
        ]
    if name == "Straddle":
        return [
            make_leg("Call", "Long", round(spot, 2), T, 1.0),
            make_leg("Put", "Long", round(spot, 2), T, 1.0),
        ]
    if name == "Synthetic Long Forward":
        return [
            make_leg("Call", "Long", round(spot, 2), T, 1.0),
            make_leg("Put", "Short", round(spot, 2), T, 1.0),
        ]
    return []


def reset_st_slider_to_spot(spot):
    st.session_state.ST_eval = float(round(spot, 4))


def build_payoff_rows(ST, legs):
    rows = []
    total_payoff = 0.0

    for i, leg in enumerate(legs, start=1):
        payoff_i = payoff_single(
            S=np.array([ST], dtype=float),
            instrument=leg["instrument"],
            side=leg["side"],
            K=float(leg["K"]),
            qty=float(leg["qty"]),
        )
        payoff_i = float(payoff_i[0])
        total_payoff += payoff_i

        rows.append(
            {
                "Leg": i,
                "Instrument": leg["instrument"],
                "Side": leg["side"],
                "Strike/Fwd": float(leg["K"]),
                "# contratos": float(leg["qty"]),
                "Payoff@S_T": payoff_i,
            }
        )

    return rows, total_payoff


# =========================================================
# SESSION STATE
# =========================================================
if "legs" not in st.session_state:
    st.session_state.legs = []

if "last_ticker" not in st.session_state:
    st.session_state.last_ticker = None

if "last_period" not in st.session_state:
    st.session_state.last_period = None

if "last_spot" not in st.session_state:
    st.session_state.last_spot = None

if "ST_eval" not in st.session_state:
    st.session_state.ST_eval = 100.0


# =========================================================
# SIDEBAR
# =========================================================
st.sidebar.header("Supuestos")

ticker = st.sidebar.selectbox(
    "Acción",
    ["AAPL", "MSFT", "NVDA", "TSLA", "SPY"],
    index=0,
)

period = st.sidebar.selectbox("Ventana histórica", ["3mo", "6mo", "1y"], index=1)

manual_spot = st.sidebar.checkbox("Editar spot manualmente", value=False)

spot_mkt, chg, chg_pct, hist = get_market_data(ticker, period=period)

if spot_mkt is None:
    st.sidebar.error("No pude descargar datos.")
    spot_mkt = 100.0
    chg = 0.0
    chg_pct = 0.0

if manual_spot:
    spot = st.sidebar.number_input(
        "Spot",
        min_value=0.01,
        value=float(round(spot_mkt, 4)),
        step=0.1,
    )
else:
    spot = float(round(spot_mkt, 4))

rate_pct = st.sidebar.slider("Tasa libre de riesgo (%)", 0.0, 15.0, 10.0, 0.25)
r = rate_pct / 100.0

default_days = st.sidebar.slider("Vencimiento base (días)", 7, 365, 90, 1)
T_base = default_days / 365.0

if st.sidebar.button("Actualizar datos", use_container_width=True):
    st.cache_data.clear()
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.caption("Visualizador de payoff al vencimiento.")


# =========================================================
# STATE SYNC
# =========================================================
if st.session_state.last_ticker is None:
    st.session_state.last_ticker = ticker
    reset_st_slider_to_spot(spot)

if st.session_state.last_period is None:
    st.session_state.last_period = period

if st.session_state.last_spot is None:
    st.session_state.last_spot = float(spot)

ticker_changed = st.session_state.last_ticker != ticker
period_changed = st.session_state.last_period != period
spot_changed = abs(float(st.session_state.last_spot) - float(spot)) > 1e-12

if ticker_changed:
    st.session_state.legs = []
    st.session_state.last_ticker = ticker
    st.session_state.last_spot = float(spot)
    reset_st_slider_to_spot(spot)
    st.rerun()

if period_changed:
    st.session_state.last_period = period
    st.session_state.last_spot = float(spot)
    reset_st_slider_to_spot(spot)
    st.rerun()

if spot_changed:
    st.session_state.last_spot = float(spot)
    reset_st_slider_to_spot(spot)

if st.session_state.last_ticker is None:
    st.session_state.last_ticker = ticker
if st.session_state.last_period is None:
    st.session_state.last_period = period


# =========================================================
# TOP SUMMARY
# =========================================================
m1, m2, m3 = st.columns(3)
m1.metric("Spot", f"{spot:,.4f}", f"{chg:,.4f} ({chg_pct:,.2f}%)")
m2.metric("Vencimiento base", f"{default_days} días")
m3.metric("Ticker", ticker)

st.markdown("---")


# =========================================================
# QUICK TEMPLATES
# =========================================================
st.subheader("Estrategias rápidas")

t1, t2, t3, t4, t5, t6, t7, t8 = st.columns(8)

if t1.button("Long Call", use_container_width=True):
    st.session_state.legs = load_template("Long Call", spot, r, T_base)
    reset_st_slider_to_spot(spot)
    st.rerun()

if t2.button("Long Put", use_container_width=True):
    st.session_state.legs = load_template("Long Put", spot, r, T_base)
    reset_st_slider_to_spot(spot)
    st.rerun()

if t3.button("Forward", use_container_width=True):
    st.session_state.legs = load_template("Long Forward", spot, r, T_base)
    reset_st_slider_to_spot(spot)
    st.rerun()

if t4.button("Protective Put", use_container_width=True):
    st.session_state.legs = load_template("Protective Put", spot, r, T_base)
    reset_st_slider_to_spot(spot)
    st.rerun()

if t5.button("Covered Call", use_container_width=True):
    st.session_state.legs = load_template("Covered Call", spot, r, T_base)
    reset_st_slider_to_spot(spot)
    st.rerun()

if t6.button("Bull Spread", use_container_width=True):
    st.session_state.legs = load_template("Bull Call Spread", spot, r, T_base)
    reset_st_slider_to_spot(spot)
    st.rerun()

if t7.button("Straddle", use_container_width=True):
    st.session_state.legs = load_template("Straddle", spot, r, T_base)
    reset_st_slider_to_spot(spot)
    st.rerun()

if t8.button("Synth Fwd", use_container_width=True):
    st.session_state.legs = load_template("Synthetic Long Forward", spot, r, T_base)
    reset_st_slider_to_spot(spot)
    st.rerun()


# =========================================================
# MAIN LAYOUT
# =========================================================
left, right = st.columns([0.95, 2.35])


# =========================================================
# LEFT PANEL
# =========================================================
with left:
    st.subheader("Builder")

    with st.expander("Agregar instrumento", expanded=True):
        with st.form("builder_form", clear_on_submit=False):
            instrument_options = ["Call", "Put", "Forward", "Stock"]
            side_options = ["Long", "Short"]

            instrument = st.selectbox("Instrumento", instrument_options, index=0)
            side = st.selectbox("Posición", side_options, index=0)

            if instrument == "Stock":
                K_default = 0.0
            elif instrument == "Forward":
                K_default = round(forward_price_no_dividends(spot, r, T_base), 2)
            else:
                K_default = round(spot, 2)

            K_form = st.number_input(
                "Strike / precio forward",
                min_value=0.0,
                value=float(K_default),
                step=1.0,
            )

            qty_form = st.number_input(
                "# contratos",
                min_value=0.0,
                value=1.0,
                step=1.0,
            )

            add_leg = st.form_submit_button("Agregar", use_container_width=True)

        if add_leg:
            st.session_state.legs.append(
                {
                    "instrument": instrument,
                    "side": side,
                    "K": float(K_form),
                    "T": float(T_base),
                    "qty": float(qty_form),
                }
            )
            st.rerun()

    with st.expander("Controles visuales", expanded=True):
        view_mode = st.radio(
            "Vista del eje X",
            ["Corto", "Medio", "Amplio", "Muy amplio"],
            index=1,
            horizontal=True,
        )
        show_total_payoff = st.checkbox("Mostrar payoff total", value=True)
        show_legs = st.checkbox("Mostrar patas individuales", value=False)

    if st.button("Vaciar estrategia", use_container_width=True):
        st.session_state.legs = []
        reset_st_slider_to_spot(spot)
        st.rerun()


# =========================================================
# RIGHT PANEL
# =========================================================
with right:
    if len(st.session_state.legs) == 0:
        st.info("Agrega instrumentos o usa una estrategia rápida.")
    else:
        S, s_min, s_max = build_price_grid_from_view(spot, view_mode, n=801)
        s_min = float(round(s_min, 4))
        s_max = float(round(s_max, 4))

        st.session_state.ST_eval = clamp(st.session_state.ST_eval, s_min, s_max)

        width = s_max - s_min
        if spot < 20:
            step_temp = 0.05
        elif spot < 200:
            step_temp = 0.1
        elif spot < 2000:
            step_temp = 1.0
        else:
            step_temp = max(round(width / 150.0, 4), 5.0)

        total_payoff, leg_payoffs = portfolio_payoff(S, st.session_state.legs)

        ST = st.slider(
            "Precio final al vencimiento (S_T)",
            min_value=s_min,
            max_value=s_max,
            step=float(step_temp),
            key="ST_eval",
            format="%.4f",
        )
        ST = float(ST)

        rows, total_payoff_ST = build_payoff_rows(ST, st.session_state.legs)
        max_payoff = float(np.max(total_payoff))
        min_payoff = float(np.min(total_payoff))

        fig = go.Figure()

        if show_legs:
            for i, leg in enumerate(st.session_state.legs):
                base_name = f"Leg {i+1}: {leg['side']} {leg['instrument']}"
                fig.add_trace(
                    go.Scatter(
                        x=S,
                        y=leg_payoffs[i],
                        mode="lines",
                        name=base_name,
                        line=dict(width=1.5, dash="dot"),
                        opacity=0.45,
                        hovertemplate="S=%{x:,.2f}<br>Payoff=%{y:,.2f}<extra></extra>",
                    )
                )

        if show_total_payoff:
            fig.add_trace(
                go.Scatter(
                    x=S,
                    y=total_payoff,
                    mode="lines",
                    name="TOTAL Payoff",
                    line=dict(width=4),
                    hovertemplate="S=%{x:,.2f}<br>Total Payoff=%{y:,.2f}<extra></extra>",
                )
            )

        fig.add_hline(y=0, line_dash="dash", line_width=1)
        fig.add_vline(x=spot, line_dash="dot", line_width=1.2)
        fig.add_vline(x=ST, line_width=2)

        fig.add_annotation(
            x=spot,
            y=1.0,
            yref="paper",
            text=f"Spot = {spot:,.2f}",
            showarrow=False,
            xanchor="left",
        )

        fig.add_annotation(
            x=ST,
            y=1.0,
            yref="paper",
            text=f"S_T = {ST:,.2f}",
            showarrow=False,
            xanchor="left",
        )

        strikes_seen = []
        for leg in st.session_state.legs:
            if leg["instrument"] in ["Call", "Put", "Forward"]:
                k = float(leg["K"])
                if all(abs(k - kk) > 1e-8 for kk in strikes_seen):
                    strikes_seen.append(k)

        for k in strikes_seen:
            fig.add_vline(x=k, line_dash="dot", line_width=1)
            fig.add_annotation(
                x=k,
                y=0.9,
                yref="paper",
                text=f"K={k:,.2f}",
                showarrow=False,
                textangle=-90,
            )

        y_all = []
        if show_total_payoff:
            y_all.extend(total_payoff.tolist())
        if show_legs:
            for arr in leg_payoffs:
                y_all.extend(arr.tolist())

        if len(y_all) > 0:
            y_min = float(np.nanmin(y_all))
            y_max = float(np.nanmax(y_all))
            pad = 0.08 * max(1.0, y_max - y_min)
            y_range = [y_min - pad, y_max + pad]
        else:
            y_range = None

        fig.update_layout(
            title="Perfil de payoff",
            xaxis_title="Precio del subyacente al vencimiento",
            yaxis_title="Payoff",
            hovermode="x unified",
            height=620,
            margin=dict(l=20, r=20, t=60, b=20),
            legend_title="Curvas",
            xaxis=dict(range=[s_min, s_max]),
            yaxis=dict(range=y_range) if y_range else None,
        )

        st.plotly_chart(fig, use_container_width=True, key="payoff_chart")

        k1, k2, k3 = st.columns(3)
        k1.metric("Payoff total @ S_T", f"{total_payoff_ST:,.2f}")
        k2.metric("Máx payoff en vista", f"{max_payoff:,.2f}")
        k3.metric("Mín payoff en vista", f"{min_payoff:,.2f}")

        st.subheader("Qué pasa en ese precio final")
        st.write(f"Evaluación actual en **S_T = {ST:,.4f}**.")

        hdr = st.columns([0.8, 1.4, 1.0, 1.2, 1.0, 1.3, 0.7])
        hdr[0].markdown("**Leg**")
        hdr[1].markdown("**Instrument**")
        hdr[2].markdown("**Side**")
        hdr[3].markdown("**Strike/Fwd**")
        hdr[4].markdown("**Qty**")
        hdr[5].markdown("**Payoff@S_T**")
        hdr[6].markdown("**Quitar**")

        st.markdown("---")

        delete_idx = None
        for i, row in enumerate(rows):
            c = st.columns([0.8, 1.4, 1.0, 1.2, 1.0, 1.3, 0.7])
            c[0].write(int(row["Leg"]))
            c[1].write(str(row["Instrument"]))
            c[2].write(str(row["Side"]))
            c[3].write(f'{row["Strike/Fwd"]:,.2f}')
            c[4].write(f'{row["# contratos"]:,.2f}')
            c[5].write(f'{row["Payoff@S_T"]:,.2f}')
            if c[6].button("✕", key=f"delete_leg_{i}", use_container_width=True):
                delete_idx = i

        st.markdown("---")

        total_cols = st.columns([0.8, 1.4, 1.0, 1.2, 1.0, 1.3, 0.7])
        total_cols[0].markdown("**TOTAL**")
        total_cols[5].markdown(f"**{total_payoff_ST:,.2f}**")

        if delete_idx is not None:
            st.session_state.legs.pop(delete_idx)
            st.rerun()


# =========================================================
# NOTES
# =========================================================
st.markdown("---")
st.write(
    """
- **Payoff**: pago al vencimiento como función de \(S_T\).
- **Call**: \(\max(S_T-K,0)\)
- **Put**: \(\max(K-S_T,0)\)
- **Forward**: \(S_T-K\)
- **Stock**: \(S_T\)
- **Short** cambia el signo del payoff.
"""
)
