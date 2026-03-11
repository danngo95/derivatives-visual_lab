import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf


# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Payoff Lab",
    page_icon="📈",
    layout="wide",
)

st.title("Payoff Lab")
st.caption("Visualizador introductorio de payoffs de derivados")


# =========================================================
# HELPERS
# =========================================================
def get_spot_from_yahoo(ticker: str):
    try:
        data = yf.download(ticker, period="10d", interval="1d", auto_adjust=False, progress=False)
        if data is None or data.empty:
            return None, None
        close_col = "Close"
        if isinstance(data.columns, pd.MultiIndex):
            if ("Close", ticker) in data.columns:
                close_series = data[("Close", ticker)]
            else:
                close_series = data.xs("Close", axis=1, level=0).iloc[:, 0]
        else:
            close_series = data[close_col]
        close_series = close_series.dropna()
        if close_series.empty:
            return None, None
        spot = float(close_series.iloc[-1])
        hist = close_series.reset_index()
        hist.columns = ["Date", "Close"]
        return spot, hist
    except Exception:
        return None, None


def payoff_single(S, instrument, side, K, premium, qty=1.0, multiplier=1.0):
    if instrument == "Call":
        raw = np.maximum(S - K, 0.0)
    elif instrument == "Put":
        raw = np.maximum(K - S, 0.0)
    elif instrument == "Forward":
        raw = S - K
    elif instrument == "Stock":
        raw = S
    else:
        raw = np.zeros_like(S)

    sign = 1.0 if side == "Long" else -1.0

    if instrument == "Stock":
        payoff = sign * raw * qty * multiplier
        pnl = sign * (S - premium) * qty * multiplier
    elif instrument == "Forward":
        payoff = sign * raw * qty * multiplier
        pnl = payoff
    else:
        payoff = sign * raw * qty * multiplier
        premium_cash = premium * qty * multiplier
        pnl = payoff - premium_cash if side == "Long" else payoff + premium_cash

    return payoff, pnl


def portfolio_payoff(S, legs):
    total_payoff = np.zeros_like(S, dtype=float)
    total_pnl = np.zeros_like(S, dtype=float)

    leg_payoffs = []
    leg_pnls = []

    for leg in legs:
        payoff_i, pnl_i = payoff_single(
            S=S,
            instrument=leg["instrument"],
            side=leg["side"],
            K=float(leg["K"]),
            premium=float(leg["premium"]),
            qty=float(leg["qty"]),
            multiplier=float(leg["multiplier"]),
        )
        total_payoff += payoff_i
        total_pnl += pnl_i
        leg_payoffs.append(payoff_i)
        leg_pnls.append(pnl_i)

    return total_payoff, total_pnl, leg_payoffs, leg_pnls


def build_price_grid(spot, low_pct=50, high_pct=50, n=801):
    s_min = max(0.0, spot * (1 - low_pct / 100))
    s_max = max(s_min + 1e-6, spot * (1 + high_pct / 100))
    return np.linspace(s_min, s_max, n), s_min, s_max


def find_break_evens(S, pnl, tol=1e-8):
    bes = []

    for i in range(len(S) - 1):
        y1, y2 = pnl[i], pnl[i + 1]
        x1, x2 = S[i], S[i + 1]

        if abs(y1) < tol:
            bes.append(x1)

        if y1 * y2 < 0:
            x_star = x1 - y1 * (x2 - x1) / (y2 - y1)
            bes.append(x_star)

    if abs(pnl[-1]) < tol:
        bes.append(S[-1])

    bes = sorted(bes)
    cleaned = []
    for x in bes:
        if not cleaned or abs(x - cleaned[-1]) > 1e-4:
            cleaned.append(x)
    return cleaned


def net_initial_cost(legs):
    total = 0.0
    for leg in legs:
        qty = float(leg["qty"])
        mult = float(leg["multiplier"])
        premium = float(leg["premium"])
        side = leg["side"]
        instrument = leg["instrument"]

        if instrument == "Forward":
            cash = 0.0
        else:
            cash = premium * qty * mult

        total += cash if side == "Long" else -cash
    return total


def payoff_at_point(S0, legs):
    payoff, pnl, _, _ = portfolio_payoff(np.array([S0]), legs)
    return float(payoff[0]), float(pnl[0])


def load_template(template_name, spot):
    if template_name == "Long Call":
        return [
            {"instrument": "Call", "side": "Long", "K": round(spot, 2), "premium": 5.0, "qty": 1.0, "multiplier": 1.0}
        ]
    if template_name == "Long Put":
        return [
            {"instrument": "Put", "side": "Long", "K": round(spot, 2), "premium": 5.0, "qty": 1.0, "multiplier": 1.0}
        ]
    if template_name == "Long Forward":
        return [
            {"instrument": "Forward", "side": "Long", "K": round(spot, 2), "premium": 0.0, "qty": 1.0, "multiplier": 1.0}
        ]
    if template_name == "Protective Put":
        return [
            {"instrument": "Stock", "side": "Long", "K": 0.0, "premium": round(spot, 2), "qty": 1.0, "multiplier": 1.0},
            {"instrument": "Put", "side": "Long", "K": round(spot, 2), "premium": 5.0, "qty": 1.0, "multiplier": 1.0},
        ]
    if template_name == "Covered Call":
        return [
            {"instrument": "Stock", "side": "Long", "K": 0.0, "premium": round(spot, 2), "qty": 1.0, "multiplier": 1.0},
            {"instrument": "Call", "side": "Short", "K": round(1.05 * spot, 2), "premium": 4.0, "qty": 1.0, "multiplier": 1.0},
        ]
    if template_name == "Bull Call Spread":
        return [
            {"instrument": "Call", "side": "Long", "K": round(0.95 * spot, 2), "premium": 8.0, "qty": 1.0, "multiplier": 1.0},
            {"instrument": "Call", "side": "Short", "K": round(1.05 * spot, 2), "premium": 3.0, "qty": 1.0, "multiplier": 1.0},
        ]
    if template_name == "Bear Put Spread":
        return [
            {"instrument": "Put", "side": "Long", "K": round(1.05 * spot, 2), "premium": 8.0, "qty": 1.0, "multiplier": 1.0},
            {"instrument": "Put", "side": "Short", "K": round(0.95 * spot, 2), "premium": 3.0, "qty": 1.0, "multiplier": 1.0},
        ]
    if template_name == "Long Straddle":
        return [
            {"instrument": "Call", "side": "Long", "K": round(spot, 2), "premium": 6.0, "qty": 1.0, "multiplier": 1.0},
            {"instrument": "Put", "side": "Long", "K": round(spot, 2), "premium": 6.0, "qty": 1.0, "multiplier": 1.0},
        ]
    return [
        {"instrument": "Call", "side": "Long", "K": round(spot, 2), "premium": 5.0, "qty": 1.0, "multiplier": 1.0}
    ]


def strategy_explanation(name):
    explanations = {
        "Long Call": "Ganas si el subyacente sube por arriba del strike; la pérdida máxima es la prima pagada.",
        "Long Put": "Ganas si el subyacente cae por debajo del strike; la pérdida máxima es la prima pagada.",
        "Long Forward": "El payoff es lineal: ganas cuando el precio final queda arriba del precio forward.",
        "Protective Put": "Combina una posición larga en el activo con un put comprado para limitar la pérdida.",
        "Covered Call": "Combina el activo con un call vendido; sacrificas parte del upside a cambio de recibir prima.",
        "Bull Call Spread": "Apuesta alcista con ganancia y pérdida acotadas usando dos calls.",
        "Bear Put Spread": "Apuesta bajista con ganancia y pérdida acotadas usando dos puts.",
        "Long Straddle": "Apuesta a movimientos grandes en cualquier dirección comprando call y put del mismo strike.",
        "Custom": "Construye tu propia estrategia sumando instrumentos.",
    }
    return explanations.get(name, "")


# =========================================================
# SESSION STATE
# =========================================================
if "legs" not in st.session_state:
    st.session_state.legs = []

if "last_template" not in st.session_state:
    st.session_state.last_template = "Custom"


# =========================================================
# SIDEBAR - MARKET DATA
# =========================================================
st.sidebar.header("Datos del subyacente")

popular_tickers = [
    "AAPL", "MSFT", "SPY", "TSLA", "NVDA", "AMZN", "GOOGL",
    "META", "QQQ", "IWM", "BTC-USD", "ETH-USD", "USDMXN=X", "^GSPC"
]

ticker = st.sidebar.selectbox("Ticker (Yahoo Finance)", popular_tickers, index=0)
manual_override = st.sidebar.checkbox("Editar spot manualmente", value=False)

spot_yahoo, hist = get_spot_from_yahoo(ticker)

if spot_yahoo is None:
    st.sidebar.error("No pude descargar el precio desde Yahoo Finance.")
    fallback_spot = 100.0
else:
    fallback_spot = round(spot_yahoo, 4)

if manual_override:
    spot = st.sidebar.number_input("Spot manual", min_value=0.0, value=float(fallback_spot), step=0.1)
else:
    spot = float(fallback_spot)

st.sidebar.metric("Spot usado", f"{spot:,.4f}")

if hist is not None:
    st.sidebar.caption("Cierre reciente tomado de Yahoo Finance")


# =========================================================
# TOP MARKET VIEW
# =========================================================
c1, c2 = st.columns([1.2, 1.8])

with c1:
    st.subheader("Subyacente")
    st.write(f"**Ticker:** `{ticker}`")
    st.write(f"**Spot actual usado en la app:** `{spot:,.4f}`")

with c2:
    if hist is not None:
        fig_hist = go.Figure()
        fig_hist.add_trace(
            go.Scatter(
                x=hist["Date"],
                y=hist["Close"],
                mode="lines",
                name="Close",
            )
        )
        fig_hist.update_layout(
            title="Cierres recientes",
            height=280,
            margin=dict(l=20, r=20, t=50, b=20),
            xaxis_title="Fecha",
            yaxis_title="Precio",
        )
        st.plotly_chart(fig_hist, use_container_width=True)


# =========================================================
# STRATEGY BUILDER
# =========================================================
st.markdown("---")
st.subheader("Strategy Builder")

left, right = st.columns([1.1, 1.9])

with left:
    template = st.selectbox(
        "Template de estrategia",
        [
            "Custom",
            "Long Call",
            "Long Put",
            "Long Forward",
            "Protective Put",
            "Covered Call",
            "Bull Call Spread",
            "Bear Put Spread",
            "Long Straddle",
        ],
        index=0,
    )

    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("Cargar template", use_container_width=True):
            st.session_state.legs = load_template(template, spot)
            st.session_state.last_template = template

    with col_b:
        if st.button("Vaciar estrategia", use_container_width=True):
            st.session_state.legs = []
            st.session_state.last_template = "Custom"

    st.info(strategy_explanation(template))

    st.markdown("### Agregar instrumento")
    with st.form("add_leg_form", clear_on_submit=False):
        instrument_new = st.selectbox("Instrumento", ["Call", "Put", "Forward", "Stock"])
        side_new = st.selectbox("Posición", ["Long", "Short"])
        K_new = st.number_input(
            "Strike / Forward price",
            min_value=0.0,
            value=float(round(spot, 2)),
            step=1.0,
        )
        premium_default = float(round(spot, 2)) if instrument_new == "Stock" else 5.0
        premium_new = st.number_input(
            "Prima / costo inicial",
            min_value=0.0,
            value=premium_default,
            step=1.0,
        )
        qty_new = st.number_input("Cantidad", min_value=0.0, value=1.0, step=1.0)
        mult_new = st.number_input("Multiplicador", min_value=0.0, value=1.0, step=1.0)

        submitted = st.form_submit_button("Agregar instrumento")
        if submitted:
            st.session_state.legs.append(
                {
                    "instrument": instrument_new,
                    "side": side_new,
                    "K": float(K_new),
                    "premium": float(premium_new),
                    "qty": float(qty_new),
                    "multiplier": float(mult_new),
                }
            )

    st.markdown("### Configuración de la gráfica")
    low_pct = st.slider("Rango hacia abajo (%)", 0, 95, 50, 5)
    high_pct = st.slider("Rango hacia arriba (%)", 0, 300, 50, 5)
    show_payoff = st.checkbox("Mostrar payoff bruto", value=True)
    show_pnl = st.checkbox("Mostrar P&L neto", value=True)
    show_legs = st.checkbox("Mostrar cada pata individualmente", value=True)
    show_break_evens = st.checkbox("Mostrar break-even(s)", value=True)

with right:
    if len(st.session_state.legs) == 0:
        st.warning("Todavía no hay instrumentos en la estrategia. Agrega uno o carga un template.")
    else:
        S, s_min, s_max = build_price_grid(spot, low_pct=low_pct, high_pct=high_pct, n=1201)

        total_payoff, total_pnl, leg_payoffs, leg_pnls = portfolio_payoff(S, st.session_state.legs)
        be_list = find_break_evens(S, total_pnl)
        net_cost = net_initial_cost(st.session_state.legs)
        payoff_spot, pnl_spot = payoff_at_point(spot, st.session_state.legs)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Payoff en spot", f"{payoff_spot:,.2f}")
        m2.metric("P&L en spot", f"{pnl_spot:,.2f}")
        m3.metric("Costo neto inicial", f"{net_cost:,.2f}")
        if len(be_list) == 0:
            m4.metric("Break-even(s)", "N/A")
        else:
            m4.metric("Break-even(s)", ", ".join([f"{x:,.2f}" for x in be_list[:3]]))

        fig = go.Figure()

        if show_legs:
            for i, leg in enumerate(st.session_state.legs):
                label = f"Leg {i+1}: {leg['side']} {leg['instrument']}"
                if show_payoff:
                    fig.add_trace(
                        go.Scatter(
                            x=S,
                            y=leg_payoffs[i],
                            mode="lines",
                            name=f"{label} | Payoff",
                            line=dict(width=1.5, dash="dot"),
                            opacity=0.55,
                        )
                    )
                if show_pnl:
                    fig.add_trace(
                        go.Scatter(
                            x=S,
                            y=leg_pnls[i],
                            mode="lines",
                            name=f"{label} | P&L",
                            line=dict(width=1.5, dash="dash"),
                            opacity=0.55,
                        )
                    )

        if show_payoff:
            fig.add_trace(
                go.Scatter(
                    x=S,
                    y=total_payoff,
                    mode="lines",
                    name="TOTAL Payoff",
                    line=dict(width=4),
                )
            )

        if show_pnl:
            fig.add_trace(
                go.Scatter(
                    x=S,
                    y=total_pnl,
                    mode="lines",
                    name="TOTAL P&L",
                    line=dict(width=4),
                )
            )

        fig.add_hline(y=0, line_dash="dash", line_width=1)
        fig.add_vline(x=spot, line_dash="dot", line_width=1.5)
        fig.add_annotation(
            x=spot,
            y=1,
            yref="paper",
            text=f"Spot = {spot:,.2f}",
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
                y=0.94,
                yref="paper",
                text=f"K={k:,.2f}",
                showarrow=False,
                textangle=-90,
            )

        if show_break_evens:
            for be in be_list:
                fig.add_vline(x=be, line_dash="dash", line_width=1)
                fig.add_annotation(
                    x=be,
                    y=0.85,
                    yref="paper",
                    text=f"BE={be:,.2f}",
                    showarrow=False,
                    textangle=-90,
                )

        fig.update_layout(
            title="Perfil de payoff / P&L de la estrategia",
            xaxis_title="Precio del subyacente al vencimiento",
            yaxis_title="Valor",
            hovermode="x unified",
            height=620,
            margin=dict(l=20, r=20, t=60, b=20),
            legend_title="Curvas",
        )

        st.plotly_chart(fig, use_container_width=True)

        st.markdown("### Instrumentos actuales")
        table_rows = []
        for i, leg in enumerate(st.session_state.legs, start=1):
            table_rows.append(
                {
                    "Leg": i,
                    "Instrument": leg["instrument"],
                    "Side": leg["side"],
                    "K": leg["K"],
                    "Premium": leg["premium"],
                    "Qty": leg["qty"],
                    "Multiplier": leg["multiplier"],
                }
            )
        st.dataframe(pd.DataFrame(table_rows), use_container_width=True)

        st.markdown("### Editar / borrar patas")
        remove_idx = st.selectbox(
            "Selecciona una pata para borrar",
            options=list(range(1, len(st.session_state.legs) + 1)),
            format_func=lambda x: f"Leg {x}",
        )
        if st.button("Borrar pata seleccionada"):
            st.session_state.legs.pop(remove_idx - 1)
            st.rerun()


# =========================================================
# PEDAGOGICAL SECTION
# =========================================================
st.markdown("---")
st.subheader("Cómo leer la gráfica")

st.write(
    """
- **Payoff**: cuánto paga el instrumento o la estrategia al vencimiento, sin considerar la prima pagada o recibida.
- **P&L**: ganancia o pérdida neta, ya incorporando la prima.
- **Long**: compras el instrumento.
- **Short**: vendes el instrumento.
- **La estrategia total** es la suma de todas las patas.
"""
)

