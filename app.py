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
st.caption("Visualizador introductorio de payoffs de derivados")


# =========================================================
# HELPERS
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
        cash = premium * qty * multiplier
        pnl = payoff - cash if side == "Long" else payoff + cash

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


def build_price_grid(spot, low_pct=40, high_pct=40, n=1201):
    s_min = max(0.0, spot * (1 - low_pct / 100.0))
    s_max = max(s_min + 1e-8, spot * (1 + high_pct / 100.0))
    return np.linspace(s_min, s_max, n), s_min, s_max


def net_initial_cost(legs):
    total = 0.0
    for leg in legs:
        inst = leg["instrument"]
        side = leg["side"]
        premium = float(leg["premium"])
        qty = float(leg["qty"])
        mult = float(leg["multiplier"])

        if inst == "Forward":
            cash = 0.0
        else:
            cash = premium * qty * mult

        total += cash if side == "Long" else -cash
    return total


def payoff_pnl_at_ST(ST, legs):
    rows = []
    total_payoff = 0.0
    total_pnl = 0.0

    for i, leg in enumerate(legs, start=1):
        payoff_i, pnl_i = payoff_single(
            S=np.array([ST]),
            instrument=leg["instrument"],
            side=leg["side"],
            K=float(leg["K"]),
            premium=float(leg["premium"]),
            qty=float(leg["qty"]),
            multiplier=float(leg["multiplier"]),
        )

        payoff_i = float(payoff_i[0])
        pnl_i = float(pnl_i[0])

        total_payoff += payoff_i
        total_pnl += pnl_i

        rows.append(
            {
                "Leg": i,
                "Instrument": leg["instrument"],
                "Side": leg["side"],
                "K": leg["K"],
                "Premium": leg["premium"],
                "Qty": leg["qty"],
                "Multiplier": leg["multiplier"],
                "Payoff@ST": payoff_i,
                "PnL@ST": pnl_i,
            }
        )

    return pd.DataFrame(rows), total_payoff, total_pnl


def find_break_evens(S, pnl, tol=1e-9):
    bes = []

    for i in range(len(S) - 1):
        x1, x2 = S[i], S[i + 1]
        y1, y2 = pnl[i], pnl[i + 1]

        if abs(y1) < tol:
            bes.append(x1)

        if y1 * y2 < 0:
            root = x1 - y1 * (x2 - x1) / (y2 - y1)
            bes.append(root)

    if abs(pnl[-1]) < tol:
        bes.append(S[-1])

    bes = sorted(bes)
    cleaned = []
    for x in bes:
        if not cleaned or abs(x - cleaned[-1]) > 1e-4:
            cleaned.append(x)

    return cleaned


def clamp(x, xmin, xmax):
    return min(max(float(x), float(xmin)), float(xmax))


def load_template(name, spot):
    if name == "Long Call":
        return [
            {"instrument": "Call", "side": "Long", "K": round(spot, 2), "premium": 5.0, "qty": 1.0, "multiplier": 1.0}
        ]
    if name == "Long Put":
        return [
            {"instrument": "Put", "side": "Long", "K": round(spot, 2), "premium": 5.0, "qty": 1.0, "multiplier": 1.0}
        ]
    if name == "Long Forward":
        return [
            {"instrument": "Forward", "side": "Long", "K": round(spot, 2), "premium": 0.0, "qty": 1.0, "multiplier": 1.0}
        ]
    if name == "Protective Put":
        return [
            {"instrument": "Stock", "side": "Long", "K": 0.0, "premium": round(spot, 2), "qty": 1.0, "multiplier": 1.0},
            {"instrument": "Put", "side": "Long", "K": round(spot, 2), "premium": 4.0, "qty": 1.0, "multiplier": 1.0},
        ]
    if name == "Covered Call":
        return [
            {"instrument": "Stock", "side": "Long", "K": 0.0, "premium": round(spot, 2), "qty": 1.0, "multiplier": 1.0},
            {"instrument": "Call", "side": "Short", "K": round(1.05 * spot, 2), "premium": 4.0, "qty": 1.0, "multiplier": 1.0},
        ]
    if name == "Bull Call Spread":
        return [
            {"instrument": "Call", "side": "Long", "K": round(0.95 * spot, 2), "premium": 7.0, "qty": 1.0, "multiplier": 1.0},
            {"instrument": "Call", "side": "Short", "K": round(1.05 * spot, 2), "premium": 3.0, "qty": 1.0, "multiplier": 1.0},
        ]
    if name == "Bear Put Spread":
        return [
            {"instrument": "Put", "side": "Long", "K": round(1.05 * spot, 2), "premium": 7.0, "qty": 1.0, "multiplier": 1.0},
            {"instrument": "Put", "side": "Short", "K": round(0.95 * spot, 2), "premium": 3.0, "qty": 1.0, "multiplier": 1.0},
        ]
    if name == "Long Straddle":
        return [
            {"instrument": "Call", "side": "Long", "K": round(spot, 2), "premium": 5.0, "qty": 1.0, "multiplier": 1.0},
            {"instrument": "Put", "side": "Long", "K": round(spot, 2), "premium": 5.0, "qty": 1.0, "multiplier": 1.0},
        ]
    return []


def set_builder_defaults_from_spot(spot):
    st.session_state.new_K = float(round(spot, 2))
    inst = st.session_state.get("new_instrument", "Call")

    if inst == "Stock":
        st.session_state.new_premium = float(round(spot, 2))
    elif inst == "Forward":
        st.session_state.new_premium = 0.0
    else:
        if "new_premium" not in st.session_state or st.session_state.new_premium in [0.0, float(round(spot, 2))]:
            st.session_state.new_premium = 5.0

    if "new_qty" not in st.session_state:
        st.session_state.new_qty = 1.0
    if "new_multiplier" not in st.session_state:
        st.session_state.new_multiplier = 1.0


def reset_ST_if_needed(s_min, s_max, spot):
    if "ST_eval" not in st.session_state:
        st.session_state.ST_eval = float(round(spot, 4))
    st.session_state.ST_eval = clamp(st.session_state.ST_eval, s_min, s_max)


# =========================================================
# SESSION STATE
# =========================================================
if "legs" not in st.session_state:
    st.session_state.legs = []

if "last_ticker" not in st.session_state:
    st.session_state.last_ticker = None

if "last_period" not in st.session_state:
    st.session_state.last_period = None

if "new_instrument" not in st.session_state:
    st.session_state.new_instrument = "Call"
if "new_side" not in st.session_state:
    st.session_state.new_side = "Long"
if "new_K" not in st.session_state:
    st.session_state.new_K = 100.0
if "new_premium" not in st.session_state:
    st.session_state.new_premium = 5.0
if "new_qty" not in st.session_state:
    st.session_state.new_qty = 1.0
if "new_multiplier" not in st.session_state:
    st.session_state.new_multiplier = 1.0


# =========================================================
# SIDEBAR
# =========================================================
st.sidebar.header("Datos del mercado")

tickers = [
    "AAPL", "MSFT", "SPY", "TSLA", "NVDA", "AMZN", "GOOGL",
    "META", "QQQ", "IWM", "BTC-USD", "ETH-USD", "USDMXN=X", "^GSPC"
]

ticker = st.sidebar.selectbox(
    "Ticker",
    tickers,
    index=12 if "USDMXN=X" in tickers else 0,
)

period_options = ["1mo", "3mo", "6mo", "1y"]
period = st.sidebar.radio("Histórico", period_options, index=2, horizontal=True)

manual_spot = st.sidebar.checkbox("Editar spot manualmente", value=False)

spot_mkt, chg, chg_pct, hist = get_market_data(ticker, period=period)

if spot_mkt is None:
    st.sidebar.error("No pude descargar datos de Yahoo Finance.")
    spot_mkt = 100.0
    chg = 0.0
    chg_pct = 0.0

if manual_spot:
    spot = st.sidebar.number_input(
        "Spot manual",
        min_value=0.0,
        value=float(round(spot_mkt, 4)),
        step=0.1,
    )
else:
    spot = float(round(spot_mkt, 4))

st.session_state.current_spot = float(spot)

if st.session_state.last_ticker is None:
    st.session_state.last_ticker = ticker
    set_builder_defaults_from_spot(spot)

if st.session_state.last_ticker != ticker:
    st.session_state.legs = []
    st.session_state.last_ticker = ticker
    set_builder_defaults_from_spot(spot)
    st.session_state.ST_eval = float(round(spot, 4))
    st.rerun()

if st.session_state.last_period is None:
    st.session_state.last_period = period

if st.session_state.last_period != period:
    st.session_state.last_period = period
    st.session_state.ST_eval = float(round(spot, 4))
    st.rerun()

if st.sidebar.button("Actualizar datos", use_container_width=True):
    st.cache_data.clear()
    st.rerun()


# =========================================================
# TOP BAR
# =========================================================
m1, m2, m3, m4 = st.columns([1.2, 1, 1, 1])

with m1:
    st.markdown(f"### `{ticker}`")

with m2:
    st.metric("Spot", f"{spot:,.4f}")

with m3:
    st.metric("Cambio", f"{chg:,.4f}")

with m4:
    st.metric("% cambio", f"{chg_pct:,.2f}%")


# =========================================================
# HIST CHART
# =========================================================
if hist is not None:
    fig_hist = go.Figure()

    fig_hist.add_trace(
        go.Scatter(
            x=hist["Date"],
            y=hist["Close"],
            mode="lines",
            name="Close",
            line=dict(width=3),
            hovertemplate="Fecha=%{x}<br>Close=%{y:,.4f}<extra></extra>",
        )
    )

    fig_hist.add_hline(y=spot, line_dash="dot", line_width=1.2)

    fig_hist.add_annotation(
        x=hist["Date"].iloc[-1],
        y=spot,
        text=f"Spot {spot:,.4f}",
        showarrow=False,
        xanchor="left",
    )

    fig_hist.update_layout(
        title="Cierres recientes",
        height=320,
        margin=dict(l=20, r=20, t=50, b=20),
        xaxis_title="Fecha",
        yaxis_title="Precio",
        hovermode="x unified",
    )

    st.plotly_chart(fig_hist, use_container_width=True, key=f"hist_{ticker}_{period}")

st.markdown("---")


# =========================================================
# QUICK TEMPLATES
# =========================================================
st.subheader("Estrategias rápidas")

t1, t2, t3, t4, t5, t6, t7, t8 = st.columns(8)

if t1.button("Long Call", use_container_width=True):
    st.session_state.legs = load_template("Long Call", spot)
    st.session_state.ST_eval = float(round(spot, 4))
    st.rerun()

if t2.button("Long Put", use_container_width=True):
    st.session_state.legs = load_template("Long Put", spot)
    st.session_state.ST_eval = float(round(spot, 4))
    st.rerun()

if t3.button("Forward", use_container_width=True):
    st.session_state.legs = load_template("Long Forward", spot)
    st.session_state.ST_eval = float(round(spot, 4))
    st.rerun()

if t4.button("Protective Put", use_container_width=True):
    st.session_state.legs = load_template("Protective Put", spot)
    st.session_state.ST_eval = float(round(spot, 4))
    st.rerun()

if t5.button("Covered Call", use_container_width=True):
    st.session_state.legs = load_template("Covered Call", spot)
    st.session_state.ST_eval = float(round(spot, 4))
    st.rerun()

if t6.button("Bull Spread", use_container_width=True):
    st.session_state.legs = load_template("Bull Call Spread", spot)
    st.session_state.ST_eval = float(round(spot, 4))
    st.rerun()

if t7.button("Bear Spread", use_container_width=True):
    st.session_state.legs = load_template("Bear Put Spread", spot)
    st.session_state.ST_eval = float(round(spot, 4))
    st.rerun()

if t8.button("Straddle", use_container_width=True):
    st.session_state.legs = load_template("Long Straddle", spot)
    st.session_state.ST_eval = float(round(spot, 4))
    st.rerun()


# =========================================================
# MAIN LAYOUT
# =========================================================
left, right = st.columns([1.05, 2.1])


# =========================================================
# LEFT PANEL
# =========================================================
with left:
    st.subheader("Builder")

    with st.expander("Agregar instrumento", expanded=True):
        with st.form("builder_form", clear_on_submit=False):
            instrument_options = ["Call", "Put", "Forward", "Stock"]
            side_options = ["Long", "Short"]

            current_inst = st.session_state.new_instrument if st.session_state.new_instrument in instrument_options else "Call"
            current_side = st.session_state.new_side if st.session_state.new_side in side_options else "Long"

            inst_form = st.selectbox(
                "Instrumento",
                instrument_options,
                index=instrument_options.index(current_inst),
            )

            side_form = st.selectbox(
                "Posición",
                side_options,
                index=side_options.index(current_side),
            )

            if inst_form == "Stock":
                k_default = 0.0
                premium_default = float(round(spot, 2))
            elif inst_form == "Forward":
                k_default = float(round(spot, 2))
                premium_default = 0.0
            else:
                k_default = float(round(st.session_state.new_K, 2))
                premium_default = float(st.session_state.new_premium)

            K_form = st.number_input(
                "Strike / Forward",
                min_value=0.0,
                value=float(k_default),
                step=1.0,
            )

            premium_form = st.number_input(
                "Prima / costo inicial",
                min_value=0.0,
                value=float(premium_default),
                step=1.0,
            )

            qty_form = st.number_input(
                "Cantidad",
                min_value=0.0,
                value=float(st.session_state.new_qty),
                step=1.0,
            )

            multiplier_form = st.number_input(
                "Multiplicador",
                min_value=0.0,
                value=float(st.session_state.new_multiplier),
                step=1.0,
            )

            add_leg = st.form_submit_button("Agregar", use_container_width=True)

        if add_leg:
            st.session_state.legs.append(
                {
                    "instrument": inst_form,
                    "side": side_form,
                    "K": float(K_form),
                    "premium": float(premium_form),
                    "qty": float(qty_form),
                    "multiplier": float(multiplier_form),
                }
            )

            st.session_state.new_instrument = inst_form
            st.session_state.new_side = side_form

            if inst_form == "Stock":
                st.session_state.new_K = 0.0
                st.session_state.new_premium = float(round(spot, 2))
            elif inst_form == "Forward":
                st.session_state.new_K = float(round(spot, 2))
                st.session_state.new_premium = 0.0
            else:
                st.session_state.new_K = float(round(spot, 2))
                st.session_state.new_premium = 5.0

            st.session_state.new_qty = 1.0
            st.session_state.new_multiplier = 1.0
            st.rerun()

    with st.expander("Controles visuales", expanded=True):
        low_pct = st.slider("Rango abajo (%)", 0, 95, 40, 5)
        high_pct = st.slider("Rango arriba (%)", 0, 250, 40, 5)

        show_total_payoff = st.checkbox("Mostrar total payoff", value=True)
        show_total_pnl = st.checkbox("Mostrar total P&L", value=True)
        show_legs = st.checkbox("Mostrar patas individuales", value=True)
        show_break_evens = st.checkbox("Mostrar break-even(s)", value=True)

    colx, coly = st.columns(2)

    with colx:
        if st.button("Vaciar estrategia", use_container_width=True):
            st.session_state.legs = []
            st.session_state.ST_eval = float(round(spot, 4))
            st.rerun()

    with coly:
        if len(st.session_state.legs) > 0:
            remove_idx = st.selectbox(
                "Borrar leg",
                options=list(range(1, len(st.session_state.legs) + 1)),
                format_func=lambda x: f"Leg {x}",
                key=f"remove_leg_selector_{len(st.session_state.legs)}",
            )
            if st.button("Borrar seleccionada", use_container_width=True):
                st.session_state.legs.pop(remove_idx - 1)
                st.rerun()


# =========================================================
# RIGHT PANEL
# =========================================================
with right:
    if len(st.session_state.legs) == 0:
        st.info("Agrega instrumentos o usa los botones de estrategias rápidas.")
    else:
        S, s_min, s_max = build_price_grid(spot, low_pct, high_pct, n=1201)
        reset_ST_if_needed(s_min, s_max, spot)

        step_temp = max((s_max - s_min) / 200.0, 0.01)

        ST = st.slider(
            "Precio final al vencimiento (S_T)",
            min_value=float(round(s_min, 4)),
            max_value=float(round(s_max, 4)),
            value=float(round(st.session_state.ST_eval, 4)),
            step=float(step_temp),
        )

        st.session_state.ST_eval = float(ST)

        st.caption(
            "Mueve el slider y observa cómo cambia el payoff de la estrategia."
        )

        total_payoff, total_pnl, leg_payoffs, leg_pnls = portfolio_payoff(S, st.session_state.legs)
        be_list = find_break_evens(S, total_pnl)
        net_cost = net_initial_cost(st.session_state.legs)

        detail_df, total_payoff_ST, total_pnl_ST = payoff_pnl_at_ST(ST, st.session_state.legs)

        max_profit = float(np.max(total_pnl))
        max_loss = float(np.min(total_pnl))

        r1, r2, r3, r4, r5 = st.columns(5)
        r1.metric("Costo inicial neto", f"{net_cost:,.2f}")
        r2.metric("Payoff total @ S_T", f"{total_payoff_ST:,.2f}")
        r3.metric("P&L total @ S_T", f"{total_pnl_ST:,.2f}")
        r4.metric("Max P&L en rango", f"{max_profit:,.2f}")
        r5.metric("Min P&L en rango", f"{max_loss:,.2f}")

        fig = go.Figure()

        if show_legs:
            for i, leg in enumerate(st.session_state.legs):
                base_name = f"Leg {i+1}: {leg['side']} {leg['instrument']}"

                fig.add_trace(
                    go.Scatter(
                        x=S,
                        y=leg_payoffs[i],
                        mode="lines",
                        name=f"{base_name} | payoff",
                        line=dict(width=1.5, dash="dot"),
                        opacity=0.40,
                        visible=True if show_total_payoff else "legendonly",
                        hovertemplate="S=%{x:,.2f}<br>Payoff=%{y:,.2f}<extra></extra>",
                    )
                )

                fig.add_trace(
                    go.Scatter(
                        x=S,
                        y=leg_pnls[i],
                        mode="lines",
                        name=f"{base_name} | P&L",
                        line=dict(width=1.5, dash="dash"),
                        opacity=0.40,
                        visible=True if show_total_pnl else "legendonly",
                        hovertemplate="S=%{x:,.2f}<br>P&L=%{y:,.2f}<extra></extra>",
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

        if show_total_pnl:
            fig.add_trace(
                go.Scatter(
                    x=S,
                    y=total_pnl,
                    mode="lines",
                    name="TOTAL P&L",
                    line=dict(width=4),
                    hovertemplate="S=%{x:,.2f}<br>Total P&L=%{y:,.2f}<extra></extra>",
                )
            )

        fig.add_hline(y=0, line_dash="dash", line_width=1)

        fig.add_vline(x=spot, line_dash="dot", line_width=1.4)
        fig.add_annotation(
            x=spot,
            y=1.0,
            yref="paper",
            text=f"Spot = {spot:,.2f}",
            showarrow=False,
            xanchor="left",
        )

        fig.add_vline(x=ST, line_dash="solid", line_width=2)
        fig.add_annotation(
            x=ST,
            y=0.93,
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
                y=0.86,
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
                    y=0.77,
                    yref="paper",
                    text=f"BE={be:,.2f}",
                    showarrow=False,
                    textangle=-90,
                )

        fig.add_trace(
            go.Scatter(
                x=[ST],
                y=[total_payoff_ST],
                mode="markers",
                marker=dict(size=11),
                name="Total payoff @ S_T",
                visible=True if show_total_payoff else "legendonly",
                hovertemplate="S_T=%{x:,.2f}<br>Total Payoff=%{y:,.2f}<extra></extra>",
            )
        )

        fig.add_trace(
            go.Scatter(
                x=[ST],
                y=[total_pnl_ST],
                mode="markers",
                marker=dict(size=11),
                name="Total P&L @ S_T",
                visible=True if show_total_pnl else "legendonly",
                hovertemplate="S_T=%{x:,.2f}<br>Total P&L=%{y:,.2f}<extra></extra>",
            )
        )

        y_all = []
        if show_total_payoff:
            y_all.extend(total_payoff.tolist())
            y_all.append(total_payoff_ST)
        if show_total_pnl:
            y_all.extend(total_pnl.tolist())
            y_all.append(total_pnl_ST)
        if show_legs:
            for arr in leg_payoffs + leg_pnls:
                y_all.extend(arr.tolist())

        if len(y_all) > 0:
            y_min = float(np.nanmin(y_all))
            y_max = float(np.nanmax(y_all))
            pad = 0.08 * max(1.0, y_max - y_min)
            y_range = [y_min - pad, y_max + pad]
        else:
            y_range = None

        fig.update_layout(
            title="Perfil de payoff / P&L",
            xaxis_title="Precio del subyacente al vencimiento",
            yaxis_title="Valor",
            hovermode="x unified",
            height=700,
            margin=dict(l=20, r=20, t=60, b=20),
            legend_title="Curvas",
            xaxis=dict(range=[s_min, s_max]),
            yaxis=dict(range=y_range) if y_range else None,
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
            key=f"payoff_chart_{ticker}_{period}_{low_pct}_{high_pct}_{len(st.session_state.legs)}_{round(spot, 4)}",
        )

        st.subheader("Qué pasa en ese precio final")
        st.write(f"Actualmente estás evaluando la estrategia en **S_T = {ST:,.4f}**.")

        st.dataframe(
            detail_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "K": st.column_config.NumberColumn(format="%.2f"),
                "Premium": st.column_config.NumberColumn(format="%.2f"),
                "Qty": st.column_config.NumberColumn(format="%.2f"),
                "Multiplier": st.column_config.NumberColumn(format="%.2f"),
                "Payoff@ST": st.column_config.NumberColumn(format="%.2f"),
                "PnL@ST": st.column_config.NumberColumn(format="%.2f"),
            },
        )

        st.subheader("Patas actuales")

        current_legs_df = pd.DataFrame(
            [
                {
                    "Leg": i + 1,
                    "Instrument": leg["instrument"],
                    "Side": leg["side"],
                    "K": leg["K"],
                    "Premium": leg["premium"],
                    "Qty": leg["qty"],
                    "Multiplier": leg["multiplier"],
                }
                for i, leg in enumerate(st.session_state.legs)
            ]
        )

        st.dataframe(
            current_legs_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "K": st.column_config.NumberColumn(format="%.2f"),
                "Premium": st.column_config.NumberColumn(format="%.2f"),
                "Qty": st.column_config.NumberColumn(format="%.2f"),
                "Multiplier": st.column_config.NumberColumn(format="%.2f"),
            },
        )


# =========================================================
# TEACHING NOTES
# =========================================================
st.markdown("---")

st.write(
    """
- **Payoff**: pago bruto al vencimiento.  
- **P&L**: ganancia o pérdida neta, incorporando la prima o costo inicial.  
- **Long**: compras el instrumento.  
- **Short**: vendes el instrumento.
"""
)
