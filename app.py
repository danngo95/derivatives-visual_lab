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
st.caption("Visualizador introductorio de payoffs con una acción sin dividendos")


# =========================================================
# MATH HELPERS
# =========================================================
def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def bs_call_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
    if T <= 0:
        return max(S - K, 0.0)
    if sigma <= 1e-12:
        return max(S - K * math.exp(-r * T), 0.0)

    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return S * norm_cdf(d1) - K * math.exp(-r * T) * norm_cdf(d2)


def bs_put_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
    if T <= 0:
        return max(K - S, 0.0)
    if sigma <= 1e-12:
        return max(K * math.exp(-r * T) - S, 0.0)

    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return K * math.exp(-r * T) * norm_cdf(-d2) - S * norm_cdf(-d1)


def forward_price_no_dividends(S: float, r: float, T: float) -> float:
    return S * math.exp(r * T)


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
            return None, None, None, None, None

        if isinstance(data.columns, pd.MultiIndex):
            if ("Close", ticker) in data.columns:
                close = data[("Close", ticker)].dropna()
            else:
                close = data.xs("Close", axis=1, level=0).iloc[:, 0].dropna()
        else:
            close = data["Close"].dropna()

        if close.empty:
            return None, None, None, None, None

        spot = float(close.iloc[-1])
        prev = float(close.iloc[-2]) if len(close) >= 2 else spot
        chg = spot - prev
        chg_pct = (chg / prev * 100.0) if prev != 0 else 0.0

        log_ret = np.log(close / close.shift(1)).dropna()
        hist_vol_20 = float(log_ret.tail(20).std() * np.sqrt(252)) if len(log_ret) >= 20 else float(log_ret.std() * np.sqrt(252))
        hist_vol_60 = float(log_ret.tail(60).std() * np.sqrt(252)) if len(log_ret) >= 60 else hist_vol_20

        hist = close.reset_index()
        hist.columns = ["Date", "Close"]

        return spot, chg, chg_pct, hist, {"vol20": hist_vol_20, "vol60": hist_vol_60}

    except Exception:
        return None, None, None, None, None


# =========================================================
# PAYOFF HELPERS
# =========================================================
def estimated_premium(leg, spot, r, sigma):
    inst = leg["instrument"]
    K = float(leg["K"])
    T = float(leg["T"])

    if inst == "Call":
        return bs_call_price(spot, K, T, r, sigma)
    if inst == "Put":
        return bs_put_price(spot, K, T, r, sigma)
    if inst == "Forward":
        return 0.0
    if inst == "Stock":
        return spot
    return 0.0


def payoff_single(S, instrument, side, K, premium, qty=1.0):
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
        payoff = sign * raw * qty
        pnl = sign * (S - premium) * qty
    elif instrument == "Forward":
        payoff = sign * raw * qty
        pnl = payoff
    else:
        payoff = sign * raw * qty
        cash = premium * qty
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
        )
        total_payoff += payoff_i
        total_pnl += pnl_i
        leg_payoffs.append(payoff_i)
        leg_pnls.append(pnl_i)

    return total_payoff, total_pnl, leg_payoffs, leg_pnls


def build_price_grid_from_view(spot, view_mode="Medio", n=601):
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


def net_initial_cost(legs):
    total = 0.0
    for leg in legs:
        inst = leg["instrument"]
        side = leg["side"]
        premium = float(leg["premium"])
        qty = float(leg["qty"])

        cash = 0.0 if inst == "Forward" else premium * qty
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
                "Strike/Fwd": leg["K"],
                "T (años)": leg["T"],
                "Prima": leg["premium"],
                "# contratos": leg["qty"],
                "Payoff@S_T": payoff_i,
                "P&L@S_T": pnl_i,
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


def make_leg(instrument, side, K, T, qty, spot, r, sigma):
    leg = {
        "instrument": instrument,
        "side": side,
        "K": float(K),
        "T": float(T),
        "qty": float(qty),
    }
    leg["premium"] = estimated_premium(leg, spot, r, sigma)
    return leg


def load_template(name, spot, r, sigma, T):
    if name == "Long Call":
        return [make_leg("Call", "Long", round(spot, 2), T, 1.0, spot, r, sigma)]
    if name == "Long Put":
        return [make_leg("Put", "Long", round(spot, 2), T, 1.0, spot, r, sigma)]
    if name == "Long Forward":
        fwd = forward_price_no_dividends(spot, r, T)
        return [make_leg("Forward", "Long", round(fwd, 2), T, 1.0, spot, r, sigma)]
    if name == "Protective Put":
        return [
            make_leg("Stock", "Long", 0.0, T, 1.0, spot, r, sigma),
            make_leg("Put", "Long", round(spot, 2), T, 1.0, spot, r, sigma),
        ]
    if name == "Covered Call":
        return [
            make_leg("Stock", "Long", 0.0, T, 1.0, spot, r, sigma),
            make_leg("Call", "Short", round(1.05 * spot, 2), T, 1.0, spot, r, sigma),
        ]
    if name == "Bull Call Spread":
        return [
            make_leg("Call", "Long", round(0.95 * spot, 2), T, 1.0, spot, r, sigma),
            make_leg("Call", "Short", round(1.05 * spot, 2), T, 1.0, spot, r, sigma),
        ]
    if name == "Bear Put Spread":
        return [
            make_leg("Put", "Long", round(1.05 * spot, 2), T, 1.0, spot, r, sigma),
            make_leg("Put", "Short", round(0.95 * spot, 2), T, 1.0, spot, r, sigma),
        ]
    if name == "Straddle":
        return [
            make_leg("Call", "Long", round(spot, 2), T, 1.0, spot, r, sigma),
            make_leg("Put", "Long", round(spot, 2), T, 1.0, spot, r, sigma),
        ]
    if name == "Synthetic Long Forward":
        return [
            make_leg("Call", "Long", round(spot, 2), T, 1.0, spot, r, sigma),
            make_leg("Put", "Short", round(spot, 2), T, 1.0, spot, r, sigma),
        ]
    return []


def reset_st_slider_to_spot(spot):
    st.session_state.ST_eval = float(round(spot, 4))


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

period = st.sidebar.selectbox("Ventana histórica para datos", ["3mo", "6mo", "1y"], index=1)

manual_spot = st.sidebar.checkbox("Editar spot manualmente", value=False)

spot_mkt, chg, chg_pct, hist, vol_info = get_market_data(ticker, period=period)

if spot_mkt is None:
    st.sidebar.error("No pude descargar datos.")
    spot_mkt = 100.0
    chg = 0.0
    chg_pct = 0.0
    vol_info = {"vol20": 0.20, "vol60": 0.20}

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

vol_source = st.sidebar.radio(
    "Volatilidad estimada",
    ["20 días", "60 días", "Manual"],
    index=1,
)

if vol_source == "20 días":
    sigma = float(max(vol_info["vol20"], 1e-4))
elif vol_source == "60 días":
    sigma = float(max(vol_info["vol60"], 1e-4))
else:
    sigma_manual_pct = st.sidebar.slider("Volatilidad manual (%)", 1.0, 150.0, 25.0, 1.0)
    sigma = sigma_manual_pct / 100.0

default_days = st.sidebar.slider("Vencimiento base (días)", 7, 365, 90, 1)
T_base = default_days / 365.0

if st.sidebar.button("Actualizar datos", use_container_width=True):
    st.cache_data.clear()
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.caption("Se asume una acción sin dividendos.")


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

if st.session_state.last_ticker != ticker:
    st.session_state.legs = []
    st.session_state.last_ticker = ticker
    st.session_state.last_spot = float(spot)
    reset_st_slider_to_spot(spot)
    st.rerun()

if st.session_state.last_period != period:
    st.session_state.last_period = period
    st.session_state.last_spot = float(spot)
    reset_st_slider_to_spot(spot)
    st.rerun()

if abs(float(st.session_state.last_spot) - float(spot)) > 1e-12:
    st.session_state.last_spot = float(spot)
    reset_st_slider_to_spot(spot)


# =========================================================
# TOP SUMMARY
# =========================================================
call_atm = bs_call_price(spot, spot, T_base, r, sigma)
put_atm = bs_put_price(spot, spot, T_base, r, sigma)
fwd_theo = forward_price_no_dividends(spot, r, T_base)
pvK = spot * math.exp(-r * T_base)
parity_lhs = call_atm - put_atm
parity_rhs = spot - spot * math.exp(-r * T_base)

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Spot", f"{spot:,.4f}", f"{chg:,.4f} ({chg_pct:,.2f}%)")
m2.metric("Vol. estimada", f"{sigma*100:,.1f}%")
m3.metric("Vencimiento base", f"{default_days} días")
m4.metric("Call ATM estimado", f"{call_atm:,.2f}")
m5.metric("Put ATM estimado", f"{put_atm:,.2f}")

with st.expander("Relaciones teóricas útiles", expanded=False):
    c1, c2, c3 = st.columns(3)
    c1.metric("Forward teórico", f"{fwd_theo:,.4f}")
    c2.metric("c - p", f"{parity_lhs:,.4f}")
    c3.metric("S₀ - K e^(-rT) con K=S₀", f"{parity_rhs:,.4f}")

st.markdown("---")


# =========================================================
# QUICK TEMPLATES
# =========================================================
st.subheader("Estrategias rápidas")

t1, t2, t3, t4, t5, t6, t7, t8 = st.columns(8)

if t1.button("Long Call", use_container_width=True):
    st.session_state.legs = load_template("Long Call", spot, r, sigma, T_base)
    reset_st_slider_to_spot(spot)
    st.rerun()

if t2.button("Long Put", use_container_width=True):
    st.session_state.legs = load_template("Long Put", spot, r, sigma, T_base)
    reset_st_slider_to_spot(spot)
    st.rerun()

if t3.button("Forward", use_container_width=True):
    st.session_state.legs = load_template("Long Forward", spot, r, sigma, T_base)
    reset_st_slider_to_spot(spot)
    st.rerun()

if t4.button("Protective Put", use_container_width=True):
    st.session_state.legs = load_template("Protective Put", spot, r, sigma, T_base)
    reset_st_slider_to_spot(spot)
    st.rerun()

if t5.button("Covered Call", use_container_width=True):
    st.session_state.legs = load_template("Covered Call", spot, r, sigma, T_base)
    reset_st_slider_to_spot(spot)
    st.rerun()

if t6.button("Bull Spread", use_container_width=True):
    st.session_state.legs = load_template("Bull Call Spread", spot, r, sigma, T_base)
    reset_st_slider_to_spot(spot)
    st.rerun()

if t7.button("Straddle", use_container_width=True):
    st.session_state.legs = load_template("Straddle", spot, r, sigma, T_base)
    reset_st_slider_to_spot(spot)
    st.rerun()

if t8.button("Synth Fwd", use_container_width=True):
    st.session_state.legs = load_template("Synthetic Long Forward", spot, r, sigma, T_base)
    reset_st_slider_to_spot(spot)
    st.rerun()


# =========================================================
# MAIN LAYOUT
# =========================================================
left, right = st.columns([1.0, 2.2])


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

            days_form = st.number_input(
                "Vencimiento (días)",
                min_value=1,
                value=int(default_days),
                step=1,
            )

            qty_form = st.number_input(
                "# contratos",
                min_value=0.0,
                value=1.0,
                step=1.0,
            )

            T_form = float(days_form) / 365.0
            tmp_leg = {
                "instrument": instrument,
                "side": side,
                "K": float(K_form),
                "T": T_form,
                "qty": float(qty_form),
            }
            premium_est = estimated_premium(tmp_leg, spot, r, sigma)

            if instrument == "Forward":
                st.info(f"Prima inicial estimada: 0.00")
            elif instrument == "Stock":
                st.info(f"Costo inicial estimado: {premium_est:,.2f}")
            else:
                st.info(f"Prima estimada: {premium_est:,.2f}")

            add_leg = st.form_submit_button("Agregar", use_container_width=True)

        if add_leg:
            st.session_state.legs.append(
                {
                    "instrument": instrument,
                    "side": side,
                    "K": float(K_form),
                    "T": T_form,
                    "qty": float(qty_form),
                    "premium": float(premium_est),
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
        st.caption("Controla qué tan lejos del spot quieres ver el payoff.")

        show_total_payoff = st.checkbox("Mostrar total payoff", value=True)
        show_total_pnl = st.checkbox("Mostrar total P&L", value=True)
        show_legs = st.checkbox("Mostrar patas individuales", value=False)
        show_break_evens = st.checkbox("Mostrar break-even(s)", value=True)

    a, b = st.columns(2)

    with a:
        if st.button("Vaciar estrategia", use_container_width=True):
            st.session_state.legs = []
            reset_st_slider_to_spot(spot)
            st.rerun()

    with b:
        if len(st.session_state.legs) > 0:
            remove_idx = st.selectbox(
                "Borrar leg",
                options=list(range(1, len(st.session_state.legs) + 1)),
                format_func=lambda x: f"Leg {x}",
                key="remove_leg_selector",
            )
            if st.button("Borrar seleccionada", use_container_width=True):
                st.session_state.legs.pop(remove_idx - 1)
                st.rerun()

    if len(st.session_state.legs) > 0:
        st.subheader("Patas actuales")
        current_legs_df = pd.DataFrame(
            [
                {
                    "Leg": i + 1,
                    "Instrument": leg["instrument"],
                    "Side": leg["side"],
                    "Strike/Fwd": leg["K"],
                    "T (años)": leg["T"],
                    "Prima": leg["premium"],
                    "# contratos": leg["qty"],
                }
                for i, leg in enumerate(st.session_state.legs)
            ]
        )

        st.dataframe(
            current_legs_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Strike/Fwd": st.column_config.NumberColumn(format="%.2f"),
                "T (años)": st.column_config.NumberColumn(format="%.3f"),
                "Prima": st.column_config.NumberColumn(format="%.2f"),
                "# contratos": st.column_config.NumberColumn(format="%.2f"),
            },
        )


# =========================================================
# RIGHT PANEL
# =========================================================
with right:
    if len(st.session_state.legs) == 0:
        st.info("Agrega instrumentos o usa una estrategia rápida.")
    else:
        S, s_min, s_max = build_price_grid_from_view(spot, view_mode, n=601)
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

        total_payoff, total_pnl, leg_payoffs, leg_pnls = portfolio_payoff(S, st.session_state.legs)
        be_list = find_break_evens(S, total_pnl)

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
                        opacity=0.45,
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
                        opacity=0.45,
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
        fig.add_vline(x=spot, line_dash="dot", line_width=1.3)

        fig.add_annotation(
            x=spot,
            y=1.0,
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
                y=0.9,
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
                    y=0.78,
                    yref="paper",
                    text=f"BE={be:,.2f}",
                    showarrow=False,
                    textangle=-90,
                )

        y_all = []
        if show_total_payoff:
            y_all.extend(total_payoff.tolist())
        if show_total_pnl:
            y_all.extend(total_pnl.tolist())
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
            height=620,
            margin=dict(l=20, r=20, t=60, b=20),
            legend_title="Curvas",
            xaxis=dict(range=[s_min, s_max]),
            yaxis=dict(range=y_range) if y_range else None,
        )

        st.plotly_chart(fig, use_container_width=True, key="payoff_chart")

        ST = st.slider(
            "Precio final al vencimiento (S_T)",
            min_value=s_min,
            max_value=s_max,
            step=float(step_temp),
            key="ST_eval",
            format="%.4f",
        )
        ST = float(ST)

        detail_df, total_payoff_ST, total_pnl_ST = payoff_pnl_at_ST(ST, st.session_state.legs)
        net_cost = net_initial_cost(st.session_state.legs)
        max_profit = float(np.max(total_pnl))
        max_loss = float(np.min(total_pnl))

        st.caption("Ahora sí, estos números corresponden al valor elegido en el slider de S_T.")

        r1, r2, r3, r4, r5 = st.columns(5)
        r1.metric("Costo inicial neto", f"{net_cost:,.2f}")
        r2.metric("Payoff total @ S_T", f"{total_payoff_ST:,.2f}")
        r3.metric("P&L total @ S_T", f"{total_pnl_ST:,.2f}")
        r4.metric("Max P&L en vista", f"{max_profit:,.2f}")
        r5.metric("Min P&L en vista", f"{max_loss:,.2f}")

        st.subheader("Qué pasa en ese precio final")
        st.write(f"Evaluación actual en **S_T = {ST:,.4f}**.")

        st.dataframe(
            detail_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Strike/Fwd": st.column_config.NumberColumn(format="%.2f"),
                "T (años)": st.column_config.NumberColumn(format="%.3f"),
                "Prima": st.column_config.NumberColumn(format="%.2f"),
                "# contratos": st.column_config.NumberColumn(format="%.2f"),
                "Payoff@S_T": st.column_config.NumberColumn(format="%.2f"),
                "P&L@S_T": st.column_config.NumberColumn(format="%.2f"),
            },
        )


# =========================================================
# NOTES
# =========================================================
st.markdown("---")
st.write(
    """
- **Payoff**: pago bruto al vencimiento.  
- **P&L**: ganancia o pérdida neta, incorporando la prima o costo inicial.  
- **Forward**: aquí se usa la convención sin costo inicial y con payoff \(S_T-K\).  
- **Acción sin dividendos**: útil para explorar forward teórico y paridad put-call.  
"""
)
