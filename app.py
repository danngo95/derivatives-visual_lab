import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st


# -----------------------------
# Page setup
# -----------------------------
st.set_page_config(
    page_title="Derivatives Visual Lab",
    page_icon="📈",
    layout="wide",
)

st.title("Derivatives Visual Lab")
st.caption("Payoff Visualizer · Hedging Simulator · Strategy Builder")


# -----------------------------
# Helpers
# -----------------------------
def payoff_at_expiry(S, instrument, K, premium, side):
    if instrument == "Call":
        raw = np.maximum(S - K, 0.0)
    elif instrument == "Put":
        raw = np.maximum(K - S, 0.0)
    elif instrument == "Forward":
        raw = S - K
    else:
        raw = np.zeros_like(S)

    signed_payoff = raw if side == "Long" else -raw

    if instrument == "Forward":
        signed_pnl = signed_payoff
    else:
        signed_pnl = signed_payoff - premium if side == "Long" else signed_payoff + premium

    return signed_payoff, signed_pnl


def build_grid(spot, range_pct, n=201):
    s_min = max(0.0, spot * (1 - range_pct / 100))
    s_max = spot * (1 + range_pct / 100)
    if s_max <= s_min:
        s_max = s_min + 1.0
    return np.linspace(s_min, s_max, n), s_min, s_max


def single_break_even(instrument, K, premium):
    if instrument == "Call":
        return K + premium
    if instrument == "Put":
        return K - premium
    return K


def strategy_value(S, legs):
    total_payoff = np.zeros_like(S, dtype=float)
    total_pnl = np.zeros_like(S, dtype=float)

    for leg in legs:
        payoff, pnl = payoff_at_expiry(
            S=S,
            instrument=leg["instrument"],
            K=float(leg["K"]),
            premium=float(leg["premium"]),
            side=leg["side"],
        )
        qty = float(leg["qty"])
        total_payoff += qty * payoff
        total_pnl += qty * pnl

    return total_payoff, total_pnl


def plot_lines(x, series_dict, title, x_label="Underlying price at expiry", y_label="Value", vlines=None):
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for label, y in series_dict.items():
        ax.plot(x, y, linewidth=2.5, label=label)

    ax.axhline(0, linestyle="--", linewidth=1)

    if vlines:
        for label, x0 in vlines:
            if x0 is not None and np.isfinite(x0):
                ax.axvline(x0, linestyle=":", linewidth=1.2)
                ax.text(x0, ax.get_ylim()[1] * 0.92, label, rotation=90, va="top", ha="right")

    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.grid(True, alpha=0.3)
    ax.legend()
    st.pyplot(fig)


def default_strategy(name):
    if name == "Long Straddle":
        return [
            {"instrument": "Call", "side": "Long", "K": 100.0, "premium": 8.0, "qty": 1.0},
            {"instrument": "Put", "side": "Long", "K": 100.0, "premium": 7.0, "qty": 1.0},
        ]
    if name == "Bull Call Spread":
        return [
            {"instrument": "Call", "side": "Long", "K": 95.0, "premium": 10.0, "qty": 1.0},
            {"instrument": "Call", "side": "Short", "K": 105.0, "premium": 5.0, "qty": 1.0},
        ]
    if name == "Risk Reversal":
        return [
            {"instrument": "Call", "side": "Long", "K": 105.0, "premium": 4.0, "qty": 1.0},
            {"instrument": "Put", "side": "Short", "K": 95.0, "premium": 4.0, "qty": 1.0},
        ]
    return [
        {"instrument": "Call", "side": "Long", "K": 100.0, "premium": 8.0, "qty": 1.0},
    ]


# -----------------------------
# Tabs
# -----------------------------
tab1, tab2, tab3 = st.tabs(
    ["Payoff Visualizer", "Hedging Simulator", "Strategy Builder"]
)


# -----------------------------
# 1) Payoff Visualizer
# -----------------------------
with tab1:
    st.subheader("Payoff Visualizer")

    c1, c2 = st.columns([1, 2])

    with c1:
        instrument = st.selectbox("Instrument", ["Call", "Put", "Forward"], key="pv_instrument")
        side = st.radio("Side", ["Long", "Short"], horizontal=True, key="pv_side")
        K = st.number_input("Strike / Forward price", value=100.0, step=1.0, key="pv_k")
        spot = st.number_input("Current spot", value=100.0, step=1.0, key="pv_spot")
        premium = st.number_input(
            "Premium",
            value=8.0,
            step=1.0,
            disabled=(instrument == "Forward"),
            key="pv_premium",
        )
        range_pct = st.slider("Underlying range around spot (%)", 10, 200, 50, 5, key="pv_range")
        show_pnl = st.checkbox("Show P&L line", value=True, key="pv_show_pnl")

    S, s_min, s_max = build_grid(spot, range_pct)
    payoff, pnl = payoff_at_expiry(S, instrument, K, premium, side)
    payoff_spot, pnl_spot = payoff_at_expiry(np.array([spot]), instrument, K, premium, side)
    be = single_break_even(instrument, K, premium)

    with c2:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Payoff at spot", f"{payoff_spot[0]:,.2f}")
        m2.metric("P&L at spot", f"{pnl_spot[0]:,.2f}")
        m3.metric("Break-even", f"{be:,.2f}")
        m4.metric("Range", f"{s_min:,.2f} to {s_max:,.2f}")

        series = {"Payoff": payoff}
        if show_pnl:
            series["P&L"] = pnl

        plot_lines(
            x=S,
            series_dict=series,
            title="Expiration profile",
            vlines=[("K", K), ("Spot", spot), ("BE", be)],
        )

        df = pd.DataFrame({"S_T": S, "Payoff": payoff, "PnL": pnl})
        with st.expander("See data table"):
            st.dataframe(df, use_container_width=True)


# -----------------------------
# 2) Hedging Simulator
# -----------------------------
with tab2:
    st.subheader("Hedging Simulator")

    c1, c2 = st.columns([1, 2])

    with c1:
        instrument_h = st.selectbox("Instrument", ["Call", "Put", "Forward"], key="hd_instrument")
        side_h = st.radio("Side", ["Long", "Short"], horizontal=True, key="hd_side")
        K_h = st.number_input("Strike", value=100.0, step=1.0, key="hd_k")
        spot_h = st.number_input("Current spot", value=100.0, step=1.0, key="hd_spot")
        premium_h = st.number_input(
            "Premium",
            value=8.0,
            step=1.0,
            disabled=(instrument_h == "Forward"),
            key="hd_premium",
        )
        hedge_delta = st.slider("Stock hedge delta", -2.0, 2.0, 0.5, 0.05, key="hd_delta")
        hedge_scale = st.number_input("Contracts / hedge scale", value=1.0, step=1.0, key="hd_scale")
        range_pct_h = st.slider("Underlying range around spot (%)", 10, 150, 40, 5, key="hd_range")

        st.info("Modelo simple: P&L de la opción + cobertura lineal en stock alrededor del spot actual.")

    S_h, smin_h, smax_h = build_grid(spot_h, range_pct_h)
    _, option_pnl = payoff_at_expiry(S_h, instrument_h, K_h, premium_h, side_h)
    stock_hedge = hedge_scale * hedge_delta * (S_h - spot_h)
    combined = option_pnl + stock_hedge

    _, option_pnl_spot = payoff_at_expiry(np.array([spot_h]), instrument_h, K_h, premium_h, side_h)

    with c2:
        m1, m2, m3 = st.columns(3)
        m1.metric("Option P&L at spot", f"{option_pnl_spot[0]:,.2f}")
        m2.metric("Hedge P&L at spot", f"{0.0:,.2f}")
        m3.metric("Combined P&L at spot", f"{option_pnl_spot[0]:,.2f}")

        plot_lines(
            x=S_h,
            series_dict={
                "Option P&L": option_pnl,
                "Stock Hedge": stock_hedge,
                "Combined Hedged P&L": combined,
            },
            title="Hedge comparison",
            vlines=[("K", K_h), ("Spot", spot_h)],
        )

        df_h = pd.DataFrame(
            {
                "S_T": S_h,
                "Option_PnL": option_pnl,
                "Stock_Hedge": stock_hedge,
                "Combined": combined,
            }
        )
        with st.expander("See data table"):
            st.dataframe(df_h, use_container_width=True)


# -----------------------------
# 3) Strategy Builder
# -----------------------------
with tab3:
    st.subheader("Strategy Builder")

    if "strategy_legs" not in st.session_state:
        st.session_state.strategy_legs = default_strategy("Long Straddle")

    left, right = st.columns([1.15, 1.85])

    with left:
        template = st.selectbox(
            "Template",
            ["Custom", "Long Straddle", "Bull Call Spread", "Risk Reversal"],
            key="sb_template",
        )

        if template != "Custom":
            if st.button("Load template"):
                st.session_state.strategy_legs = default_strategy(template)

        spot_s = st.number_input("Current spot", value=100.0, step=1.0, key="sb_spot")
        range_pct_s = st.slider("Underlying range around spot (%)", 10, 200, 50, 5, key="sb_range")
        show_pnl_s = st.checkbox("Show total P&L", value=True, key="sb_show_pnl")

        n_legs = st.number_input("Number of legs", min_value=1, max_value=8, value=len(st.session_state.strategy_legs), step=1)

        current = st.session_state.strategy_legs
        if n_legs > len(current):
            for _ in range(n_legs - len(current)):
                current.append(
                    {"instrument": "Call", "side": "Long", "K": 100.0, "premium": 5.0, "qty": 1.0}
                )
        elif n_legs < len(current):
            current = current[:n_legs]
            st.session_state.strategy_legs = current

        updated_legs = []
        for i, leg in enumerate(st.session_state.strategy_legs):
            st.markdown(f"**Leg {i+1}**")
            a, b = st.columns(2)
            instrument_i = a.selectbox(
                f"Instrument {i+1}",
                ["Call", "Put", "Forward"],
                index=["Call", "Put", "Forward"].index(leg["instrument"]),
                key=f"inst_{i}",
            )
            side_i = b.selectbox(
                f"Side {i+1}",
                ["Long", "Short"],
                index=["Long", "Short"].index(leg["side"]),
                key=f"side_{i}",
            )

            c, d, e = st.columns(3)
            K_i = c.number_input(f"Strike {i+1}", value=float(leg["K"]), step=1.0, key=f"K_{i}")
            premium_i = d.number_input(
                f"Premium {i+1}",
                value=float(leg["premium"]),
                step=1.0,
                disabled=(instrument_i == "Forward"),
                key=f"prem_{i}",
            )
            qty_i = e.number_input(f"Qty {i+1}", value=float(leg["qty"]), step=1.0, key=f"qty_{i}")

            updated_legs.append(
                {
                    "instrument": instrument_i,
                    "side": side_i,
                    "K": K_i,
                    "premium": premium_i,
                    "qty": qty_i,
                }
            )
            st.divider()

        st.session_state.strategy_legs = updated_legs

    S_s, smin_s, smax_s = build_grid(spot_s, range_pct_s)
    total_payoff, total_pnl = strategy_value(S_s, st.session_state.strategy_legs)
    payoff_spot_s, pnl_spot_s = strategy_value(np.array([spot_s]), st.session_state.strategy_legs)

    with right:
        m1, m2 = st.columns(2)
        m1.metric("Strategy payoff at spot", f"{payoff_spot_s[0]:,.2f}")
        m2.metric("Strategy P&L at spot", f"{pnl_spot_s[0]:,.2f}")

        series_s = {"Total Payoff": total_payoff}
        if show_pnl_s:
            series_s["Total P&L"] = total_pnl

        plot_lines(
            x=S_s,
            series_dict=series_s,
            title="Strategy profile",
            vlines=[("Spot", spot_s)],
        )

        df_s = pd.DataFrame({"S_T": S_s, "Total_Payoff": total_payoff, "Total_PnL": total_pnl})
        with st.expander("See data table"):
            st.dataframe(df_s, use_container_width=True)