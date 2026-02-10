import streamlit as st

st.set_page_config(page_title="Micro Futures Notional Calculator", layout="wide")

st.title("Micro Futures Notional Value Calculator")
st.write(
    "Calculate the notional value of micro futures contracts offered by CME Group. "
    "Enter the current price for each contract to see the notional value per contract "
    "and for your desired number of contracts."
)

# Micro futures contract specifications:
# (display_name, symbol, multiplier, default_price, description)
MICRO_CONTRACTS = [
    ("Micro E-mini S&P 500", "MES", 5, 5950.00, "1/10th of E-mini S&P 500"),
    ("Micro E-mini Nasdaq-100", "MNQ", 2, 21300.00, "1/10th of E-mini Nasdaq-100"),
    ("Micro E-mini Dow Jones", "MYM", 0.50, 43800.00, "1/10th of E-mini Dow Jones"),
    ("Micro E-mini Russell 2000", "M2K", 5, 2250.00, "1/10th of E-mini Russell 2000"),
    ("Micro WTI Crude Oil", "MCL", 100, 72.50, "1/10th of standard WTI (100 barrels)"),
    ("Micro Gold", "MGC", 10, 2750.00, "10 troy ounces"),
    ("Micro Silver", "SIL", 1000, 31.50, "1,000 troy ounces"),
    ("Micro Copper", "MHG", 2500, 4.25, "2,500 pounds"),
    ("Micro Bitcoin", "MBT", 0.10, 97000.00, "1/10th of 1 Bitcoin"),
    ("Micro Ether", "MET", 0.10, 2700.00, "1/10th of 50 Ether"),
    ("Micro EUR/USD", "M6E", 12500, 1.08, "12,500 euros"),
    ("Micro GBP/USD", "M6B", 6250, 1.27, "6,250 British pounds"),
    ("Micro AUD/USD", "M6A", 10000, 0.65, "10,000 Australian dollars"),
    ("Micro USD/JPY", "M6J", 1250000, 0.0067, "1,250,000 Japanese yen (inverted)"),
    ("Micro USD/CAD", "M6C", 10000, 0.72, "10,000 Canadian dollars (inverted)"),
    ("Micro 10-Year Yield", "10Y", 1000, 4.25, "$1,000 x yield index"),
    ("Micro 2-Year Yield", "2YY", 1000, 4.10, "$1,000 x yield index"),
]

st.divider()

# Global controls
col_controls_1, col_controls_2 = st.columns([1, 3])
with col_controls_1:
    default_qty = st.number_input(
        "Default number of contracts", min_value=1, max_value=1000, value=1, step=1
    )

st.divider()

# Build the table header
header_cols = st.columns([2, 1, 1.2, 1.2, 1, 1.5])
header_cols[0].markdown("**Contract**")
header_cols[1].markdown("**Symbol**")
header_cols[2].markdown("**Multiplier**")
header_cols[3].markdown("**Price**")
header_cols[4].markdown("**Qty**")
header_cols[5].markdown("**Notional Value**")

st.divider()

total_notional = 0.0

for name, symbol, multiplier, default_price, description in MICRO_CONTRACTS:
    cols = st.columns([2, 1, 1.2, 1.2, 1, 1.5])

    with cols[0]:
        st.markdown(f"**{name}**")
        st.caption(description)

    with cols[1]:
        st.code(symbol, language=None)

    with cols[2]:
        if multiplier >= 1:
            st.write(f"${multiplier:,.2f}")
        else:
            st.write(f"${multiplier}")

    with cols[3]:
        price = st.number_input(
            f"Price ({symbol})",
            min_value=0.0,
            value=default_price,
            step=default_price * 0.001,
            format="%.4f" if default_price < 1 else "%.2f",
            label_visibility="collapsed",
            key=f"price_{symbol}",
        )

    with cols[4]:
        qty = st.number_input(
            f"Qty ({symbol})",
            min_value=0,
            max_value=10000,
            value=default_qty,
            step=1,
            label_visibility="collapsed",
            key=f"qty_{symbol}",
        )

    notional = price * multiplier * qty
    total_notional += notional

    with cols[5]:
        st.markdown(f"### ${notional:,.2f}")

st.divider()

# Summary
st.subheader("Portfolio Summary")
summary_cols = st.columns(3)
with summary_cols[0]:
    active_contracts = sum(
        1
        for _, symbol, *_ in MICRO_CONTRACTS
        if st.session_state.get(f"qty_{symbol}", default_qty) > 0
    )
    st.metric("Active Contracts", active_contracts)
with summary_cols[1]:
    st.metric("Total Notional Value", f"${total_notional:,.2f}")
with summary_cols[2]:
    total_qty = sum(
        st.session_state.get(f"qty_{symbol}", default_qty)
        for _, symbol, *_ in MICRO_CONTRACTS
    )
    st.metric("Total Contracts", total_qty)

st.divider()

# Reference information
with st.expander("About Micro Futures Contracts"):
    st.markdown("""
**What are Micro Futures?**

Micro futures are smaller-sized versions of standard futures contracts, typically
1/10th the size of their full-sized counterparts. They were introduced by CME Group
to make futures trading more accessible to individual traders.

**How is Notional Value Calculated?**

> **Notional Value = Current Price x Contract Multiplier x Number of Contracts**

For example, if the Micro E-mini S&P 500 (MES) is trading at 5,950 with a multiplier
of $5, one contract has a notional value of:

> 5,950 x $5 = **$29,750**

**Why Does Notional Value Matter?**

- It represents the total market exposure of your position
- It helps you understand the actual leverage in your account
- Margin requirements are typically a small fraction of the notional value
- Risk management decisions should account for full notional exposure

**Disclaimer:** Default prices are approximate and for illustration only. Always
verify current prices with your broker or exchange data provider.
""")
