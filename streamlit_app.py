import streamlit as st
import yfinance as yf

st.set_page_config(page_title="Micro Futures Notional Calculator", layout="wide")

st.title("Micro Futures Notional Value Calculator")
st.write(
    "Calculate the notional value of micro futures contracts offered by CME Group. "
    "Prices are fetched automatically from Yahoo Finance. You can also override them manually."
)

# Mapping from micro futures symbol to Yahoo Finance ticker.
# "invert" means we need 1/price (e.g. USD/JPY -> JPY/USD).
YAHOO_TICKERS = {
    "MES": {"ticker": "ES=F"},
    "MNQ": {"ticker": "NQ=F"},
    "MYM": {"ticker": "YM=F"},
    "M2K": {"ticker": "RTY=F"},
    "MCL": {"ticker": "CL=F"},
    "MGC": {"ticker": "GC=F"},
    "SIL": {"ticker": "SI=F"},
    "MHG": {"ticker": "HG=F"},
    "MBT": {"ticker": "BTC-USD"},
    "MET": {"ticker": "ETH-USD"},
    "M6E": {"ticker": "EURUSD=X"},
    "M6B": {"ticker": "GBPUSD=X"},
    "M6A": {"ticker": "AUDUSD=X"},
    "M6J": {"ticker": "JPY=X", "invert": True},
    "M6C": {"ticker": "CAD=X", "invert": True},
    "10Y": {"ticker": "^TNX"},
    "2YY": {"ticker": "^IRX"},
}

# Fallback prices if Yahoo Finance is unavailable
FALLBACK_PRICES = {
    "MES": 5950.00, "MNQ": 21300.00, "MYM": 43800.00, "M2K": 2250.00,
    "MCL": 72.50, "MGC": 2750.00, "SIL": 31.50, "MHG": 4.25,
    "MBT": 97000.00, "MET": 2700.00, "M6E": 1.08, "M6B": 1.27,
    "M6A": 0.65, "M6J": 0.0067, "M6C": 0.72, "10Y": 4.25, "2YY": 4.10,
}


@st.cache_data(ttl=300)  # Cache for 5 minutes
def fetch_all_prices():
    """Fetch latest prices for all tickers from Yahoo Finance."""
    tickers = " ".join(info["ticker"] for info in YAHOO_TICKERS.values())
    prices = {}
    try:
        data = yf.download(tickers, period="1d", progress=False)
        close = data["Close"]
        for symbol, info in YAHOO_TICKERS.items():
            try:
                col = info["ticker"]
                val = close[col].dropna().iloc[-1]
                if info.get("invert"):
                    val = 1.0 / val
                prices[symbol] = float(val)
            except (KeyError, IndexError):
                prices[symbol] = FALLBACK_PRICES[symbol]
    except Exception:
        return FALLBACK_PRICES.copy()
    return prices


# Micro futures contract specifications:
# (display_name, symbol, multiplier, description)
MICRO_CONTRACTS = [
    ("Micro E-mini S&P 500", "MES", 5, "1/10th of E-mini S&P 500"),
    ("Micro E-mini Nasdaq-100", "MNQ", 2, "1/10th of E-mini Nasdaq-100"),
    ("Micro E-mini Dow Jones", "MYM", 0.50, "1/10th of E-mini Dow Jones"),
    ("Micro E-mini Russell 2000", "M2K", 5, "1/10th of E-mini Russell 2000"),
    ("Micro WTI Crude Oil", "MCL", 100, "1/10th of standard WTI (100 barrels)"),
    ("Micro Gold", "MGC", 10, "10 troy ounces"),
    ("Micro Silver", "SIL", 1000, "1,000 troy ounces"),
    ("Micro Copper", "MHG", 2500, "2,500 pounds"),
    ("Micro Bitcoin", "MBT", 0.10, "1/10th of 1 Bitcoin"),
    ("Micro Ether", "MET", 0.10, "1/10th of 50 Ether"),
    ("Micro EUR/USD", "M6E", 12500, "12,500 euros"),
    ("Micro GBP/USD", "M6B", 6250, "6,250 British pounds"),
    ("Micro AUD/USD", "M6A", 10000, "10,000 Australian dollars"),
    ("Micro USD/JPY", "M6J", 1250000, "1,250,000 Japanese yen (inverted)"),
    ("Micro USD/CAD", "M6C", 10000, "10,000 Canadian dollars (inverted)"),
    ("Micro 10-Year Yield", "10Y", 1000, "$1,000 x yield index"),
    ("Micro 2-Year Yield", "2YY", 1000, "$1,000 x yield index"),
]

# Fetch live prices
live_prices = fetch_all_prices()

st.divider()

# Global controls
col_controls_1, col_controls_2, col_controls_3 = st.columns([1, 1, 2])
with col_controls_1:
    default_qty = st.number_input(
        "Default number of contracts", min_value=1, max_value=1000, value=1, step=1
    )
with col_controls_2:
    if st.button("Refresh Prices"):
        st.cache_data.clear()
        st.rerun()

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

for name, symbol, multiplier, description in MICRO_CONTRACTS:
    cols = st.columns([2, 1, 1.2, 1.2, 1, 1.5])

    fetched_price = live_prices.get(symbol, FALLBACK_PRICES[symbol])

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
            value=fetched_price,
            step=fetched_price * 0.001,
            format="%.4f" if fetched_price < 1 else "%.2f",
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

# ── Kelly Criterion Leverage Analysis ──
st.header("Kelly Criterion Leverage Analysis")
st.write(
    "The Kelly Criterion determines the optimal leverage ratio that maximizes "
    "long-term geometric growth of your portfolio. Enter your account details "
    "and market assumptions below."
)

kelly_input_cols = st.columns(2)

with kelly_input_cols[0]:
    st.subheader("Account Inputs")
    nlv = st.number_input(
        "Net Liquidation Value ($)",
        min_value=0.0,
        value=100000.00,
        step=1000.00,
        format="%.2f",
        help="Your total account equity / net liquidation value.",
    )
    exposure_method = st.radio(
        "Exposure method",
        ["Use total notional from contracts above", "Enter SPX Delta manually"],
        help="Choose how to determine your portfolio's dollar exposure.",
    )
    if exposure_method == "Enter SPX Delta manually":
        spx_delta = st.number_input(
            "SPX Beta-Weighted Delta ($)",
            min_value=0.0,
            value=total_notional,
            step=1000.00,
            format="%.2f",
            help="Your portfolio's total dollar delta expressed in SPX-equivalent terms.",
        )
    else:
        spx_delta = total_notional
        st.info(f"Using total notional from above: **${total_notional:,.2f}**")

with kelly_input_cols[1]:
    st.subheader("Market Assumptions")
    expected_return = st.number_input(
        "Expected annual return (%)",
        min_value=0.0,
        max_value=100.0,
        value=10.0,
        step=0.5,
        format="%.1f",
        help="Long-term expected annual return of the portfolio. S&P 500 historical average is ~10%.",
    )
    risk_free_rate = st.number_input(
        "Risk-free rate (%)",
        min_value=0.0,
        max_value=50.0,
        value=4.5,
        step=0.25,
        format="%.2f",
        help="Current risk-free rate (e.g. T-bill yield).",
    )
    annual_volatility = st.number_input(
        "Expected annual volatility (%)",
        min_value=0.1,
        max_value=200.0,
        value=16.0,
        step=0.5,
        format="%.1f",
        help="Expected annualized standard deviation. S&P 500 long-term average is ~15-17%.",
    )

# Calculate Kelly
mu_excess = (expected_return - risk_free_rate) / 100.0
sigma = annual_volatility / 100.0
kelly_optimal = mu_excess / (sigma ** 2) if sigma > 0 else 0.0
half_kelly = kelly_optimal / 2.0

# Current leverage
current_leverage = spx_delta / nlv if nlv > 0 else 0.0

st.divider()

# Results
result_cols = st.columns(3)
with result_cols[0]:
    st.metric("Full Kelly Leverage", f"{kelly_optimal:.2f}x")
with result_cols[1]:
    st.metric("Half Kelly Leverage", f"{half_kelly:.2f}x")
with result_cols[2]:
    leverage_delta = current_leverage - kelly_optimal
    st.metric(
        "Your Current Leverage",
        f"{current_leverage:.2f}x",
        delta=f"{leverage_delta:+.2f}x vs Kelly",
        delta_color="inverse",
    )

# Warnings
if nlv > 0 and current_leverage > 0:
    if current_leverage > kelly_optimal:
        st.error(
            f"**Above Full Kelly!** Your leverage ({current_leverage:.2f}x) exceeds the "
            f"Kelly optimal ({kelly_optimal:.2f}x). This level of leverage is expected to "
            f"**reduce** long-term geometric growth and significantly increases the risk of "
            f"large drawdowns. Consider reducing exposure."
        )
    elif current_leverage > half_kelly:
        st.warning(
            f"**Above Half Kelly.** Your leverage ({current_leverage:.2f}x) is between "
            f"Half Kelly ({half_kelly:.2f}x) and Full Kelly ({kelly_optimal:.2f}x). "
            f"You are in the aggressive zone — returns are still positive but with "
            f"substantially higher volatility. Many practitioners target Half Kelly "
            f"as a practical maximum."
        )
    else:
        st.success(
            f"**At or below Half Kelly.** Your leverage ({current_leverage:.2f}x) is at or "
            f"below the Half Kelly level ({half_kelly:.2f}x). This is a conservative "
            f"position that sacrifices some expected growth for meaningfully lower "
            f"volatility and drawdown risk."
        )

# Kelly detail table
with st.expander("Kelly Calculation Details"):
    st.markdown(f"""
| Parameter | Value |
|---|---|
| Expected Return (μ) | {expected_return:.1f}% |
| Risk-Free Rate (r) | {risk_free_rate:.2f}% |
| Excess Return (μ - r) | {expected_return - risk_free_rate:.2f}% |
| Volatility (σ) | {annual_volatility:.1f}% |
| Variance (σ²) | {annual_volatility**2 / 100:.2f}% |
| **Full Kelly (f* = (μ-r) / σ²)** | **{kelly_optimal:.2f}x** |
| **Half Kelly (f*/2)** | **{half_kelly:.2f}x** |
| Net Liquidation Value | ${nlv:,.2f} |
| Portfolio Exposure (SPX Delta) | ${spx_delta:,.2f} |
| **Current Leverage** | **{current_leverage:.2f}x** |
| Kelly-Optimal Notional | ${nlv * kelly_optimal:,.2f} |
| Half-Kelly Notional | ${nlv * half_kelly:,.2f} |
""")

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

**Data Source:** Prices are fetched from Yahoo Finance and cached for 5 minutes.
Click "Refresh Prices" to get the latest quotes. Prices may be delayed.
Always verify current prices with your broker or exchange data provider.
""")

with st.expander("About the Kelly Criterion"):
    st.markdown("""
**What is the Kelly Criterion?**

The Kelly Criterion, developed by John L. Kelly Jr. in 1956, determines the
optimal fraction of capital to allocate to a risky bet (or investment) to
maximize the long-term geometric growth rate of wealth.

**Kelly Leverage Formula**

For a continuously-compounding portfolio with normally-distributed returns:

> **f* = (μ - r) / σ²**

Where:
- **f*** = optimal leverage ratio
- **μ** = expected annual return of the asset/portfolio
- **r** = risk-free rate
- **σ²** = variance (volatility squared) of the asset/portfolio

**Why Half Kelly?**

Full Kelly is theoretically optimal but assumes perfect knowledge of the true
return and volatility parameters. In practice:

- Parameter estimates are uncertain — small errors in μ or σ can lead to
  significant over-leveraging
- Full Kelly produces large drawdowns (~50%+ is common)
- Half Kelly captures ~75% of the growth rate with substantially less variance
- Most professional allocators and practitioners use Half Kelly or less

**Example: S&P 500**

Using approximate long-run parameters (10% return, 4.5% risk-free, 16% vol):

> f* = (10% - 4.5%) / (16%)² = 5.5% / 2.56% = **2.15x**
>
> Half Kelly = **1.07x**

This suggests a portfolio fully invested in an S&P 500 index at 1x leverage is
roughly at the Half Kelly level — a reasonable, moderate position.

**SPX Delta / Beta-Weighted Delta**

If your portfolio contains multiple assets (futures, options, stocks), you can
convert the total exposure to an SPX-equivalent dollar delta. This beta-weighted
delta represents how much your portfolio would move for a $1 move in the S&P 500,
giving a single measure of equity-like exposure to compare against Kelly.
""")

