from datetime import date, timedelta

import numpy as np
import pandas as pd
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



@st.cache_data(ttl=300)
def fetch_market_assumptions():
    """Fetch market-derived defaults for Kelly assumptions.

    Returns dict with:
      - risk_free_rate: 13-week T-bill yield (^IRX)
      - implied_vol: VIX level (^VIX)
      - realized_vol: trailing 1-year realized volatility of S&P 500
      - trailing_return: trailing 1-year annualized return of S&P 500
      - expected_return: risk-free rate + long-run equity risk premium (5.5%)
    """
    defaults = {
        "risk_free_rate": 4.5,
        "implied_vol": 16.0,
        "realized_vol": 16.0,
        "trailing_return": 10.0,
        "expected_return": 10.0,
    }
    try:
        # Fetch VIX and T-bill yield
        data = yf.download("^VIX ^IRX", period="5d", progress=False)
        close = data["Close"]

        # Risk-free rate from 13-week T-bill (^IRX is quoted in %, e.g. 4.5)
        try:
            defaults["risk_free_rate"] = round(float(close["^IRX"].dropna().iloc[-1]), 2)
        except (KeyError, IndexError):
            pass

        # Implied vol from VIX
        try:
            defaults["implied_vol"] = round(float(close["^VIX"].dropna().iloc[-1]), 1)
        except (KeyError, IndexError):
            pass

        # Trailing 1-year S&P 500 data for realized vol and return
        spx = yf.download("^GSPC", period="1y", progress=False)
        spx_close = spx["Close"]
        if hasattr(spx_close, "columns"):
            spx_close = spx_close.iloc[:, 0]
        if len(spx_close) > 20:
            daily_returns = spx_close.pct_change().dropna()
            realized = float(daily_returns.std() * np.sqrt(252) * 100)
            defaults["realized_vol"] = round(realized, 1)

            total_return = float(spx_close.iloc[-1] / spx_close.iloc[0] - 1)
            annualized = total_return * (252 / len(daily_returns)) * 100
            defaults["trailing_return"] = round(annualized, 1)

        # Expected return = risk-free + equity risk premium (long-run ~5.5%)
        ERP = 5.5
        defaults["expected_return"] = round(defaults["risk_free_rate"] + ERP, 1)

    except Exception:
        pass
    return defaults


# ── Expiration date calculation ──

def _nth_weekday(year, month, weekday, n):
    """Return the nth occurrence of a weekday in a given month (1-indexed)."""
    first = date(year, month, 1)
    # Days until the first occurrence of the target weekday
    offset = (weekday - first.weekday()) % 7
    d = first + timedelta(days=offset + 7 * (n - 1))
    return d


def _third_friday(year, month):
    """3rd Friday of a given month."""
    return _nth_weekday(year, month, 4, 3)  # 4 = Friday


def _last_business_day(year, month):
    """Last business day (Mon-Fri) of a given month."""
    if month == 12:
        d = date(year + 1, 1, 1) - timedelta(days=1)
    else:
        d = date(year, month + 1, 1) - timedelta(days=1)
    while d.weekday() > 4:  # Sat=5, Sun=6
        d -= timedelta(days=1)
    return d


def _last_friday(year, month):
    """Last Friday of a given month."""
    d = _last_business_day(year, month)
    while d.weekday() != 4:
        d -= timedelta(days=1)
    return d


def next_expiration(symbol, ref_date=None):
    """Calculate the next expiration date for a micro futures contract.

    Rules by contract type:
    - Equity indices (MES, MNQ, MYM, M2K): Quarterly (H/M/U/Z), 3rd Friday
    - MCL: Monthly, ~3 business days before 25th of prior month (simplified: 21st or prior biz day)
    - MGC: Even months (G/J/M/Q/V/Z), 3rd-to-last business day of month prior
    - SIL: Monthly, 3rd-to-last business day of month prior
    - MHG: Monthly, 3rd-to-last business day of month prior
    - MBT, MET: Monthly, last Friday of the contract month
    - FX micros (M6E, M6B, M6A, M6J, M6C): Quarterly (H/M/U/Z), 3rd Friday (2 biz days prior for settlement)
    - Yield micros (10Y, 2YY): Quarterly (H/M/U/Z), 3rd Friday

    Simplified: we return the primary expiration date for the next available contract.
    """
    if ref_date is None:
        ref_date = date.today()

    quarterly_months = [3, 6, 9, 12]
    even_months = [2, 4, 6, 8, 10, 12]

    if symbol in ("MES", "MNQ", "MYM", "M2K", "M6E", "M6B", "M6A", "M6J", "M6C", "10Y", "2YY"):
        # Quarterly: 3rd Friday of Mar, Jun, Sep, Dec
        for offset in range(12):
            m = ((ref_date.month - 1 + offset) % 12) + 1
            y = ref_date.year + ((ref_date.month - 1 + offset) // 12)
            if m in quarterly_months:
                exp = _third_friday(y, m)
                if exp > ref_date:
                    return exp

    elif symbol == "MCL":
        # Monthly: trading terminates ~4 business days before the 25th of the month prior
        # to the contract month. Simplified: we look for the next monthly cycle.
        for offset in range(1, 14):
            m = ((ref_date.month - 1 + offset) % 12) + 1
            y = ref_date.year + ((ref_date.month - 1 + offset) // 12)
            # Expiration is in the month *before* the contract month
            exp_m = m - 1 if m > 1 else 12
            exp_y = y if m > 1 else y - 1
            d = date(exp_y, exp_m, 25)
            # Back up 4 business days
            biz = 0
            while biz < 4:
                d -= timedelta(days=1)
                if d.weekday() < 5:
                    biz += 1
            if d > ref_date:
                return d

    elif symbol in ("MGC",):
        # Even months: 3rd-to-last business day of the month prior to contract month
        for offset in range(12):
            m = ((ref_date.month - 1 + offset) % 12) + 1
            y = ref_date.year + ((ref_date.month - 1 + offset) // 12)
            if m in even_months:
                exp_m = m - 1 if m > 1 else 12
                exp_y = y if m > 1 else y - 1
                d = _last_business_day(exp_y, exp_m)
                for _ in range(2):  # go back 2 more biz days
                    d -= timedelta(days=1)
                    while d.weekday() > 4:
                        d -= timedelta(days=1)
                if d > ref_date:
                    return d

    elif symbol in ("SIL", "MHG"):
        # Monthly: 3rd-to-last business day of the month prior to the contract month
        for offset in range(1, 14):
            m = ((ref_date.month - 1 + offset) % 12) + 1
            y = ref_date.year + ((ref_date.month - 1 + offset) // 12)
            exp_m = m - 1 if m > 1 else 12
            exp_y = y if m > 1 else y - 1
            d = _last_business_day(exp_y, exp_m)
            for _ in range(2):
                d -= timedelta(days=1)
                while d.weekday() > 4:
                    d -= timedelta(days=1)
            if d > ref_date:
                return d

    elif symbol in ("MBT", "MET"):
        # Monthly: last Friday of the contract month
        for offset in range(13):
            m = ((ref_date.month - 1 + offset) % 12) + 1
            y = ref_date.year + ((ref_date.month - 1 + offset) // 12)
            exp = _last_friday(y, m)
            if exp > ref_date:
                return exp

    return None


# ── Contract specifications ──
# tick_size: minimum price increment
# tick_value: dollar value of one tick
# maint_margin: approximate maintenance margin per contract (subject to change)
# asset_class: asset category for grouping
# spx_beta: approximate beta relative to S&P 500 (for beta-weighted delta)

CONTRACT_SPECS = {
    "MES": {"tick_size": "0.25 pts",  "tick_inc": 0.25,    "tick_value": 1.25,  "maint_margin": 1500, "asset_class": "Equity",  "spx_beta": 1.00},
    "MNQ": {"tick_size": "0.25 pts",  "tick_inc": 0.25,    "tick_value": 0.50,  "maint_margin": 2100, "asset_class": "Equity",  "spx_beta": 1.20},
    "MYM": {"tick_size": "1.0 pt",    "tick_inc": 1.0,     "tick_value": 0.50,  "maint_margin": 1000, "asset_class": "Equity",  "spx_beta": 0.90},
    "M2K": {"tick_size": "0.10 pts",  "tick_inc": 0.10,    "tick_value": 0.50,  "maint_margin": 750,  "asset_class": "Equity",  "spx_beta": 1.20},
    "MCL": {"tick_size": "$0.01",     "tick_inc": 0.01,    "tick_value": 1.00,  "maint_margin": 950,  "asset_class": "Energy",  "spx_beta": 0.60},
    "MGC": {"tick_size": "$0.10",     "tick_inc": 0.10,    "tick_value": 1.00,  "maint_margin": 1550, "asset_class": "Metal",   "spx_beta": 0.15},
    "SIL": {"tick_size": "$0.005",    "tick_inc": 0.005,   "tick_value": 5.00,  "maint_margin": 1900, "asset_class": "Metal",   "spx_beta": 0.30},
    "MHG": {"tick_size": "$0.0005",   "tick_inc": 0.0005,  "tick_value": 1.25,  "maint_margin": 700,  "asset_class": "Metal",   "spx_beta": 0.50},
    "MBT": {"tick_size": "$5.00",     "tick_inc": 5.0,     "tick_value": 0.50,  "maint_margin": 5100, "asset_class": "Crypto",  "spx_beta": 2.00},
    "MET": {"tick_size": "$0.50",     "tick_inc": 0.50,    "tick_value": 0.05,  "maint_margin": 350,  "asset_class": "Crypto",  "spx_beta": 2.50},
    "M6E": {"tick_size": "$0.0001",   "tick_inc": 0.0001,  "tick_value": 1.25,  "maint_margin": 290,  "asset_class": "FX",      "spx_beta": 0.10},
    "M6B": {"tick_size": "$0.0001",   "tick_inc": 0.0001,  "tick_value": 0.625, "maint_margin": 200,  "asset_class": "FX",      "spx_beta": 0.10},
    "M6A": {"tick_size": "$0.0001",   "tick_inc": 0.0001,  "tick_value": 1.00,  "maint_margin": 180,  "asset_class": "FX",      "spx_beta": 0.20},
    "M6J": {"tick_size": "¥0.01",     "tick_inc": 0.000001, "tick_value": 1.25, "maint_margin": 250,  "asset_class": "FX",      "spx_beta": 0.10},
    "M6C": {"tick_size": "$0.0001",   "tick_inc": 0.0001,  "tick_value": 1.00,  "maint_margin": 160,  "asset_class": "FX",      "spx_beta": 0.15},
    "10Y": {"tick_size": "0.01 pts",  "tick_inc": 0.01,    "tick_value": 10.00, "maint_margin": 440,  "asset_class": "Rates",   "spx_beta": -0.30},
    "2YY": {"tick_size": "0.01 pts",  "tick_inc": 0.01,    "tick_value": 10.00, "maint_margin": 340,  "asset_class": "Rates",   "spx_beta": -0.15},
}


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

def _snap_to_tick(price, tick_inc):
    """Round a price to the nearest tick increment."""
    if tick_inc <= 0:
        return price
    return round(round(price / tick_inc) * tick_inc, 10)


# Fetch live prices
live_prices = fetch_all_prices()

st.divider()

# Global controls
col_controls_1, col_controls_2, col_controls_3 = st.columns([1, 1, 2])
with col_controls_1:
    default_qty = st.number_input(
        "Default number of contracts", min_value=0, max_value=1000, value=0, step=1
    )
# When default qty changes, push it into all individual contract qty widgets
prev_default = st.session_state.get("_prev_default_qty")
if prev_default is not None and default_qty != prev_default:
    for _, symbol, *_ in MICRO_CONTRACTS:
        key = f"qty_{symbol}"
        if key in st.session_state:
            st.session_state[key] = default_qty
st.session_state["_prev_default_qty"] = default_qty
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
total_margin = 0.0
total_beta_weighted_delta = 0.0
class_breakdown = {}  # {class: {"notional": ..., "beta_delta": ..., "margin": ..., "qty": ...}}

for name, symbol, multiplier, description in MICRO_CONTRACTS:
    cols = st.columns([2, 1, 1.2, 1.2, 1, 1.5])

    spec = CONTRACT_SPECS.get(symbol, {})
    tick_inc = spec.get("tick_inc", 0)
    fetched_price = live_prices.get(symbol, FALLBACK_PRICES[symbol])
    # Snap fetched price to the nearest valid tick increment
    if tick_inc > 0:
        fetched_price = _snap_to_tick(fetched_price, tick_inc)

    # Determine decimal places from tick increment
    if tick_inc > 0:
        tick_str = f"{tick_inc:.10f}".rstrip("0")
        decimals = len(tick_str.split(".")[-1]) if "." in tick_str else 0
    else:
        decimals = 4 if fetched_price < 1 else 2
    fmt = f"%.{decimals}f"

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
            step=tick_inc if tick_inc > 0 else fetched_price * 0.001,
            format=fmt,
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
    beta = spec.get("spx_beta", 0.0)
    beta_delta = notional * beta
    contract_margin = spec.get("maint_margin", 0) * qty
    total_notional += notional
    total_margin += contract_margin
    total_beta_weighted_delta += beta_delta

    # Accumulate per-class
    cls = spec.get("asset_class", "Other")
    if cls not in class_breakdown:
        class_breakdown[cls] = {"notional": 0.0, "beta_delta": 0.0, "margin": 0.0, "qty": 0}
    class_breakdown[cls]["notional"] += notional
    class_breakdown[cls]["beta_delta"] += beta_delta
    class_breakdown[cls]["margin"] += contract_margin
    class_breakdown[cls]["qty"] += qty

    with cols[5]:
        st.markdown(f"### ${notional:,.2f}")

st.divider()

# Summary
st.subheader("Portfolio Summary")
summary_cols = st.columns(5)
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
    st.metric("Beta-Weighted Delta", f"${total_beta_weighted_delta:,.2f}")
with summary_cols[3]:
    total_qty = sum(
        st.session_state.get(f"qty_{symbol}", default_qty)
        for _, symbol, *_ in MICRO_CONTRACTS
    )
    st.metric("Total Contracts", total_qty)
with summary_cols[4]:
    st.metric("Total Maint. Margin", f"${total_margin:,.0f}")

# Asset class breakdown
if any(v["qty"] > 0 for v in class_breakdown.values()):
    class_order = ["Equity", "Energy", "Metal", "Crypto", "FX", "Rates"]
    breakdown_rows = []
    for cls in class_order:
        if cls in class_breakdown and class_breakdown[cls]["qty"] > 0:
            d = class_breakdown[cls]
            breakdown_rows.append({
                "Asset Class": cls,
                "Contracts": d["qty"],
                "Notional": f"${d['notional']:,.2f}",
                "Beta-Wtd Delta": f"${d['beta_delta']:,.2f}",
                "Maint. Margin": f"${d['margin']:,.0f}",
                "% of Notional": f"{d['notional'] / total_notional * 100:.1f}%" if total_notional > 0 else "—",
            })
    if breakdown_rows:
        st.caption("Breakdown by Asset Class")
        st.dataframe(pd.DataFrame(breakdown_rows), use_container_width=True, hide_index=True)

st.divider()

# ── Contract Specifications ──
st.header("Contract Specifications")
st.write(
    "Tick size, tick value, maintenance margin, and next expiration for each contract. "
    "Margins are approximate and subject to change by the exchange — always verify with your broker."
)

today = date.today()
specs_data = []
for name, symbol, multiplier, description in MICRO_CONTRACTS:
    spec = CONTRACT_SPECS.get(symbol, {})
    exp = next_expiration(symbol, today)
    if exp:
        days_to_exp = (exp - today).days
        exp_str = exp.strftime("%b %d, %Y")
        dte_str = f"{days_to_exp}d"
    else:
        exp_str = "—"
        dte_str = "—"
    specs_data.append({
        "Contract": name,
        "Symbol": symbol,
        "Class": spec.get("asset_class", "—"),
        "SPX Beta": f"{spec.get('spx_beta', 0):.2f}",
        "Tick Size": spec.get("tick_size", "—"),
        "Tick Value": f"${spec.get('tick_value', 0):.2f}",
        "Maint. Margin": f"${spec.get('maint_margin', 0):,.0f}",
        "Next Expiration": exp_str,
        "DTE": dte_str,
    })

specs_df = pd.DataFrame(specs_data)
st.dataframe(specs_df, use_container_width=True, hide_index=True)

st.caption(
    "Maintenance margins are approximate values from CME Group and are updated periodically. "
    "Actual margin requirements may differ based on your broker and account type. "
    "Expiration dates are calculated based on standard CME contract cycle rules."
)

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
    if total_margin > 0:
        margin_pct = (total_margin / nlv * 100) if nlv > 0 else 0.0
        st.info(
            f"Total maintenance margin from contracts above: **${total_margin:,.0f}** "
            f"({margin_pct:.1f}% of NLV)"
        )
    exposure_method = st.radio(
        "Exposure method",
        [
            "Use beta-weighted delta from contracts above",
            "Use total notional from contracts above",
            "Enter SPX Delta in dollars",
            "Enter SPX Delta in shares",
        ],
        help="Choose how to determine your portfolio's dollar exposure.",
    )
    spx_price = live_prices.get("MES", FALLBACK_PRICES["MES"])
    if exposure_method == "Enter SPX Delta in dollars":
        spx_delta = st.number_input(
            "SPX Beta-Weighted Delta ($)",
            min_value=0.0,
            value=total_notional,
            step=1000.00,
            format="%.2f",
            help="Your portfolio's total dollar delta expressed in SPX-equivalent terms.",
        )
    elif exposure_method == "Enter SPX Delta in shares":
        spx_shares = st.number_input(
            "SPX Delta (shares)",
            min_value=0.0,
            value=0.0,
            step=1.0,
            format="%.2f",
            help="Number of SPX-equivalent shares. Multiplied by the current SPX price to get dollar delta.",
        )
        spx_delta = spx_shares * spx_price
        st.info(
            f"**{spx_shares:,.2f}** shares x **${spx_price:,.2f}** (SPX) = "
            f"**${spx_delta:,.2f}** dollar delta"
        )
    elif exposure_method == "Use total notional from contracts above":
        spx_delta = total_notional
        st.info(f"Using total notional (unweighted): **${total_notional:,.2f}**")
    else:
        spx_delta = total_beta_weighted_delta
        st.info(f"Using beta-weighted delta: **${total_beta_weighted_delta:,.2f}**")

with kelly_input_cols[1]:
    st.subheader("Market Assumptions")
    market = fetch_market_assumptions()
    st.caption(
        f"Defaults from live data — T-bill: {market['risk_free_rate']:.2f}%, "
        f"VIX: {market['implied_vol']:.1f}%, "
        f"Realized vol: {market['realized_vol']:.1f}%, "
        f"Trailing S&P return: {market['trailing_return']:.1f}%"
    )
    expected_return = st.number_input(
        "Expected annual return (%)",
        min_value=0.0,
        max_value=100.0,
        value=market["expected_return"],
        step=0.5,
        format="%.1f",
        help=(
            f"Default: risk-free ({market['risk_free_rate']:.2f}%) + equity risk premium (5.5%) "
            f"= {market['expected_return']:.1f}%. "
            f"Trailing 1-year S&P 500 return: {market['trailing_return']:.1f}%."
        ),
    )
    risk_free_rate = st.number_input(
        "Risk-free rate (%)",
        min_value=0.0,
        max_value=50.0,
        value=market["risk_free_rate"],
        step=0.25,
        format="%.2f",
        help="Live default from 13-week T-bill yield (^IRX).",
    )
    vol_source = st.radio(
        "Volatility source",
        ["VIX (implied)", "Realized (trailing 1Y)", "Custom"],
        help=(
            f"VIX (implied): {market['implied_vol']:.1f}% — "
            f"Realized (trailing 1Y): {market['realized_vol']:.1f}%"
        ),
        horizontal=True,
    )
    if vol_source == "VIX (implied)":
        default_vol = market["implied_vol"]
    elif vol_source == "Realized (trailing 1Y)":
        default_vol = market["realized_vol"]
    else:
        default_vol = 16.0
    annual_volatility = st.number_input(
        "Expected annual volatility (%)",
        min_value=0.1,
        max_value=200.0,
        value=default_vol,
        step=0.5,
        format="%.1f",
        help="Annualized standard deviation. Override the default from the selected source above.",
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

