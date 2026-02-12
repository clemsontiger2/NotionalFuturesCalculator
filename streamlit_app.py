from datetime import date, timedelta
from io import BytesIO

import numpy as np
import pandas as pd
import plotly.graph_objects as go
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


@st.cache_data(ttl=300)
def fetch_rolling_betas(window=60):
    """Calculate rolling beta of each contract vs S&P 500.

    Uses *window* trading days of daily returns. Returns a dict
    mapping micro-futures symbol -> beta (float). Falls back to
    the static CONTRACT_SPECS beta if data is insufficient.
    """
    static = {sym: spec.get("spx_beta", 0.0) for sym, spec in CONTRACT_SPECS.items()}
    try:
        # Build ticker list: all contract tickers + SPX benchmark
        all_tickers = [info["ticker"] for info in YAHOO_TICKERS.values()]
        all_tickers.append("^GSPC")
        data = yf.download(
            " ".join(all_tickers),
            period="1y",
            progress=False,
        )
        close = data["Close"]
        if hasattr(close, "columns") is False:
            return static

        # SPX returns
        spx = close["^GSPC"].dropna()
        spx_ret = spx.pct_change().dropna()
        if len(spx_ret) < window:
            return static

        spx_window = spx_ret.iloc[-window:]
        spx_var = spx_window.var()
        if spx_var == 0:
            return static

        betas = {}
        for symbol, info in YAHOO_TICKERS.items():
            try:
                col = info["ticker"]
                series = close[col].dropna()
                if info.get("invert"):
                    series = 1.0 / series
                ret = series.pct_change().dropna()
                # Align with SPX window
                common = ret.index.intersection(spx_window.index)
                if len(common) < window * 0.5:
                    betas[symbol] = static.get(symbol, 0.0)
                    continue
                asset_ret = ret.loc[common]
                spx_aligned = spx_window.loc[common]
                cov = asset_ret.cov(spx_aligned)
                var = spx_aligned.var()
                betas[symbol] = round(float(cov / var), 2) if var > 0 else static.get(symbol, 0.0)
            except Exception:
                betas[symbol] = static.get(symbol, 0.0)
        return betas
    except Exception:
        return static


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
# spx_beta: static fallback beta relative to S&P 500 (overridden by rolling beta when enabled)

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
    ("Micro 10-Year Yield", "10Y", 1000, "\\$1,000 x yield index"),
    ("Micro 2-Year Yield", "2YY", 1000, "\\$1,000 x yield index"),
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
st.subheader("Settings")
col_controls_1, col_controls_2, col_controls_3 = st.columns([1, 1, 2])
with col_controls_1:
    default_qty = st.number_input(
        "Default number of contracts",
        min_value=0,
        max_value=1000,
        value=0,
        step=1,
        help="Sets the starting quantity for every contract across all asset classes. Change this to quickly populate all contracts at once.",
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
with col_controls_3:
    margin_mode = st.radio(
        "Margin calculation mode",
        ["Exchange estimates", "Formula-based (auto)", "Custom (per contract)"],
        horizontal=True,
        help=(
            "**Exchange estimates**: approximate maintenance margins from CME. "
            "**Formula-based**: Notional × daily vol × Z₉₉ × buffer — auto-updates with VIX. "
            "**Custom**: enter your own margin per contract."
        ),
    )

# Beta source toggle
beta_cols = st.columns([1, 1, 2])
with beta_cols[0]:
    beta_source = st.radio(
        "SPX Beta source",
        ["Static", "Rolling (live)"],
        horizontal=True,
        help=(
            "**Static**: hardcoded approximate betas from long-run estimates. "
            "**Rolling**: calculates beta from recent price history via Yahoo Finance."
        ),
    )
with beta_cols[1]:
    if beta_source == "Rolling (live)":
        beta_window = st.number_input(
            "Lookback (trading days)",
            min_value=20,
            max_value=252,
            value=60,
            step=10,
            help="Number of trading days for the rolling beta window. 60 ≈ 3 months.",
        )
    else:
        beta_window = 60

# Fetch or use static betas
if beta_source == "Rolling (live)":
    live_betas = fetch_rolling_betas(window=beta_window)
else:
    live_betas = {sym: spec.get("spx_beta", 0.0) for sym, spec in CONTRACT_SPECS.items()}

# Margin mode parameters
z_99 = 2.326
daily_vol_formula = 0.0
margin_buffer = 2.0
custom_margins = {}

if margin_mode == "Formula-based (auto)":
    market_early = fetch_market_assumptions()
    formula_cols = st.columns(3)
    with formula_cols[0]:
        margin_vol = st.number_input(
            "Annualized vol for margin (%)",
            min_value=1.0,
            max_value=200.0,
            value=market_early["implied_vol"],
            step=0.5,
            format="%.1f",
            help="Default from VIX. Used: Notional × (vol/√252) × 2.326 × buffer.",
        )
    with formula_cols[1]:
        margin_buffer = st.number_input(
            "Buffer factor",
            min_value=1.0,
            max_value=5.0,
            value=2.0,
            step=0.1,
            format="%.1f",
            help="Exchanges typically set margins at ~2× the statistical 99% daily VaR.",
        )
    daily_vol_formula = (margin_vol / 100) / np.sqrt(252)
    with formula_cols[2]:
        st.info(
            f"Margin ≈ Notional × {daily_vol_formula:.4%} × {z_99} × {margin_buffer:.1f}\n\n"
            f"Daily vol: {margin_vol:.1f}% / √252 = {daily_vol_formula * 100:.3f}%"
        )
elif margin_mode == "Custom (per contract)":
    with st.expander("Custom Margin per Contract", expanded=True):
        custom_cols = st.columns(4)
        for i, (name_c, symbol_c, _mult, _desc) in enumerate(MICRO_CONTRACTS):
            spec_c = CONTRACT_SPECS.get(symbol_c, {})
            with custom_cols[i % 4]:
                custom_margins[symbol_c] = st.number_input(
                    f"{symbol_c} margin ($)",
                    min_value=0.0,
                    value=float(spec_c.get("maint_margin", 0)),
                    step=50.0,
                    format="%.0f",
                    key=f"custom_margin_{symbol_c}",
                )

st.divider()

st.subheader("Contracts")

# Group contracts by asset class for tabbed display
_CLASS_ORDER = ["Equity", "Energy", "Metal", "Crypto", "FX", "Rates"]
_contracts_by_class = {}
for _n, _s, _m, _d in MICRO_CONTRACTS:
    _cls = CONTRACT_SPECS.get(_s, {}).get("asset_class", "Other")
    _contracts_by_class.setdefault(_cls, []).append((_n, _s, _m, _d))

total_notional = 0.0
total_margin = 0.0
total_beta_weighted_delta = 0.0
class_breakdown = {}  # {class: {"notional": ..., "beta_delta": ..., "margin": ..., "qty": ...}}
contract_rows = []  # per-contract data for grouped display

asset_tabs = st.tabs(_CLASS_ORDER)
for _tab, _tab_cls in zip(asset_tabs, _CLASS_ORDER):
    with _tab:
        # Table header inside each tab
        header_cols = st.columns([2, 1, 1.2, 1.2, 1, 1.5])
        header_cols[0].markdown("**Contract**")
        header_cols[1].markdown("**Symbol**")
        header_cols[2].markdown("**Multiplier**")
        header_cols[3].markdown("**Price**")
        header_cols[4].markdown("**Qty**")
        header_cols[5].markdown("**Notional Value**")
        st.divider()

        for name, symbol, multiplier, description in _contracts_by_class.get(_tab_cls, []):
            cols = st.columns([2, 1, 1.2, 1.2, 1, 1.5])

            spec = CONTRACT_SPECS.get(symbol, {})
            tick_inc = spec.get("tick_inc", 0)
            fetched_price = live_prices.get(symbol, FALLBACK_PRICES[symbol])
            if tick_inc > 0:
                fetched_price = _snap_to_tick(fetched_price, tick_inc)

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
            beta = live_betas.get(symbol, spec.get("spx_beta", 0.0))
            beta_delta = notional * beta

            # Margin calculation based on selected mode
            if margin_mode == "Exchange estimates":
                contract_margin = spec.get("maint_margin", 0) * qty
            elif margin_mode == "Formula-based (auto)":
                contract_margin = abs(notional) * daily_vol_formula * z_99 * margin_buffer if notional > 0 else 0.0
            else:  # Custom (per contract)
                contract_margin = custom_margins.get(symbol, spec.get("maint_margin", 0)) * qty
            total_notional += notional
            total_margin += contract_margin
            total_beta_weighted_delta += beta_delta

            cls = spec.get("asset_class", "Other")
            if cls not in class_breakdown:
                class_breakdown[cls] = {"notional": 0.0, "beta_delta": 0.0, "margin": 0.0, "qty": 0}
            class_breakdown[cls]["notional"] += notional
            class_breakdown[cls]["beta_delta"] += beta_delta
            class_breakdown[cls]["margin"] += contract_margin
            class_breakdown[cls]["qty"] += qty

            if qty > 0:
                contract_rows.append({
                    "asset_class": cls,
                    "Symbol": symbol,
                    "Contract": name,
                    "Price": price,
                    "Qty": qty,
                    "Notional": round(notional, 2),
                    "Beta": beta,
                    "Beta-Wtd Delta": round(beta_delta, 2),
                    "Margin": round(contract_margin, 2),
                })

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

# Asset class breakdown — grouped accordion
if contract_rows:
    st.subheader("Portfolio Composition")
    class_order = ["Equity", "Energy", "Metal", "Crypto", "FX", "Rates"]
    contracts_df = pd.DataFrame(contract_rows)
    for cls in class_order:
        if cls not in class_breakdown or class_breakdown[cls]["qty"] == 0:
            continue
        d = class_breakdown[cls]
        pct = f"{d['notional'] / total_notional * 100:.1f}%" if total_notional > 0 else ""
        with st.expander(
            f"{cls}  —  Notional: ${d['notional']:,.0f}  |  "
            f"Beta-Wtd: ${d['beta_delta']:,.0f}  |  {pct} of portfolio",
            expanded=True,
        ):
            class_df = contracts_df[contracts_df["asset_class"] == cls][
                ["Symbol", "Contract", "Price", "Qty", "Notional", "Beta", "Beta-Wtd Delta"]
            ]
            st.dataframe(
                class_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Price": st.column_config.NumberColumn(format="$%.2f"),
                    "Notional": st.column_config.NumberColumn(format="$%.2f"),
                    "Beta-Wtd Delta": st.column_config.NumberColumn(format="$%.2f"),
                },
            )
            st.caption(f"Maintenance margin for {cls}: ${d['margin']:,.0f}")

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
    # Effective margin per contract based on margin mode
    if margin_mode == "Exchange estimates":
        eff_margin = spec.get("maint_margin", 0)
    elif margin_mode == "Formula-based (auto)":
        spec_price = live_prices.get(symbol, FALLBACK_PRICES[symbol])
        one_notional = spec_price * multiplier
        eff_margin = round(abs(one_notional) * daily_vol_formula * z_99 * margin_buffer, 0) if one_notional > 0 else 0
    else:  # Custom
        eff_margin = custom_margins.get(symbol, spec.get("maint_margin", 0))
    specs_data.append({
        "Contract": name,
        "Symbol": symbol,
        "Class": spec.get("asset_class", "—"),
        "SPX Beta": f"{live_betas.get(symbol, spec.get('spx_beta', 0)):.2f}",
        "Tick Size": spec.get("tick_size", "—"),
        "Tick Value": f"${spec.get('tick_value', 0):.2f}",
        "Maint. Margin": f"${eff_margin:,.0f}",
        "Next Expiration": exp_str,
        "DTE": dte_str,
    })

specs_df = pd.DataFrame(specs_data)
st.dataframe(specs_df, use_container_width=True, hide_index=True)

if margin_mode == "Exchange estimates":
    st.caption(
        "Maintenance margins are approximate values from CME Group and are updated periodically. "
        "Actual margin requirements may differ based on your broker and account type. "
        "Expiration dates are calculated based on standard CME contract cycle rules."
    )
elif margin_mode == "Formula-based (auto)":
    st.caption(
        f"Margins calculated via formula: Notional × {daily_vol_formula:.4%} × {z_99} × {margin_buffer:.1f}. "
        "Formula margins approximate exchange requirements but may differ. Verify with your broker."
    )
else:
    st.caption(
        "Margins reflect your custom entries. "
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
            f"Total maintenance margin from contracts above: **\\${total_margin:,.0f}** "
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
            f"**{spx_shares:,.2f}** shares x **\\${spx_price:,.2f}** (SPX) = "
            f"**\\${spx_delta:,.2f}** dollar delta"
        )
    elif exposure_method == "Use total notional from contracts above":
        spx_delta = total_notional
        st.info(f"Using total notional (unweighted): **\\${total_notional:,.2f}**")
    else:
        spx_delta = total_beta_weighted_delta
        st.info(f"Using beta-weighted delta: **\\${total_beta_weighted_delta:,.2f}**")

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

    st.markdown("---")
    st.markdown("### Why Over-Kelly Destroys Long-Term Growth")
    st.markdown(r"""
**The key insight:** Compounding returns are *geometric*, not arithmetic.
Volatility creates a **drag** on compound growth that most people underestimate.

The expected geometric (compound) growth rate of a leveraged portfolio is:

$$g \approx L(\mu - r) + r - \tfrac{1}{2} L^2 \sigma^2$$

Where **L** is your leverage ratio. Notice two competing forces:

- **L(μ − r)**: Leverage *multiplies* your excess return — more leverage, more return
- **½ L² σ²**: Leverage *squares* your volatility drag — the penalty grows **quadratically**

At low leverage, the linear return benefit dominates. But as you increase leverage,
the quadratic drag catches up and eventually overwhelms the return. The Kelly
optimal is the exact point where these forces balance. Beyond it, **every
additional unit of leverage reduces your long-term compound growth.**

At 2× Kelly, the volatility drag exactly cancels the excess return — your
expected geometric growth drops to **zero**, equivalent to holding T-bills.
Above 2× Kelly, you're expected to **lose money** over time despite the
underlying asset having a positive return.
""")

    # Concrete leverage comparison table
    st.markdown("### Leverage Impact on Your Portfolio")
    leverage_levels = [0.5, 0.75, 1.0, half_kelly, kelly_optimal, kelly_optimal * 1.5, kelly_optimal * 2.0]
    leverage_labels = ["0.50x", "0.75x", "1.00x", f"{half_kelly:.2f}x (½K)", f"{kelly_optimal:.2f}x (1K)", f"{kelly_optimal * 1.5:.2f}x (1.5K)", f"{kelly_optimal * 2:.2f}x (2K)"]
    # Sort by leverage level so the table is always in ascending order
    sorted_pairs = sorted(zip(leverage_levels, leverage_labels), key=lambda x: x[0])
    leverage_levels, leverage_labels = zip(*sorted_pairs)
    lev_rows = []
    for lev, label in zip(leverage_levels, leverage_labels):
        g_arith = lev * mu_excess + (risk_free_rate / 100)
        vol_drag = 0.5 * (lev ** 2) * (sigma ** 2)
        g_geo = g_arith - vol_drag
        lev_rows.append({
            "Leverage": label,
            "Arithmetic Return": f"{g_arith * 100:.2f}%",
            "Volatility Drag": f"-{vol_drag * 100:.2f}%",
            "Geometric Growth (g)": f"{g_geo * 100:.2f}%",
        })
    lev_df = pd.DataFrame(lev_rows)
    st.dataframe(lev_df, use_container_width=True, hide_index=True)
    st.caption(
        "Arithmetic return grows linearly with leverage. Volatility drag grows "
        "quadratically (L²). At Full Kelly, geometric growth is maximized. "
        "At 2× Kelly, geometric growth is approximately zero."
    )

    st.markdown(r"""
### Why Half Kelly?

Full Kelly maximizes long-term growth but assumes **perfect knowledge** of
μ and σ. In reality:

- **Parameter uncertainty**: Small errors in estimated return or volatility
  can push you into the over-Kelly zone without realizing it
- **Extreme drawdowns**: Full Kelly portfolios regularly experience **50%+
  drawdowns** — psychologically and practically devastating
- **Half Kelly captures ~75% of the growth rate** with roughly **half the
  volatility and drawdown** — a much better risk/reward tradeoff
- Most institutional allocators and professional traders use Half Kelly or less

**Rule of thumb:** If you're unsure about your estimates, Half Kelly is
the maximum leverage you should consider.
""")

st.divider()

# ── Risk Analysis ──

# Calculate risk metrics first (needed for gauge)
true_leverage = total_beta_weighted_delta / nlv if nlv > 0 else 0.0
raw_leverage = total_notional / nlv if nlv > 0 else 0.0

# Value at Risk (95% 1-day)
z_score_95 = 1.645
daily_vol = sigma / np.sqrt(252) if sigma > 0 else 0.0
portfolio_var_95 = abs(total_beta_weighted_delta) * daily_vol * z_score_95

# Distance to margin call
excess_liquidity = nlv - total_margin
if total_notional > 0:
    dist_to_margin_call = excess_liquidity / total_notional
else:
    dist_to_margin_call = 1.0

# Determine status
if true_leverage <= half_kelly:
    badge_color = "#2d6a4f"
    status_label = "Conservative"
    status_msg = "Within safe sizing parameters."
elif true_leverage <= kelly_optimal:
    badge_color = "#e67e22"
    status_label = "Moderate"
    status_msg = "Between Half Kelly and Full Kelly. Elevated risk."
else:
    badge_color = "#e74c3c"
    status_label = "Aggressive"
    status_msg = "Exceeding Full Kelly. High risk of drawdown."

# Header with leverage badge
st.markdown(
    f"## Risk Analysis &nbsp; "
    f"<span style='background:{badge_color}; color:white; padding:4px 12px; "
    f"border-radius:12px; font-size:0.7em; vertical-align:middle;'>"
    f"{true_leverage:.2f}x Lev</span>",
    unsafe_allow_html=True,
)
st.write(
    "The gauge below shows your **beta-adjusted leverage** relative to the Kelly Criterion targets "
    "from your market assumptions. The green zone (0x to Half Kelly) is conservative, "
    "orange (Half Kelly to Full Kelly) is moderate, and red (beyond Full Kelly) indicates "
    "over-leverage that is expected to hurt long-term growth. The diamond marks your current position."
)

# Leverage gauge
if kelly_optimal > 0:
    max_gauge = max(kelly_optimal * 1.5, true_leverage * 1.2, 0.5)
    bar_h = 0.4

    fig_gauge = go.Figure()

    # Colored zones
    fig_gauge.add_shape(
        type="rect", x0=0, x1=half_kelly, y0=0, y1=bar_h,
        fillcolor="rgba(0,204,150,0.35)", line_width=0,
    )
    fig_gauge.add_shape(
        type="rect", x0=half_kelly, x1=kelly_optimal, y0=0, y1=bar_h,
        fillcolor="rgba(255,161,90,0.35)", line_width=0,
    )
    fig_gauge.add_shape(
        type="rect", x0=kelly_optimal, x1=max_gauge, y0=0, y1=bar_h,
        fillcolor="rgba(239,85,59,0.35)", line_width=0,
    )

    # Half Kelly dashed marker
    fig_gauge.add_shape(
        type="line", x0=half_kelly, x1=half_kelly, y0=-0.05, y1=bar_h + 0.05,
        line=dict(color="white", width=1.5, dash="dash"),
    )
    # Full Kelly solid marker
    fig_gauge.add_shape(
        type="line", x0=kelly_optimal, x1=kelly_optimal, y0=-0.05, y1=bar_h + 0.05,
        line=dict(color="#EF553B", width=2.5),
    )

    # Current leverage diamond
    m_color = "#00CC96" if true_leverage <= half_kelly else (
        "#FFA15A" if true_leverage <= kelly_optimal else "#EF553B"
    )
    clamped = min(true_leverage, max_gauge * 0.98)
    fig_gauge.add_trace(go.Scatter(
        x=[clamped], y=[bar_h / 2],
        mode="markers",
        marker=dict(size=16, color=m_color, symbol="diamond",
                    line=dict(color="white", width=1.5)),
        showlegend=False,
        hovertemplate=f"Current: {true_leverage:.2f}x<extra></extra>",
    ))

    # Labels
    fig_gauge.add_annotation(
        x=0, y=-0.18, text="0x", showarrow=False,
        font=dict(color="gray", size=11), xanchor="left",
    )
    safe_mid = half_kelly / 2 if half_kelly > 0 else 0.1
    fig_gauge.add_annotation(
        x=safe_mid, y=bar_h + 0.15, text="Safe Zone", showarrow=False,
        font=dict(color="gray", size=11),
    )
    fig_gauge.add_annotation(
        x=max_gauge, y=-0.18, text="Max Risk", showarrow=False,
        font=dict(color="gray", size=11), xanchor="right",
    )
    fig_gauge.add_annotation(
        x=half_kelly, y=-0.18,
        text=f"Target (½K): {half_kelly:.2f}x", showarrow=False,
        font=dict(color="#00CC96", size=11),
    )
    fig_gauge.add_annotation(
        x=kelly_optimal, y=-0.18,
        text=f"Limit (1K): {kelly_optimal:.2f}x", showarrow=False,
        font=dict(color="#EF553B", size=11),
    )

    fig_gauge.update_layout(
        height=130,
        margin=dict(l=10, r=10, t=5, b=35),
        xaxis=dict(visible=False, range=[-0.03 * max_gauge, max_gauge * 1.03]),
        yaxis=dict(visible=False, range=[-0.3, bar_h + 0.25]),
        template="plotly_dark",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )

    st.plotly_chart(fig_gauge, use_container_width=True, config={"displayModeBar": False})

# Status message
if true_leverage <= half_kelly:
    st.info(f"**{status_label}** — {status_msg}")
elif true_leverage <= kelly_optimal:
    st.warning(f"**{status_label}** — {status_msg}")
else:
    st.error(f"**{status_label}** — {status_msg}")

risk_cols = st.columns(3)
with risk_cols[0]:
    raw_delta_str = f"vs {raw_leverage:.2f}x raw" if abs(true_leverage - raw_leverage) > 0.01 else None
    st.metric(
        label="True Leverage (Beta-Adj)",
        value=f"{true_leverage:.2f}x",
        delta=raw_delta_str,
        delta_color="off",
        help="Leverage adjusted for the volatility of your assets relative to the S&P 500.",
    )

with risk_cols[1]:
    st.metric(
        label="1-Day VaR (95%)",
        value=f"${portfolio_var_95:,.0f}",
        delta=f"{portfolio_var_95 / nlv * 100:.2f}% of NLV" if nlv > 0 else None,
        delta_color="inverse",
        help="There is a 5% chance you will lose more than this amount in a single day.",
    )

with risk_cols[2]:
    st.metric(
        label="Margin Buffer",
        value=f"{dist_to_margin_call:.1%}" if total_notional > 0 else "No positions",
        delta=f"\\${excess_liquidity:,.0f} excess liquidity" if total_margin > 0 else None,
        delta_color="normal",
        help="How far your portfolio can drop before a margin call. Higher = safer.",
    )

# Warnings
if nlv > 0 and total_margin > 0:
    if dist_to_margin_call < 0:
        st.error(
            f"**Margin call!** Your maintenance margin (${total_margin:,.0f}) exceeds "
            f"your NLV (${nlv:,.2f}). You have negative excess liquidity."
        )
    elif dist_to_margin_call < 0.10:
        st.error(
            f"**Very close to margin call.** Only a **{dist_to_margin_call:.1%}** portfolio "
            f"decline would trigger a margin call. Excess liquidity: ${excess_liquidity:,.0f}."
        )
    elif dist_to_margin_call < 0.25:
        st.warning(
            f"**Elevated margin risk.** A **{dist_to_margin_call:.1%}** decline would trigger "
            f"a margin call. Consider reducing positions or adding capital."
        )

with st.expander("Risk Calculation Details"):
    st.markdown(f"""
| Metric | Value |
|---|---|
| Net Liquidation Value | ${nlv:,.2f} |
| Total Notional (raw) | ${total_notional:,.2f} |
| Beta-Weighted Notional | ${total_beta_weighted_delta:,.2f} |
| Raw Leverage | {raw_leverage:.2f}x |
| **True Leverage (Beta-Adj)** | **{true_leverage:.2f}x** |
| Annual Volatility (σ) | {annual_volatility:.1f}% |
| Daily Volatility (σ/√252) | {daily_vol * 100:.3f}% |
| Z-score (95%) | {z_score_95} |
| **1-Day VaR (95%)** | **${portfolio_var_95:,.2f}** |
| Total Maint. Margin | ${total_margin:,.0f} |
| Excess Liquidity | ${excess_liquidity:,.0f} |
| **Margin Buffer** | **{dist_to_margin_call:.2%}** |
""")

# ── Volatility-Based Trailing Stops ──
if total_notional > 0 and nlv > 0 and daily_vol > 0:
    st.subheader("Volatility-Based Trailing Stops")
    st.write(
        "Trailing stop levels derived from your volatility assumption. "
        "A common institutional approach uses 2x–3x the daily expected move "
        "to set stop distances that respect normal market noise."
    )

    daily_move_pct = daily_vol * 100  # daily vol as %
    tight_mult = 2.0
    loose_mult = 3.0
    tight_stop_pct = daily_move_pct * tight_mult
    loose_stop_pct = daily_move_pct * loose_mult

    # Portfolio-level dollar impact
    tight_stop_dollar = nlv * (tight_stop_pct / 100)
    loose_stop_dollar = nlv * (loose_stop_pct / 100)

    stop_cols = st.columns(3)
    with stop_cols[0]:
        st.metric(
            label="Daily Expected Move",
            value=f"{daily_move_pct:.2f}%",
            delta=f"from {annual_volatility:.1f}% annual vol",
            delta_color="off",
            help=f"Annual vol ({annual_volatility:.1f}%) / √252 = {daily_move_pct:.2f}% daily.",
        )
    with stop_cols[1]:
        st.metric(
            label="Tight Stop (2x Daily Vol)",
            value=f"{tight_stop_pct:.2f}%",
            delta=f"\\${tight_stop_dollar:,.0f} on \\${nlv:,.0f} NLV",
            delta_color="off",
            help="2x daily volatility — tighter stop, more frequent triggers, less drawdown per event.",
        )
    with stop_cols[2]:
        st.metric(
            label="Loose Stop (3x Daily Vol)",
            value=f"{loose_stop_pct:.2f}%",
            delta=f"\\${loose_stop_dollar:,.0f} on \\${nlv:,.0f} NLV",
            delta_color="off",
            help="3x daily volatility — wider stop, fewer false triggers, larger drawdown per event.",
        )

    # Per-contract stop price levels
    if contract_rows:
        with st.expander("Per-Contract Stop Levels"):
            stop_data = []
            for row in contract_rows:
                sym = row["Symbol"]
                p = row["Price"]
                tight_price = p * (1 - tight_stop_pct / 100)
                loose_price = p * (1 - loose_stop_pct / 100)
                spec = CONTRACT_SPECS.get(sym, {})
                tick_inc = spec.get("tick_inc", 0)
                if tick_inc > 0:
                    tight_price = _snap_to_tick(tight_price, tick_inc)
                    loose_price = _snap_to_tick(loose_price, tick_inc)
                stop_data.append({
                    "Symbol": sym,
                    "Price": p,
                    "Tight Stop": tight_price,
                    "Loose Stop": loose_price,
                })
            stop_df = pd.DataFrame(stop_data)
            st.dataframe(stop_df, use_container_width=True, hide_index=True)
            st.caption(
                f"Tight Stop = 2x daily vol ({tight_stop_pct:.2f}%). "
                f"Loose Stop = 3x daily vol ({loose_stop_pct:.2f}%). "
            )
            st.caption(
                "Stop prices are based on the portfolio-level volatility assumption. "
                "Individual contracts may have different realized volatility — "
                "consider adjusting per-asset if needed."
            )

# ── Portfolio Stress Test ──
if total_notional > 0 and nlv > 0:
    st.subheader("Portfolio Stress Test")

    moves = np.linspace(-0.30, 0.30, 61)
    pnl = total_beta_weighted_delta * moves
    equity_curve = nlv + pnl

    fig = go.Figure()

    # Projected equity line
    fig.add_trace(go.Scatter(
        x=moves * 100,
        y=equity_curve,
        mode="lines",
        name="Projected Equity",
        line=dict(color="#00CC96", width=3),
    ))

    # Margin call level
    fig.add_trace(go.Scatter(
        x=[moves[0] * 100, moves[-1] * 100],
        y=[total_margin, total_margin],
        mode="lines",
        name="Margin Call Level",
        line=dict(color="#EF553B", width=2, dash="dash"),
    ))

    # Current position marker
    fig.add_trace(go.Scatter(
        x=[0],
        y=[nlv],
        mode="markers+text",
        text=["Current"],
        textposition="top center",
        marker=dict(size=10, color="white"),
        name="Current Status",
    ))

    # 1-Day VaR marker
    if portfolio_var_95 > 0:
        var_move_pct = -portfolio_var_95 / total_beta_weighted_delta * 100 if total_beta_weighted_delta != 0 else 0
        fig.add_trace(go.Scatter(
            x=[var_move_pct],
            y=[nlv - portfolio_var_95],
            mode="markers+text",
            text=["95% VaR"],
            textposition="bottom center",
            marker=dict(size=8, color="#FFA15A", symbol="diamond"),
            name="1-Day VaR (95%)",
        ))

    fig.update_layout(
        title="Projected Account Value vs. Market Move (SPX-equivalent)",
        xaxis_title="Market Move (%)",
        yaxis_title="Account Equity ($)",
        yaxis_tickprefix="$",
        xaxis_ticksuffix="%",
        hovermode="x unified",
        template="plotly_dark",
        height=450,
    )

    # Liquidation zone shading
    if min(equity_curve) < total_margin:
        margin_cross_pct = -dist_to_margin_call * 100
        fig.add_vrect(
            x0=margin_cross_pct,
            x1=moves[0] * 100,
            fillcolor="red",
            opacity=0.15,
            layer="below",
            line_width=0,
            annotation_text="LIQUIDATION ZONE",
            annotation_position="top left",
        )

    st.plotly_chart(fig, use_container_width=True)

st.divider()

# ── What-If Scenario Simulator ──
if total_notional > 0 and nlv > 0:
    st.header("What-If Scenario Simulator")
    st.write(
        "Simulate a market shock and VIX spike to see if your account survives. "
        "Adjust the sliders to stress-test your portfolio in real time."
    )

    wi_cols = st.columns(2)
    with wi_cols[0]:
        wi_market_move = st.slider(
            "S&P 500 market move",
            min_value=-30.0,
            max_value=10.0,
            value=0.0,
            step=0.5,
            format="%.1f%%",
            help="Simulate an instantaneous market move. Negative = crash, positive = rally.",
        )
    with wi_cols[1]:
        wi_vix = st.slider(
            "VIX spikes to",
            min_value=10.0,
            max_value=90.0,
            value=annual_volatility,
            step=1.0,
            format="%.0f",
            help="Simulate a volatility spike. Current assumption shown as default.",
        )

    # Shocked P&L: beta-weighted delta * market move
    wi_move_dec = wi_market_move / 100.0
    wi_pnl = total_beta_weighted_delta * wi_move_dec
    wi_equity = nlv + wi_pnl

    # Stressed VaR with shocked VIX
    wi_daily_vol = (wi_vix / 100.0) / np.sqrt(252)
    wi_var_95 = abs(total_beta_weighted_delta) * wi_daily_vol * 1.645
    wi_var_99 = abs(total_beta_weighted_delta) * wi_daily_vol * 2.326

    # Margin under formula mode scales with vol; exchange/custom stays fixed
    if margin_mode == "Formula-based (auto)":
        wi_daily_formula = (wi_vix / 100.0) / np.sqrt(252)
        wi_margin = abs(total_notional) * wi_daily_formula * z_99 * margin_buffer
    else:
        wi_margin = total_margin

    wi_excess = wi_equity - wi_margin
    wi_surviving = wi_equity > wi_margin

    # Display results
    st.markdown("---")
    wi_result_cols = st.columns(4)
    with wi_result_cols[0]:
        pnl_color = "normal" if wi_pnl >= 0 else "inverse"
        st.metric(
            label="Scenario P&L",
            value=f"${wi_pnl:,.0f}",
            delta=f"{wi_market_move:+.1f}% move",
            delta_color=pnl_color,
        )
    with wi_result_cols[1]:
        st.metric(
            label="Surviving Equity",
            value=f"${wi_equity:,.0f}",
            delta=f"{wi_pnl / nlv * 100:+.1f}% of NLV" if nlv > 0 else None,
            delta_color="normal" if wi_pnl >= 0 else "inverse",
        )
    with wi_result_cols[2]:
        st.metric(
            label="Stressed 1-Day VaR (95%)",
            value=f"${wi_var_95:,.0f}",
            delta=f"{wi_var_95 / wi_equity * 100:.1f}% of equity" if wi_equity > 0 else "N/A",
            delta_color="inverse",
        )
    with wi_result_cols[3]:
        st.metric(
            label="Excess Liquidity After Shock",
            value=f"${wi_excess:,.0f}",
            delta="Surviving" if wi_surviving else "LIQUIDATED",
            delta_color="normal" if wi_surviving else "inverse",
        )

    # Survival verdict
    if not wi_surviving:
        st.error(
            f"**MARGIN CALL.** After a **{wi_market_move:+.1f}%** move, equity drops to "
            f"\\${wi_equity:,.0f} — below the maintenance margin of \\${wi_margin:,.0f}. "
            f"Your positions would be liquidated."
        )
    elif wi_excess < nlv * 0.10:
        st.warning(
            f"**Close call.** After a **{wi_market_move:+.1f}%** move, you'd have only "
            f"\\${wi_excess:,.0f} excess liquidity. Another "
            f"{wi_excess / total_notional:.1%} decline triggers liquidation."
        )
    else:
        st.success(
            f"**Survives.** After a **{wi_market_move:+.1f}%** move with VIX at {wi_vix:.0f}, "
            f"equity is \\${wi_equity:,.0f} with \\${wi_excess:,.0f} excess liquidity."
        )

    # Scenario detail table
    with st.expander("Scenario Details"):
        st.markdown(f"""
| Parameter | Current | After Shock |
|---|---|---|
| Market Move | — | {wi_market_move:+.1f}% |
| Volatility (annualized) | {annual_volatility:.1f}% | {wi_vix:.0f}% |
| Daily Vol | {daily_vol * 100:.3f}% | {wi_daily_vol * 100:.3f}% |
| Portfolio Equity | ${nlv:,.0f} | ${wi_equity:,.0f} |
| P&L | — | ${wi_pnl:,.0f} |
| 1-Day VaR (95%) | ${portfolio_var_95:,.0f} | ${wi_var_95:,.0f} |
| 1-Day VaR (99%) | — | ${wi_var_99:,.0f} |
| Maint. Margin | ${total_margin:,.0f} | ${wi_margin:,.0f} |
| Excess Liquidity | ${excess_liquidity:,.0f} | ${wi_excess:,.0f} |
| **Survival** | — | **{"YES" if wi_surviving else "NO — LIQUIDATED"}** |
""")

    # Quick scenario presets
    st.caption(
        "**Common scenarios:** "
        "Flash crash (-5%, VIX 35) · Correction (-10%, VIX 40) · "
        "Bear market (-20%, VIX 55) · Black swan (-30%, VIX 80)"
    )

st.divider()

# ── Monte Carlo Simulation ──
if total_notional > 0 and nlv > 0 and sigma > 0:
    st.header("Monte Carlo Simulation")
    st.write(
        "10,000 simulated portfolio paths over 5 years using your current leverage, "
        "return assumptions, and volatility. Shows the range of possible outcomes "
        "and the probability of ruin."
    )

    mc_cols = st.columns(3)
    with mc_cols[0]:
        mc_n_paths = st.number_input(
            "Paths", min_value=1000, max_value=50000, value=10000, step=1000,
            help="Number of random simulations to run.",
        )
    with mc_cols[1]:
        mc_years = st.number_input(
            "Horizon (years)", min_value=1, max_value=30, value=5, step=1,
            help="Simulation horizon in years.",
        )
    with mc_cols[2]:
        _ruin_default = max(1.0, round(total_margin / nlv * 100, 0)) if nlv > 0 and total_margin > 0 else 25.0
        mc_ruin_pct = st.number_input(
            "Ruin threshold (% of NLV)",
            min_value=1.0, max_value=100.0, value=_ruin_default,
            step=1.0, format="%.0f",
            help="Equity level considered 'ruin' (default = margin call level as % of NLV, minimum 5%).",
        )

    # Tail risk settings
    tail_cols = st.columns(3)
    with tail_cols[0]:
        mc_distribution = st.radio(
            "Return distribution",
            ["Normal", "Student-t (fat tails)"],
            horizontal=True,
            help=(
                "**Normal**: standard Gaussian returns. "
                "**Student-t**: heavier tails — extreme moves are more likely, "
                "matching real market behavior better."
            ),
        )
    with tail_cols[1]:
        if mc_distribution == "Student-t (fat tails)":
            mc_df = st.number_input(
                "Degrees of freedom (ν)",
                min_value=3, max_value=30, value=5, step=1,
                help=(
                    "Lower = fatter tails. ν=3–5 matches typical equity markets. "
                    "ν=5 is a common default. ν→∞ converges to Normal."
                ),
            )
        else:
            mc_df = 30  # effectively normal
    with tail_cols[2]:
        mc_jump_risk = st.checkbox(
            "Enable jump risk",
            value=False,
            help=(
                "Adds rare crash events (Poisson jumps). "
                "On average ~2 jumps per year, each causing "
                "a sudden -5% to -15% drop — simulating flash crashes, "
                "black swan events, and gap risk."
            ),
        )
        if mc_jump_risk:
            mc_jump_intensity = st.slider(
                "Avg jumps/year", min_value=0.5, max_value=5.0, value=2.0, step=0.5,
                help="Average number of crash-type jump events per year.",
            )
            mc_jump_mean = st.slider(
                "Avg jump size", min_value=-0.20, max_value=-0.01, value=-0.07, step=0.01,
                format="%.0f%%" if False else "%.2f",
                help="Mean jump magnitude (negative = crash). -0.07 = average 7% drop per jump.",
            )

    mc_n_days = int(mc_years * 252)
    mc_leverage = current_leverage
    mc_rf_daily = (risk_free_rate / 100) / 252
    mc_mu_daily = (expected_return / 100) / 252
    mc_sigma_daily = sigma / np.sqrt(252)
    mc_ruin_level = nlv * (mc_ruin_pct / 100)

    # Simulate
    rng = np.random.default_rng(seed=42)

    # Generate random shocks based on selected distribution
    if mc_distribution == "Student-t (fat tails)":
        # Student-t: scale so variance = 1 (same as Normal for fair comparison)
        raw_t = rng.standard_t(df=mc_df, size=(mc_n_days, mc_n_paths))
        # Variance of t(df) = df/(df-2), so scale down to unit variance
        t_scale = np.sqrt(mc_df / (mc_df - 2)) if mc_df > 2 else 1.0
        z = raw_t / t_scale
    else:
        z = rng.standard_normal((mc_n_days, mc_n_paths))

    # Underlying daily returns
    underlying_returns = mc_mu_daily + mc_sigma_daily * z

    # Add jump risk if enabled
    if mc_jump_risk:
        jump_lambda = mc_jump_intensity / 252  # daily probability
        jump_occurs = rng.poisson(jump_lambda, size=(mc_n_days, mc_n_paths))
        jump_sizes = rng.normal(mc_jump_mean, abs(mc_jump_mean) * 0.5, size=(mc_n_days, mc_n_paths))
        underlying_returns = underlying_returns + jump_occurs * jump_sizes

    # Leveraged portfolio daily returns: r_p = L * r_asset + (1 - L) * r_f
    port_returns = mc_leverage * underlying_returns + (1 - mc_leverage) * mc_rf_daily

    # Compound to equity curves
    growth_factors = 1 + port_returns
    cum_growth = np.cumprod(growth_factors, axis=0)
    equity_paths = nlv * cum_growth

    # Prepend initial NLV row
    equity_paths = np.vstack([np.full(mc_n_paths, nlv), equity_paths])

    # ── Metrics ──

    # 1. Geometric CAGR per path
    final_values = equity_paths[-1, :]
    cagrs = (final_values / nlv) ** (1 / mc_years) - 1
    median_cagr = np.median(cagrs)
    p5_cagr = np.percentile(cagrs, 5)
    p95_cagr = np.percentile(cagrs, 95)

    # 2. Maximum drawdown per path
    running_max = np.maximum.accumulate(equity_paths, axis=0)
    drawdowns = (equity_paths - running_max) / running_max
    max_dd_per_path = drawdowns.min(axis=0)
    median_max_dd = np.median(max_dd_per_path)
    p95_max_dd = np.percentile(max_dd_per_path, 5)  # worst 5% of drawdowns

    # 3. Ruin probability
    min_equity_per_path = equity_paths.min(axis=0)
    ruin_count = int(np.sum(min_equity_per_path < mc_ruin_level))
    ruin_prob = ruin_count / mc_n_paths

    # Display metrics
    mc_metric_cols = st.columns(4)
    with mc_metric_cols[0]:
        st.metric(
            label="Expected Geometric CAGR",
            value=f"{median_cagr:.1%}",
            delta=f"5th–95th: {p5_cagr:.1%} to {p95_cagr:.1%}",
            delta_color="off",
            help="Median annualized compound growth rate across all simulated paths.",
        )
    with mc_metric_cols[1]:
        st.metric(
            label="Expected Max Drawdown",
            value=f"{median_max_dd:.1%}",
            delta=f"Worst 5%: {p95_max_dd:.1%}",
            delta_color="off",
            help="Median worst peak-to-trough decline across all paths.",
        )
    with mc_metric_cols[2]:
        ruin_color = "off"
        if ruin_prob > 0.10:
            ruin_color = "inverse"
        st.metric(
            label=f"Ruin Probability ({mc_years}yr)",
            value=f"{ruin_prob:.1%}",
            delta=f"{ruin_count:,} of {mc_n_paths:,} paths",
            delta_color=ruin_color,
            help=f"Fraction of paths where equity drops below ${mc_ruin_level:,.0f} ({mc_ruin_pct:.0f}% of NLV).",
        )
    with mc_metric_cols[3]:
        median_terminal = np.median(final_values)
        st.metric(
            label=f"Median Terminal Value ({mc_years}yr)",
            value=f"${median_terminal:,.0f}",
            delta=f"{(median_terminal / nlv - 1):+.0%} total return",
            delta_color="normal" if median_terminal >= nlv else "inverse",
            help="Median ending portfolio value across all paths.",
        )

    # Ruin warning
    if ruin_prob > 0.25:
        st.error(
            f"**High ruin risk.** {ruin_prob:.0%} of simulated paths hit the ruin threshold "
            f"(\\${mc_ruin_level:,.0f}) within {mc_years} years. At {mc_leverage:.2f}x leverage, "
            f"the expected max drawdown is {median_max_dd:.0%}. Consider reducing leverage."
        )
    elif ruin_prob > 0.05:
        st.warning(
            f"**Moderate ruin risk.** {ruin_prob:.1%} of paths hit ruin within {mc_years} years. "
            f"Expected max drawdown: {median_max_dd:.0%}."
        )
    elif mc_leverage > 0:
        st.success(
            f"**Low ruin risk.** Only {ruin_prob:.1%} of paths hit ruin over {mc_years} years. "
            f"Expected CAGR: {median_cagr:.1%}, expected max drawdown: {median_max_dd:.0%}."
        )

    # ── Fan chart ──
    days = np.arange(equity_paths.shape[0])
    years_axis = days / 252

    p5 = np.percentile(equity_paths, 5, axis=1)
    p25 = np.percentile(equity_paths, 25, axis=1)
    p50 = np.percentile(equity_paths, 50, axis=1)
    p75 = np.percentile(equity_paths, 75, axis=1)
    p95 = np.percentile(equity_paths, 95, axis=1)

    fig_mc = go.Figure()

    # 5th–95th percentile band
    fig_mc.add_trace(go.Scatter(
        x=np.concatenate([years_axis, years_axis[::-1]]),
        y=np.concatenate([p95, p5[::-1]]),
        fill="toself",
        fillcolor="rgba(0,204,150,0.1)",
        line=dict(width=0),
        name="5th–95th percentile",
        hoverinfo="skip",
    ))

    # 25th–75th percentile band
    fig_mc.add_trace(go.Scatter(
        x=np.concatenate([years_axis, years_axis[::-1]]),
        y=np.concatenate([p75, p25[::-1]]),
        fill="toself",
        fillcolor="rgba(0,204,150,0.25)",
        line=dict(width=0),
        name="25th–75th percentile",
        hoverinfo="skip",
    ))

    # Median path
    fig_mc.add_trace(go.Scatter(
        x=years_axis, y=p50,
        mode="lines",
        name="Median path",
        line=dict(color="#00CC96", width=2.5),
    ))

    # Ruin level
    fig_mc.add_trace(go.Scatter(
        x=[0, years_axis[-1]],
        y=[mc_ruin_level, mc_ruin_level],
        mode="lines",
        name="Ruin threshold",
        line=dict(color="#EF553B", width=2, dash="dash"),
    ))

    # Starting NLV
    fig_mc.add_trace(go.Scatter(
        x=[0, years_axis[-1]],
        y=[nlv, nlv],
        mode="lines",
        name="Starting NLV",
        line=dict(color="gray", width=1, dash="dot"),
    ))

    fig_mc.update_layout(
        title=f"Simulated Portfolio Equity ({mc_n_paths:,} paths, {mc_leverage:.2f}x leverage)",
        xaxis_title="Years",
        yaxis_title="Portfolio Value ($)",
        yaxis_tickprefix="$",
        hovermode="x unified",
        template="plotly_dark",
        height=450,
    )

    st.plotly_chart(fig_mc, use_container_width=True)

    with st.expander("Simulation Parameters"):
        _dist_label = mc_distribution
        if mc_distribution == "Student-t (fat tails)":
            _dist_label = f"Student-t (ν={mc_df})"
        _jump_label = "Off"
        if mc_jump_risk:
            _jump_label = f"On — {mc_jump_intensity:.1f} jumps/yr, avg size {mc_jump_mean:.0%}"
        st.markdown(f"""
| Parameter | Value |
|---|---|
| Paths simulated | {mc_n_paths:,} |
| Horizon | {mc_years} years ({mc_n_days:,} trading days) |
| Leverage (current) | {mc_leverage:.2f}x |
| Distribution | {_dist_label} |
| Jump risk | {_jump_label} |
| Expected return (μ) | {expected_return:.1f}% annual → {mc_mu_daily * 100:.4f}% daily |
| Volatility (σ) | {annual_volatility:.1f}% annual → {mc_sigma_daily * 100:.3f}% daily |
| Risk-free rate | {risk_free_rate:.2f}% annual → {mc_rf_daily * 100:.5f}% daily |
| Ruin threshold | ${mc_ruin_level:,.0f} ({mc_ruin_pct:.0f}% of NLV) |
| **Median CAGR** | **{median_cagr:.2%}** |
| **Median max drawdown** | **{median_max_dd:.1%}** |
| **Ruin probability** | **{ruin_prob:.2%}** ({ruin_count:,} paths) |
| Median terminal value | ${median_terminal:,.0f} |
| 5th pctl terminal | ${np.percentile(final_values, 5):,.0f} |
| 95th pctl terminal | ${np.percentile(final_values, 95):,.0f} |
""")

st.divider()

# ── Portfolio Insights ──
st.header("Portfolio Insights")

if not contract_rows:
    st.info("Add positions above to see portfolio insights.")
else:
    insights = []  # list of (icon, title, message, severity)

    # ── Leverage assessment ──
    if nlv > 0 and current_leverage > 0:
        if current_leverage > kelly_optimal:
            insights.append((
                "leverage",
                "Leverage exceeds Full Kelly",
                f"Current leverage ({current_leverage:.2f}x) is above the Full Kelly "
                f"optimal ({kelly_optimal:.2f}x). This is expected to **reduce** long-term "
                f"geometric growth and significantly increases drawdown risk. "
                f"Consider reducing exposure by "
                f"${(current_leverage - kelly_optimal) * nlv:,.0f} notional.",
                "error",
            ))
        elif current_leverage > half_kelly:
            insights.append((
                "leverage",
                "Leverage above Half Kelly",
                f"Current leverage ({current_leverage:.2f}x) is between Half Kelly "
                f"({half_kelly:.2f}x) and Full Kelly ({kelly_optimal:.2f}x). "
                f"You're capturing most of the Kelly growth rate but with higher "
                f"volatility. Most practitioners cap at Half Kelly.",
                "warning",
            ))
        else:
            insights.append((
                "leverage",
                "Leverage within safe range",
                f"Current leverage ({current_leverage:.2f}x) is at or below Half Kelly "
                f"({half_kelly:.2f}x). This is a conservative position that balances "
                f"growth with drawdown protection.",
                "success",
            ))

    # ── Margin utilization ──
    if nlv > 0 and total_margin > 0:
        margin_util = total_margin / nlv
        if margin_util > 0.80:
            insights.append((
                "margin",
                "Very high margin utilization",
                f"Margin uses **{margin_util:.0%}** of NLV (\\${total_margin:,.0f} / "
                f"\\${nlv:,.0f}). A small adverse move could trigger a margin call. "
                f"Consider reducing positions or adding capital.",
                "error",
            ))
        elif margin_util > 0.50:
            insights.append((
                "margin",
                "Elevated margin utilization",
                f"Margin uses **{margin_util:.0%}** of NLV. You have "
                f"\\${excess_liquidity:,.0f} excess liquidity — a "
                f"{dist_to_margin_call:.1%} decline triggers a margin call.",
                "warning",
            ))
        else:
            insights.append((
                "margin",
                "Margin utilization healthy",
                f"Margin uses **{margin_util:.0%}** of NLV with "
                f"\\${excess_liquidity:,.0f} excess liquidity.",
                "success",
            ))

    # ── Concentration risk ──
    active_classes = {cls: d for cls, d in class_breakdown.items() if d["qty"] > 0}
    if total_notional > 0 and active_classes:
        max_cls = max(active_classes, key=lambda c: abs(active_classes[c]["notional"]))
        max_pct = abs(active_classes[max_cls]["notional"]) / total_notional
        if len(active_classes) == 1:
            insights.append((
                "diversification",
                "Single asset class",
                f"All exposure is in **{max_cls}**. Consider diversifying across "
                f"asset classes to reduce correlation risk.",
                "warning",
            ))
        elif max_pct > 0.70:
            insights.append((
                "diversification",
                f"Concentrated in {max_cls}",
                f"**{max_cls}** represents {max_pct:.0%} of total notional. "
                f"High concentration increases risk if that sector moves against you.",
                "warning",
            ))
        elif len(active_classes) >= 3:
            insights.append((
                "diversification",
                "Good diversification",
                f"Exposure across **{len(active_classes)} asset classes** "
                f"({', '.join(active_classes.keys())}). Largest: {max_cls} at {max_pct:.0%}.",
                "success",
            ))

    # ── Correlated positions ──
    equity_syms = [r["Symbol"] for r in contract_rows if r["asset_class"] == "Equity"]
    if len(equity_syms) >= 3:
        insights.append((
            "correlation",
            "Multiple correlated equity indices",
            f"You hold {len(equity_syms)} equity index futures ({', '.join(equity_syms)}). "
            f"These are highly correlated — in a broad selloff they will all move "
            f"together, amplifying losses.",
            "warning",
        ))

    crypto_syms = [r["Symbol"] for r in contract_rows if r["asset_class"] == "Crypto"]
    if len(crypto_syms) >= 2:
        insights.append((
            "correlation",
            "Multiple crypto positions",
            f"You hold {len(crypto_syms)} crypto futures ({', '.join(crypto_syms)}). "
            f"BTC and ETH are highly correlated and have ~2x+ SPX beta — "
            f"consider this amplified risk in your sizing.",
            "warning",
        ))

    # ── VaR assessment ──
    if nlv > 0 and portfolio_var_95 > 0:
        var_pct = portfolio_var_95 / nlv
        if var_pct > 0.05:
            insights.append((
                "var",
                "High daily Value at Risk",
                f"Your 1-day 95% VaR is **{var_pct:.1%}** of NLV (\\${portfolio_var_95:,.0f}). "
                f"A 5%+ daily VaR means you could lose more than this on 1-in-20 trading "
                f"days. In a month, expect at least one such day.",
                "error",
            ))
        elif var_pct > 0.02:
            insights.append((
                "var",
                "Moderate daily Value at Risk",
                f"Your 1-day 95% VaR is **{var_pct:.1%}** of NLV (\\${portfolio_var_95:,.0f}).",
                "warning",
            ))
        else:
            insights.append((
                "var",
                "Low daily Value at Risk",
                f"Your 1-day 95% VaR is **{var_pct:.1%}** of NLV (\\${portfolio_var_95:,.0f}). "
                f"Daily risk is well-contained.",
                "success",
            ))

    # ── Single position dominance ──
    if len(contract_rows) > 1 and total_notional > 0:
        biggest = max(contract_rows, key=lambda r: abs(r["Notional"]))
        big_pct = abs(biggest["Notional"]) / total_notional
        if big_pct > 0.60:
            insights.append((
                "sizing",
                f"{biggest['Symbol']} dominates portfolio",
                f"**{biggest['Symbol']}** is {big_pct:.0%} of total notional "
                f"(\\${biggest['Notional']:,.0f}). A single position this large means "
                f"the portfolio's risk profile is essentially that one contract.",
                "warning",
            ))

    # ── Display insights ──
    for _cat, title, msg, severity in insights:
        if severity == "error":
            st.error(f"**{title}** — {msg}")
        elif severity == "warning":
            st.warning(f"**{title}** — {msg}")
        else:
            st.success(f"**{title}** — {msg}")

    # ── Copyable summary for external AI ──
    with st.expander("Copy Portfolio Summary for AI Chat"):
        lines = []
        lines.append("=== PORTFOLIO POSITIONS ===")
        for row in contract_rows:
            lines.append(
                f"  {row['Symbol']} ({row['Contract']}): {row['Qty']} contracts, "
                f"Notional ${row['Notional']:,.2f}, Beta {row['Beta']:.2f}, "
                f"Beta-Wtd Delta ${row['Beta-Wtd Delta']:,.2f}, Margin ${row['Margin']:,.2f}"
            )
        lines.append("")
        lines.append("=== PORTFOLIO SUMMARY ===")
        lines.append(f"Net Liquidation Value: ${nlv:,.2f}")
        lines.append(f"Total Notional Exposure: ${total_notional:,.2f}")
        lines.append(f"Beta-Weighted Delta (SPX): ${total_beta_weighted_delta:,.2f}")
        lines.append(f"Total Maintenance Margin: ${total_margin:,.0f}")
        lines.append(f"Beta Source: {beta_source}" + (f" ({beta_window}-day window)" if beta_source == "Rolling (live)" else ""))
        lines.append(f"Margin Mode: {margin_mode}")
        lines.append("")
        lines.append("=== ASSET CLASS BREAKDOWN ===")
        for cls in ["Equity", "Energy", "Metal", "Crypto", "FX", "Rates"]:
            if cls in class_breakdown and class_breakdown[cls]["qty"] > 0:
                d = class_breakdown[cls]
                pct = d["notional"] / total_notional * 100 if total_notional > 0 else 0
                lines.append(
                    f"  {cls}: Notional ${d['notional']:,.0f} ({pct:.1f}%), "
                    f"Beta-Wtd ${d['beta_delta']:,.0f}, Margin ${d['margin']:,.0f}"
                )
        lines.append("")
        lines.append("=== KELLY CRITERION ===")
        lines.append(f"Full Kelly: {kelly_optimal:.2f}x | Half Kelly: {half_kelly:.2f}x")
        lines.append(f"Current Leverage: {current_leverage:.2f}x | Status: {status_label}")
        lines.append(f"Expected Return: {expected_return:.1f}% | Vol: {annual_volatility:.1f}% | Rf: {risk_free_rate:.2f}%")
        lines.append("")
        lines.append("=== RISK METRICS ===")
        lines.append(f"True Leverage (Beta-Adj): {true_leverage:.2f}x | Raw: {raw_leverage:.2f}x")
        lines.append(f"1-Day VaR (95%): ${portfolio_var_95:,.2f}")
        lines.append(f"Margin Buffer: {dist_to_margin_call:.1%}")
        lines.append(f"Excess Liquidity: ${excess_liquidity:,.0f}")
        lines.append("")
        lines.append("=== VOLATILITY-BASED TRAILING STOPS ===")
        if daily_vol > 0:
            _dm = daily_vol * 100
            lines.append(f"Daily Expected Move: {_dm:.2f}% (from {annual_volatility:.1f}% annual)")
            lines.append(f"Tight Stop (2x): {_dm * 2:.2f}% = ${nlv * _dm * 2 / 100:,.0f}")
            lines.append(f"Loose Stop (3x): {_dm * 3:.2f}% = ${nlv * _dm * 3 / 100:,.0f}")
        else:
            lines.append("N/A — no volatility data")
        lines.append("")
        lines.append("=== MONTE CARLO SIMULATION ===")
        try:
            lines.append(f"Paths: {mc_n_paths:,} | Horizon: {mc_years} years | Leverage: {mc_leverage:.2f}x")
            lines.append(f"Distribution: {mc_distribution}" + (f" (df={mc_df})" if mc_distribution == "Student-t (fat tails)" else ""))
            lines.append(f"Jump Risk: {'On — ' + f'{mc_jump_intensity:.1f} jumps/yr, avg {mc_jump_mean:.0%}' if mc_jump_risk else 'Off'}")
            lines.append(f"Expected Geometric CAGR: {median_cagr:.2%} (5th pctl: {p5_cagr:.2%}, 95th: {p95_cagr:.2%})")
            lines.append(f"Expected Max Drawdown: {median_max_dd:.1%} (worst 5%: {p95_max_dd:.1%})")
            lines.append(f"Ruin Probability ({mc_years}yr): {ruin_prob:.2%} ({ruin_count:,} of {mc_n_paths:,} paths)")
            lines.append(f"Ruin Threshold: ${mc_ruin_level:,.0f} ({mc_ruin_pct:.0f}% of NLV)")
            lines.append(f"Median Terminal Value: ${median_terminal:,.0f}")
            lines.append(f"5th Percentile Terminal: ${np.percentile(final_values, 5):,.0f}")
            lines.append(f"95th Percentile Terminal: ${np.percentile(final_values, 95):,.0f}")
        except NameError:
            lines.append("N/A — simulation requires positions, NLV, and volatility")
        summary_text = "\n".join(lines)
        st.code(summary_text, language="text")
        st.caption("Copy the summary above and paste into ChatGPT, Claude, or any AI assistant for deeper analysis.")

st.divider()

# ── Export to Excel ──
st.header("Export to Excel")
st.write("Download your full portfolio analysis as a multi-sheet Excel workbook.")

if st.button("Generate Excel Report", type="primary"):
    buf = BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        # Sheet 1: Positions
        if contract_rows:
            pos_df = pd.DataFrame(contract_rows)
            pos_df = pos_df[["Symbol", "Contract", "asset_class", "Price", "Qty",
                             "Notional", "Beta", "Beta-Wtd Delta", "Margin"]]
            pos_df.rename(columns={"asset_class": "Asset Class"}, inplace=True)
            pos_df.to_excel(writer, sheet_name="Positions", index=False)

        # Sheet 2: Portfolio Summary
        summary_data = {
            "Metric": [
                "Net Liquidation Value",
                "Total Notional Exposure",
                "Beta-Weighted Delta (SPX)",
                "Total Maintenance Margin",
                "Margin Mode",
                "Beta Source",
                "Active Contracts",
                "Total Contracts",
            ],
            "Value": [
                nlv,
                total_notional,
                total_beta_weighted_delta,
                total_margin,
                margin_mode,
                beta_source + (f" ({beta_window}d)" if beta_source == "Rolling (live)" else ""),
                sum(1 for r in contract_rows),
                sum(r["Qty"] for r in contract_rows) if contract_rows else 0,
            ],
        }
        pd.DataFrame(summary_data).to_excel(writer, sheet_name="Summary", index=False)

        # Sheet 3: Asset Class Breakdown
        class_rows = []
        for cls in ["Equity", "Energy", "Metal", "Crypto", "FX", "Rates"]:
            if cls in class_breakdown and class_breakdown[cls]["qty"] > 0:
                d = class_breakdown[cls]
                pct = d["notional"] / total_notional * 100 if total_notional > 0 else 0
                class_rows.append({
                    "Asset Class": cls,
                    "Notional": round(d["notional"], 2),
                    "% of Portfolio": round(pct, 1),
                    "Beta-Wtd Delta": round(d["beta_delta"], 2),
                    "Margin": round(d["margin"], 2),
                    "Contracts": d["qty"],
                })
        if class_rows:
            pd.DataFrame(class_rows).to_excel(writer, sheet_name="Asset Classes", index=False)

        # Sheet 4: Kelly Analysis
        kelly_data = {
            "Parameter": [
                "Expected Return (μ)", "Risk-Free Rate (r)", "Excess Return (μ-r)",
                "Volatility (σ)", "Variance (σ²)",
                "Full Kelly Leverage", "Half Kelly Leverage", "Current Leverage",
                "Kelly Status", "Kelly-Optimal Notional", "Half-Kelly Notional",
            ],
            "Value": [
                f"{expected_return:.1f}%", f"{risk_free_rate:.2f}%",
                f"{expected_return - risk_free_rate:.2f}%",
                f"{annual_volatility:.1f}%", f"{annual_volatility**2 / 100:.2f}%",
                f"{kelly_optimal:.2f}x", f"{half_kelly:.2f}x", f"{current_leverage:.2f}x",
                status_label, f"${nlv * kelly_optimal:,.2f}", f"${nlv * half_kelly:,.2f}",
            ],
        }
        pd.DataFrame(kelly_data).to_excel(writer, sheet_name="Kelly Analysis", index=False)

        # Sheet 5: Risk Metrics
        risk_data = {
            "Metric": [
                "True Leverage (Beta-Adj)", "Raw Leverage",
                "1-Day VaR (95%)", "VaR % of NLV",
                "Margin Buffer", "Excess Liquidity",
                "Total Maint. Margin",
            ],
            "Value": [
                f"{true_leverage:.2f}x", f"{raw_leverage:.2f}x",
                f"${portfolio_var_95:,.2f}",
                f"{portfolio_var_95 / nlv * 100:.2f}%" if nlv > 0 else "N/A",
                f"{dist_to_margin_call:.2%}",
                f"${excess_liquidity:,.0f}", f"${total_margin:,.0f}",
            ],
        }
        pd.DataFrame(risk_data).to_excel(writer, sheet_name="Risk Metrics", index=False)

        # Sheet 6: Trailing Stops
        if daily_vol > 0 and contract_rows:
            _dm = daily_vol * 100
            stop_export = []
            for row in contract_rows:
                sym = row["Symbol"]
                p = row["Price"]
                spec = CONTRACT_SPECS.get(sym, {})
                tick_inc = spec.get("tick_inc", 0)
                tp = p * (1 - _dm * 2 / 100)
                lp = p * (1 - _dm * 3 / 100)
                if tick_inc > 0:
                    tp = _snap_to_tick(tp, tick_inc)
                    lp = _snap_to_tick(lp, tick_inc)
                stop_export.append({
                    "Symbol": sym,
                    "Current Price": p,
                    f"Tight Stop (2x={_dm*2:.2f}%)": round(tp, 6),
                    f"Loose Stop (3x={_dm*3:.2f}%)": round(lp, 6),
                })
            pd.DataFrame(stop_export).to_excel(writer, sheet_name="Trailing Stops", index=False)

        # Sheet 7: Contract Specifications
        specs_export = []
        for name, symbol, multiplier, _desc in MICRO_CONTRACTS:
            spec = CONTRACT_SPECS.get(symbol, {})
            exp = next_expiration(symbol, date.today())
            specs_export.append({
                "Contract": name,
                "Symbol": symbol,
                "Multiplier": multiplier,
                "Asset Class": spec.get("asset_class", ""),
                "SPX Beta": live_betas.get(symbol, spec.get("spx_beta", 0)),
                "Tick Size": spec.get("tick_size", ""),
                "Tick Value": spec.get("tick_value", 0),
                "Maint. Margin": spec.get("maint_margin", 0),
                "Next Expiration": exp.strftime("%Y-%m-%d") if exp else "",
                "DTE": (exp - date.today()).days if exp else "",
            })
        pd.DataFrame(specs_export).to_excel(writer, sheet_name="Contract Specs", index=False)

        # Sheet 8: Monte Carlo Results (if available)
        try:
            mc_data = {
                "Parameter": [
                    "Paths", "Horizon", "Leverage", "Distribution",
                    "Jump Risk", "Ruin Threshold",
                    "Median CAGR", "5th Pctl CAGR", "95th Pctl CAGR",
                    "Median Max Drawdown", "Worst 5% Max Drawdown",
                    "Ruin Probability", "Ruin Paths",
                    "Median Terminal Value", "5th Pctl Terminal", "95th Pctl Terminal",
                ],
                "Value": [
                    mc_n_paths, f"{mc_years} years", f"{mc_leverage:.2f}x",
                    mc_distribution + (f" (df={mc_df})" if mc_distribution == "Student-t (fat tails)" else ""),
                    f"On — {mc_jump_intensity:.1f}/yr, avg {mc_jump_mean:.0%}" if mc_jump_risk else "Off",
                    f"${mc_ruin_level:,.0f} ({mc_ruin_pct:.0f}%)",
                    f"{median_cagr:.2%}", f"{p5_cagr:.2%}", f"{p95_cagr:.2%}",
                    f"{median_max_dd:.1%}", f"{p95_max_dd:.1%}",
                    f"{ruin_prob:.2%}", ruin_count,
                    f"${median_terminal:,.0f}",
                    f"${np.percentile(final_values, 5):,.0f}",
                    f"${np.percentile(final_values, 95):,.0f}",
                ],
            }
            pd.DataFrame(mc_data).to_excel(writer, sheet_name="Monte Carlo", index=False)
        except NameError:
            pass

    buf.seek(0)
    st.download_button(
        label="Download Excel Report",
        data=buf,
        file_name=f"portfolio_report_{date.today().strftime('%Y%m%d')}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
    st.success("Report generated! Click the download button above.")

st.divider()

# Reference information
with st.expander("About Micro Futures Contracts"):
    st.markdown("""
**What are Micro Futures?**

Micro futures are smaller-sized versions of standard futures contracts, typically
1/10th the size of their full-sized counterparts. They were introduced by CME Group
to make futures trading more accessible to individual traders.

**Why Do People Trade Futures?**

- **Capital efficiency** — Futures require only a small margin deposit (often 3–12%
  of notional value) to control a large position. A trader with \$10,000 can get
  exposure equivalent to \$100,000+ in stocks, freeing up capital for other uses.
- **Tax advantages (U.S.)** — Futures are taxed under the 60/40 rule (Section 1256):
  60% of gains are taxed as long-term capital gains and 40% as short-term,
  regardless of holding period. This can result in a lower blended tax rate
  compared to stocks held under one year.
- **Nearly 24-hour access** — Futures trade almost around the clock (Sunday evening
  through Friday afternoon), allowing traders to react to overnight news and global
  events without waiting for the stock market to open.
- **No pattern day trader rule** — Unlike stocks, futures have no PDT rule requiring
  a \$25,000 minimum to day trade. You can actively trade with a smaller account.
- **True portfolio hedging** — Futures let you hedge market exposure precisely. A
  long stock portfolio can be hedged by shorting index futures rather than selling
  positions and triggering taxable events.
- **Short selling without borrowing** — Going short a futures contract is as simple
  as going long. There are no borrow fees, hard-to-borrow lists, or uptick rules.
- **Diversification across asset classes** — From a single account, you can trade
  equities, commodities, currencies, interest rates, and crypto — all with the same
  margin framework.
- **Transparent, centralized pricing** — Futures trade on regulated exchanges (CME)
  with centralized order books, no payment for order flow concerns, and tight
  bid-ask spreads on liquid contracts.

**How is Notional Value Calculated?**

> **Notional Value = Current Price x Contract Multiplier x Number of Contracts**

For example, if the Micro E-mini S&P 500 (MES) is trading at 5,950 with a multiplier
of \$5, one contract has a notional value of:

> 5,950 x \$5 = **\$29,750**

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
    if 0.7 <= half_kelly <= 1.3:
        _kelly_narrative = "roughly at the Half Kelly level — a reasonable, moderate position."
    elif 1.0 < half_kelly:
        _kelly_narrative = f"at {1.0/kelly_optimal:.0%} of Full Kelly — a conservative position."
    else:
        _kelly_narrative = f"at {1.0/kelly_optimal:.0%} of Full Kelly — above Half Kelly, so consider reducing leverage."
    st.markdown(f"""
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

**Example with your current inputs:**

Using your parameters ({expected_return:.1f}% return, {risk_free_rate:.1f}% risk-free, {annual_volatility:.1f}% vol):

> f* = ({expected_return:.1f}% - {risk_free_rate:.1f}%) / ({annual_volatility:.1f}%)² = {mu_excess*100:.1f}% / {sigma**2*100:.2f}% = **{kelly_optimal:.2f}x**
>
> Half Kelly = **{half_kelly:.2f}x**

This means a portfolio fully invested in an S&P 500 index at 1x leverage is
{_kelly_narrative}

**SPX Delta / Beta-Weighted Delta**

If your portfolio contains multiple assets (futures, options, stocks), you can
convert the total exposure to an SPX-equivalent dollar delta. This beta-weighted
delta represents how much your portfolio would move for a \$1 move in the S&P 500,
giving a single measure of equity-like exposure to compare against Kelly.
""")

