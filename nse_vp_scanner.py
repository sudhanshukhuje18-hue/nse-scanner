"""
NSE Volume Profile Scanner (with click-to-chart)
------------------------------------------------
Finds NSE-listed stocks trading ABOVE their N-day volume profile level (POC),
but not more than X% above it (band is editable). Click a row to see its chart.

Install:  pip install streamlit yfinance pandas numpy requests plotly
Run:      streamlit run nse_vp_scanner.py
"""

import io
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import streamlit.components.v1 as components
import yfinance as yf
from plotly.subplots import make_subplots

st.set_page_config(page_title="NSE Volume Profile Scanner", layout="wide")

NSE_LIST_URL = "https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv"
HEADERS = {"User-Agent": "Mozilla/5.0"}


# ---------------------------------------------------------------- data
@st.cache_data(ttl=24 * 3600, show_spinner=False)
def get_nse_symbols() -> list[str]:
    r = requests.get(NSE_LIST_URL, headers=HEADERS, timeout=30)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text))
    df.columns = [c.strip() for c in df.columns]
    df = df[df["SERIES"].str.strip() == "EQ"]
    return sorted(df["SYMBOL"].str.strip().unique().tolist())


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def download_prices(symbols: tuple, chunk: int = 100) -> dict:
    """Daily OHLCV for every symbol (~9 months so 100+ trading days exist)."""
    out = {}
    prog = st.progress(0.0, text="Downloading price data...")
    for i in range(0, len(symbols), chunk):
        batch = list(symbols[i:i + chunk])
        tickers = [s + ".NS" for s in batch]
        try:
            data = yf.download(
                tickers, period="9mo", interval="1d", group_by="ticker",
                auto_adjust=False, threads=True, progress=False,
            )
        except Exception:
            continue
        for s, t in zip(batch, tickers):
            try:
                d = data[t] if len(tickers) > 1 else data
                d = d[["Open", "High", "Low", "Close", "Volume"]].dropna()
                if len(d):
                    out[s] = d
            except Exception:
                pass
        prog.progress(min((i + chunk) / len(symbols), 1.0),
                      text=f"Downloaded {min(i + chunk, len(symbols))}/{len(symbols)}")
    prog.empty()
    return out


# ------------------------------------------------------ volume profile
def volume_profile(df: pd.DataFrame, lookback: int, bins: int, va_pct: float):
    """Each day's volume is spread evenly across the price bins its
    high-low range touches. Returns dict with profile + POC/VAH/VAL."""
    d = df.tail(lookback)
    if len(d) < lookback * 0.8:          # not enough history (new listing)
        return None
    lo, hi = d["Low"].min(), d["High"].max()
    if hi <= lo:
        return None
    edges = np.linspace(lo, hi, bins + 1)
    mids = (edges[:-1] + edges[1:]) / 2
    vol = np.zeros(bins)

    lows = np.clip(np.searchsorted(edges, d["Low"].values, side="right") - 1, 0, bins - 1)
    highs = np.clip(np.searchsorted(edges, d["High"].values, side="right") - 1, 0, bins - 1)
    for l, h, v in zip(lows, highs, d["Volume"].values):
        vol[l:h + 1] += v / (h - l + 1)

    poc_i = int(vol.argmax())
    target, covered = vol.sum() * va_pct, vol[poc_i]
    lo_i = hi_i = poc_i
    while covered < target and (lo_i > 0 or hi_i < bins - 1):
        down = vol[lo_i - 1] if lo_i > 0 else -1
        up = vol[hi_i + 1] if hi_i < bins - 1 else -1
        if up >= down:
            hi_i += 1; covered += vol[hi_i]
        else:
            lo_i -= 1; covered += vol[lo_i]
    return {"mids": mids, "vol": vol, "poc": mids[poc_i],
            "vah": mids[hi_i], "val": mids[lo_i], "lo_i": lo_i, "hi_i": hi_i, "poc_i": poc_i}


@st.cache_data(show_spinner=False)
def build_table(_prices: dict, key: int, lookback: int, bins: int, va_pct: float):
    rows = []
    for sym, df in _prices.items():
        vp = volume_profile(df, lookback, bins, va_pct)
        if vp is None:
            continue
        d = df.tail(lookback)
        rows.append({
            "Symbol": sym,
            "Close": round(df["Close"].iloc[-1], 2),
            "POC": round(vp["poc"], 2),
            "VAH": round(vp["vah"], 2),
            "VAL": round(vp["val"], 2),
            "AvgVol": int(d["Volume"].mean()),
            "AvgTurnover(Cr)": round((d["Close"] * d["Volume"]).mean() / 1e7, 2),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------- chart
def make_chart(sym: str, df: pd.DataFrame, lookback: int, bins: int, va_pct: float):
    vp = volume_profile(df, lookback, bins, va_pct)
    d = df.tail(lookback)

    fig = make_subplots(rows=1, cols=2, shared_yaxes=True,
                        column_widths=[0.82, 0.18], horizontal_spacing=0.01)

    fig.add_trace(go.Candlestick(
        x=d.index, open=d["Open"], high=d["High"], low=d["Low"], close=d["Close"],
        name=sym, increasing_line_color="#26a69a", decreasing_line_color="#ef5350"), row=1, col=1)

    # volume profile bars (right panel): value area shaded, POC in red
    colors = []
    for i in range(len(vp["mids"])):
        if i == vp["poc_i"]:
            colors.append("#ef5350")
        elif vp["lo_i"] <= i <= vp["hi_i"]:
            colors.append("#5c7cfa")
        else:
            colors.append("#adb5bd")
    step = vp["mids"][1] - vp["mids"][0]
    fig.add_trace(go.Bar(x=vp["vol"], y=vp["mids"], orientation="h", width=step * 0.95,
                         marker_color=colors, name="Volume profile",
                         hovertemplate="Price %{y:.2f}<br>Vol %{x:,.0f}<extra></extra>"),
                  row=1, col=2)

    for lvl, name, color, dash in [(vp["poc"], "POC", "#ef5350", "solid"),
                                   (vp["vah"], "VAH", "#5c7cfa", "dash"),
                                   (vp["val"], "VAL", "#5c7cfa", "dash")]:
        fig.add_hline(y=lvl, line_color=color, line_dash=dash, line_width=1.2,
                      annotation_text=f"{name} {lvl:.2f}", annotation_position="top left",
                      row=1, col=1)

    fig.update_layout(title=f"{sym} – last {lookback} days with volume profile",
                      height=650, showlegend=False, margin=dict(l=10, r=10, t=50, b=10),
                      xaxis_rangeslider_visible=False)
    fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])], row=1, col=1)
    return fig


# ---------------------------------------------------- TradingView
def tv_symbol(sym: str) -> str:
    # TradingView writes hyphens as underscores (BAJAJ-AUTO -> BAJAJ_AUTO)
    return "NSE:" + sym.replace("-", "_").replace("&", "_")


def tv_widget(sym: str, height: int = 700):
    html = f"""
    <div style="height:{height}px;width:100%">
      <div id="tv_chart" style="height:{height}px;width:100%"></div>
      <script src="https://s3.tradingview.com/tv.js"></script>
      <script>
        new TradingView.widget({{
          autosize: true, symbol: "{tv_symbol(sym)}", interval: "D",
          timezone: "Asia/Kolkata", theme: "dark", style: "1", locale: "en",
          enable_publishing: false, allow_symbol_change: true,
          studies: ["Volume@tv-basicstudies", "VbPVisible@tv-volumebyprice"],
          container_id: "tv_chart"
        }});
      </script>
    </div>
    """
    components.html(html, height=height)


def tv_grid(symbols: list, interval: str, vp: bool, cols: int, cell_h: int = 400):
    """Grid of TradingView Advanced Chart widgets (same approach as the Forex Monitor)."""
    import json
    rows = -(-len(symbols) // cols)
    cfg = {"symbols": [tv_symbol(x) for x in symbols], "interval": interval, "vp": vp}
    html = """
    <style>
      body{margin:0;background:#0e1117}
      .grid{display:grid;grid-template-columns:repeat(__COLS__,1fr);gap:6px}
      .cell{height:__H__px;background:#131722;border:1px solid #2a2e39}
      .tradingview-widget-container,.tradingview-widget-container__widget{width:100%;height:100%}
    </style>
    <div class="grid" id="g"></div>
    <script>
      const cfg = __CFG__;
      cfg.symbols.forEach(sym => {
        const cell = document.createElement("div"); cell.className = "cell";
        const box = document.createElement("div"); box.className = "tradingview-widget-container";
        const inner = document.createElement("div"); inner.className = "tradingview-widget-container__widget";
        box.appendChild(inner);
        const sc = document.createElement("script");
        sc.src = "https://s3.tradingview.com/external-embedding/embed-widget-advanced-chart.js";
        sc.async = true;
        sc.innerHTML = JSON.stringify({
          autosize: true, symbol: sym, interval: cfg.interval, timezone: "Asia/Kolkata",
          theme: "dark", style: "1", locale: "en", allow_symbol_change: false,
          studies: cfg.vp ? ["VbPVisible@tv-volumebyprice"] : [],
          hide_side_toolbar: true, withdateranges: false, save_image: false,
          support_host: "https://www.tradingview.com"
        });
        box.appendChild(sc); cell.appendChild(box);
        document.getElementById("g").appendChild(cell);
      });
    </script>
    """
    html = (html.replace("__COLS__", str(cols)).replace("__H__", str(cell_h))
                .replace("__CFG__", json.dumps(cfg)))
    components.html(html, height=rows * (cell_h + 6) + 10, scrolling=True)


# ------------------------------------------------------------------ UI
st.title("NSE Volume Profile Scanner")

with st.sidebar:
    st.header("Settings")
    lookback = st.number_input("Volume profile lookback (trading days)", 20, 250, 100, 5)
    bins = st.number_input("Price bins", 20, 200, 60, 10)
    ref_label = st.selectbox(
        "Compare price against",
        ["POC (highest-volume price)", "VAH (value area high)", "VAL (value area low)"],
    )
    ref = ref_label.split()[0]
    va_pct = st.slider("Value area %", 50, 90, 70) / 100

    st.subheader("Band (% above reference)")
    band = st.slider("Min % – Max %", 0.0, 50.0, (0.0, 5.0), 0.5)

    st.subheader("Liquidity filters")
    min_price = st.number_input("Min price (₹)", 0.0, value=20.0)
    min_turnover = st.number_input("Min avg daily turnover (₹ Cr)", 0.0, value=1.0)

    if st.button("Refresh data"):
        st.cache_data.clear()
        st.rerun()

symbols = get_nse_symbols()
st.caption(f"{len(symbols)} NSE EQ-series stocks in universe")

prices = download_prices(tuple(symbols))
table = build_table(prices, len(prices), int(lookback), int(bins), va_pct)

pct_col = "% vs " + ref
table[pct_col] = ((table["Close"] - table[ref]) / table[ref] * 100).round(2)

res = table[
    (table[pct_col] >= band[0]) & (table[pct_col] <= band[1])
    & (table["Close"] >= min_price)
    & (table["AvgTurnover(Cr)"] >= min_turnover)
].sort_values(pct_col).reset_index(drop=True)

c1, c2, c3 = st.columns(3)
c1.metric("Stocks scanned", len(table))
c2.metric("Matches", len(res))
c3.metric("Band", f"{band[0]}% to {band[1]}% above {ref}")

st.subheader("Results  (click a row to open its chart)")
event = st.dataframe(res, use_container_width=True, height=400,
                     on_select="rerun", selection_mode="single-row", key="results_tbl")
d1, d2, _ = st.columns([1, 1.6, 3])
d1.download_button("Download CSV", res.to_csv(index=False), "vp_scan.csv", "text/csv")
tv_list = ",".join(tv_symbol(x) for x in res["Symbol"])
d2.download_button("Download TradingView watchlist (.txt)", tv_list,
                   "NSE_VP_watchlist.txt", "text/plain")
with st.expander("Or copy symbols for TradingView"):
    st.code(tv_list, language=None)

tab_grid, tab_one = st.tabs(["TradingView chart grid (all matches)", "Single stock chart"])

with tab_grid:
    if len(res) == 0:
        st.info("No stocks match the current band / filters.")
    else:
        g1, g2, g3, g4 = st.columns(4)
        per_page = g1.selectbox("Charts per page", [4, 6, 9, 12, 16], index=2)
        tf_label = g2.selectbox("Timeframe", ["1D", "1W", "4h", "1h", "15m"])
        tf = {"1D": "D", "1W": "W", "4h": "240", "1h": "60", "15m": "15"}[tf_label]
        vp_on = g3.checkbox("Volume profile on charts", value=True)
        pages = max(1, -(-len(res) // per_page))
        page = g4.number_input(f"Page (of {pages})", 1, pages, 1)
        chunk = res["Symbol"].iloc[(page - 1) * per_page: page * per_page].tolist()
        cols = 2 if per_page <= 4 else 3 if per_page <= 9 else 4
        st.caption(f"Showing {len(chunk)} of {len(res)} matches, sorted closest to POC first. "
                   "Changing the band or filters in the sidebar updates this list.")
        tv_grid(chunk, tf, vp_on, cols)

with tab_one:
    selected = event.selection.rows
    if selected and len(res):
        sym = res.iloc[selected[0]]["Symbol"]
        row = res.iloc[selected[0]]
        st.subheader(f"{sym}  ·  Close ₹{row['Close']}  ·  POC ₹{row['POC']}  ·  {row[pct_col]}% vs {ref}")
        view = st.radio("Chart", ["TradingView", "Built-in (with POC / value area)"], horizontal=True)
        if view == "TradingView":
            tv_widget(sym)
            st.markdown(f"[Open full chart on TradingView ↗](https://www.tradingview.com/chart/?symbol={tv_symbol(sym)})")
            st.caption(f"Draw the POC line (₹{row['POC']}) on TradingView, or switch to the built-in chart to see it automatically. "
                       "If the volume profile isn't shown, add it via Indicators → 'Volume Profile Visible Range'.")
        else:
            st.plotly_chart(make_chart(sym, prices[sym], int(lookback), int(bins), va_pct),
                            use_container_width=True)
            st.caption("Red bar/line = POC · Blue = value area (VAH/VAL) · Grey = outside value area")
    else:
        st.info("Click a row in the table above to see the chart.")


st.caption(
    "Profile is built from daily bars (volume spread across each day's high–low range), "
    "so it approximates an intraday-built profile. Data: Yahoo Finance. Not investment advice."
)
