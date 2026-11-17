"""
NSE Volume Profile Scanner (with click-to-chart)
------------------------------------------------
Finds NSE-listed stocks trading ABOVE their N-day volume profile level (POC),
but not more than X% above it (band is editable). Click a row to see its chart.

Install:  pip install streamlit yfinance pandas numpy requests plotly
Run:      streamlit run nse_vp_scanner.py
"""

import io
import re
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots

st.set_page_config(page_title="NSE Volume Profile Scanner", layout="wide")

NSE_LIST_URL = "https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv"
HEADERS = {"User-Agent": "Mozilla/5.0"}


# ---------------------------------------------------------------- data
LOCAL_LIST = Path(__file__).parent / "EQUITY_L.csv"


def _clean_list(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [c.strip() for c in df.columns]
    df["SYMBOL"] = df["SYMBOL"].astype(str).str.strip()
    df["SERIES"] = df["SERIES"].astype(str).str.strip()
    if "NAME OF COMPANY" not in df.columns:
        df["NAME OF COMPANY"] = ""
    return df[["SYMBOL", "NAME OF COMPANY", "SERIES"]]


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def load_list() -> pd.DataFrame:
    """Try NSE live; if NSE blocks us (common on cloud servers), use EQUITY_L.csv from the repo."""
    try:
        sess = requests.Session()
        sess.headers.update({"User-Agent": "Mozilla/5.0", "Referer": "https://www.nseindia.com/"})
        r = sess.get(NSE_LIST_URL, timeout=10)
        r.raise_for_status()
        return _clean_list(pd.read_csv(io.StringIO(r.text)))
    except Exception:
        pass
    if LOCAL_LIST.exists():
        return _clean_list(pd.read_csv(LOCAL_LIST))
    raise RuntimeError("NO_LIST")


def get_nse_symbols(series: tuple) -> list[str]:
    df = load_list()
    return sorted(df[df["SERIES"].isin(series)]["SYMBOL"].unique().tolist())


def yf_ticker(sym: str) -> str:
    """NSE symbols get .NS; entries that already contain a dot (e.g. 512399.BO) are used as-is."""
    return sym if "." in sym else sym + ".NS"


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def download_prices(symbols: tuple, chunk: int = 100) -> dict:
    """Daily OHLCV for every symbol (~9 months so 100+ trading days exist)."""
    out = {}
    prog = st.progress(0.0, text="Downloading price data...")
    for i in range(0, len(symbols), chunk):
        batch = list(symbols[i:i + chunk])
        tickers = [yf_ticker(s) for s in batch]
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
def build_table(_prices: dict, key: int, lookback: int, bins: int, va_pct: float,
                n_res: int = 20, k_sup: int = 5):
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
            "Res1D": round(df["High"].iloc[-(n_res + 1):-1].max(), 2),   # highest high of prior N days
            "MinCloseK": round(df["Close"].tail(k_sup).min(), 2),         # lowest close in last K days
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------- chart
def make_chart(sym: str, df: pd.DataFrame, lookback: int, bins: int, va_pct: float, height: int = 650):
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
                      height=height, showlegend=False, margin=dict(l=10, r=10, t=50, b=10),
                      xaxis_rangeslider_visible=False)
    fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])], row=1, col=1)
    return fig


# ---------------------------------------------------- TradingView
def tv_symbol(sym: str) -> str:
    if sym.endswith(".BO"):
        return "BSE:" + sym[:-3]
    # TradingView writes hyphens as underscores (BAJAJ-AUTO -> BAJAJ_AUTO)
    return "NSE:" + sym.replace("-", "_").replace("&", "_")



def _pick(data, t):
    try:
        return data[t]
    except KeyError:
        return data


@st.cache_data(ttl=900, show_spinner=False)
def get_15m_levels(symbols: tuple, bars: int, chunk: int = 50) -> dict:
    """For each symbol: (latest 15m close, highest high of the previous `bars` 15m bars)."""
    out = {}
    for i in range(0, len(symbols), chunk):
        batch = list(symbols[i:i + chunk])
        tickers = [yf_ticker(x) for x in batch]
        try:
            data = yf.download(tickers, period="30d", interval="15m", group_by="ticker",
                               auto_adjust=False, threads=True, progress=False)
        except Exception:
            continue
        for x, t in zip(batch, tickers):
            try:
                d = _pick(data, t)[["High", "Close"]].dropna()
                if len(d) > bars + 1:
                    out[x] = (float(d["Close"].iloc[-1]), float(d["High"].iloc[-(bars + 1):-1].max()))
            except Exception:
                pass
    return out


MY_LIST_FILES = [Path(__file__).parent / "my_stocks.csv", Path(__file__).parent / "my_stocks.txt"]
BSE_MAP_FILE = Path(__file__).parent / "bse_scrips.csv"


def _split_sym(sym: str, ex: str = ""):
    sym = sym.strip().upper()
    if ":" in sym:
        ex, sym = sym.split(":", 1)
    sym = sym.strip()
    if sym.endswith(".NS"):
        sym = sym[:-3]
    return sym, ex.strip().upper()


def read_custom_df(raw: str, name: str = "") -> pd.DataFrame:
    """CSV with a Symbol column (TradingView screener export works as-is), or a TradingView watchlist .txt.
    Returns columns: sym, ex (exchange if known), turn (today's turnover in Cr, if Price & Volume columns exist)."""
    raw = raw.lstrip("\ufeff")
    rows = []
    if name.lower().endswith(".csv"):
        df = pd.read_csv(io.StringIO(raw))
        df.columns = [str(c).strip() for c in df.columns]
        low = {c.lower(): c for c in df.columns}
        col = next((low[k] for k in ("symbol", "ticker", "scrip code", "security code", "code", "security id")
                    if k in low), df.columns[0])
        exc, pcol = low.get("exchange"), low.get("price")
        vcol = next((c for c in df.columns if c.lower().startswith("volume")), None)
        for i, v in df[col].dropna().items():
            sym, ex = _split_sym(str(v), str(df.at[i, exc]) if exc else "")
            turn = np.nan
            if pcol and vcol:
                turn = (pd.to_numeric(df.at[i, pcol], errors="coerce")
                        * pd.to_numeric(df.at[i, vcol], errors="coerce") / 1e7)
            rows.append((sym, ex, turn))
    else:
        for tok in re.split(r"[,\n\r\t ]+", raw):
            if tok and not tok.startswith("#"):
                sym, ex = _split_sym(tok)
                rows.append((sym, ex, np.nan))
    out = pd.DataFrame(rows, columns=["sym", "ex", "turn"])
    out = out[(out["sym"] != "") & (out["sym"] != "NAN")]
    return out.sort_values("turn", ascending=False).drop_duplicates("sym").reset_index(drop=True)


def read_bse_map(raw: str) -> dict:
    """BSE 'List of scrips' or Bhavcopy CSV -> {ticker/Security Id: BSE code}."""
    df = pd.read_csv(io.StringIO(raw.lstrip("\ufeff")), index_col=False)  # BSE rows carry trailing commas
    low = {str(c).strip().lower(): c for c in df.columns}
    idc = next((low[k] for k in ("security id", "tckrsymb", "scrip id", "scrip_id", "symbol") if k in low), None)
    cc = next((low[k] for k in ("security code", "fininstrmid", "sc_code", "scrip code", "scrip_cd", "code")
               if k in low), None)
    if not idc or not cc:
        return {}
    codes = pd.to_numeric(df[cc], errors="coerce")
    out = {}
    for sid, code in zip(df[idc].astype(str).str.strip().str.upper(), codes):
        if code == code:
            out[sid] = str(int(code))
    return out


BSE_API = ("https://api.bseindia.com/BseIndiaAPI/api/ListofScripData/w"
           "?Group=&Scripcode=&industry=&segment=Equity&status=Active")


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def load_bse_map_live() -> dict:
    """Best effort: BSE often blocks cloud servers. Returns {} if it does."""
    try:
        r = requests.get(BSE_API, timeout=15, headers={
            "User-Agent": "Mozilla/5.0", "Referer": "https://www.bseindia.com/",
            "Origin": "https://www.bseindia.com"})
        r.raise_for_status()
        out = {}
        for row in r.json():
            low = {str(k).lower(): v for k, v in row.items()}
            sid = low.get("scrip_id") or low.get("security id")
            code = str(low.get("scrip_cd") or low.get("security code") or "").strip()
            if sid and code.isdigit():
                out[str(sid).strip().upper()] = code
        return out if len(out) > 1000 else {}
    except Exception:
        return {}


# ------------------------------------------------------------------ UI
st.title("NSE Volume Profile Scanner")

with st.sidebar:
    st.header("Settings")
    series_sel = st.multiselect("NSE series to include", ["EQ", "BE", "BZ", "SM", "ST"], default=["EQ"],
                                help="EQ = normal stocks. BE/BZ = trade-for-trade or surveillance stocks. "
                                "SM/ST = SME platform. Adding series means a bigger (slower) download.")
    extra_txt = st.text_area(
        "Extra stocks to scan (comma-separated)", value="",
        help="Add stocks missing from the NSE list. BSE-only stocks: BSE code + .BO, e.g. 512399.BO. "
             "NSE stocks: just the symbol, e.g. SERA.")
    universe_mode = st.radio("Stock universe", ["NSE list", "NSE list + my list", "My list only"],
                             index=1 if any(f.exists() for f in MY_LIST_FILES) else 0,
                             help="'My list' = the file you upload below, or my_stocks.csv / my_stocks.txt in the repository.")
    up = st.file_uploader("My stock list (.csv or .txt)", type=["csv", "txt"],
                          help="CSV with a Symbol column (optional Exchange column), or a TradingView watchlist .txt. "
                               "BSE stocks: use the 6-digit code, e.g. 512399.")
    custom_df = pd.DataFrame(columns=["sym", "ex", "turn"])
    try:
        if up is not None:
            custom_df = read_custom_df(up.getvalue().decode("utf-8", "ignore"), up.name)
        else:
            for f in MY_LIST_FILES:
                if f.exists():
                    custom_df = read_custom_df(f.read_text(encoding="utf-8", errors="ignore"), f.name)
                    break
    except Exception as e:
        st.error(f"Could not read your list: {e}")
    min_turn_pre = st.number_input(
        "My list: skip stocks with today's turnover below (₹ Cr)", 0.0, value=0.2, step=0.1,
        help="Uses Price × Volume from your CSV to skip dead stocks before downloading (much faster). Set 0 to scan all.")
    bse_up = st.file_uploader("BSE scrip list (optional, .csv)", type=["csv"],
                              help="BSE's 'List of scrips' file (columns Security Code, Security Id). "
                                   "Lets the scanner find BSE-only stocks from your list. Or put it in the repo as bse_scrips.csv.")
    bse_map = {}
    try:
        if bse_up is not None:
            bse_map = read_bse_map(bse_up.getvalue().decode("utf-8", "ignore"))
        elif BSE_MAP_FILE.exists():
            bse_map = read_bse_map(BSE_MAP_FILE.read_text(encoding="utf-8", errors="ignore"))
        else:
            bse_map = load_bse_map_live()
    except Exception as e:
        st.error(f"Could not read the BSE list: {e}")
    pin_extra = st.checkbox("Always show my extra stocks (even if they fail filters)", value=True,
                            help="Shows them at the top with Passes = False so you can see their numbers.")
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

    st.subheader("Support / resistance filters")
    use_d_res = st.checkbox("Price above 1D resistance", value=False,
                            help="Latest close > highest high of the previous N daily candles.")
    n_res = st.number_input("1D resistance: highest high of last N days", 5, 100, 20, 5)
    use_sup = st.checkbox("Support held at POC", value=False,
                          help="No daily close below POC (minus tolerance) in the last K days.")
    k_sup = st.number_input("Support check: last K days", 1, 60, 5)
    sup_tol = st.number_input("Support tolerance (% below POC)", 0.0, 5.0, 1.0, 0.5)
    use_15 = st.checkbox("Price above 15-min resistance", value=False,
                         help="Latest 15-min close > highest high of the previous N 15-min bars. "
                              "Checked only on stocks that pass the other filters.")
    bars15 = st.number_input("15-min resistance: last N bars (26 bars ≈ 1 day)", 10, 500, 100, 10)

    if st.button("Refresh data"):
        st.cache_data.clear()
        st.rerun()

try:
    full_list = load_list()
    symbols = get_nse_symbols(tuple(series_sel or ['EQ']))
    extra_list = []
    for t in extra_txt.replace("\n", ",").split(","):
        t = t.strip().upper()
        if t.endswith(".NS"):
            t = t[:-3]
        if t:
            extra_list.append(t)
    # ---- resolve my list (TradingView names) to Yahoo tickers: NSE first, then BSE via scrip list
    custom_syms, unresolved, skipped = [], [], 0
    nse_all = set(full_list["SYMBOL"])
    for r in custom_df.itertuples(index=False):
        if r.turn == r.turn and r.turn < min_turn_pre:
            skipped += 1
            continue
        t = r.sym
        if t.endswith(".BO"):
            custom_syms.append(t)
        elif t.isdigit():
            custom_syms.append(t + ".BO")
        else:
            hit = next((c for c in (t, t.replace("_", "-")) if c in nse_all), None)
            if hit and r.ex != "BSE":
                custom_syms.append(hit)
            elif t in bse_map:
                custom_syms.append(bse_map[t] + ".BO")
            else:
                unresolved.append(t)
    if len(custom_df):
        st.caption(f"My list: {len(custom_df)} symbols → {len(custom_syms)} to scan, "
                   f"{skipped} skipped (low turnover), {len(unresolved)} not matched to NSE/BSE")
        if unresolved:
            with st.expander(f"{len(unresolved)} symbols not matched (add a BSE scrip list to resolve BSE-only ones)"):
                st.write(", ".join(unresolved[:500]) + (" ..." if len(unresolved) > 500 else ""))
    if universe_mode == "My list only":
        symbols = []
    if universe_mode != "NSE list":
        symbols = list(symbols) + custom_syms
    symbols = sorted(set(symbols) | set(extra_list))
except RuntimeError:
    st.error("Could not load the NSE stock list. Upload 'EQUITY_L.csv' (from "
             "https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv) to the app's GitHub repository.")
    st.stop()
st.caption(f"{len(symbols)} NSE stocks in universe (series: {', '.join(series_sel or ['EQ'])})")

prices = download_prices(tuple(symbols))
table = build_table(prices, len(prices), int(lookback), int(bins), va_pct, int(n_res), int(k_sup))

pct_col = "% vs " + ref
table[pct_col] = ((table["Close"] - table[ref]) / table[ref] * 100).round(2)

res = table[
    (table[pct_col] >= band[0]) & (table[pct_col] <= band[1])
    & (table["Close"] >= min_price)
    & (table["AvgTurnover(Cr)"] >= min_turnover)
].sort_values(pct_col).reset_index(drop=True)


# ---- support / resistance filters
if use_d_res:
    res = res[res["Close"] > res["Res1D"]]
if use_sup:
    res = res[res["MinCloseK"] >= res["POC"] * (1 - sup_tol / 100)]
res = res.reset_index(drop=True)

if use_15 and len(res):
    if len(res) > 300:
        st.warning(f"{len(res)} stocks passed the other filters; checking 15-min data for the 300 closest "
                   "to POC only. Narrow the band or turn on other filters to cover all.")
        res = res.head(300)
    with st.spinner("Checking 15-minute resistance..."):
        lv = get_15m_levels(tuple(res["Symbol"]), int(bars15))
    res["Close15m"] = res["Symbol"].map(lambda x: lv.get(x, (np.nan, np.nan))[0]).round(2)
    res["Res15m"] = res["Symbol"].map(lambda x: lv.get(x, (np.nan, np.nan))[1]).round(2)
    res = res[res["Close15m"] > res["Res15m"]].reset_index(drop=True)

res["Passes"] = True
if pin_extra and extra_list:
    in_table = set(table["Symbol"])
    no_data = [x for x in extra_list if x not in in_table]
    if no_data:
        st.warning("No usable Yahoo data (or under 80 days of history) for: " + ", ".join(no_data))
    missing = table[table["Symbol"].isin(extra_list) & ~table["Symbol"].isin(res["Symbol"])].copy()
    if len(missing):
        missing["Passes"] = False
        res = pd.concat([missing, res], ignore_index=True)

c1, c2, c3 = st.columns(3)
c1.metric("Stocks scanned", len(table))
c2.metric("Matches", int(res["Passes"].sum()))
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

def tv_link(sym):
    return f"https://www.tradingview.com/chart/?symbol={tv_symbol(sym)}"


tab_grid, tab_one = st.tabs(["Chart grid (all matches)", "Single stock chart"])

with tab_grid:
    if len(res) == 0:
        st.info("No stocks match the current band / filters.")
    else:
        g1, g2 = st.columns(2)
        per_page = g1.selectbox("Charts per page", [4, 6, 9, 12], index=1)
        pages = max(1, -(-len(res) // per_page))
        page = g2.number_input(f"Page (of {pages})", 1, pages, 1)
        chunk = res.iloc[(page - 1) * per_page: page * per_page]
        ncols = 2 if per_page <= 4 else 3
        st.caption(f"Showing {len(chunk)} of {len(res)} matches, closest to POC first. "
                   "Red = POC · Blue = value area · Grey = outside value area.")
        rows_ = list(chunk.itertuples(index=False))
        for i in range(0, len(rows_), ncols):
            cols = st.columns(ncols)
            for col, r in zip(cols, rows_[i:i + ncols]):
                with col:
                    st.plotly_chart(
                        make_chart(r.Symbol, prices[r.Symbol], int(lookback), int(bins), va_pct, height=360),
                        use_container_width=True, key=f"grid_{r.Symbol}")
                    st.markdown(f"**{r.Symbol}** · ₹{r.Close} · "
                                f"[Open on TradingView ↗]({tv_link(r.Symbol)})")

with tab_one:
    selected = event.selection.rows
    if selected and len(res):
        row = res.iloc[selected[0]]
        sym = row["Symbol"]
        st.subheader(f"{sym}  ·  Close ₹{row['Close']}  ·  POC ₹{row['POC']}  ·  {row[pct_col]}% vs {ref}")
        st.plotly_chart(make_chart(sym, prices[sym], int(lookback), int(bins), va_pct),
                        use_container_width=True, key="single_chart")
        st.markdown(f"[Open {sym} on TradingView ↗]({tv_link(sym)})")
        st.caption("Red bar/line = POC · Blue = value area (VAH/VAL) · Grey = outside value area")
    else:
        st.info("Click a row in the table above to see the chart.")



with st.expander("Why isn't a stock showing? (diagnose)"):
    q = st.text_input("NSE symbol, BSE code with .BO (e.g. 512399.BO), or part of the name").strip().upper()
    if q:
        fl = full_list
        hits = fl[fl["SYMBOL"].str.upper().str.contains(q, regex=False)
                  | fl["NAME OF COMPANY"].astype(str).str.upper().str.contains(q, regex=False)]
        custom = q in (set(extra_list) | set(custom_syms))
        if not custom:
            if hits.empty:
                st.error("Not found in the NSE equity list file. It may be listed only on BSE, be a very recent "
                         "listing, or the EQUITY_L.csv copy in the repository is old. For a BSE-only stock, add "
                         "its BSE code with .BO (e.g. 512399.BO) to 'Extra stocks to scan' in the sidebar.")
            else:
                st.write("Matches in the NSE list:")
                st.dataframe(hits.head(15), use_container_width=True, hide_index=True)
        exact = hits[hits["SYMBOL"].str.upper() == q]
        if custom or not exact.empty:
            ser = "extra" if custom else exact.iloc[0]["SERIES"]
            if not custom and ser not in (series_sel or ["EQ"]):
                st.warning(f"{q} is in series **{ser}**, which is not selected. Add it under "
                           "'NSE series to include' in the sidebar.")
            elif q not in prices:
                st.warning("No price data came back from Yahoo Finance for this symbol "
                           "(not listed there, delisted, or the download was throttled). Try Refresh data.")
            elif q not in set(table["Symbol"]):
                st.warning(f"Fewer than {int(lookback * 0.8)} trading days of history, "
                           f"so a {int(lookback)}-day profile can't be built.")
            else:
                r = table[table["Symbol"] == q].iloc[0]
                checks = [
                    (f"{pct_col} within {band[0]}% to {band[1]}%", band[0] <= r[pct_col] <= band[1], f"{r[pct_col]}%"),
                    (f"Price >= {min_price}", r["Close"] >= min_price, f"{r['Close']}"),
                    (f"Avg turnover >= {min_turnover} Cr", r["AvgTurnover(Cr)"] >= min_turnover, f"{r['AvgTurnover(Cr)']} Cr"),
                ]
                if use_d_res:
                    checks.append(("Close above 1D resistance", r["Close"] > r["Res1D"], f"close {r['Close']} vs resistance {r['Res1D']}"))
                if use_sup:
                    checks.append(("Support held at POC", r["MinCloseK"] >= r["POC"] * (1 - sup_tol / 100),
                                   f"lowest close {r['MinCloseK']} vs POC {r['POC']}"))
                st.write(f"**{q}**: close {r['Close']}, POC {r['POC']}, VAH {r['VAH']}, VAL {r['VAL']}")
                for label, ok, detail in checks:
                    st.write(("✅ " if ok else "❌ ") + f"{label}  ({detail})")
                if use_15:
                    st.caption("The 15-minute check is not shown here; it runs only on stocks that pass the other filters.")
        elif not hits.empty:
            st.info("Type the exact symbol from the table above to run the full check.")

st.caption(
    "Profile is built from daily bars (volume spread across each day's high–low range), "
    "so it approximates an intraday-built profile. Data: Yahoo Finance. Not investment advice."
)
