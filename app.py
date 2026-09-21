import streamlit as st
import pandas as pd
import numpy as np
import requests
import xml.etree.ElementTree as ET
from datetime import date, timedelta
from urllib.parse import quote

st.set_page_config(
    page_title="KEN AI 台股智慧分析",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- 專業深色金融介面 ----------
st.markdown("""
<style>
.stApp {
    background:
      radial-gradient(circle at 85% 10%, rgba(37,99,235,.18), transparent 28%),
      radial-gradient(circle at 15% 85%, rgba(14,165,233,.10), transparent 25%),
      linear-gradient(135deg,#07111f 0%,#0b1728 52%,#08121f 100%);
    color:#e8eef8;
}
[data-testid="stSidebar"] {
    background:linear-gradient(180deg,#0c1728 0%,#101d31 100%);
    border-right:1px solid rgba(148,163,184,.15);
}
[data-testid="stSidebar"] * { color:#e8eef8; }
.block-container {max-width:1280px;padding-top:1.5rem;padding-bottom:3rem;}
h1,h2,h3 {color:#f8fafc !important;}
.hero {
    padding:24px 28px;border:1px solid rgba(96,165,250,.22);border-radius:22px;
    background:linear-gradient(135deg,rgba(30,64,175,.22),rgba(15,23,42,.80));
    box-shadow:0 18px 55px rgba(0,0,0,.25);margin-bottom:18px;
}
.hero-title {font-size:2.15rem;font-weight:800;letter-spacing:.02em;color:#fff;}
.hero-sub {color:#9fb2cc;margin-top:6px;}
.card {
    background:rgba(15,29,49,.78);border:1px solid rgba(148,163,184,.15);
    border-radius:18px;padding:18px;box-shadow:0 12px 30px rgba(0,0,0,.18);
}
.signal {
    padding:20px;border-radius:18px;text-align:center;border:1px solid rgba(148,163,184,.16);
    background:rgba(15,29,49,.82);min-height:132px;
}
.signal-title {font-size:.92rem;color:#9fb2cc;}
.signal-main {font-size:1.55rem;font-weight:800;margin-top:10px;}
.up {color:#34d399}.watch {color:#fbbf24}.down {color:#fb7185}
.muted {color:#91a4bf;font-size:.9rem;}
.pricebox {
    padding:14px 16px;border-radius:14px;background:rgba(15,29,49,.75);
    border:1px solid rgba(148,163,184,.14);
}
div[data-testid="stMetric"] {
    background:rgba(15,29,49,.75);border:1px solid rgba(148,163,184,.14);
    padding:14px 16px;border-radius:16px;
}
div[data-testid="stMetric"] label {color:#9fb2cc !important;}
div[data-testid="stMetricValue"] {color:#f8fafc !important;}
.stButton>button {
    width:100%;border:0;border-radius:12px;font-weight:700;
    background:linear-gradient(90deg,#2563eb,#0ea5e9);color:white;padding:.65rem 1rem;
}
a {color:#60a5fa !important;}
</style>
""", unsafe_allow_html=True)

API = "https://api.finmindtrade.com/api/v4/data"

def get_data(dataset, stock, start, end, token=""):
    params = {
        "dataset": dataset, "data_id": stock,
        "start_date": start, "end_date": end
    }
    if token:
        params["token"] = token
    r = requests.get(API, params=params, timeout=20)
    r.raise_for_status()
    obj = r.json()
    if obj.get("status") != 200:
        raise RuntimeError(obj.get("msg", "FinMind 資料取得失敗"))
    return pd.DataFrame(obj.get("data", []))

def add_indicators(df):
    df = df.copy()
    for c in ["open","max","min","close","Trading_Volume"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    for n in [5,10,20,60]:
        df[f"MA{n}"] = df["close"].rolling(n).mean()

    delta = df["close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df["RSI14"] = 100 - (100 / (1 + rs))

    low9 = df["min"].rolling(9).min()
    high9 = df["max"].rolling(9).max()
    rsv = (df["close"] - low9) / (high9 - low9).replace(0, np.nan) * 100
    df["K"] = rsv.ewm(alpha=1/3, adjust=False).mean()
    df["D"] = df["K"].ewm(alpha=1/3, adjust=False).mean()

    ema12 = df["close"].ewm(span=12, adjust=False).mean()
    ema26 = df["close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_SIGNAL"] = df["MACD"].ewm(span=9, adjust=False).mean()
    return df

def trend_score(df, days):
    d = df.tail(days)
    last = df.iloc[-1]
    score = 0
    reasons = []
    if last["close"] > last.get("MA5", np.nan):
        score += 1; reasons.append("股價站上5日線")
    else:
        score -= 1
    if last["close"] > last.get("MA20", np.nan):
        score += 1; reasons.append("股價站上20日線")
    else:
        score -= 1
    if days >= 60 and pd.notna(last.get("MA60")):
        score += 1 if last["close"] > last["MA60"] else -1
    if pd.notna(last.get("RSI14")):
        if 50 <= last["RSI14"] <= 70: score += 1
        elif last["RSI14"] < 40: score -= 1
    if pd.notna(last.get("MACD")) and pd.notna(last.get("MACD_SIGNAL")):
        score += 1 if last["MACD"] > last["MACD_SIGNAL"] else -1
    if len(d) >= 6:
        score += 1 if d["close"].iloc[-1] > d["close"].iloc[0] else -1
    return score, reasons

def label_signal(score):
    if score >= 3:
        return "偏多", "🟢", "up"
    if score <= -3:
        return "偏空", "🔴", "down"
    return "觀望", "🟡", "watch"

def rss_news(query, limit=6):
    # 公開 Google News RSS 搜尋；只做新聞索引與連結，不代表來源內容已被驗證
    url = "https://news.google.com/rss/search?q=" + quote(query) + "&hl=zh-TW&gl=TW&ceid=TW:zh-Hant"
    try:
        r = requests.get(url, timeout=12, headers={"User-Agent":"Mozilla/5.0"})
        r.raise_for_status()
        root = ET.fromstring(r.content)
        items = []
        for item in root.findall(".//item")[:limit]:
            items.append({
                "title": item.findtext("title",""),
                "link": item.findtext("link",""),
                "pubDate": item.findtext("pubDate","")
            })
        return items
    except Exception:
        return []

def render_signal(title, score, period):
    label, icon, cls = label_signal(score)
    st.markdown(
        f"""<div class="signal">
        <div class="signal-title">{title}</div>
        <div class="signal-main {cls}">{icon} {label}</div>
        <div class="muted">{period}</div>
        </div>""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
  <div class="hero-title">📈 KEN AI 台股智慧分析系統</div>
  <div class="hero-sub">市場數據｜技術分析｜法人籌碼｜新聞情報｜短中長期趨勢研判</div>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header("🔎 分析條件")
    stock_id = st.text_input("股票代碼", "6213", help="例如：2330、6213")
    stock_name = st.text_input("股票名稱（新聞搜尋用）", "聯茂")
    holding = st.radio("持股狀態", ["尚未持有", "已持有"])
    cost = st.number_input("持有成本（元）", min_value=0.0, value=0.0, step=1.0)
    shares = st.number_input("持有張數", min_value=0.0, value=0.0, step=1.0)
    token = st.text_input("FinMind Token（建議填入）", type="password",
                          help="請勿把 Token 寫進公開 GitHub 程式碼。")
    run = st.button("🚀 開始智慧分析", type="primary")

if not run:
    st.info("👈 請在左側輸入股票代碼、名稱與持股條件，再按「開始智慧分析」。")
    c1,c2,c3 = st.columns(3)
    with c1:
        st.markdown('<div class="card"><h3>📊 技術面</h3><p class="muted">MA、RSI、KD、MACD、支撐與壓力。</p></div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="card"><h3>🏦 籌碼面</h3><p class="muted">法人買賣超與市場量價結構。</p></div>', unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="card"><h3>📰 情報面</h3><p class="muted">公開新聞、財經節目與市場討論索引。</p></div>', unsafe_allow_html=True)
    st.caption("本系統為研究與資訊整理工具；趨勢訊號不代表保證獲利或個別投資建議。")
    st.stop()

end = date.today()
start = end - timedelta(days=220)

try:
    price = get_data("TaiwanStockPrice", stock_id, str(start), str(end), token)
    if price.empty:
        st.error("查不到此股票的行情資料，請確認股票代碼或 FinMind Token。")
        st.stop()
    price = price.sort_values("date").reset_index(drop=True)
    price = add_indicators(price)
except Exception as e:
    st.error(f"行情資料取得失敗：{e}")
    st.info("若未填 FinMind Token，建議先申請免費 Token 後再試。")
    st.stop()

last = price.iloc[-1]
prev = price.iloc[-2] if len(price) >= 2 else last
chg = float(last["close"] - prev["close"])
chg_pct = (chg / prev["close"] * 100) if prev["close"] else 0

short_score,_ = trend_score(price, 10)
mid_score,_ = trend_score(price, 40)
long_score,_ = trend_score(price, 100)

# 法人資料
inst_text = "資料暫缺"
inst_net = 0
try:
    inst = get_data("TaiwanStockInstitutionalInvestorsBuySell", stock_id,
                    str(end - timedelta(days=20)), str(end), token)
    if not inst.empty and {"buy","sell"}.issubset(inst.columns):
        inst["buy"] = pd.to_numeric(inst["buy"], errors="coerce").fillna(0)
        inst["sell"] = pd.to_numeric(inst["sell"], errors="coerce").fillna(0)
        inst["net"] = inst["buy"] - inst["sell"]
        inst_net = float(inst.tail(15)["net"].sum())
        inst_text = "近期待偏多" if inst_net > 0 else "近期待偏空" if inst_net < 0 else "中性"
except Exception:
    inst = pd.DataFrame()

# 關鍵價位
tail20 = price.tail(20)
tail60 = price.tail(60)
support1 = float(tail20["min"].min())
resist1 = float(tail20["max"].max())
support2 = float(tail60["min"].min())
resist2 = float(tail60["max"].max())

st.subheader(f"{stock_id} {stock_name}｜最新市場概況")
m1,m2,m3,m4 = st.columns(4)
m1.metric("最新收盤", f"{last['close']:.2f}", f"{chg:+.2f} ({chg_pct:+.2f}%)")
m2.metric("成交量", f"{int(last['Trading_Volume']):,}")
m3.metric("RSI(14)", f"{last['RSI14']:.1f}" if pd.notna(last["RSI14"]) else "—")
m4.metric("法人近期淨額", f"{inst_net:,.0f}" if inst_text != "資料暫缺" else "資料暫缺")

st.markdown("### 🤖 短／中／長期趨勢訊號")
a,b,c = st.columns(3)
with a: render_signal("短線趨勢", short_score, "約 1～5 個交易日")
with b: render_signal("中線趨勢", mid_score, "約 2～6 週")
with c: render_signal("長線趨勢", long_score, "約 3～12 個月")

combined = short_score + mid_score + long_score
overall_label, overall_icon, overall_cls = label_signal(round(combined/3))
st.markdown(
    f"""<div class="card" style="margin-top:16px">
    <div class="muted">綜合市場訊號</div>
    <div style="font-size:1.8rem;font-weight:800" class="{overall_cls}">
    {overall_icon} {overall_label}
    </div>
    <div class="muted">依目前量價與技術資料計算；不是保證未來漲跌。</div>
    </div>""", unsafe_allow_html=True)

st.markdown("### 🎯 關鍵價位")
k1,k2,k3,k4,k5 = st.columns(5)
k1.metric("第二支撐", f"{support2:.2f}")
k2.metric("第一支撐", f"{support1:.2f}")
k3.metric("目前價格", f"{last['close']:.2f}")
k4.metric("第一壓力", f"{resist1:.2f}")
k5.metric("第二壓力", f"{resist2:.2f}")

if holding == "已持有" and cost > 0:
    pnl_pct = (last["close"] - cost) / cost * 100
    st.markdown("### 💼 我的持股")
    h1,h2,h3 = st.columns(3)
    h1.metric("成本", f"{cost:.2f}")
    h2.metric("目前損益率", f"{pnl_pct:+.2f}%")
    h3.metric("張數", f"{shares:g}")

st.markdown("### 📉 價格與均線")
chart_cols = ["date","close","MA5","MA20","MA60"]
chart = price[[c for c in chart_cols if c in price.columns]].copy()
chart["date"] = pd.to_datetime(chart["date"])
chart = chart.set_index("date")
st.line_chart(chart, use_container_width=True)

st.markdown("### 🧭 技術面摘要")
t1,t2,t3,t4 = st.columns(4)
t1.metric("MA5", f"{last['MA5']:.2f}" if pd.notna(last["MA5"]) else "—")
t2.metric("MA20", f"{last['MA20']:.2f}" if pd.notna(last["MA20"]) else "—")
t3.metric("KD", f"K {last['K']:.1f} / D {last['D']:.1f}" if pd.notna(last["K"]) else "—")
t4.metric("MACD", f"{last['MACD']:.2f}" if pd.notna(last["MACD"]) else "—")

st.markdown("### 🏦 法人籌碼")
st.write(f"近期法人整體狀態：**{inst_text}**")
if isinstance(inst, pd.DataFrame) and not inst.empty:
    show_cols = [c for c in ["date","name","buy","sell","net"] if c in inst.columns]
    st.dataframe(inst[show_cols].tail(20), use_container_width=True, hide_index=True)

st.markdown("### 📰 公開新聞與市場情報")
news_query = f"{stock_id} {stock_name} 股票"
news = rss_news(news_query, 6)
if news:
    for x in news:
        st.markdown(f"**{x['title']}**  \n{x['pubDate']}  \n[查看來源]({x['link']})")
else:
    st.info("目前無法取得新聞索引。")

q1,q2 = st.columns(2)
with q1:
    st.markdown("#### 📺 錢線百分百相關")
    tv = rss_news(f'"錢線百分百" {stock_id} {stock_name}', 4)
    if tv:
        for x in tv:
            st.markdown(f"• [{x['title']}]({x['link']})")
    else:
        st.caption("目前公開新聞索引未找到近期相關內容。")
with q2:
    st.markdown("#### 💬 股市爆料同學會／論壇相關")
    forum = rss_news(f'"股市爆料同學會" {stock_id} {stock_name}', 4)
    if forum:
        for x in forum:
            st.markdown(f"• [{x['title']}]({x['link']})")
    else:
        st.caption("目前公開新聞索引未找到近期相關內容。")

st.markdown("### 🧠 綜合研判說明")
short_l = label_signal(short_score)[0]
mid_l = label_signal(mid_score)[0]
long_l = label_signal(long_score)[0]
st.markdown(f"""
<div class="card">
<b>短線：</b>{short_l}。主要觀察短期均線、RSI、MACD、成交量與近期價格動能。<br><br>
<b>中線：</b>{mid_l}。主要觀察 20／60 日結構、法人籌碼與波段高低點。<br><br>
<b>長線：</b>{long_l}。目前版本以較長期價格趨勢作為參考；若要提高長線分析品質，
後續應再加入月營收、EPS、財報、產業展望與公司重大訊息。<br><br>
<b>訊號轉換：</b>若有效突破壓力區且量能配合，偏多訊號可能增強；
若跌破重要支撐，風險訊號可能提高。
</div>
""", unsafe_allow_html=True)

st.warning(
    "重要：新聞、節目與論壇內容只作資訊索引，論壇意見不等於事實。"
    "本系統的偏多／觀望／偏空為資料訊號，不保證未來漲跌，也不構成個別投資建議。"
)
