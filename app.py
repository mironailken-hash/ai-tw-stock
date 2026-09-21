import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import date, timedelta

st.set_page_config(page_title="AI 台股智慧分析", page_icon="📈", layout="wide")

st.markdown("""
<style>
.block-container {max-width: 1180px; padding-top: 1.5rem;}
div[data-testid="stMetric"] {background:#f7f7f8; border:1px solid #e7e7ea; padding:14px; border-radius:14px;}
.signal {padding:14px 18px; border-radius:14px; background:#f5f5f5; border:1px solid #ddd; margin:8px 0 18px;}
.small {color:#666; font-size:.9rem;}
</style>
""", unsafe_allow_html=True)

st.title("📈 AI 台股智慧分析")
st.caption("繁體中文版｜行情・技術面・法人籌碼・關鍵價位・持有情境")

with st.sidebar:
    st.header("分析條件")
    stock_id = st.text_input("股票代碼", "6213", help="例如：2330、6213")
    cost = st.number_input("我的持有成本（元）", min_value=0.0, value=0.0, step=1.0)
    horizon = st.radio("操作週期", ["短線（1～5個交易日）", "波段（約2～6週）", "中期（約1～3個月）"])
    token = st.text_input("FinMind Token（建議填入）", type="password", help="免費註冊後可取得 Token；不會顯示在畫面上。")
    run = st.button("開始分析", type="primary", use_container_width=True)

API = "https://api.finmindtrade.com/api/v4/data"

def get_data(dataset, stock, start, end, token=""):
    params = {"dataset": dataset, "data_id": stock, "start_date": start, "end_date": end}
    if token:
        params["token"] = token
    r = requests.get(API, params=params, timeout=20)
    r.raise_for_status()
    payload = r.json()
    if payload.get("status") != 200:
        raise RuntimeError(payload.get("msg", "資料來源回傳錯誤"))
    return pd.DataFrame(payload.get("data", []))

def calc_indicators(df):
    d = df.copy().sort_values("date")
    d["close"] = pd.to_numeric(d["close"], errors="coerce")
    d["max"] = pd.to_numeric(d["max"], errors="coerce")
    d["min"] = pd.to_numeric(d["min"], errors="coerce")
    d["Trading_Volume"] = pd.to_numeric(d["Trading_Volume"], errors="coerce")
    for n in [5,10,20,60]:
        d[f"MA{n}"] = d["close"].rolling(n).mean()

    delta = d["close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    d["RSI14"] = 100 - (100 / (1 + rs))

    low9 = d["min"].rolling(9).min()
    high9 = d["max"].rolling(9).max()
    rsv = (d["close"] - low9) / (high9 - low9).replace(0, np.nan) * 100
    d["K"] = rsv.ewm(com=2, adjust=False).mean()
    d["D"] = d["K"].ewm(com=2, adjust=False).mean()

    ema12 = d["close"].ewm(span=12, adjust=False).mean()
    ema26 = d["close"].ewm(span=26, adjust=False).mean()
    d["MACD"] = ema12 - ema26
    d["MACD_SIGNAL"] = d["MACD"].ewm(span=9, adjust=False).mean()
    return d

def support_resistance(d):
    recent20 = d.tail(20)
    recent60 = d.tail(60)
    support1 = recent20["min"].min()
    support2 = recent60["min"].min()
    resist1 = recent20["max"].max()
    resist2 = recent60["max"].max()
    return support1, support2, resist1, resist2

def trend_score(row):
    score, reasons = 0, []
    for ma in ["MA5","MA10","MA20"]:
        if pd.notna(row[ma]):
            if row["close"] > row[ma]:
                score += 1; reasons.append(f"股價站上 {ma.replace('MA','')} 日均線")
            else:
                score -= 1; reasons.append(f"股價低於 {ma.replace('MA','')} 日均線")
    if pd.notna(row["RSI14"]):
        if 50 <= row["RSI14"] <= 70:
            score += 1; reasons.append("RSI 位於偏強區")
        elif row["RSI14"] < 40:
            score -= 1; reasons.append("RSI 偏弱")
        elif row["RSI14"] > 75:
            reasons.append("RSI 偏高，留意短線過熱")
    if pd.notna(row["MACD"]) and pd.notna(row["MACD_SIGNAL"]):
        if row["MACD"] > row["MACD_SIGNAL"]:
            score += 1; reasons.append("MACD 動能偏正向")
        else:
            score -= 1; reasons.append("MACD 動能偏弱")
    signal = "偏多 🟢" if score >= 3 else ("偏弱 🔴" if score <= -2 else "中性觀察 🟡")
    return signal, score, reasons

if run:
    if not stock_id.isdigit():
        st.error("請輸入台股數字代碼，例如 2330 或 6213。")
        st.stop()

    end = date.today()
    start = end - timedelta(days=180)
    try:
        price = get_data("TaiwanStockPrice", stock_id, start.isoformat(), end.isoformat(), token)
        if price.empty:
            st.error("找不到這檔股票的行情資料，請確認股票代碼或 Token。")
            st.stop()

        d = calc_indicators(price)
        last = d.iloc[-1]
        prev = d.iloc[-2] if len(d) > 1 else last
        change = last["close"] - prev["close"]
        pct = (change / prev["close"] * 100) if prev["close"] else 0
        s1, s2, r1, r2 = support_resistance(d)
        signal, score, reasons = trend_score(last)

        st.subheader(f"股票代碼：{stock_id}")
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("最新收盤", f"{last['close']:.2f} 元", f"{change:+.2f}（{pct:+.2f}%）")
        c2.metric("成交量", f"{last['Trading_Volume']/1000:,.0f} 張")
        c3.metric("5 日均線", f"{last['MA5']:.2f}" if pd.notna(last["MA5"]) else "—")
        c4.metric("20 日均線", f"{last['MA20']:.2f}" if pd.notna(last["MA20"]) else "—")

        st.markdown(f'<div class="signal"><b>今日技術訊號：{signal}</b><br><span class="small">此為規則式市場狀態整理，不代表保證獲利或個人化買賣指示。</span></div>', unsafe_allow_html=True)

        a,b = st.columns(2)
        with a:
            st.markdown("### 🎯 關鍵價位")
            st.write(f"**第一壓力區參考：** {r1:.2f} 元")
            st.write(f"**較長區間壓力參考：** {r2:.2f} 元")
            st.write(f"**第一支撐區參考：** {s1:.2f} 元")
            st.write(f"**較長區間防守參考：** {s2:.2f} 元")
            st.caption("價位以近 20 / 60 個交易日高低點簡化計算，後續可升級成樞紐點、成交密集區與波動度模型。")
        with b:
            st.markdown("### 📊 技術指標")
            st.write(f"RSI(14)：**{last['RSI14']:.1f}**" if pd.notna(last["RSI14"]) else "RSI(14)：—")
            st.write(f"KD：K **{last['K']:.1f}** / D **{last['D']:.1f}**" if pd.notna(last["K"]) else "KD：—")
            st.write(f"MACD：**{last['MACD']:.2f}**" if pd.notna(last["MACD"]) else "MACD：—")
            st.write(f"60 日均線：**{last['MA60']:.2f}**" if pd.notna(last["MA60"]) else "60 日均線：—")

        if cost > 0:
            pnl = (last["close"] / cost - 1) * 100
            st.markdown("### 💼 我的持有狀況")
            x,y,z = st.columns(3)
            x.metric("持有成本", f"{cost:.2f} 元")
            y.metric("目前價格", f"{last['close']:.2f} 元")
            z.metric("價格相對成本", f"{pnl:+.2f}%")

        st.markdown("### 🤖 中文綜合摘要")
        direction = "目前技術結構相對偏強" if score >= 3 else ("目前技術結構相對偏弱" if score <= -2 else "目前技術結構較為中性")
        st.write(f"**{direction}。** 你選擇的觀察週期為「{horizon}」。")
        st.write("主要依據：" + "；".join(reasons[:6]) + "。")
        st.write(f"短期可優先觀察 **{s1:.2f} 元附近支撐** 與 **{r1:.2f} 元附近壓力**。若價格有效突破或跌破關鍵區域，應搭配成交量與後續數日走勢重新判讀，而不是只依單一價位決定。")

        st.markdown("### 📈 近 90 個交易日")
        chart = d.tail(90).set_index("date")[["close","MA5","MA20","MA60"]].rename(columns={
            "close":"收盤價","MA5":"5日線","MA20":"20日線","MA60":"60日線"
        })
        st.line_chart(chart)

        st.markdown("### 🏦 法人籌碼")
        try:
            inst = get_data("TaiwanStockInstitutionalInvestorsBuySell", stock_id, (end-timedelta(days=30)).isoformat(), end.isoformat(), token)
            if not inst.empty:
                inst["buy"] = pd.to_numeric(inst["buy"], errors="coerce")
                inst["sell"] = pd.to_numeric(inst["sell"], errors="coerce")
                inst["買賣超(張)"] = (inst["buy"] - inst["sell"]) / 1000
                latest_date = inst["date"].max()
                show = inst[inst["date"] == latest_date][["name","買賣超(張)"]].rename(columns={"name":"法人"})
                st.dataframe(show, use_container_width=True, hide_index=True)
            else:
                st.info("目前未取得法人資料。")
        except Exception:
            st.info("法人資料目前未取得；不影響技術面分析。")

        st.warning("⚠️ 本工具為資訊整理與研究用途。支撐、壓力、技術指標及趨勢訊號皆可能失效，不能保證未來價格或投資結果。")

    except Exception as e:
        st.error(f"讀取資料失敗：{e}")
        st.info("請確認網路連線、股票代碼，以及 FinMind Token 是否有效。")
else:
    st.info("👈 請在左側輸入股票代碼與條件，再按「開始分析」。")
    st.markdown("""
### 第一版功能
- 台股行情與成交量
- 5 / 10 / 20 / 60 日均線
- RSI、KD、MACD
- 近 20 / 60 日支撐與壓力
- 個人成本相對價格
- 短線／波段／中期觀察情境
- 三大法人資料（資料來源支援時）
- 全繁體中文分析摘要

下一版可加入：**台指期夜盤、重大新聞、融資融券、大盤環境、台積電 ADR、AI 新聞摘要與自選股清單**。
""")
