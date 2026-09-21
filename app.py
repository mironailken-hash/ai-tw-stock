import streamlit as st
import pandas as pd
import numpy as np
import requests
import xml.etree.ElementTree as ET
import base64
from pathlib import Path
from datetime import date, timedelta
from urllib.parse import quote

st.set_page_config(page_title="KEN AI 台股智慧分析 V4 Premium", page_icon="📈", layout="wide")

API = "https://api.finmindtrade.com/api/v4/data"

def local_image_b64(path):
    try:
        return base64.b64encode(Path(path).read_bytes()).decode()
    except Exception:
        return ""

BANNER_B64 = local_image_b64("assets/banner.png")

# =========================
# 專業深色介面
# =========================
st.markdown("""
<style>
.stApp{
    background:
      radial-gradient(circle at 82% -5%,rgba(25,118,190,.25),transparent 28%),
      radial-gradient(circle at 13% 9%,rgba(229,190,72,.12),transparent 20%),
      radial-gradient(circle at 50% 110%,rgba(12,78,120,.15),transparent 34%),
      repeating-linear-gradient(90deg,rgba(255,255,255,.018) 0,rgba(255,255,255,.018) 1px,transparent 1px,transparent 74px),
      repeating-linear-gradient(0deg,rgba(255,255,255,.014) 0,rgba(255,255,255,.014) 1px,transparent 1px,transparent 74px),
      linear-gradient(135deg,#030914 0%,#071523 46%,#040b15 100%);
    color:#F4F7FB;
}
.block-container{max-width:1380px;padding-top:1.1rem;padding-bottom:3rem;}
[data-testid="stSidebar"]{background:#071321;border-right:1px solid #1d3148;}
[data-testid="stSidebar"] *{color:#F4F7FB;}

.hero{
    position:relative;overflow:hidden;
    background:
      radial-gradient(circle at 88% 25%,rgba(231,194,87,.17),transparent 18%),
      linear-gradient(110deg,rgba(12,31,51,.99),rgba(5,14,25,.99));
    border:1px solid rgba(223,188,86,.40);
    border-radius:22px;padding:25px 30px;margin-bottom:16px;
    box-shadow:0 22px 65px rgba(0,0,0,.38), inset 0 1px 0 rgba(255,255,255,.05);
}
.hero-title{font-size:34px;font-weight:900;letter-spacing:.5px;color:#fff;}
.hero-sub{color:#9db5ce;margin-top:4px;font-size:14px;}
.gold{color:#E9C65C;}

.panel{
    background:linear-gradient(145deg,rgba(15,31,51,.98),rgba(9,21,36,.98));
    border:1px solid #20354d;border-radius:18px;padding:20px;
    box-shadow:0 10px 30px rgba(0,0,0,.20);margin:8px 0 14px;
}
.action-title{font-size:31px;font-weight:900;margin:3px 0;}
.action-sub{font-size:15px;color:#a8bbcf;}
.price{font-size:31px;font-weight:900;}
.kicker{font-size:12px;color:#7f9ab7;letter-spacing:1.2px;font-weight:700;}
.level{font-size:22px;font-weight:850;color:#fff;}
.small{color:#94a9bf;font-size:13px;}
.decision{
    background:linear-gradient(115deg,rgba(13,35,57,.99),rgba(6,18,31,.99));
    border:1px solid rgba(223,188,86,.44);border-radius:20px;padding:22px 25px;
    box-shadow:0 18px 50px rgba(0,0,0,.32);margin:7px 0 16px;
}
.decision-grid{display:grid;grid-template-columns:1.45fr .55fr;gap:18px;align-items:center}
.decision-status{font-size:36px;font-weight:950;line-height:1.12;color:#fff}
.decision-score{text-align:right;font-size:42px;font-weight:950;color:#E9C65C}
.decision-note{font-size:16px;color:#c8d5e3;margin-top:8px}
.level-grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px;margin-top:17px}
.levelbox{background:rgba(3,12,22,.55);border:1px solid #263d56;border-radius:13px;padding:13px}
@media(max-width:800px){
 .decision-grid{grid-template-columns:1fr}.decision-score{text-align:left}
 .level-grid{grid-template-columns:1fr}.hero-title{font-size:27px}.decision-status{font-size:29px}
}

div[data-testid="stTextInput"] input,
div[data-testid="stNumberInput"] input{
    background:#0A1828 !important;
    color:#FFFFFF !important;
    border:1px solid #B99B43 !important;
    border-radius:10px !important;
    font-weight:700 !important;
}
div[data-testid="stTextInput"] input::placeholder{color:#C7D0DA !important;opacity:1 !important;}
div[data-testid="stTextInput"] input:focus,
div[data-testid="stNumberInput"] input:focus{
    border:1px solid #F0CC61 !important;
    box-shadow:0 0 0 2px rgba(240,204,97,.16) !important;
}
div[data-testid="stSelectbox"] > div > div{
    background:#0A1828 !important;color:#fff !important;border-color:#B99B43 !important;
}
.stButton > button{
    background:linear-gradient(90deg,#D4B14E,#F0D276) !important;
    color:#111820 !important;border:0 !important;border-radius:10px !important;
    font-weight:900 !important;min-height:45px;
}
.stButton > button:hover{box-shadow:0 0 20px rgba(233,198,92,.25);}
[data-testid="stMetric"]{
    background:#0A1828;border:1px solid #20354d;border-radius:13px;padding:12px;
}
[data-testid="stMetricValue"]{color:#fff;}
details{background:#091725;border:1px solid #20354d;border-radius:14px;padding:5px 12px;}
hr{border-color:#20354d;}
</style>
""", unsafe_allow_html=True)

# =========================
# 資料與計算
# =========================
def fm(dataset, sid, start, end, token=""):
    params={"dataset":dataset,"data_id":sid,"start_date":str(start),"end_date":str(end)}
    if token: params["token"]=token
    try:
        r=requests.get(API,params=params,timeout=18)
        j=r.json()
        if r.status_code==200 and j.get("status")==200:
            return pd.DataFrame(j.get("data",[]))
    except Exception:
        pass
    return pd.DataFrame()

@st.cache_data(ttl=86400)
def stock_table():
    rows=[]
    urls=[
        "https://openapi.twse.com.tw/v1/opendata/t187ap03_L",
        "https://openapi.twse.com.tw/v1/exchangeReport/STOCK_DAY_ALL"
    ]
    for url in urls:
        try:
            data=requests.get(url,timeout=12).json()
            for x in data:
                sid=str(x.get("公司代號",x.get("Code",""))).strip()
                name=str(x.get("公司簡稱",x.get("Name",""))).strip()
                if sid and name: rows.append((sid,name))
            if rows: break
        except Exception: pass
    return pd.DataFrame(rows,columns=["代號","名稱"]).drop_duplicates() if rows else pd.DataFrame(columns=["代號","名稱"])

def resolve_stock(q):
    q=q.strip()
    m=stock_table()
    if q.isdigit():
        hit=m[m["代號"]==q]
        return q,(str(hit.iloc[0]["名稱"]) if len(hit) else "")
    if len(m):
        hit=m[m["名稱"].str.contains(q,case=False,na=False)]
        if len(hit): return str(hit.iloc[0]["代號"]),str(hit.iloc[0]["名稱"])
    return "",q

def add_indicators(df):
    d=df.copy()
    d["date"]=pd.to_datetime(d["date"])
    d=d.sort_values("date")
    c=pd.to_numeric(d["close"],errors="coerce")
    for n in [5,10,20,60]:
        d[f"MA{n}"]=c.rolling(n).mean()
    delta=c.diff()
    gain=delta.clip(lower=0).rolling(14).mean()
    loss=(-delta.clip(upper=0)).rolling(14).mean()
    d["RSI"]=100-(100/(1+gain/loss.replace(0,np.nan)))
    d["MACD"]=c.ewm(span=12,adjust=False).mean()-c.ewm(span=26,adjust=False).mean()
    d["SIGNAL"]=d["MACD"].ewm(span=9,adjust=False).mean()
    return d

def score_trend(d, n):
    x=d.tail(n); r=x.iloc[-1]; s=50; close=float(r["close"])
    for ma,w in [("MA5",7),("MA20",11),("MA60",12)]:
        if pd.notna(r.get(ma)): s += w if close>float(r[ma]) else -w
    if pd.notna(r["RSI"]):
        if 50<=r["RSI"]<=70: s+=8
        elif r["RSI"]<40: s-=8
        elif r["RSI"]>78: s-=4
    if pd.notna(r["MACD"]) and pd.notna(r["SIGNAL"]):
        s += 9 if r["MACD"]>r["SIGNAL"] else -9
    if len(x)>5:
        s += 8 if close>float(x.iloc[0]["close"]) else -8
    return int(np.clip(s,0,100))

def trend_label(s):
    if s>=78:return "強勢偏多","🟢"
    if s>=60:return "偏多","🟢"
    if s>=42:return "觀望","🟡"
    if s>=25:return "偏空","🔴"
    return "強勢偏空","🔴"

def zh_institution_name(x):
    """FinMind 常見法人名稱轉繁體中文顯示。"""
    s=str(x)
    pairs={
        "Foreign_Investor":"外資",
        "Foreign_Dealer_Self":"外資自營商",
        "Investment_Trust":"投信",
        "Dealer_self":"自營商（自行買賣）",
        "Dealer_Hedging":"自營商（避險）",
        "Dealer":"自營商",
        "Foreign Investor":"外資",
        "Investment Trust":"投信",
        "Dealer self":"自營商（自行買賣）",
        "Dealer Hedging":"自營商（避險）",
    }
    return pairs.get(s,s)

def institutional_score(inst):
    if inst.empty:return 50,0
    buy=[c for c in inst.columns if "buy" in c.lower()]
    sell=[c for c in inst.columns if "sell" in c.lower()]
    if not buy or not sell:return 50,0
    b=inst[buy].apply(pd.to_numeric,errors="coerce").fillna(0).sum(axis=1)
    s=inst[sell].apply(pd.to_numeric,errors="coerce").fillna(0).sum(axis=1)
    net=float((b-s).tail(5).sum())
    return (72 if net>0 else 28 if net<0 else 50),net

def rss(q,n=5):
    try:
        url=f"https://news.google.com/rss/search?q={quote(q)}&hl=zh-TW&gl=TW&ceid=TW:zh-Hant"
        root=ET.fromstring(requests.get(url,timeout=12,headers={"User-Agent":"Mozilla/5.0"}).content)
        return [{"title":x.findtext("title",""),"link":x.findtext("link","")} for x in root.findall(".//item")[:n]]
    except Exception:return []

def attack_status(short, close, support, resistance, vol_ratio):
    breakout=max(resistance, close*1.015)
    pull_lo=support
    pull_hi=support*1.025
    weak=support*0.985
    if short>=78 and close>=resistance and vol_ratio>=1.2:
        return "🚀 短線進攻訊號成立","多方動能與突破條件較完整",breakout,pull_lo,pull_hi,weak
    if short>=65:
        return "🟢 等待突破進攻","偏多，但等待突破確認可降低假突破風險",breakout,pull_lo,pull_hi,weak
    if short>=42:
        return "🟡 觀望","短線條件尚未形成一致方向",breakout,pull_lo,pull_hi,weak
    if short>=25:
        return "⚠️ 轉弱警戒","短線結構偏弱，先等待重新站回關鍵區",breakout,pull_lo,pull_hi,weak
    return "🔴 短線弱勢","目前短線動能明顯偏弱",breakout,pull_lo,pull_hi,weak

# =========================
# Header + Search
# =========================
hero_bg = (
    f"linear-gradient(90deg,rgba(3,10,18,.98) 0%,rgba(3,10,18,.82) 42%,rgba(3,10,18,.18) 100%),"
    f"url('data:image/png;base64,{BANNER_B64}')"
    if BANNER_B64 else
    "linear-gradient(110deg,#07182a,#020811)"
)
st.markdown(f"""
<div class="hero" style="min-height:235px;background-image:{hero_bg};background-size:cover;background-position:center 27%;display:flex;align-items:center;">
  <div style="max-width:680px">
    <div class="kicker">TAIWAN EQUITY INTELLIGENCE TERMINAL</div>
    <div class="hero-title"><span class="gold">KEN AI</span> 台股智慧決策系統</div>
    <div class="hero-sub" style="font-size:16px;margin-top:10px">市場訊號 × 法人籌碼 × 趨勢結構 × 風險驗證</div>
    <div style="margin-top:18px;color:#e8d18b;font-weight:800">用條件確認趨勢，不用情緒猜行情</div>
  </div>
</div>
""",unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## 股票搜尋")
    q=st.text_input("輸入股票代號或名稱",value="",placeholder="例如：6213、聯茂、台積電")
    own=st.selectbox("持股狀態",["尚未持有","已持有"])
    cost=st.number_input("持有成本",min_value=0.0,value=0.0,step=0.5,disabled=own=="尚未持有")
    shares=st.number_input("持有股數",min_value=0,value=0,step=100,disabled=own=="尚未持有")
    try:
        token = st.secrets.get("FINMIND_TOKEN", "")
    except Exception:
        token = ""
    run=st.button("開始分析",use_container_width=True)

if not run:
    st.markdown("""
    <div class="panel">
      <div class="kicker">QUICK DECISION</div>
      <div class="action-title">搜尋一檔股票，先看短線總結</div>
      <div class="action-sub">首頁只保留：目前狀態、何時轉強、支撐壓力、短中長線。詳細模型收在展開面板內。</div>
    </div>
    """,unsafe_allow_html=True)
    st.stop()

if not q.strip():
    st.warning("請先輸入股票代號或名稱。")
    st.stop()

sid,name=resolve_stock(q)
if not sid:
    st.error("找不到股票名稱。請改輸入股票代號，例如 6213。")
    st.stop()

today=date.today()
price=fm("TaiwanStockPrice",sid,today-timedelta(days=330),today,token)
if price.empty:
    st.error("目前無法取得股價資料。請確認股票代號，或填入自己的 FinMind Token 後再試。")
    st.stop()

d=add_indicators(price)
r=d.iloc[-1]
close=float(r["close"])
prev=float(d.iloc[-2]["close"]) if len(d)>1 else close
chg=(close/prev-1)*100 if prev else 0
vols=pd.to_numeric(d["Trading_Volume"],errors="coerce")
vol=float(vols.iloc[-1])
avg20=float(vols.tail(20).mean()) if len(vols) else 0
vol_ratio=vol/avg20 if avg20 else 1

support=float(pd.to_numeric(d.tail(20)["min"],errors="coerce").min())
resistance=float(pd.to_numeric(d.tail(20)["max"],errors="coerce").max())
support60=float(pd.to_numeric(d.tail(60)["min"],errors="coerce").min())
resistance60=float(pd.to_numeric(d.tail(60)["max"],errors="coerce").max())

short=score_trend(d,10)
mid=score_trend(d,30)
long=score_trend(d,90)

inst=fm("TaiwanStockInstitutionalInvestorsBuySell",sid,today-timedelta(days=35),today,token)
inst_score,inst_net=institutional_score(inst)

heat=int(np.clip(50+(12 if abs(chg)>2 else 0)+(12 if vol_ratio>=1.2 else 0),0,100))
risk=int(np.clip(50+abs(chg)*4+(8 if pd.notna(r["RSI"]) and (r["RSI"]>75 or r["RSI"]<30) else 0),0,100))
tech=int(round(short*.5+mid*.3+long*.2))
overall=int(np.clip(round(tech*.58+inst_score*.27+heat*.15-(risk-50)*.08),0,100))
overall_label,overall_icon=trend_label(overall)

status,status_reason,breakout,pull_lo,pull_hi,weak=attack_status(short,close,support,resistance,vol_ratio)

# =========================
# 簡潔首頁
# =========================
st.markdown(f"## {sid} {name or q}")
m1,m2,m3,m4=st.columns(4)
m1.metric("最新收盤",f"{close:.2f}",f"{chg:+.2f}%")
m2.metric("短線強度",f"{short}/100")
m3.metric("量能比",f"{vol_ratio:.2f}x")
m4.metric("AI 綜合訊號",f"{overall}/100")

# 首屏決策卡：先回答「現在是否具備短線進攻條件」
if short >= 78 and close >= resistance and vol_ratio >= 1.2:
    verdict = "短線進攻訊號成立"
    verdict_note = "突破、量能與短線模型同時達標；仍需留意跌回突破區後的失效風險。"
elif short >= 65:
    verdict = "偏強｜等待突破確認"
    verdict_note = "方向偏多，但尚未同時滿足突破與量能確認，不把『偏多』直接當成已成立的進攻訊號。"
elif short >= 42:
    verdict = "觀望｜尚未形成進攻訊號"
    verdict_note = "短線訊號尚未形成一致優勢；等待突破或拉回止穩後再重新判讀。"
else:
    verdict = "弱勢｜尚未形成進攻訊號"
    verdict_note = "短線結構偏弱，優先等待趨勢修復，而不是追價。"

st.markdown(f"""
<div class="decision">
  <div style="display:inline-block;background:linear-gradient(90deg,#E8C35A,#F5DC8B);
    color:#08111D;padding:7px 14px;border-radius:8px;font-size:14px;font-weight:950;
    letter-spacing:.8px;box-shadow:0 0 20px rgba(232,195,90,.22);margin-bottom:12px">
    AI ACTION CENTER｜短線市場訊號
    </div>
  <div class="decision-grid">
    <div>
      <div class="decision-status">{verdict}</div>
      <div class="decision-note">{verdict_note}</div>
    </div>
    <div class="decision-score">{short}<span style="font-size:18px;color:#9db2c8"> / 100</span></div>
  </div>
  <div class="level-grid">
    <div class="levelbox"><div class="small">突破確認價</div><div class="level">{breakout:.2f}</div><div class="small">突破且量能同步增強，再重新確認進攻訊號</div></div>
    <div class="levelbox"><div class="small">拉回觀察區</div><div class="level">{pull_lo:.2f} ～ {pull_hi:.2f}</div><div class="small">回測止穩且技術轉強，可形成另一種轉強劇本</div></div>
    <div class="levelbox"><div class="small">轉弱警戒</div><div class="level">{weak:.2f}</div><div class="small">跌破後目前短線劇本失效，重新評估</div></div>
  </div>
</div>
""",unsafe_allow_html=True)

st.markdown("### 趨勢燈號")
c1,c2,c3=st.columns(3)
for col,title,score,period in zip([c1,c2,c3],["短線","中線","長線"],[short,mid,long],["1–10 交易日","2–6 週","1–6 個月"]):
    lab,ico=trend_label(score)
    with col:
        st.markdown(f"""<div class="panel"><div class="kicker">{period}</div>
        <div style="font-size:24px;font-weight:900">{ico} {title}｜{lab}</div>
        <div class="gold" style="font-size:25px;font-weight:900">{score}/100</div></div>""",unsafe_allow_html=True)

st.markdown("### 關鍵價位")
a,b,c,e=st.columns(4)
a.metric("20日支撐",f"{support:.2f}")
b.metric("20日壓力",f"{resistance:.2f}")
c.metric("60日支撐",f"{support60:.2f}")
e.metric("60日壓力",f"{resistance60:.2f}")

if own=="已持有" and cost>0:
    st.markdown("### 我的持股")
    a,b,c=st.columns(3)
    a.metric("成本",f"{cost:.2f}")
    b.metric("目前報酬",f"{(close/cost-1)*100:+.2f}%")
    c.metric("估算損益",f"{(close-cost)*shares:+,.0f} 元")

# =========================
# 專業資訊總覽（全部攤開，不使用下拉展開）
# =========================
st.markdown("### AI 模型面板")
st.markdown(f"""<div class="panel">
<div class="kicker">MODEL CONSENSUS</div>
<div style="font-size:26px;font-weight:900">{overall_icon} AI 綜合判斷：{overall}/100｜{overall_label}</div>
</div>""",unsafe_allow_html=True)
models=[("技術模型",tech),("法人籌碼",inst_score),("市場熱度",heat),("風險防守",100-risk)]
mc=st.columns(4)
for col,(title,score) in zip(mc,models):
    lab,ico=trend_label(score)
    with col:
        st.markdown(f"""<div class="panel"><div class="kicker">{title}</div>
        <div style="font-size:25px;font-weight:900">{score}/100</div>
        <div>{ico} {lab}</div></div>""",unsafe_allow_html=True)

st.markdown("### 價格趨勢")
st.line_chart(d.set_index("date")[["close","MA5","MA20","MA60"]].tail(120),use_container_width=True)

st.markdown("### 法人籌碼")
st.markdown(f"""<div class="panel"><div class="kicker">INSTITUTIONAL FLOW</div>
<div style="font-size:23px;font-weight:900">近 5 日法人代理淨額：{inst_net:,.0f}</div></div>""",unsafe_allow_html=True)
if not inst.empty:
    inst_show=inst.copy()
    if "name" in inst_show.columns:
        inst_show["name"]=inst_show["name"].map(zh_institution_name)
        inst_show=inst_show.rename(columns={"name":"法人名稱"})
    inst_show=inst_show.rename(columns={"date":"日期","buy":"買進","sell":"賣出"})
    wanted=[x for x in ["日期","法人名稱","買進","賣出"] if x in inst_show.columns]
    st.dataframe(inst_show[wanted].tail(20) if wanted else inst_show.tail(20),
                 use_container_width=True,hide_index=True)

st.markdown("### 公開市場情報")
news=rss(f"{sid} {name} 台股",6)
if news:
    for n in news: st.markdown(f"- [{n['title']}]({n['link']})")
else:
    st.caption("目前未取得相關公開新聞索引。")

n1,n2=st.columns(2)
with n1:
    st.markdown("#### 錢線百分百相關索引")
    tv=rss(f'"錢線百分百" {sid} {name}',4)
    if tv:
        for n in tv: st.markdown(f"- [{n['title']}]({n['link']})")
    else: st.caption("目前沒有近期相關公開索引。")
with n2:
    st.markdown("#### 股市爆料同學會相關索引")
    forum=rss(f'"股市爆料同學會" {sid} {name}',4)
    if forum:
        for n in forum: st.markdown(f"- [{n['title']}]({n['link']})")
    else: st.caption("目前沒有近期相關公開索引。")

st.markdown(f"""
<div class="panel">
<div class="kicker">FINAL SUMMARY｜市場總結</div>
<div style="font-size:25px;font-weight:900">{overall_icon} {overall_label}｜AI 訊號 {overall}/100</div>
<div class="action-sub">短線目前為「{status.replace("🚀 ","").replace("🟢 ","").replace("🟡 ","").replace("⚠️ ","").replace("🔴 ","")}」。
重點不是預測哪一天一定上漲，而是等待價格、量能與技術條件觸發後再更新訊號。</div>
</div>
""",unsafe_allow_html=True)

st.warning("本系統為市場研究與資訊整理工具。『進攻／觀望／轉弱』代表模型市場訊號，不保證未來漲跌，也不構成個別投資建議。")
