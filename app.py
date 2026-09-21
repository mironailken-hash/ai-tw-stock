import streamlit as st
import pandas as pd
import numpy as np
import requests
import xml.etree.ElementTree as ET
from datetime import date, timedelta
from urllib.parse import quote

st.set_page_config(page_title="KEN AI 台股智慧分析 V3",page_icon="🧧",layout="wide",initial_sidebar_state="expanded")

st.markdown("""
<style>
.stApp{background:radial-gradient(circle at 85% 8%,rgba(212,175,55,.13),transparent 25%),linear-gradient(135deg,#06101d,#0b192b 55%,#07111e);color:#eaf2ff}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#071321,#0d1a2c);border-right:1px solid #20324a}
[data-testid="stSidebar"] *{color:#eaf2ff}.block-container{max-width:1450px;padding-top:1rem}
.hero{padding:25px 30px;border:1px solid rgba(212,175,55,.38);border-radius:22px;background:linear-gradient(120deg,#10213a,#081322);box-shadow:0 18px 55px #0005;margin-bottom:18px}
.hero-title{font-size:38px;font-weight:900;color:#fff}.gold{color:#f3cf62}.sub{color:#8fbce8}
.card,.model{background:linear-gradient(145deg,rgba(17,33,55,.96),rgba(9,21,37,.96));border:1px solid #263951;border-radius:18px;padding:18px;margin:8px 0;box-shadow:0 10px 28px #0004}
.model{border-radius:14px;padding:15px}.signal{font-size:34px;font-weight:900}.muted{color:#8fa8c5}
</style>
""",unsafe_allow_html=True)

API="https://api.finmindtrade.com/api/v4/data"

def fm(dataset,sid,start,end,token=""):
    p={"dataset":dataset,"data_id":sid,"start_date":str(start),"end_date":str(end)}
    if token:p["token"]=token
    try:
        j=requests.get(API,params=p,timeout=18).json()
        return pd.DataFrame(j.get("data",[])) if j.get("status")==200 else pd.DataFrame()
    except:return pd.DataFrame()

@st.cache_data(ttl=86400)
def stocks():
    rows=[]
    for u in ["https://openapi.twse.com.tw/v1/opendata/t187ap03_L","https://openapi.twse.com.tw/v1/exchangeReport/STOCK_DAY_ALL"]:
        try:
            for x in requests.get(u,timeout=12).json():
                sid=str(x.get("公司代號",x.get("Code",""))).strip()
                name=str(x.get("公司簡稱",x.get("Name",""))).strip()
                if sid and name:rows.append((sid,name))
            if rows:break
        except:pass
    return pd.DataFrame(rows,columns=["代號","名稱"]).drop_duplicates() if rows else pd.DataFrame(columns=["代號","名稱"])

def resolve(q):
    q=q.strip(); m=stocks()
    if q.isdigit():
        h=m[m["代號"]==q]
        return q,(h.iloc[0]["名稱"] if len(h) else "")
    if len(m):
        h=m[m["名稱"].str.contains(q,case=False,na=False)]
        if len(h):return str(h.iloc[0]["代號"]),str(h.iloc[0]["名稱"])
    return "",q

def ind(df):
    d=df.copy();d["date"]=pd.to_datetime(d["date"]);d=d.sort_values("date")
    c=pd.to_numeric(d["close"],errors="coerce")
    for n in [5,10,20,60]:d[f"MA{n}"]=c.rolling(n).mean()
    z=c.diff();g=z.clip(lower=0).rolling(14).mean();l=(-z.clip(upper=0)).rolling(14).mean()
    d["RSI"]=100-100/(1+g/l.replace(0,np.nan))
    lo=pd.to_numeric(d["min"],errors="coerce").rolling(9).min();hi=pd.to_numeric(d["max"],errors="coerce").rolling(9).max()
    rsv=(c-lo)/(hi-lo).replace(0,np.nan)*100;d["K"]=rsv.ewm(alpha=1/3,adjust=False).mean();d["D"]=d["K"].ewm(alpha=1/3,adjust=False).mean()
    d["MACD"]=c.ewm(span=12,adjust=False).mean()-c.ewm(span=26,adjust=False).mean()
    d["SIGNAL"]=d["MACD"].ewm(span=9,adjust=False).mean()
    return d

def tscore(d,n):
    x=d.tail(n);r=x.iloc[-1];s=50;c=float(r["close"])
    for ma,w in [("MA5",6),("MA20",10),("MA60",12)]:
        if pd.notna(r.get(ma)):s+=w if c>r[ma] else -w
    if pd.notna(r["RSI"]):
        s+=8 if 50<=r["RSI"]<=70 else (-8 if r["RSI"]<40 else (-3 if r["RSI"]>78 else 0))
    if pd.notna(r["MACD"]) and pd.notna(r["SIGNAL"]):s+=8 if r["MACD"]>r["SIGNAL"] else -8
    if len(x)>5:s+=8 if c>float(x.iloc[0]["close"]) else -8
    return int(np.clip(s,0,100))

def label(s):
    if s>=78:return "強勢偏多","🟢"
    if s>=60:return "偏多","🟢"
    if s>=42:return "中性","🟡"
    if s>=25:return "偏空","🔴"
    return "強勢偏空","🔴"

def rss(q,n=6):
    try:
        u=f"https://news.google.com/rss/search?q={quote(q)}&hl=zh-TW&gl=TW&ceid=TW:zh-Hant"
        root=ET.fromstring(requests.get(u,timeout=12,headers={"User-Agent":"Mozilla/5.0"}).content)
        return [{"title":x.findtext("title",""),"link":x.findtext("link","")} for x in root.findall(".//item")[:n]]
    except:return []

def institutional_score(x):
    if x.empty:return 50,0
    buy=[c for c in x if "buy" in c.lower()];sell=[c for c in x if "sell" in c.lower()]
    if not buy or not sell:return 50,0
    b=x[buy].apply(pd.to_numeric,errors="coerce").fillna(0).sum(axis=1)
    s=x[sell].apply(pd.to_numeric,errors="coerce").fillna(0).sum(axis=1)
    net=float((b-s).tail(5).sum())
    return (72 if net>0 else 28 if net<0 else 50),net

st.markdown("""<div class="hero"><div class="hero-title">🧧 <span class="gold">財神金庫</span>　KEN AI 台股智慧分析 V3　🪙 🪙 🪙</div>
<div class="sub">TAIWAN STOCK INTELLIGENCE TERMINAL ｜ 技術 × 籌碼 × 新聞 × 風險 × 多模型共識</div></div>""",unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## 🔎 智慧搜尋")
    q=st.text_input("股票代號或名稱",value="6213",placeholder="6213、聯茂、台積電")
    own=st.radio("持股狀態",["尚未持有","已持有"])
    cost=st.number_input("持有成本（元）",min_value=0.0,value=0.0,step=0.5,disabled=own=="尚未持有")
    shares=st.number_input("持有股數",min_value=0,value=0,step=100,disabled=own=="尚未持有")
    token=st.text_input("FinMind Token（建議填入）",type="password")
    run=st.button("🚀 啟動 V3 多模型分析",use_container_width=True)

if not run:
    st.markdown("""<div class="card"><h2>🧠 AI MULTI-MODEL ANALYSIS</h2>
    <p>輸入股票代號或名稱，分析短／中／長線、技術、法人籌碼、新聞索引、量價熱度與風險。</p>
    <p>🟢 強勢偏多　🟢 偏多　🟡 中性　🔴 偏空　🔴 強勢偏空</p></div>""",unsafe_allow_html=True)
    st.stop()

sid,name=resolve(q)
if not sid:
    st.error("找不到股票名稱對應代號，請改用股票代號。");st.stop()

today=date.today();p=fm("TaiwanStockPrice",sid,today-timedelta(days=330),today,token)
if p.empty:
    st.error("抓不到股價資料。請確認代號，或輸入自己的 FinMind Token。");st.stop()

d=ind(p);r=d.iloc[-1];close=float(r["close"]);prev=float(d.iloc[-2]["close"]);chg=(close/prev-1)*100
v=pd.to_numeric(d["Trading_Volume"],errors="coerce");vol=float(v.iloc[-1])
low20=float(pd.to_numeric(d.tail(20)["min"]).min());high20=float(pd.to_numeric(d.tail(20)["max"]).max())
low60=float(pd.to_numeric(d.tail(60)["min"]).min());high60=float(pd.to_numeric(d.tail(60)["max"]).max())

inst=fm("TaiwanStockInstitutionalInvestorsBuySell",sid,today-timedelta(days=35),today,token)
isc,inet=institutional_score(inst)
short,mid,long=tscore(d,10),tscore(d,30),tscore(d,90)
tech=int(round(short*.45+mid*.35+long*.20))
heat=int(np.clip(50+(12 if abs(chg)>2 else 0)+(12 if vol>v.tail(20).mean() else 0),0,100))
risk=int(np.clip(50+abs(chg)*4+(8 if pd.notna(r["RSI"]) and (r["RSI"]>75 or r["RSI"]<30) else 0),0,100))
# 未接可靠結構化來源的模型保持中性，避免製造假分數
basic=newss=globalm=50
overall=int(np.clip(round(tech*.42+isc*.25+heat*.13+basic*.08+newss*.07+globalm*.05-(risk-50)*.10),0,100))
olab,oico=label(overall)

st.markdown(f"## {sid} {name or q}　｜　{pd.to_datetime(r['date']).date()}")
a,b,c,e=st.columns(4)
a.metric("最新收盤",f"{close:.2f}",f"{chg:+.2f}%")
b.metric("RSI(14)",f"{float(r['RSI']):.1f}" if pd.notna(r["RSI"]) else "-")
c.metric("法人近5日代理淨額",f"{inet:,.0f}")
e.metric("綜合模型",f"{overall}/100")

st.markdown(f"""<div class="card"><div class="muted">MODEL CONSENSUS｜多模型決策中心</div>
<div class="signal">{oico} {olab}　<span class="gold">{overall}/100</span></div>
<div>訊號明確，但不把尚未取得的資料假裝成已分析結果。</div></div>""",unsafe_allow_html=True)

models=[("📈 技術模型",tech),("🏦 法人籌碼模型",isc),("💰 基本面模型*",basic),("📰 新聞情緒模型*",newss),("🌎 國際市場模型*",globalm),("🔥 市場熱度模型",heat)]
cols=st.columns(3)
for i,(n,s) in enumerate(models):
    la,ic=label(s)
    with cols[i%3]:
        st.markdown(f'<div class="model"><b>{n}</b><br><span style="font-size:28px;font-weight:900">{s}/100</span><br>{ic} {la}</div>',unsafe_allow_html=True)
        st.progress(s/100)
st.caption("* 尚未串接可靠結構化來源的模型維持 50 中性，不用猜測製造分數。")

st.markdown("### ⏱️ 短／中／長線")
cols=st.columns(3)
for col,title,s,period in zip(cols,["短線","中線","長線"],[short,mid,long],["1–10 個交易日","2–6 週","1–6 個月"]):
    la,ic=label(s)
    with col:st.markdown(f'<div class="card"><span class="muted">{period}</span><h2>{title}：{ic} {la}</h2><span class="gold" style="font-size:28px;font-weight:900">{s}/100</span></div>',unsafe_allow_html=True)

st.markdown("### 🎯 關鍵價位")
x1,x2,x3,x4=st.columns(4)
x1.metric("20日重要支撐",f"{low20:.2f}");x2.metric("20日主要壓力",f"{high20:.2f}")
x3.metric("60日重要支撐",f"{low60:.2f}");x4.metric("60日主要壓力",f"{high60:.2f}")
if overall>=60:st.info(f"目前偏多；若有效跌破 {low20:.2f}，多方訊號需要重新評估。")
elif overall<=41:st.warning(f"目前偏空；若有效突破 {high20:.2f}，偏空訊號需要重新評估。")
else:st.info(f"目前中性；優先觀察 {low20:.2f}～{high20:.2f} 哪一側先有效突破。")

if own=="已持有" and cost>0:
    pnl=(close/cost-1)*100
    st.markdown("### 💼 持股面板")
    a,b,c=st.columns(3);a.metric("成本",f"{cost:.2f}");b.metric("未實現報酬率",f"{pnl:+.2f}%");c.metric("估算損益",f"{(close-cost)*shares:+,.0f} 元")

st.markdown("### 📊 120 日價格與均線")
st.line_chart(d.set_index("date")[["close","MA5","MA20","MA60"]].tail(120),use_container_width=True)

st.markdown("### 📰 公開新聞情報")
news=rss(f"{sid} {name} 台股",7)
if news:
    for n in news:st.markdown(f"- [{n['title']}]({n['link']})")
else:st.caption("目前未取得相關公開新聞索引。")

c1,c2=st.columns(2)
with c1:
    st.markdown("#### 📺 錢線百分百")
    z=rss(f'"錢線百分百" {sid} {name}',4)
    if z:
        for n in z:st.markdown(f"- [{n['title']}]({n['link']})")
    else:st.caption("目前沒有近期相關公開索引。")
with c2:
    st.markdown("#### 💬 股市爆料同學會")
    z=rss(f'"股市爆料同學會" {sid} {name}',4)
    if z:
        for n in z:st.markdown(f"- [{n['title']}]({n['link']})")
    else:st.caption("目前沒有近期相關公開索引。")

reasons=[]
if tech>=60:reasons.append("技術結構偏多")
elif tech<=41:reasons.append("技術結構偏空")
if isc>=60:reasons.append("法人籌碼偏正向")
elif isc<=41:reasons.append("法人籌碼偏弱")
if heat>=60:reasons.append("量價熱度提高")
reason="、".join(reasons) if reasons else "主要模型目前沒有形成高度一致方向"

st.markdown(f"""### ⚡ 犀利結論
<div class="card"><h2>{oico} {olab}｜{overall}/100</h2>
<b>核心理由：</b>{reason}。<br><br>
<b>短線：</b>{label(short)[1]} {label(short)[0]}　｜　
<b>中線：</b>{label(mid)[1]} {label(mid)[0]}　｜　
<b>長線：</b>{label(long)[1]} {label(long)[0]}<br><br>
<b>關鍵區間：</b>{low20:.2f} ～ {high20:.2f}</div>""",unsafe_allow_html=True)

st.warning("本系統為市場研究與資訊整理工具。偏多／中性／偏空是模型市場訊號，不保證未來漲跌，也不構成個別投資建議。新聞、節目及論壇為公開索引，論壇意見不等同事實。")
