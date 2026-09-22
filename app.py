import streamlit as st
import requests
import pandas as pd
import numpy as np
import re
import json
import plotly.graph_objects as go
SKLEARN_IMPORT_ERROR = ""
try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import brier_score_loss
    SKLEARN_OK = True
except Exception as _skerr:
    SKLEARN_OK = False
    SKLEARN_IMPORT_ERROR = f"{type(_skerr).__name__}: {_skerr}"

import xml.etree.ElementTree as ET
import base64
from pathlib import Path
from datetime import date, timedelta
from urllib.parse import quote

st.set_page_config(page_title="KEN AI 百億台股智慧決策系統", page_icon="📈", layout="wide")

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
.block-container{max-width:1380px;padding-top:.55rem;padding-bottom:3rem;}

/* Streamlit 系統頂部列：移除白色背景與多餘留白 */
header[data-testid="stHeader"]{
    background:transparent !important;
    height:0 !important;
}
[data-testid="stToolbar"]{
    visibility:hidden !important;
    height:0 !important;
}
[data-testid="stDecoration"]{display:none !important;}
#MainMenu{visibility:hidden !important;}
footer{visibility:hidden !important;}
.stApp > header{background:transparent !important;}

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

/* ===== V4.4 深色金融資訊區 ===== */
.section-pro{
    background:
      radial-gradient(circle at 92% 0%,rgba(31,113,170,.12),transparent 28%),
      linear-gradient(145deg,rgba(8,26,43,.98),rgba(4,15,27,.98));
    border:1px solid rgba(221,183,68,.72);
    border-radius:18px;
    padding:16px 20px 12px 20px;
    margin:20px 0 8px 0;
    box-shadow:0 14px 35px rgba(0,0,0,.25), inset 0 1px 0 rgba(255,255,255,.025);
}
.section-pro-title{
    color:#F1CB5C;
    font-size:24px;
    font-weight:950;
    letter-spacing:.4px;
}
.section-pro-sub{
    color:#8FA9C1;
    font-size:13px;
    margin-top:3px;
}

/* 圖表外框與底色 */
[data-testid="stArrowVegaLiteChart"],
[data-testid="stVegaLiteChart"],
[data-testid="stLineChart"]{
    background:linear-gradient(145deg,#071827,#04111E) !important;
    border:1px solid #1D3A53 !important;
    border-radius:16px !important;
    padding:12px !important;
    box-shadow:0 12px 30px rgba(0,0,0,.22) !important;
}

/* Dataframe 外層深色 */
[data-testid="stDataFrame"],
[data-testid="stDataFrameResizable"]{
    background:#061522 !important;
    border:1px solid #1D3A53 !important;
    border-radius:16px !important;
    overflow:hidden !important;
    box-shadow:0 12px 30px rgba(0,0,0,.22) !important;
}

/* 避免 Streamlit 元件容器出現突兀白底 */
[data-testid="stElementContainer"]{
    background-color:transparent;
}


/* ===== V4.5 真正全深色圖表與法人表格 ===== */
[data-testid="stPlotlyChart"]{
    background:linear-gradient(145deg,#071827,#04111E) !important;
    border:1px solid #1D3A53 !important;
    border-radius:16px !important;
    padding:10px !important;
    box-shadow:0 12px 30px rgba(0,0,0,.22) !important;
}
.inst-table-wrap{
    width:100%;
    overflow-x:auto;
    background:linear-gradient(145deg,#071827,#04111E);
    border:1px solid #1D3A53;
    border-radius:16px;
    box-shadow:0 12px 30px rgba(0,0,0,.22);
}
.inst-table{
    width:100%;
    border-collapse:collapse;
    color:#DDE8F2;
    font-size:14px;
}
.inst-table th{
    background:#0B2033;
    color:#F0C95C;
    text-align:left;
    padding:13px 14px;
    border-bottom:1px solid #29445D;
    white-space:nowrap;
}
.inst-table td{
    padding:12px 14px;
    border-bottom:1px solid rgba(75,111,140,.20);
    background:rgba(6,21,34,.72);
    white-space:nowrap;
}
.inst-table tr:hover td{background:#0B2134;}
.inst-table tr:last-child td{border-bottom:0;}
.tw-red{color:#FF6666;font-weight:850;}
.tw-green{color:#45D483;font-weight:850;}
.muted{color:#A4B5C5;font-weight:750;}


/* ===== V4.6 Mobile Responsive ===== */
@media (max-width: 768px) {
  .block-container{
    max-width:100% !important;
    padding-left:.65rem !important;
    padding-right:.65rem !important;
    padding-top:.35rem !important;
  }

  /* Sidebar remains usable on mobile */
  [data-testid="stSidebar"]{
    min-width:82vw !important;
    max-width:82vw !important;
  }

  /* Stack Streamlit columns vertically */
  [data-testid="stHorizontalBlock"]{
    flex-direction:column !important;
    gap:.55rem !important;
  }
  [data-testid="column"]{
    width:100% !important;
    flex:1 1 100% !important;
    min-width:100% !important;
  }

  .hero{
    min-height:185px !important;
    padding:20px 16px !important;
    border-radius:14px !important;
    background-position:62% center !important;
  }
  .hero-title{
    font-size:28px !important;
    line-height:1.15 !important;
    max-width:82% !important;
  }
  .hero-sub{
    font-size:12px !important;
    line-height:1.5 !important;
    max-width:78% !important;
  }
  .kicker{font-size:10px !important;}

  .panel,
  .section-pro{
    padding:14px !important;
    border-radius:14px !important;
  }
  .section-pro-title{font-size:20px !important;}

  /* Avoid giant metric text on small screens */
  [data-testid="stMetricValue"]{
    font-size:28px !important;
  }

  /* Plotly responsive height/width */
  [data-testid="stPlotlyChart"]{
    width:100% !important;
    overflow:hidden !important;
    padding:4px !important;
  }
  [data-testid="stPlotlyChart"] > div{
    width:100% !important;
  }

  /* Institutional table can scroll horizontally without breaking page */
  .inst-table-wrap{
    width:100% !important;
    overflow-x:auto !important;
    -webkit-overflow-scrolling:touch;
  }
  .inst-table{
    min-width:690px !important;
    font-size:12px !important;
  }
  .inst-table th,
  .inst-table td{
    padding:9px 10px !important;
  }

  /* Inputs/buttons larger for touch */
  .stButton > button{
    min-height:46px !important;
    font-size:15px !important;
  }
  div[data-baseweb="input"] input{
    font-size:16px !important;
  }

  h1{font-size:28px !important;}
  h2{font-size:24px !important;}
  h3{font-size:20px !important;}
}


/* V4.7 手機快速重新搜尋 */
div[data-testid="stForm"]{
    background:linear-gradient(145deg,#081827,#06121F);
    border:1px solid rgba(221,183,68,.58);
    border-radius:14px;
    padding:10px 12px 4px 12px;
    margin:8px 0 16px 0;
}
@media (max-width:768px){
    div[data-testid="stForm"]{
        position:sticky;
        top:6px;
        z-index:999;
        box-shadow:0 10px 28px rgba(0,0,0,.38);
    }
    div[data-testid="stForm"] [data-testid="stHorizontalBlock"]{
        flex-direction:row !important;
        align-items:end !important;
        gap:8px !important;
    }
    div[data-testid="stForm"] [data-testid="column"]:first-child{
        width:72% !important;
        min-width:72% !important;
        flex:0 0 72% !important;
    }
    div[data-testid="stForm"] [data-testid="column"]:last-child{
        width:28% !important;
        min-width:28% !important;
        flex:0 0 28% !important;
    }
}


/* ===== V5 主畫面搜尋 ===== */
.search-title-box{
    margin-top:14px;
    margin-bottom:6px;
}
.search-main-title{
    color:#F3D36C;
    font-size:24px;
    font-weight:950;
}
div[data-testid="stForm"]{
    position:relative !important;
    top:auto !important;
    z-index:auto !important;
    background:linear-gradient(145deg,#081827,#06121F) !important;
    border:1px solid rgba(221,183,68,.72) !important;
    border-radius:15px !important;
    padding:12px !important;
    margin:0 0 18px 0 !important;
}
div[data-baseweb="input"] > div{
    background:#071522 !important;
    border:1px solid #D9B84E !important;
}
div[data-baseweb="input"] input{
    color:#FFFFFF !important;
    -webkit-text-fill-color:#FFFFFF !important;
    font-size:16px !important;
}
div[data-baseweb="input"] input::placeholder{
    color:#AFC0D0 !important;
    opacity:1 !important;
}
@media (max-width:768px){
    .hero{margin-bottom:8px !important;}
    .search-main-title{font-size:21px !important;}
    div[data-testid="stForm"]{
        position:relative !important;
        top:auto !important;
        width:100% !important;
    }
    div[data-testid="stForm"] [data-testid="stHorizontalBlock"]{
        flex-direction:column !important;
    }
}


/* ===== V5.1 精簡專業版 ===== */
.data-badge{
    display:inline-block;
    padding:5px 10px;
    border:1px solid rgba(221,183,68,.55);
    border-radius:999px;
    background:#071522;
    color:#DCC66A;
    font-size:12px;
    margin:4px 4px 4px 0;
}
@media(max-width:768px){
    .data-badge{font-size:11px;padding:4px 8px;}
}


/* ===== V5.2 搜尋按鈕黑金修正 ===== */
div[data-testid="stForm"] button,
div[data-testid="stForm"] button[kind="secondaryFormSubmit"],
div[data-testid="stForm"] button[kind="primaryFormSubmit"]{
    background:linear-gradient(135deg,#C69B2D,#F1D56A) !important;
    color:#071522 !important;
    border:1px solid #E7C75B !important;
    border-radius:10px !important;
    font-weight:900 !important;
    min-height:44px !important;
    box-shadow:0 5px 16px rgba(0,0,0,.28) !important;
}
div[data-testid="stForm"] button:hover{
    background:linear-gradient(135deg,#E0B63E,#FFE58A) !important;
    color:#020B12 !important;
    border-color:#FFE58A !important;
}
div[data-testid="stForm"] button:focus,
div[data-testid="stForm"] button:active{
    background:linear-gradient(135deg,#B98B22,#E8C858) !important;
    color:#071522 !important;
    border-color:#F4D66A !important;
}
div[data-testid="stForm"] button p{
    color:#071522 !important;
    font-weight:900 !important;
}


.top-signature{
 width:100%;text-align:right;color:#D8C47A;font-size:11px;
 letter-spacing:.08em;font-weight:650;opacity:.88;padding:2px 4px 7px;
}
@media(max-width:768px){
 .top-signature{text-align:center;font-size:10px;letter-spacing:.04em;padding-bottom:6px;}
}


/* ===== V6 LIVE PRICE + DAY TRADE RADAR ===== */
.v6-live-grid{
 display:grid;grid-template-columns:1fr 1.45fr;gap:12px;margin:10px 0 14px;
}
.v6-live-card{
 background:linear-gradient(145deg,#071522,#091B2A);
 border:1px solid rgba(221,183,68,.65);border-radius:16px;padding:16px 18px;
 box-shadow:0 10px 28px rgba(0,0,0,.24);
}
.v6-price{font-size:40px;font-weight:950;color:#F5F7FA;line-height:1.05;margin-top:6px;}
.v6-change{font-size:22px;font-weight:900;margin-top:5px;}
.v6-change.up{color:#FF5B61}.v6-change.down{color:#49D17D}
.v6-dt{font-size:25px;font-weight:950;color:#F2D56B;margin:7px 0 5px;}
.v6-score{font-size:15px;font-weight:800;color:#E9EEF3;margin-bottom:7px;}
.v6-meta{font-size:12px;line-height:1.65;color:#AFC0D0;}
@media(max-width:768px){
 .v6-live-grid{grid-template-columns:1fr;gap:8px;}
 .v6-price{font-size:34px}.v6-dt{font-size:22px}
}


.v7-prob{font-size:16px;color:#F4F6F8;margin:6px 0;}
.v7-prob b{font-size:23px;color:#F2D56B;}
.v7-event-card{
 background:linear-gradient(145deg,#08131f,#0b1c2b);
 border:1px solid rgba(221,183,68,.48);border-radius:16px;
 padding:15px 18px;margin:0 0 13px;
}
.v7-event-title{font-size:21px;font-weight:900;color:#F2D56B;margin:6px 0;}
.v7-beta{font-size:14px;font-weight:800;color:#E7EDF3;margin-top:7px;}


.v7-prob-legend{
 background:rgba(221,183,68,.08);border:1px solid rgba(221,183,68,.28);
 color:#C8D3DD;border-radius:10px;padding:8px 12px;margin:4px 0 10px;
 font-size:12px;line-height:1.55;
}


.v8-data-card{background:linear-gradient(145deg,#071522,#0a1d2d);border:1px solid rgba(221,183,68,.42);
border-radius:16px;padding:16px 18px;margin:8px 0 13px}
.v8-data-title{font-size:21px;font-weight:900;color:#F2D56B;margin:6px 0}
.v8-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin-top:10px}
.v8-grid>div{background:#06111c;border:1px solid #1e3a52;border-radius:10px;padding:9px;color:#dce7ef;font-size:12px}
@media(max-width:700px){.v8-grid{grid-template-columns:1fr 1fr}}


.v9-prob-rule{
 background:#07131f;border:1px solid rgba(221,183,68,.50);border-radius:12px;
 padding:10px 14px;margin:8px 0 12px;color:#dce7ef;font-size:12px;line-height:1.65}
.v9-prob-rule b{color:#f2d56b}.v9-prob-rule span{color:#9fb3c4}


.v92-stale{background:rgba(255,184,77,.08);border:1px solid rgba(255,184,77,.35);
border-radius:10px;padding:9px 12px;margin:8px 0;color:#e8d9b5;font-size:12px}

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


@st.cache_data(ttl=1800, show_spinner=False)
def _v133_twse_month(sid, y, m):
    """TWSE public monthly STOCK_DAY fallback for listed stocks."""
    try:
        date=f"{int(y):04d}{int(m):02d}01"
        u="https://www.twse.com.tw/exchangeReport/STOCK_DAY"
        r=requests.get(u,params={"response":"json","date":date,"stockNo":str(sid)},timeout=12)
        j=r.json()
        rows=j.get("data") or []
        if not rows: return pd.DataFrame()
        out=[]
        for z in rows:
            try:
                roc=z[0].split("/")
                gy=int(roc[0])+1911
                d=f"{gy:04d}-{int(roc[1]):02d}-{int(roc[2]):02d}"
                def n(v):
                    return float(str(v).replace(",","").replace("--","nan"))
                out.append({
                    "date":d,
                    "Trading_Volume":n(z[1]),
                    "open":n(z[3]),"max":n(z[4]),"min":n(z[5]),"close":n(z[6])
                })
            except Exception:
                continue
        return pd.DataFrame(out)
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=1800, show_spinner=False)
def _v133_twse_history(sid, months=48):
    """Build multi-year listed-stock daily history from TWSE public monthly endpoint."""
    import time as _time
    now=_time.gmtime(_time.time()+8*3600)
    y,m=now.tm_year,now.tm_mon
    parts=[]
    for k in range(int(months)):
        mm=m-k
        yy=y+(mm-1)//12
        mm=(mm-1)%12+1
        q=_v133_twse_month(sid,yy,mm)
        if q is not None and len(q): parts.append(q)
    if not parts: return pd.DataFrame()
    x=pd.concat(parts,ignore_index=True)
    x=x.drop_duplicates("date").sort_values("date").reset_index(drop=True)
    return x

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


def realtime_quote(sid):
    """TWSE MIS 公開盤中行情。上市先查 tse，再查 otc；失敗則回傳空資料。"""
    headers={"User-Agent":"Mozilla/5.0","Referer":"https://mis.twse.com.tw/"}
    for market in ["tse","otc"]:
        try:
            url="https://mis.twse.com.tw/stock/api/getStockInfo.jsp"
            params={"ex_ch":f"{market}_{sid}.tw","json":"1","delay":"0"}
            j=requests.get(url,params=params,headers=headers,timeout=8).json()
            msg=j.get("msgArray",[])
            if msg:
                x=msg[0]
                def num(v):
                    try:
                        # MIS sometimes returns '-' or comma separated strings
                        return float(str(v).replace(",","")) if str(v) not in ["","-","--"] else np.nan
                    except Exception:
                        return np.nan
                z=num(x.get("z"))
                y=num(x.get("y"))
                o=num(x.get("o"))
                h=num(x.get("h"))
                l=num(x.get("l"))
                v=num(x.get("v"))
                t=x.get("t","")
                d=x.get("d","")
                name=x.get("n","")
                if pd.isna(z):
                    # If no last trade, use best bid/ask midpoint when available
                    bids=str(x.get("b","")).split("_")
                    asks=str(x.get("a","")).split("_")
                    bid=num(bids[0]) if bids else np.nan
                    ask=num(asks[0]) if asks else np.nan
                    if pd.notna(bid) and pd.notna(ask): z=(bid+ask)/2
                    elif pd.notna(bid): z=bid
                    elif pd.notna(ask): z=ask
                return {"price":z,"prev":y,"open":o,"high":h,"low":l,"volume":v,
                        "time":f"{d} {t}".strip(),"market":market,"name":name}
        except Exception:
            pass
    return {}




def _v10_features(df):
    """Create strictly backward-looking daily features."""
    x=df.copy().sort_values("date").reset_index(drop=True)
    c=x["close"].astype(float)
    h=x["max"].astype(float) if "max" in x else c
    l=x["min"].astype(float) if "min" in x else c
    v=x["Trading_Volume"].astype(float) if "Trading_Volume" in x else x.get("volume", 0)
    v=pd.Series(v, index=x.index).astype(float)
    ret=c.pct_change()
    x["f_ret1"]=ret
    x["f_ret5"]=c.pct_change(5)
    x["f_ret20"]=c.pct_change(20)
    x["f_ma5"]=c/c.rolling(5).mean()-1
    x["f_ma20"]=c/c.rolling(20).mean()-1
    x["f_vol20"]=ret.rolling(20).std()
    x["f_range"]=(h-l)/c.replace(0,np.nan)
    x["f_volratio"]=v/v.rolling(20).mean()
    x["y1"]=(c.shift(-1)>c).astype(float)
    x["y5"]=(c.shift(-5)>c).astype(float)
    return x

def _v10_walk_forward_probability(df,horizon=1):
    diag={"raw_rows":0,"feature_rows":0,"train_rows":0,"oos_rows":0,"reason":""}
    try:
        diag["raw_rows"]=0 if df is None else len(df)
        if not SKLEARN_OK:
            diag["reason"]="scikit-learn 未載入" + (f"｜{SKLEARN_IMPORT_ERROR}" if SKLEARN_IMPORT_ERROR else "")
            return None,diag
        z=_v10_features(df)
        target="y1" if horizon==1 else "y5"
        feats=["f_ret1","f_ret5","f_ret20","f_ma5","f_ma20","f_vol20","f_range","f_volratio"]
        diag["feature_rows"]=len(z.dropna(subset=feats))
        if diag["feature_rows"]<260:
            diag["reason"]=f"有效模型樣本僅 {diag['feature_rows']} 筆，需要至少 260 筆"
            return None,diag
        train=z.iloc[:-horizon].dropna(subset=feats+[target]).copy()
        diag["train_rows"]=len(train)
        if len(train)<240:
            diag["reason"]=f"可訓練樣本僅 {len(train)} 筆，需要至少 240 筆"
            return None,diag
        start_i=max(180,int(len(train)*0.55))
        step=10 if len(train)<700 else 20
        probs=[]; actual=[]
        for i in range(start_i,len(train),step):
            tr=train.iloc[:i]
            te=train.iloc[i:min(i+step,len(train))]
            if len(te)==0 or tr[target].nunique()<2: continue
            model=Pipeline([("scaler",StandardScaler()),
                            ("lr",LogisticRegression(max_iter=1000,class_weight="balanced"))])
            model.fit(tr[feats],tr[target])
            probs.extend(model.predict_proba(te[feats])[:,1].tolist())
            actual.extend(te[target].astype(int).tolist())
        diag["oos_rows"]=len(actual)
        if len(actual)<50:
            diag["reason"]=f"Walk-forward 驗證僅 {len(actual)} 筆，需要至少 50 筆"
            return None,diag
        final=Pipeline([("scaler",StandardScaler()),
                        ("lr",LogisticRegression(max_iter=1000,class_weight="balanced"))])
        final.fit(train[feats],train[target])
        latest=z.iloc[[-1]][feats]
        prob=float(final.predict_proba(latest)[:,1][0])
        brier=float(brier_score_loss(actual,probs))
        bins=[0,.4,.5,.6,1.000001]
        gaps=[]; weights=[]
        pa=np.asarray(probs); ya=np.asarray(actual)
        for lo,hi in zip(bins[:-1],bins[1:]):
            mask=(pa>=lo)&(pa<hi)
            if mask.sum()>=10:
                gaps.append(abs(float(pa[mask].mean())-float(ya[mask].mean())))
                weights.append(int(mask.sum()))
        gap=float(np.average(gaps,weights=weights)) if gaps else np.nan
        if brier<=.23 and (np.isnan(gap) or gap<=.10): status="良好"
        elif brier<=.26: status="普通"
        else: status="不足"
        diag["reason"]="模型驗證完成"
        result={"prob":prob,"n":len(actual),"brier":brier,"gap":gap,"status":status,
                "period_start":str(train["date"].iloc[0])[:10] if "date" in train else "",
                "period_end":str(train["date"].iloc[-1])[:10] if "date" in train else ""}
        return result,diag
    except Exception as e:
        diag["reason"]=f"模型計算失敗：{type(e).__name__}"
        return None,diag

def _v10_probability_panel(df):
    st.markdown("## AI 機率模型｜V14")
    st.caption("盤前也可計算：這裡使用已完成的歷史日線。盤中即時資料屬另一套模型，不會混入此處。")
    r1,d1=_v10_walk_forward_probability(df,1)
    r5,d5=_v10_walk_forward_probability(df,5)
    c1,c2=st.columns(2)
    with c1:
        st.caption("下一交易日上漲機率")
        if r1:
            st.markdown(f"### {r1['prob']*100:.1f}%")
            st.caption(f"驗證樣本 {r1['n']}｜Brier {r1['brier']:.3f}｜校準狀態 {r1['status']}")
        else:
            st.markdown("### 資料不足")
            st.caption(d1["reason"])
    with c2:
        st.caption("5日上漲機率")
        if r5:
            st.markdown(f"### {r5['prob']*100:.1f}%")
            st.caption(f"驗證樣本 {r5['n']}｜Brier {r5['brier']:.3f}｜校準狀態 {r5['status']}")
        else:
            st.markdown("### 資料不足")
            st.caption(d5["reason"])
    st.markdown("#### 模型資料診斷")
    dx1,dx2,dx3,dx4=st.columns(4)
    dx1.metric("歷史交易日", d1["raw_rows"])
    dx2.metric("有效特徵樣本", d1["feature_rows"])
    dx3.metric("可訓練樣本", d1["train_rows"])
    dx4.metric("Walk-forward驗證", d1["oos_rows"])
    if r1 is None or r5 is None:
        st.warning(f"明日模型：{d1['reason']}｜5日模型：{d5['reason']}")
    else:
        st.success("歷史樣本與 Walk-forward 驗證均已通過。")
    st.caption("這些是經 Walk-forward 歷史驗證的條件機率估計，不保證未來結果；目前尚未加入獨立的 post-hoc 機率校準層，市場結構改變時仍可能失效。")
    return r1,r5

def _v13_load_ledger():
    try:
        if os.path.exists(V13_LEDGER_PATH):
            with open(V13_LEDGER_PATH,"r",encoding="utf-8") as f:
                x=json.load(f)
                return x if isinstance(x,list) else []
    except Exception:
        pass
    return []

def _v13_save_ledger(rows):
    try:
        with open(V13_LEDGER_PATH,"w",encoding="utf-8") as f:
            json.dump(rows[-3000:],f,ensure_ascii=False,indent=2)
    except Exception:
        pass

def _v13_record_prediction(sid, name, price_df, r1, r5):
    """Session ledger. Streamlit Cloud ephemeral storage may reset after redeploy/restart."""
    if price_df is None or len(price_df)==0: return
    d=str(price_df["date"].iloc[-1])[:10]
    close=float(price_df["close"].iloc[-1])
    ledger=_v13_load_ledger()
    key=f"{sid}|{d}"
    if any(x.get("key")==key for x in ledger): return
    ledger.append({
        "key":key,"stock":str(sid),"name":str(name),"date":d,"close":close,
        "p1": None if not r1 else round(float(r1["prob"]),6),
        "p5": None if not r5 else round(float(r5["prob"]),6),
        "p1_status": None if not r1 else r1["status"],
        "p5_status": None if not r5 else r5["status"],
    })
    _v13_save_ledger(ledger)

def _v13_settle_ledger(sid, price_df):
    if price_df is None or len(price_df)==0: return
    px=price_df[["date","close"]].copy()
    px["date"]=px["date"].astype(str).str[:10]
    px["close"]=pd.to_numeric(px["close"],errors="coerce")
    mp=dict(zip(px["date"],px["close"]))
    dates=list(px["date"])
    ledger=_v13_load_ledger()
    changed=False
    for row in ledger:
        if str(row.get("stock"))!=str(sid): continue
        d=row.get("date")
        if d not in dates: continue
        i=dates.index(d)
        base=float(row.get("close",np.nan))
        if row.get("p1") is not None and row.get("y1") is None and i+1<len(dates):
            row["y1"]=int(float(px["close"].iloc[i+1])>base); changed=True
        if row.get("p5") is not None and row.get("y5") is None and i+5<len(dates):
            row["y5"]=int(float(px["close"].iloc[i+5])>base); changed=True
    if changed: _v13_save_ledger(ledger)

def _v13_accuracy_panel(sid):
    rows=[x for x in _v13_load_ledger() if str(x.get("stock"))==str(sid)]
    st.markdown("## AI 實戰驗證｜V14")
    settled1=[x for x in rows if x.get("p1") is not None and x.get("y1") is not None]
    settled5=[x for x in rows if x.get("p5") is not None and x.get("y5") is not None]
    c1,c2,c3=st.columns(3)
    c1.metric("已保存預測",len(rows))
    c2.metric("明日已驗證",len(settled1))
    c3.metric("5日已驗證",len(settled5))
    if not settled1 and not settled5:
        st.caption("尚未累積足夠的實際預測結果。V14 不會用回測命中率冒充真實上線戰績。")
        return
    for label,data,pk,yk in [
        ("明日模型",settled1,"p1","y1"),("5日模型",settled5,"p5","y5")]:
        if len(data)>=10:
            pred=np.array([float(x[pk]) for x in data])
            y=np.array([int(x[yk]) for x in data])
            hit=np.mean((pred>=.5)==y)
            bs=np.mean((pred-y)**2)
            st.write(f"**{label}**｜實際方向命中率 {hit*100:.1f}%｜Brier {bs:.3f}｜樣本 {len(data)}")
        elif data:
            st.write(f"**{label}**｜已驗證 {len(data)} 筆；未滿 10 筆，不顯示命中率。")


def _v14_model_health(r1, r5):
    """Model-health gate. This is not a probability."""
    rows=[r for r in (r1,r5) if r]
    if len(rows)<2:
        return "資料不足","至少需要明日與5日兩個模型都完成驗證"
    worst=max(float(r.get("brier",1)) for r in rows)
    statuses=[str(r.get("status","")) for r in rows]
    if worst>0.26 or "不足" in statuses:
        return "警戒",f"Brier 最高 {worst:.3f}，模型表現偏弱"
    if worst>0.23 or "普通" in statuses:
        return "普通",f"Brier 最高 {worst:.3f}，機率僅作輔助"
    return "良好",f"Brier 最高 {worst:.3f}，歷史驗證相對穩定"

def _v14_unified_signal(regime, r1, r5, short_score=None, inst_score=None):
    """One final signal. No fake probability and no winner-style certainty."""
    if not r1 or not r5:
        return "觀望","機率模型資料尚未完整"
    p1=float(r1["prob"]); p5=float(r5["prob"])
    health,_=_v14_model_health(r1,r5)
    if health=="警戒":
        return "觀望","模型健康度警戒，暫不放大訊號"
    tech=0
    try:
        if short_score is not None:
            tech=1 if float(short_score)>=2 else (-1 if float(short_score)<=-2 else 0)
    except Exception:
        tech=0
    chip=0
    try:
        if inst_score is not None:
            chip=1 if float(inst_score)>0 else (-1 if float(inst_score)<0 else 0)
    except Exception:
        chip=0

    if p1>=.62 and p5>=.62 and regime!="偏空" and tech>=0:
        return "偏多確認",f"明日 {p1*100:.1f}%、5日 {p5*100:.1f}% 且市場未偏空"
    if p1>=.56 and p5>=.56 and regime!="偏空":
        return "等待買進",f"機率略偏多，但尚未達偏多確認門檻"
    if p1<=.38 and p5<=.38:
        return "風險偏高",f"明日 {p1*100:.1f}%、5日 {p5*100:.1f}%"
    if p1<=.44 and p5<=.44:
        return "減碼警戒","兩個週期的上漲機率同步偏低"
    return "觀望","多空優勢尚未拉開"

def _v14_validation_panel(r1,r5):
    health,reason=_v14_model_health(r1,r5)
    st.markdown("## 模型自我驗證｜V14")
    a,b,c=st.columns(3)
    a.metric("模型健康度",health)
    a.caption(reason)
    if r1:
        b.metric("明日 Brier",f"{float(r1['brier']):.3f}")
        b.caption(f"Walk-forward {int(r1['n'])} 筆｜{r1['status']}")
    else:
        b.metric("明日 Brier","—")
    if r5:
        c.metric("5日 Brier",f"{float(r5['brier']):.3f}")
        c.caption(f"Walk-forward {int(r5['n'])} 筆｜{r5['status']}")
    else:
        c.metric("5日 Brier","—")
    st.caption("健康度用來判斷模型是否值得信任，不是上漲機率。Brier 越低越好；市場結構改變時模型可能退化。")

def _v13_market_regime(price_df, inst_df=None):
    """Transparent regime label; not a probability."""
    try:
        c=price_df["close"].astype(float)
        r5=c.pct_change(5).iloc[-1]
        r20=c.pct_change(20).iloc[-1]
        vol=c.pct_change().rolling(20).std().iloc[-1]
        score=0
        score += 1 if r5>0 else -1
        score += 1 if r20>0 else -1
        score += -1 if vol>0.035 else 0
        if score>=2: return "偏多"
        if score<=-2: return "偏空"
        return "震盪"
    except Exception:
        return "資料不足"

def _v13_trade_plan(current, support, resistance, regime, r1, r5):
    """No fake probability; creates a transparent conditional plan."""
    try:
        p1=None if not r1 else float(r1["prob"])
        p5=None if not r5 else float(r5["prob"])
        if p1 is not None and p5 is not None and p1>=.58 and p5>=.58 and regime!="偏空":
            signal="等待偏多條件"
        elif p1 is not None and p5 is not None and p1<=.42 and p5<=.42:
            signal="風險偏高"
        else:
            signal="觀望"
        invalid=float(support)*0.985 if pd.notna(support) else np.nan
        return signal, invalid
    except Exception:
        return "觀望", np.nan

def v9_market_session():
    """台灣股市時段：直接以 Unix time 加 UTC+8 計算。"""
    import time as _time
    now = _time.gmtime(_time.time() + 8 * 3600)
    weekday = now.tm_wday
    hour = now.tm_hour
    minute = now.tm_min

    if weekday >= 5:
        return "closed", "休市"
    mins = hour * 60 + minute
    if mins < 9 * 60:
        return "pre", "盤前"
    if mins <= 13 * 60 + 30:
        return "open", "盤中"
    return "post", "盤後"

def market_is_open_tw():
    """台灣集中市場一般交易時段：平日 09:00~13:30；實際休市日由行情是否取得再做第二層判斷。"""
    from datetime import datetime
    try:
        from zoneinfo import ZoneInfo
        now=datetime.now(ZoneInfo("Asia/Taipei"))
    except Exception:
        now=datetime.now()
    mins=now.hour*60+now.minute
    return now.weekday()<5 and 9*60 <= mins <= 13*60+30, now

def daytrade_radar(close, prev, day_open, day_high, day_low, vol_ratio,
                   short_score, inst_score, support, resistance):
    """盤中當沖雷達：只做市場訊號，不假裝預知13:30收盤。"""
    if not prev or prev <= 0:
        return "⚪ 觀望", 50, "資料不足", "盤整", support, resistance

    pct=(close/prev-1)*100
    open_pct=(close/day_open-1)*100 if day_open and day_open>0 else 0
    rng=max(day_high-day_low, 0.01) if day_high and day_low else max(close*0.01,0.01)
    loc=(close-day_low)/rng if day_high and day_low else .5

    score=50
    score += max(-15,min(15,pct*4))
    score += max(-8,min(8,open_pct*3))
    score += 8 if loc>=.72 else (-8 if loc<=.28 else 0)
    score += 8 if vol_ratio>=1.5 else (4 if vol_ratio>=1.15 else (-3 if vol_ratio<.75 else 0))
    score += (short_score-50)*0.18
    score += (inst_score-50)*0.08
    score=max(0,min(100,round(score)))

    if score>=75:
        signal="🟢 偏多當沖訊號"
        direction="偏多"
        reason="價格位置、盤中動能與量能多項偏強；仍需守住盤中防守位。"
    elif score>=62:
        signal="🟡 震盪偏多"
        direction="震盪偏多"
        reason="盤中結構偏多，但尚未形成高一致性訊號。"
    elif score>=42:
        signal="⚪ 觀望"
        direction="盤整"
        reason="多空條件接近，追價風險較高。"
    elif score>=28:
        signal="🟠 震盪偏空"
        direction="震盪偏空"
        reason="盤中價格結構偏弱，反彈仍需重新確認量價。"
    else:
        signal="🔴 偏空當沖訊號"
        direction="偏空"
        reason="盤中動能與價格位置明顯偏弱，以風險控制優先。"

    intraday_def=max(day_low if day_low else support, support)
    intraday_res=min(day_high if day_high else resistance, resistance) if resistance else day_high
    return signal,score,reason,direction,intraday_def,intraday_res


def _rss_items(query, n=8):
    """Google News RSS 公開新聞搜尋。"""
    import xml.etree.ElementTree as ET
    from urllib.parse import quote_plus
    url = f"https://news.google.com/rss/search?q={quote_plus(query)}&hl=zh-TW&gl=TW&ceid=TW:zh-Hant"
    try:
        rr = requests.get(url, timeout=8, headers={"User-Agent":"Mozilla/5.0"})
        rr.raise_for_status()
        root = ET.fromstring(rr.content)
        out=[]
        for item in root.findall(".//item")[:n]:
            out.append({
                "title": (item.findtext("title") or "").strip(),
                "link": (item.findtext("link") or "").strip(),
                "pubDate": (item.findtext("pubDate") or "").strip(),
                "source": ((item.find("source").text if item.find("source") is not None else "") or "").strip()
            })
        return out
    except Exception:
        return []

def global_event_news(sid, name, n=18):
    """
    V7 全球事件情報：
    公司/產業 + 美國總統/Fed/關稅/制裁 + 戰爭/地緣政治。
    RSS 是公開新聞索引，不把單一標題當成事實本身。
    """
    queries = [
        f'"{name}" OR "{sid}" 台股 重大訊息 財報 營收 法說',
        f'"{name}" 半導體 AI 供應鏈 關稅 出口管制',
        '美國總統 發言 關稅 台灣 半導體 晶片',
        'Federal Reserve Fed 利率 美股 科技股 台灣',
        '戰爭 衝突 制裁 中東 台海 烏克蘭 油價 股市',
        'NASDAQ futures semiconductor stocks Taiwan market'
    ]
    rows=[]
    seen=set()
    for q in queries:
        for x in _rss_items(q, 5):
            key=x["title"]
            if key and key not in seen:
                seen.add(key); rows.append(x)
    return rows[:n]

def event_impact_for_stock(news_rows, sid, name):
    """
    規則型事件風險層。只根據標題做『風險/關聯』初篩，
    不把它偽裝成完整 NLP 或確定的事件方向。
    """
    if not news_rows:
        return {"score":0, "risk":"資料不足", "related":0, "negative":0, "positive":0, "items":[]}

    high_kw = ["戰爭","開戰","攻擊","空襲","飛彈","制裁","關稅","出口管制","禁令",
               "Fed","聯準會","利率","美國總統","Trump","川普","台海","地震","停工"]
    neg_kw = ["下跌","重挫","暴跌","制裁","禁令","限制","戰爭","攻擊","停工","下修",
              "衰退","虧損","裁員","調降","關稅"]
    pos_kw = ["上漲","大漲","創高","上修","成長","獲利","訂單","擴產","降息","突破"]

    related=[]
    neg=pos=0
    for x in news_rows:
        t=x["title"]
        relevance = 2 if (name in t or sid in t) else (1 if any(k.lower() in t.lower() for k in high_kw) else 0)
        if relevance:
            nn=sum(1 for k in neg_kw if k.lower() in t.lower())
            pp=sum(1 for k in pos_kw if k.lower() in t.lower())
            neg += nn*relevance
            pos += pp*relevance
            xx=dict(x); xx["relevance"]=relevance; xx["tone"]="偏空" if nn>pp else ("偏多" if pp>nn else "中性/待確認")
            related.append(xx)

    raw=pos-neg
    if abs(raw)>=8: risk="重大事件影響"
    elif abs(raw)>=3: risk="事件影響中等"
    else: risk="事件影響有限/中性"
    return {"score":max(-20,min(20,raw)), "risk":risk, "related":len(related),
            "negative":neg, "positive":pos, "items":related[:8]}

def calibrated_probability_proxy(base_score, event_score=0, completeness=1.0):
    """
    V7 beta：暫用『機率代理值』，不是已完成歷史校準的真實勝率。
    等累積回測樣本後才應改標正式『上漲機率』。
    """
    x=(base_score-50)/11.5 + event_score/14
    p=1/(1+np.exp(-x))
    # 資料不完整時往 50% 收斂
    p=.5 + (p-.5)*max(.25,min(1.0,completeness))
    return round(p*100,1)


def _safe_read_html(url, timeout=10):
    try:
        h=requests.get(url,timeout=timeout,headers={"User-Agent":"Mozilla/5.0"}).text
        return pd.read_html(h)
    except Exception:
        return []

def taifex_pc_ratio():
    """期交所臺指選擇權 Put/Call ratio，官方公開頁面。"""
    try:
        tabs=_safe_read_html("https://www.taifex.com.tw/cht/3/pcRatio")
        for df in tabs:
            cols=" ".join(map(str,df.columns))
            if "買賣權成交量比率" in cols and len(df):
                r=df.iloc[0]
                vals=[x for x in r.tolist()]
                return {"date":str(vals[0]),"vol_pc":float(str(vals[3]).replace(",","")),
                        "oi_pc":float(str(vals[6]).replace(",",""))}
    except Exception:
        pass
    return None

def taifex_foreign_tx():
    """期交所臺股期貨三大法人；抓外資未平倉多空淨額。"""
    try:
        tabs=_safe_read_html("https://www.taifex.com.tw/cht/3/futContractsDateExcel")
        for df in tabs:
            flat=" ".join(map(str,df.astype(str).values.flatten()[:500]))
            if "臺股期貨" in flat and "外資" in flat:
                # HTML欄位會因網站調整而變；採保守解析，失敗就回傳 None，不猜值。
                for _,r in df.iterrows():
                    txt=" ".join(map(str,r.tolist()))
                    if "臺股期貨" in txt and "外資" in txt:
                        nums=[]
                        for v in r.tolist():
                            z=str(v).replace(",","").strip()
                            try: nums.append(float(z))
                            except: pass
                        if nums:
                            return {"raw":txt,"net_oi":nums[-1]}
    except Exception:
        pass
    return None

def margin_finmind(sid, token=""):
    """FinMind 融資融券；資料源不可用時回 None。"""
    try:
        start=(datetime.now()-timedelta(days=45)).strftime("%Y-%m-%d")
        p={"dataset":"TaiwanStockMarginPurchaseShortSale","data_id":sid,"start_date":start}
        if token: p["token"]=token
        j=requests.get("https://api.finmindtrade.com/api/v4/data",params=p,timeout=10).json()
        d=pd.DataFrame(j.get("data",[]))
        if d.empty: return None
        r=d.iloc[-1]
        def pick(keys):
            for k in keys:
                if k in r.index and pd.notna(r[k]):
                    try:return float(r[k])
                    except:return r[k]
            return None
        return {
            "date":str(r.get("date","")),
            "margin_balance":pick(["MarginPurchaseTodayBalance","MarginPurchaseBalance"]),
            "short_balance":pick(["ShortSaleTodayBalance","ShortSaleBalance"])
        }
    except Exception:
        return None

def lending_finmind(sid, token=""):
    """FinMind 借券相關；資料源不可用時回 None。"""
    for dataset in ["TaiwanStockSecuritiesLending","TaiwanStockSecuritiesLendingShortSale"]:
        try:
            start=(datetime.now()-timedelta(days=45)).strftime("%Y-%m-%d")
            p={"dataset":dataset,"data_id":sid,"start_date":start}
            if token:p["token"]=token
            j=requests.get("https://api.finmindtrade.com/api/v4/data",params=p,timeout=10).json()
            d=pd.DataFrame(j.get("data",[]))
            if not d.empty:
                return {"date":str(d.iloc[-1].get("date","")),"dataset":dataset,"row":d.iloc[-1].to_dict()}
        except Exception:
            pass
    return None

def probability_gate(data_flags):
    """資料不足時禁止顯示買賣機率。"""
    required=["價格","歷史行情","技術面","法人"]
    missing=[k for k in required if not data_flags.get(k,False)]
    completeness=sum(bool(v) for v in data_flags.values())/max(1,len(data_flags))
    ok=(not missing) and completeness>=0.55
    return ok, completeness, missing

def broker_research(sid, name, n=8):
    """只搜尋公開券商/投顧研究索引，不把一般新聞雜訊混入模型。"""
    domains=[
        ("凱基投顧","kgisia.com.tw"),
        ("凱基證券","kgi.com.tw"),
        ("群益投顧","capitalim.com.tw"),
    ]
    out=[]
    for broker,domain in domains:
        items=rss(f'site:{domain} "{sid}" "{name}"',3)
        for it in items:
            it["broker"]=broker
            out.append(it)
    return out[:n]

def rss(q,n=5):
    try:
        url=f"https://news.google.com/rss/search?q={quote(q)}&hl=zh-TW&gl=TW&ceid=TW:zh-Hant"
        root=ET.fromstring(requests.get(url,timeout=12,headers={"User-Agent":"Mozilla/5.0"}).content)
        return [{"title":x.findtext("title",""),"link":x.findtext("link","")} for x in root.findall(".//item")[:n]]
    except Exception:return []

def attack_status(short, close, support, resistance, vol_ratio, inst_score=50):
    """V5 五級市場訊號：清楚，但以條件式模型訊號呈現。"""
    breakout=max(resistance, close*1.01)
    pull_lo=support
    pull_hi=support*1.025
    weak=support*0.985

    confirmations=0
    confirmations += 1 if short>=70 else 0
    confirmations += 1 if close>=resistance*0.995 else 0
    confirmations += 1 if vol_ratio>=1.15 else 0
    confirmations += 1 if inst_score>=55 else 0

    if short>=78 and confirmations>=3:
        label="🟢 模型訊號：可買進"
        reason="短線趨勢、價格突破、量能與籌碼中至少三項同步確認。"
    elif short>=62 and confirmations>=2:
        label="🟡 模型訊號：等待買進"
        reason="方向偏強，但條件確認尚未完整；等待突破或拉回止穩。"
    elif short>=42:
        label="⚪ 模型訊號：觀望"
        reason="多空訊號尚未形成明顯優勢。"
    elif short>=25:
        label="🟠 模型訊號：減碼警戒"
        reason="短線結構轉弱，防守條件的重要性上升。"
    else:
        label="🔴 模型訊號：賣出"
        reason="短線趨勢與動能明顯偏弱，模型進入防守狀態。"
    return label,reason,breakout,pull_lo,pull_hi,weak,confirmations


# =========================
# Header + Search
# =========================
hero_bg = (
    f"linear-gradient(90deg,rgba(3,10,18,.98) 0%,rgba(3,10,18,.82) 42%,rgba(3,10,18,.18) 100%),"
    f"url('data:image/png;base64,{BANNER_B64}')"
    if BANNER_B64 else
    "linear-gradient(110deg,#07182a,#020811)"
)
st.markdown('<div class="top-signature">投顧大師 謝子鵬・與你攜手第一個百億　｜　版權使用：社團法人台灣美業國際交流協會</div>', unsafe_allow_html=True)

st.markdown(f"""
<div class="hero" style="min-height:235px;background-image:{hero_bg};background-size:cover;background-position:center 27%;display:flex;align-items:center;">
  <div style="max-width:680px">
    <div class="kicker">TAIWAN EQUITY INTELLIGENCE TERMINAL</div>
    <div class="hero-title"><span class="gold">KEN AI 百億</span>台股智慧決策系統</div>
    <div class="hero-sub" style="font-size:16px;margin-top:10px">市場訊號 × 法人籌碼因素 × 趨勢結構 × 風險驗證</div>
    <div style="margin-top:18px;color:#e8d18b;font-weight:800">用條件確認趨勢，不用情緒猜行情</div>
  </div>
</div>
""",unsafe_allow_html=True)


# ===== V5：搜尋框固定放在主畫面，手機／電腦都直接可用 =====
try:
    token = st.secrets.get("FINMIND_TOKEN", "")
except Exception:
    token = ""

# 持股設定保留在側欄，但搜尋不再依賴側欄
with st.sidebar:
    st.markdown("## 持股設定")
    own = st.selectbox("持股狀態", ["尚未持有","已持有"], key="sidebar_own_v48")
    cost = st.number_input("持有成本", min_value=0.0, value=0.0, step=0.5,
                           disabled=own=="尚未持有", key="sidebar_cost_v48")
    shares = st.number_input("持有股數", min_value=0, value=0, step=100,
                             disabled=own=="尚未持有", key="sidebar_shares_v48")

st.markdown("""
<div class="search-title-box">
  <div class="kicker">STOCK SEARCH</div>
  <div class="search-main-title">股票搜尋</div>
</div>
""", unsafe_allow_html=True)

with st.form("main_stock_search", clear_on_submit=False):
    search_q = st.text_input(
        "輸入股票代號或名稱",
        value="",
        placeholder="例如：2330、台積電、6213、聯茂",
        key="main_stock_query_v48"
    )
    search_run = st.form_submit_button("開始分析", use_container_width=True)

if search_run:
    if search_q.strip():
        st.session_state["active_stock_v48"] = search_q.strip()
        st.query_params["stock"] = search_q.strip()
        st.session_state["active_own_v48"] = own
        st.session_state["active_cost_v48"] = cost
        st.session_state["active_shares_v48"] = shares
    else:
        st.warning("請先輸入股票代號或名稱。")

if "active_stock_v48" not in st.session_state:
    _qp_stock = st.query_params.get("stock", "")
    if _qp_stock:
        st.session_state["active_stock_v48"] = _qp_stock

if "active_stock_v48" not in st.session_state:
    st.markdown("""
    <div class="panel">
      <div class="kicker">QUICK DECISION</div>
      <div class="action-title">輸入股票代號或名稱後，按「開始分析」</div>
      <div class="action-sub">搜尋框固定在主畫面，不需要開啟側邊欄；手機與電腦使用方式相同。</div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

q = st.session_state["active_stock_v48"]
own = st.session_state.get("active_own_v48", own)
cost = st.session_state.get("active_cost_v48", cost)
shares = st.session_state.get("active_shares_v48", shares)


# 開盤中每 8 秒自動刷新；使用者不需要重按搜尋
_market_open, _tw_now = market_is_open_tw()
if _market_open:
    st.markdown('', unsafe_allow_html=True)

sid,name=resolve_stock(q)

# V7：搜尋個股 + 台灣 + 國際重大事件新聞
_v7_news = global_event_news(sid, name)
_v7_event = event_impact_for_stock(_v7_news, sid, name)
_v8_pc = taifex_pc_ratio()
_v8_tx = taifex_foreign_tx()
# TOKEN 在原程式後段才初始化，因此此處直接安全讀取 Streamlit Secrets。
# 若沒有設定 FINMIND_TOKEN，函式仍會以公開/免 Token 模式嘗試取得資料。
_v8_token = st.secrets.get("FINMIND_TOKEN", "")
_v8_margin = margin_finmind(sid, _v8_token)
_v8_lending = lending_finmind(sid, _v8_token)

if not sid:
    st.error("找不到股票名稱。請改輸入股票代號，例如 6213。")
    st.stop()

today=date.today()
price=fm("TaiwanStockPrice",sid,today-timedelta(days=1825),today,token)

# V13.5: probability model requires multi-year history; short partial data is also backfilled.
if price is None or price.empty or len(price) < 520:
    _v133_fb=_v133_twse_history(sid,60)
    if _v133_fb is not None and not _v133_fb.empty:
        if price is not None and not price.empty:
            price=pd.concat([_v133_fb,price],ignore_index=True)
            price=price.drop_duplicates("date",keep="last").sort_values("date").reset_index(drop=True)
        else:
            price=_v133_fb
        st.caption(f"歷史資料：已啟用 TWSE 公開盤後備援，共 {len(price)} 個交易日")
if price is None or price.empty:
    st.error("目前暫時無法取得足夠的歷史股價資料。系統已嘗試主要資料來源與備援來源，請稍後再試。")
    st.stop()

d=add_indicators(price)
r=d.iloc[-1]
close=float(r["close"])
prev=float(d.iloc[-2]["close"]) if len(d)>1 else close

# 盤中優先採 TWSE MIS 最新成交；休市/無成交則自動回退最新日線
rt=realtime_quote(sid)
rt_price=rt.get("price",np.nan) if rt else np.nan
rt_prev=rt.get("prev",np.nan) if rt else np.nan
if pd.notna(rt_price) and rt_price>0:
    close=float(rt_price)
    if pd.notna(rt_prev) and rt_prev>0:
        prev=float(rt_prev)
    data_time=rt.get("time","盤中")
    data_mode="盤中最新取得行情"
else:
    data_time=str(pd.to_datetime(r["date"]).date())
    data_mode="最新交易日收盤"

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
_v8_flags={
    "價格": pd.notna(close),
    "歷史行情": price is not None and len(price)>=20,
    "技術面": True,
    "法人": inst is not None and len(inst)>0,
    "融資融券": _v8_margin is not None,
    "借券": _v8_lending is not None,
    "臺指選擇權": _v8_pc is not None,
    "臺指期外資": _v8_tx is not None,
    "全球事件": len(_v7_news)>0,
}
_v8_prob_ok,_v8_complete,_v8_missing=probability_gate(_v8_flags)


status,status_reason,breakout,pull_lo,pull_hi,weak,confirmations=attack_status(
    short,close,support,resistance,vol_ratio,inst_score
)

# V6 即時價格與 AI 當沖雷達
_market_open, _tw_now = market_is_open_tw()
rt_open = rt.get("open", np.nan) if rt else np.nan
rt_high = rt.get("high", np.nan) if rt else np.nan
rt_low  = rt.get("low", np.nan) if rt else np.nan
rt_vol  = rt.get("volume", np.nan) if rt else np.nan

day_open = float(rt_open) if pd.notna(rt_open) and rt_open>0 else float(r.get("open",close))
day_high = float(rt_high) if pd.notna(rt_high) and rt_high>0 else float(r.get("max",close))
day_low  = float(rt_low) if pd.notna(rt_low) and rt_low>0 else float(r.get("min",close))

dt_signal,dt_score,dt_reason,dt_direction,dt_def,dt_res = daytrade_radar(
    close,prev,day_open,day_high,day_low,vol_ratio,short,inst_score,support,resistance
)

# V7 Beta：重大事件先影響模型上漲機率；資料完整度獨立顯示
_v7_components = {
    "個股價格": bool(pd.notna(close)),
    "即時行情": bool(rt and pd.notna(rt_price)),
    "技術面": True,
    "法人": bool(inst is not None and len(inst)>0),
    "全球事件新聞": bool(len(_v7_news)>0),
    "券商公開研究": True
}
_v7_completeness = _v8_complete
dt_score = int(max(0,min(100, dt_score + _v7_event["score"]*0.45)))
_v7_up_prob = calibrated_probability_proxy(dt_score, _v7_event["score"], _v7_completeness)
_v7_down_prob = round(100-_v7_up_prob,1)
# V8.2：所有 UI 會用到的機率文字先初始化，避免先顯示後定義造成 NameError。
_v8_day_up_txt="偏多" if (_v8_prob_ok and _v7_up_prob>=55) else ("偏空" if (_v8_prob_ok and _v7_up_prob<=45) else ("中性" if _v8_prob_ok else "資料不足"))
_v8_day_down_txt="模型傾向" if _v8_prob_ok else "—"
_v9_session,_v9_session_label=v9_market_session()
# 盤中機率只在盤中且成功取得盤中行情時顯示。
_v9_has_live = bool(rt and pd.notna(rt_price))
if _v9_session=="open" and _v9_has_live and _v8_prob_ok:
    _v9_intraday_title="13:30前"
    _v9_intraday_up=_v8_day_up_txt
    _v9_intraday_down=_v8_day_down_txt
elif _v9_session=="pre":
    _v9_intraday_title="今日盤前"
    _v9_intraday_up="開盤後計算"
    _v9_intraday_down="開盤後計算"
elif _v9_session=="post":
    _v9_intraday_title="今日盤後"
    _v9_intraday_up="盤中預測已停止"
    _v9_intraday_down="盤中預測已停止"
else:
    _v9_intraday_title=_v9_session_label
    _v9_intraday_up="即時行情不足"
    _v9_intraday_down="即時行情不足"


# 波段值在稍後正式計算；先給安全預設，避免任何前段 UI 引用失敗。
_v8_swing_up_txt="資料不足"
_v8_swing_down_txt="—"


# 波段機率代理值：以現有波段/短線上漲機率 + 法人 + 事件層建立 beta 值
_v7_swing_base = max(0,min(100, short*0.55 + mid*0.25 + inst_score*0.20))
_v7_swing_up = calibrated_probability_proxy(_v7_swing_base, _v7_event["score"]*0.7, _v7_completeness)
_v7_swing_down = round(100-_v7_swing_up,1)
_v8_swing_up_txt="偏多" if (_v8_prob_ok and _v7_swing_up>=55) else ("偏空" if (_v8_prob_ok and _v7_swing_up<=45) else ("中性" if _v8_prob_ok else "資料不足"))
_v8_swing_down_txt="模型傾向" if _v8_prob_ok else "—"



heat=int(np.clip(50+(12 if abs(chg)>2 else 0)+(12 if vol_ratio>=1.2 else 0),0,100))
risk=int(np.clip(50+abs(chg)*4+(8 if pd.notna(r["RSI"]) and (r["RSI"]>75 or r["RSI"]<30) else 0),0,100))
tech=int(round(short*.5+mid*.3+long*.2))
overall=int(np.clip(round(tech*.58+inst_score*.27+heat*.15-(risk-50)*.08),0,100))
overall_label,overall_icon=trend_label(overall)

# =========================
# 簡潔首頁
# =========================
st.markdown(f"## {sid} {name or q}")
m1,m2,m3,m4=st.columns(4)
m1.metric("最新價格",f"{close:.2f}",f"{chg:+.2f}%")

def _v9_strength_label(v):
    try:
        v=float(v)
        if v >= 75: return "強勢偏多"
        if v >= 60: return "偏多"
        if v >= 45: return "中性"
        if v >= 30: return "偏空"
        return "強勢偏空"
    except Exception:
        return "資料不足"

short_label=_v9_strength_label(short)

mid_label=_v9_strength_label(mid)
long_label=_v9_strength_label(long)
m2.metric("短線強度",f"{short_label}")
m3.metric("量能比",f"{vol_ratio:.2f}x")
m4.metric("AI 綜合判斷",overall_label)

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


# ===== V13.9 最上方：即時價格 + AI 當沖雷達 =====
# 即時價格可盤中刷新；日線真機率模型仍使用已完成日線，避免把跳動報價冒充重新校準的機率。
rt_state = "🟢 盤中最新行情" if _market_open and rt and pd.notna(rt_price) else "⚪ 非開盤時段／最新取得資料"
price_source = "TWSE MIS 盤中行情" if (_market_open and rt and pd.notna(rt_price)) else data_mode
update_text = data_time if data_time else (_tw_now.strftime("%Y-%m-%d %H:%M:%S") if _tw_now else "")
_rt_open = rt.get("open", np.nan) if rt else np.nan
_rt_high = rt.get("high", np.nan) if rt else np.nan
_rt_low = rt.get("low", np.nan) if rt else np.nan
_rt_vol = rt.get("volume", np.nan) if rt else np.nan
def _v139_fmt(v, digits=2):
    try:
        return f"{float(v):,.{digits}f}" if pd.notna(v) else "—"
    except Exception:
        return "—"
_rt_ohlv = f"開 {_v139_fmt(_rt_open)}　高 {_v139_fmt(_rt_high)}　低 {_v139_fmt(_rt_low)}　量 {_v139_fmt(_rt_vol,0)}"

st.markdown(f"""
<div class="v6-live-grid">
  <div class="v6-live-card">
    <div class="kicker">LIVE PRICE｜即時價格</div>
    <div class="v6-price">{close:.2f}</div>
    <div class="v6-change {'up' if chg>=0 else 'down'}">{chg:+.2f}%</div>
    <div class="v6-meta">{rt_state}<br>{_rt_ohlv}<br>{price_source}｜更新 {update_text}</div>
  </div>
  <div class="v6-live-card">
    <div class="kicker">SUPER DAY TRADE｜百億超級當沖雷達</div>
    <div class="v6-dt">{dt_signal}</div>
    <div class="v7-prob">{_v9_intraday_title}上漲預估機率（Beta） <b>{_v9_intraday_up}</b>　｜　下跌預估機率（Beta） <b>{_v9_intraday_down}</b></div>
    <div class="v6-score">方向：{dt_direction}｜資料狀態 {("完整" if _v7_completeness>=0.85 else ("部分缺失" if _v7_completeness>=0.55 else "不足"))}</div>
    <div class="v6-meta">{dt_reason}<br>盤中防守：{dt_def:.2f}｜壓力：{dt_res:.2f}</div>
  </div>
</div>
""", unsafe_allow_html=True)



st.markdown("""
<div class="v7-prob-legend">
<b>機率顯示說明：</b> 本頁不再顯示「分數／100」。
AI 模型目前只顯示方向與狀態，不把模型分數包裝成機率；股價與實際漲跌幅仍維持市場原始單位。
</div>
""", unsafe_allow_html=True)




# V9.2 行情時段與資料新鮮度提示
if not _v9_has_live:
    st.markdown(f"""
    <div class="v92-stale">
      ⚠️ 目前為<strong>{_v9_session_label}</strong>或尚未取得今日盤中行情。
      畫面價格來源：{price_source}｜{update_text}。
      本系統不使用前一交易日收盤價冒充今日盤中機率。
    </div>
    """,unsafe_allow_html=True)

# ===== V13 機率誠信規則 =====
st.markdown(f"""
<div class="v9-prob-rule">
 <b>V13 機率誠信規則</b>｜畫面中的「%」目前只保留實際市場百分比資料；未完成歷史回測與校準的 AI 預測不顯示 %。
 技術、法人、風險、資料完整度等內部模型因素不再以百分比冒充機率。
 <br><span>技術、法人、市場、風險、趨勢及 Beta 預測一律只顯示文字狀態；完成歷史回測與機率校準後，才啟用 AI 機率百分比。</span>
</div>
""", unsafe_allow_html=True)

# ===== V8 全市場資料引擎 =====
_v8_status="可產生機率" if _v8_prob_ok else "資料不足・暫停機率判斷"
_v8_status_icon="🟢" if _v8_prob_ok else "⚠️"
_pc_txt=(f"成交量 P/C {_v8_pc['vol_pc']:.2f}%｜未平倉 P/C {_v8_pc['oi_pc']:.2f}%" if _v8_pc else "尚未取得")
_tx_txt=(f"外資臺指期未平倉淨額 {int(_v8_tx['net_oi']):,} 口" if _v8_tx else "尚未取得")
_mg_txt=("已取得" if _v8_margin else "尚未取得")
_ld_txt=("已取得" if _v8_lending else "尚未取得")
st.markdown(f"""
<div class="v8-data-card">
 <div class="kicker">V8 MARKET DATA ENGINE｜全市場資料引擎</div>
 <div class="v8-data-title">{_v8_status_icon} {_v8_status}</div>
 <div class="v6-meta">資料狀態 {("完整" if _v8_complete>=0.85 else ("部分缺失" if _v8_complete>=0.55 else "不足"))}｜缺少必要資料：{("、".join(_v8_missing) if _v8_missing else "無")}</div>
 <div class="v8-grid">
   <div><b>臺指選擇權</b><br>{_pc_txt}</div>
   <div><b>臺指期外資</b><br>{_tx_txt}</div>
   <div><b>融資融券</b><br>{_mg_txt}</div>
   <div><b>借券資料</b><br>{_ld_txt}</div>
 </div>
</div>
""",unsafe_allow_html=True)

# ===== V7 全球事件情報 =====
_event_icon = "🚨" if _v7_event["risk"]=="重大事件影響" else ("⚠️" if _v7_event["risk"]=="事件影響中等" else "🌐")
st.markdown(f"""
<div class="v7-event-card">
 <div class="kicker">GLOBAL EVENT INTELLIGENCE｜全球事件情報</div>
 <div class="v7-event-title">{_event_icon} {_v7_event["risk"]}</div>
 <div class="v6-meta">已掃描台灣/國際新聞；與個股或重大市場事件相關 {_v7_event["related"]} 則｜
 事件偏多指標 {_v7_event["positive"]}｜偏空指標 {_v7_event["negative"]}</div>
 <div class="v7-beta">未來波段上漲預估機率（Beta） {_v8_swing_up_txt}｜下跌預估機率（Beta） {_v8_swing_down_txt}</div>
</div>
""", unsafe_allow_html=True)

if _v7_event["items"]:
    with st.expander("查看影響模型的台灣／國際重大新聞", expanded=False):
        for _x in _v7_event["items"]:
            _src = _x.get("source","")
            _tone = _x.get("tone","待確認")
            if _x.get("link"):
                st.markdown(f"- **[{_tone}]** [{_x['title']}]({_x['link']})  `{_src}`")
            else:
                st.markdown(f"- **[{_tone}]** {_x['title']}  `{_src}`")
        st.caption("新聞標題只作事件偵測與市場情緒輸入；重大事件仍應以公司、交易所、政府或可信媒體原始資訊確認。")

st.markdown(f"""
<div class="decision">
  <div style="display:inline-block;background:linear-gradient(90deg,#E8C35A,#F5DC8B);
    color:#08111D;padding:7px 14px;border-radius:8px;font-size:14px;font-weight:950;
    letter-spacing:.8px;box-shadow:0 0 20px rgba(232,195,90,.22);margin-bottom:12px">
    AI ACTION CENTER｜V14 自我驗證決策系統
    </div>
  <div class="decision-grid">
    <div>
      <div class="decision-status">{status}</div>
      <div class="decision-note">{status_reason}</div>
      <div class="small" style="margin-top:7px">資料：{data_mode}｜{data_time}｜條件確認 {confirmations}/4</div>
    </div>
    <div class="decision-score"><span style="font-size:22px">趨勢強度：</span>{short_label}</div>
  </div>
  <div class="level-grid">
    <div class="levelbox"><div class="small">突破確認價</div><div class="level">{breakout:.2f}</div><div class="small">突破且量能同步增強，再重新確認進攻訊號</div></div>
    <div class="levelbox"><div class="small">拉回觀察區</div><div class="level">{pull_lo:.2f} ～ {pull_hi:.2f}</div><div class="small">回測止穩且技術轉強，可形成另一種轉強劇本</div></div>
    <div class="levelbox"><div class="small">轉弱警戒</div><div class="level">{weak:.2f}</div><div class="small">跌破後目前短線劇本失效，重新評估</div></div>
  </div>
</div>
""",unsafe_allow_html=True)

st.markdown(f"""
<div class="panel">
<div class="kicker">FINAL SUMMARY｜市場總結</div>
<div style="font-size:25px;font-weight:900">{overall_icon} AI 綜合判斷｜{overall_label}</div>
<div class="action-sub">短線目前為「{status.replace("🚀 ","").replace("🟢 ","").replace("🟡 ","").replace("⚠️ ","").replace("🔴 ","")}」。
重點不是預測哪一天一定上漲，而是等待價格、量能與技術條件觸發後再更新訊號。</div>
</div>
""",unsafe_allow_html=True)


st.markdown("### 趨勢燈號")
c1,c2,c3=st.columns(3)
for col,title,score,period in zip([c1,c2,c3],["短線","中線","長線"],[short,mid,long],["1–10 交易日","2–6 週","1–6 個月"]):
    lab,ico=trend_label(score)
    with col:
        st.markdown(f"""<div class="panel"><div class="kicker">{period}</div>
        <div style="font-size:24px;font-weight:900">{ico} {title}｜{lab}</div>
        <div class="gold" style="font-size:22px;font-weight:900">趨勢狀態｜{lab}</div></div>""",unsafe_allow_html=True)

_v10_p1,_v10_p5=_v10_probability_panel(price)
_v13_settle_ledger(sid,price)
_v13_record_prediction(sid,name,price,_v10_p1,_v10_p5)
_v13_regime=_v13_market_regime(price,inst if "inst" in globals() else None)
try:
    _v13_signal,_v14_signal_reason=_v14_unified_signal(
        _v13_regime,_v10_p1,_v10_p5,
        short_score=short if "short" in locals() else None,
        inst_score=inst_score if "inst_score" in locals() else None
    )
    _,_v13_invalid=_v13_trade_plan(current,support,resistance,_v13_regime,_v10_p1,_v10_p5)
except Exception:
    _v13_signal,_v14_signal_reason,_v13_invalid="觀望","決策條件尚未完整",np.nan


# V13.11 判斷失效價：優先使用已計算的20日支撐，其次60日支撐，
# 再以目前價格的5%風險帶備援。這只是模型重新評估參考，不是保證停損價。
def _v1311_valid_num(x):
    try:
        v = float(x)
        return v if np.isfinite(v) and v > 0 else np.nan
    except Exception:
        return np.nan

_v1311_price = _v1311_valid_num(current_price if "current_price" in locals() else close)
_v1311_s20 = _v1311_valid_num(support20 if "support20" in locals() else np.nan)
_v1311_s60 = _v1311_valid_num(support60 if "support60" in locals() else np.nan)

_v1311_candidates = [
    v for v in (_v1311_s20, _v1311_s60)
    if np.isfinite(v) and (not np.isfinite(_v1311_price) or v < _v1311_price)
]

if _v1311_candidates:
    v1311_invalidation = max(_v1311_candidates)
    v1311_invalidation_source = "近期技術支撐"
elif np.isfinite(_v1311_price):
    v1311_invalidation = _v1311_price * 0.95
    v1311_invalidation_source = "價格風險帶"
else:
    v1311_invalidation = np.nan
    v1311_invalidation_source = "尚無足夠價格資料"

v1311_invalidation_text = (
    f"{v1311_invalidation:,.2f} 元"
    if np.isfinite(v1311_invalidation)
    else "尚未形成有效失效價"
)

st.markdown("## V14 統一決策中心")
_v13a,_v13b,_v13c=st.columns(3)
_v13a.metric("市場狀態",_v13_regime)
_v13b.metric("模型訊號",_v13_signal)

# 優先採用原模型可用的失效價；若原模型無法產生，就使用 V13.11 備援失效價。
if pd.notna(_v13_invalid):
    _v1313_invalid_text = f"{float(_v13_invalid):,.2f} 元"
    _v1313_source = "模型技術條件"
else:
    _v1313_invalid_text = v1311_invalidation_text
    _v1313_source = v1311_invalidation_source

_v13c.metric("判斷失效價", _v1313_invalid_text)
st.caption(
    f"統一訊號依據：{_v14_signal_reason}。"
    f" 判斷失效價依據：{_v1313_source}；跌破後應重新評估目前模型判斷。"
)
_v14_validation_panel(_v10_p1,_v10_p5)
_v13_accuracy_panel(sid)
st.caption("實戰預測目前仍使用 Streamlit 執行環境暫存；網站重新部署或休眠後可能重置。要永久保存戰績，需要再連接外部資料庫。")


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
<div style="font-size:26px;font-weight:900">{overall_icon} AI 綜合判斷｜{overall_label}</div>
</div>""",unsafe_allow_html=True)
models=[("技術面因素",tech),("法人籌碼因素",inst_score),("市場環境因素",heat),("風險因素",100-risk)]
mc=st.columns(4)
for col,(title,score) in zip(mc,models):
    lab,ico=trend_label(score)
    with col:
        st.markdown(f"""<div class="panel"><div class="kicker">{title}</div>
        <div style="font-size:25px;font-weight:900">{ico} {lab}</div></div>""",unsafe_allow_html=True)

st.markdown("""
<div class="section-pro">
  <div class="section-pro-title">📈 價格趨勢</div>
  <div class="section-pro-sub">K線趨勢與均線結構｜掌握價格方向與波動變化</div>
</div>
""", unsafe_allow_html=True)
chart_df = d.tail(120).copy()
fig = go.Figure()
line_defs = [
    ("close", "收盤價", "#F2C94C", 3.0),
    ("MA5", "MA5", "#52B6FF", 1.8),
    ("MA20", "MA20", "#2F80ED", 1.8),
    ("MA60", "MA60", "#EB5757", 1.8),
]
for key, label, color, width in line_defs:
    if key in chart_df.columns:
        fig.add_trace(go.Scatter(
            x=chart_df["date"], y=chart_df[key],
            mode="lines", name=label,
            line=dict(color=color, width=width),
            hovertemplate="%{x|%Y-%m-%d}<br>"+label+"：%{y:.2f}<extra></extra>"
        ))
fig.update_layout(
    height=430,
    margin=dict(l=20,r=20,t=24,b=20),
    paper_bgcolor="#061522",
    plot_bgcolor="#061522",
    font=dict(color="#D9E5F0"),
    legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0),
    hovermode="x unified",
    xaxis=dict(showgrid=False, color="#8EA7BE", zeroline=False),
    yaxis=dict(gridcolor="rgba(110,145,175,.14)", color="#8EA7BE", zeroline=False),
)
st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

st.markdown("""
<div class="section-pro">
  <div class="section-pro-title">▥ 法人籌碼因素</div>
  <div class="section-pro-sub">外資・投信・自營商｜觀察近期資金方向</div>
</div>
""", unsafe_allow_html=True)
st.markdown(f"""<div class="panel"><div class="kicker">INSTITUTIONAL FLOW</div>
<div style="font-size:23px;font-weight:900">近 5 日法人代理淨額：{inst_net:,.0f}</div></div>""",unsafe_allow_html=True)
if not inst.empty:
    inst_show=inst.copy()
    if "name" in inst_show.columns:
        inst_show["name"]=inst_show["name"].map(zh_institution_name)
        inst_show=inst_show.rename(columns={"name":"法人名稱"})
    inst_show=inst_show.rename(columns={"date":"日期","buy":"買進","sell":"賣出"})
    wanted=[x for x in ["日期","法人名稱","買進","賣出"] if x in inst_show.columns]
    table_df = (inst_show[wanted].tail(20) if wanted else inst_show.tail(20)).copy()

# 數值欄位格式與淨額
for col in ["買進","賣出"]:
    # V13.4: institutional table defensive initialization
    if "table_df" not in locals() or table_df is None:
        table_df = inst.copy() if "inst" in locals() and isinstance(inst, pd.DataFrame) else pd.DataFrame()
    if col in table_df.columns:
        table_df[col] = pd.to_numeric(table_df[col], errors="coerce").fillna(0)

if "買進" in table_df.columns and "賣出" in table_df.columns:
    table_df["淨買賣"] = table_df["買進"] - table_df["賣出"]

def money_cell(v, net=False):
    try:
        v=float(v)
        if net:
            cls="tw-red" if v>0 else "tw-green" if v<0 else "muted"
            sign="+" if v>0 else ""
            return f'<span class="{cls}">{sign}{v:,.0f}</span>'
        return f'{v:,.0f}'
    except Exception:
        return str(v)

headers = "".join(f"<th>{c}</th>" for c in table_df.columns)
rows = []
for _, rr in table_df.iterrows():
    cells=[]
    for c in table_df.columns:
        val=rr[c]
        if c=="淨買賣":
            shown=money_cell(val, True)
        elif c in ["買進","賣出"]:
            shown=money_cell(val)
        else:
            shown=str(val)
        cells.append(f"<td>{shown}</td>")
    rows.append("<tr>"+"".join(cells)+"</tr>")

dark_table = f"""
<div class="inst-table-wrap">
<table class="inst-table">
<thead><tr>{headers}</tr></thead>
<tbody>{''.join(rows)}</tbody>
</table>
</div>
"""
st.markdown(dark_table, unsafe_allow_html=True)

st.markdown("### 券商公開研究")
research_items=broker_research(sid,name,8)
if research_items:
    for item in research_items:
        st.markdown(f"- **{item['broker']}**｜[{item['title']}]({item['link']})")
else:
    st.caption("目前未找到與此個股直接相關的近期公開券商研究索引。一般新聞不列入，避免資訊雜訊。")



st.warning("「可買進／等待買進／觀望／減碼警戒／賣出」為程式依最新取得或最新交易日市場資料計算的模型訊號，不是保證獲利或個人化投資指示；盤中行情與券商公開研究可能有延遲或資料缺漏。")
