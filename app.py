import os
import json
import time
import serial
import serial.tools.list_ports
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from sklearn.ensemble import IsolationForest
from openai import AzureOpenAI
from dotenv import load_dotenv
import requests
import joblib

load_dotenv()
st.set_page_config(page_title="Battery Sentinel 3.0", layout="wide")

LOG_PATH = "control_log.csv"

# ==========================================
# 1. Isolation Forest (데이터셋 학습 모델)
# ==========================================
@st.cache_resource
def init_anomaly_model():
    path = "isolation_forest_ev.pkl"
    if os.path.exists(path):
        return joblib.load(path), path
    np.random.seed(42)
    X = np.column_stack([
        np.random.normal(28, 2.5, 800),
        np.abs(np.random.normal(0.6, 0.3, 800)),
        np.random.normal(0.02, 0.03, 800),
        np.random.normal(0.5, 0.15, 800)
    ])
    m = IsolationForest(n_estimators=100, contamination=0.03, random_state=42)
    m.fit(X)
    return m, "fallback_synthetic"

iso, model_source = init_anomaly_model()

# ==========================================
# 2. 열·제어
# ==========================================
C_TH = 40.0
K_COOL = 0.012
HEAT_GAIN = 2.2
DT = 1.0

def power_from_state(score, tmax, r_est, dtdt):
    if tmax >= 58 or dtdt >= 1.5 or score >= 70:
        return 0.0, "hard"
    if tmax >= 48 or score >= 55:
        return 0.0, "soft"
    tf = 1.0 if tmax <= 35 else max(0.0, 1.0 - (tmax - 35) / (48 - 35))
    if score <= 15:
        sf = 1.0
    elif score >= 50:
        sf = 0.2
    else:
        sf = 1.0 - (score - 15) / (50 - 15) * 0.7
    rf = 1.0 if r_est <= 0.85 else max(0.3, 1.0 - (r_est - 0.85) * 0.5)
    return float(np.clip(100.0 * min(tf, sf, rf), 0, 100)), "ok"

def status_from_power(power):
    if power >= 85:
        return "⚡ 고속 충전"
    if power >= 50:
        return "🔶 저속 충전"
    if power > 1:
        return "⚠️ 제한 충전"
    return "⏸️ 일시 정지"

def append_log(row: dict):
    df = pd.DataFrame([row])
    header = not os.path.exists(LOG_PATH)
    df.to_csv(LOG_PATH, mode="a", header=header, index=False)

# ==========================================
# 3. AI / Telegram
# ==========================================
def get_client():
    k, e = os.getenv("AZURE_OPENAI_API_KEY"), os.getenv("AZURE_OPENAI_ENDPOINT")
    if not k or not e:
        return None
    return AzureOpenAI(
        api_key=k,
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview"),
        azure_endpoint=e
    )

def ask_brief(s):
    c = get_client()
    if not c:
        return "AI 설정 없음 (.env 확인)"
    try:
        p = (f"충전전원={s['Power']:.0f}% Tmax={s['T_max']:.1f}°C ΔT={s['Delta_T']:.2f} "
             f"R={s['R_est']:.2f}Ω AI점수={s['AI_Score']:.1f}% 상태={s['Status']}. "
             f"고속/저속/일시정지/차단 관점에서 3문장.")
        r = c.chat.completions.create(
            model=os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini"),
            messages=[{"role": "user", "content": p}],
            temperature=0.3, max_tokens=350
        )
        return r.choices[0].message.content
    except Exception as ex:
        return str(ex)

def ask_decision(s):
    c = get_client()
    d = {"recommended_power": 100, "should_block": False, "reasoning": ""}
    if not c:
        return d
    try:
        p = f"""JSON만.
Tmax={s['T_max']:.1f} ΔT={s['Delta_T']:.2f} R={s['R_est']:.2f} score={s['AI_Score']:.1f}
정상→100, 위험→감속/차단. 온도가 높으면 100을 주지 말 것.
{{"recommended_power":0-100,"should_block":false,"reasoning":""}}"""
        r = c.chat.completions.create(
            model=os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini"),
            messages=[{"role": "user", "content": p}],
            temperature=0.2, max_tokens=200,
            response_format={"type": "json_object"}
        )
        return json.loads(r.choices[0].message.content)
    except Exception:
        return d

def send_tg(msg):
    token, cid = os.getenv("TELEGRAM_BOT_TOKEN"), os.getenv("TELEGRAM_CHAT_ID")
    if not token or not cid:
        return
    try:
        requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": cid, "text": msg}, timeout=4
        )
    except Exception:
        pass

# ==========================================
# 4. Session
# ==========================================
for k, v in {
    "hist": pd.DataFrame(columns=["Time","T1","T2","T3","T4","Delta_T","V","I","R","Score","Power","H1","H2","Relay"]),
    "sim": [26.0, 26.1, 26.0, 25.5],
    "report": "대기",
    "decision": None,
    "serA": None,
    "serB": None,
    "last_ai": 0,
    "last_brief": 0,
    "last_status": "",
    "last_alert": 0,
    "latched": False,
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ==========================================
# 5. Sidebar
# ==========================================
st.sidebar.title("제어판")
st.sidebar.caption(f"모델: {model_source}")
mode = st.sidebar.radio("모드", ["시뮬레이션", "하드웨어"])
ports = [p.device for p in serial.tools.list_ports.comports()]
if mode == "하드웨어":
    pa = st.sidebar.selectbox("ESP32 A (히터)", ports or ["COM3"])
    pb = st.sidebar.selectbox("ESP32 B (센서·릴레이)", ports or ["COM4"])
    if st.sidebar.button("A 연결"):
        try:
            if st.session_state.serA:
                st.session_state.serA.close()
            st.session_state.serA = serial.Serial(pa, 115200, timeout=0.5)
            st.sidebar.success("A 연결")
        except Exception as e:
            st.sidebar.error(str(e))
    if st.sidebar.button("B 연결"):
        try:
            if st.session_state.serB:
                st.session_state.serB.close()
            st.session_state.serB = serial.Serial(pb, 115200, timeout=0.5)
            st.sidebar.success("B 연결")
        except Exception as e:
            st.sidebar.error(str(e))

scen = st.sidebar.selectbox("시나리오", ["A.정상", "B.저항↑", "C.HotSpot", "D.과전압"])
local_ai = st.sidebar.toggle("로컬 AI 감속", True)
ai_auto = st.sidebar.toggle("AI 자동분석", True)
ai_ctrl = st.sidebar.toggle("AI 제어반영", False)
run = st.sidebar.checkbox("실시간", False)
if st.sidebar.button("🔓 차단 해제"):
    st.session_state.latched = False
    st.sidebar.success("리셋됨")

# ==========================================
# 6. I/O
# ==========================================
def send_heaters(h1, h2):
    if st.session_state.serA and st.session_state.serA.is_open:
        try:
            st.session_state.serA.write(f"CMD:{int(h1*2.55)},{int(h2*2.55)}\n".encode())
        except Exception:
            pass

def send_relay(on: bool):
    if st.session_state.serB and st.session_state.serB.is_open:
        try:
            st.session_state.serB.write(f"RELAY:{1 if on else 0}\n".encode())
        except Exception:
            pass

# ==========================================
# 7. Step
# ==========================================
def step():
    prev = st.session_state.hist.iloc[-1] if len(st.session_state.hist) else None
    p_power = float(prev["Power"]) if prev is not None else 0.0

    if mode == "시뮬레이션":
        t1, t2, t3, t4 = st.session_state.sim
        i_cmd = 1.5 * (p_power / 100.0)
        r = 1.20 if scen.startswith("B") else 0.65 if scen.startswith("C") else 0.50
        p_heat = (i_cmd ** 2) * r * HEAT_GAIN

        def update_t(t, extra=0.0):
            return t + ((p_heat + extra) / C_TH - K_COOL * (t - t4)) * DT

        t1, t2 = update_t(t1), update_t(t2)
        extra3 = 0.45 if scen.startswith("C") and p_power > 10 else 0.0
        t3 = update_t(t3, extra3)
        t4 = t4 + 0.002 * ((t1 + t2 + t3) / 3.0 - t4)
        t1, t2, t3, t4 = [max(20.0, x) for x in (t1, t2, t3, t4)]
        st.session_state.sim = [t1, t2, t3, t4]
        v = 18.0 if scen.startswith("D") else 12.0
        i = i_cmd
    else:
        t1 = t2 = t3 = t4 = 25.0
        v, i, r = 12.0, 0.0, 0.5
        if st.session_state.serB and st.session_state.serB.is_open:
            try:
                line = st.session_state.serB.readline().decode(errors="ignore").strip()
                if line.startswith("{"):
                    d = json.loads(line)
                    t1, t2, t3, t4 = d.get("T1", 25), d.get("T2", 25), d.get("T3", 25), d.get("T4", 25)
                    v, i = d.get("V", 12), d.get("I", 0) / 1000.0
                    r = (v / i) if i > 0.05 else 0.5
            except Exception:
                pass

    tmax = max(t1, t2, t3)
    dtv = tmax - min(t1, t2, t3)
    dtdt = 0.02
    if len(st.session_state.hist) >= 3:
        dtdt = (tmax - max(st.session_state.hist.iloc[-3][["T1", "T2", "T3"]])) / 3.0

    # 데이터셋 모델 입력 feature
    feat = [tmax, dtv, dtdt, r]
    raw = iso.decision_function([feat])[0]
    score = float(np.clip((0.05 - raw) * 180, 0, 100))

    if st.session_state.latched:
        power, status, kind = 0.0, "🛑 차단 (리셋 필요)", "hard"
    else:
        power, kind = power_from_state(score, tmax, r, dtdt) if local_ai else (100.0, "ok")
        if scen.startswith("D") and v >= 17:
            kind = "hard"
        if kind == "hard":
            power = 0.0
            st.session_state.latched = True
            status = "🛑 열폭주 차단"
        elif kind == "soft":
            power = 0.0
            status = "⏸️ 일시 정지 (냉각 대기)"
        else:
            status = status_from_power(power)
        if ai_ctrl and st.session_state.decision and not st.session_state.latched:
            dec = st.session_state.decision
            if dec.get("should_block"):
                power, status = 0.0, "🛑 AI 차단"
                st.session_state.latched = True
            else:
                power = min(power, float(dec.get("recommended_power", power)))
                status = status_from_power(power)

    alpha = 0.2
    power = 0.0 if st.session_state.latched else float(np.clip(p_power * (1 - alpha) + power * alpha, 0, 100))
    blocked = st.session_state.latched or power <= 1
    h1 = 0.0 if blocked else power * 0.85
    h2 = 45.0 if (scen.startswith("C") and not st.session_state.latched and power > 15) else 0.0

    send_heaters(h1, h2)
    send_relay(not st.session_state.latched)

    if st.session_state.latched and time.time() - st.session_state.last_alert > 25:
        send_tg(f"열폭주 차단 Tmax {tmax:.1f} score {score:.0f}%")
        st.session_state.last_alert = time.time()

    # 데이터셋 활용 로그 (2초마다)
    if int(time.time()) % 2 == 0:
        append_log({
            "time": time.strftime("%H:%M:%S"),
            "mode": mode,
            "model": model_source,
            "T_max": round(tmax, 2),
            "Delta_T": round(dtv, 2),
            "dT_dt": round(dtdt, 3),
            "R_est": round(r, 3),
            "raw_score": round(float(raw), 4),
            "AI_Score": round(score, 1),
            "Power": round(power, 1),
            "kind": kind,
            "status": status,
        })

    row = pd.DataFrame([{
        "Time": time.strftime("%H:%M:%S"),
        "T1": t1, "T2": t2, "T3": t3, "T4": t4,
        "Delta_T": dtv, "V": v, "I": i, "R": r,
        "Score": score, "Power": power, "H1": h1, "H2": h2,
        "Relay": int(not st.session_state.latched)
    }])
    st.session_state.hist = pd.concat([st.session_state.hist, row], ignore_index=True).tail(90)
    return {
        "T_max": tmax, "Delta_T": dtv, "dT_dt": dtdt, "R_est": r,
        "AI_Score": score, "Status": status, "Power": power, "H1": h1, "H2": h2,
        "raw": float(raw), "feat": feat
    }

# ==========================================
# 8. Run
# ==========================================
if run or len(st.session_state.hist) == 0:
    latest = step()
else:
    h = st.session_state.hist.iloc[-1]
    latest = {
        "T_max": max(h.T1, h.T2, h.T3), "Delta_T": h.Delta_T, "dT_dt": 0.02,
        "R_est": h.R, "AI_Score": h.Score, "Status": "일시정지",
        "Power": h.Power, "H1": h.H1, "H2": h.H2, "raw": 0.0, "feat": []
    }

# 자동 브리핑: 시뮬 20초 / 하드웨어 80초 / 상태 변화 시
if run:
    interval = 20 if mode == "시뮬레이션" else 80
    changed = latest["Status"] != st.session_state.last_status
    if changed or (time.time() - st.session_state.last_brief > interval):
        st.session_state.report = ask_brief(latest)
        st.session_state.last_brief = time.time()
        st.session_state.last_status = latest["Status"]

if (run and ai_auto and not st.session_state.latched and latest["AI_Score"] >= 18
        and time.time() - st.session_state.last_ai > 25):
    st.session_state.decision = ask_decision(latest)
    st.session_state.last_ai = time.time()

# ==========================================
# 9. UI
# ==========================================
st.title("🛡️ Battery Sentinel 3.0")
st.caption("데이터셋 기반 이상점수 · 고속/저속/일시정지/차단")

if "차단" in latest["Status"]:
    st.error(latest["Status"])
elif any(x in latest["Status"] for x in ("일시", "저속", "제한")):
    st.warning(latest["Status"])
else:
    st.success(latest["Status"])

cols = st.columns(6)
cols[0].metric("위험도", f"{latest['AI_Score']:.1f}%")
cols[1].metric("충전", f"{latest['Power']:.0f}%")
cols[2].metric("Tmax", f"{latest['T_max']:.1f}°C")
cols[3].metric("ΔT", f"{latest['Delta_T']:.2f}°C")
cols[4].metric("H1", f"{latest['H1']:.0f}%")
cols[5].metric("H2", f"{latest['H2']:.0f}%")

c1, c2 = st.columns(2)
with c1:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=st.session_state.hist.Time, y=st.session_state.hist.Score, name="위험도"))
    fig.add_trace(go.Scatter(x=st.session_state.hist.Time, y=st.session_state.hist.Power, name="충전%"))
    fig.update_layout(height=260, margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig, use_container_width=True)
with c2:
    fig2 = go.Figure()
    for col in ["T1", "T2", "T3", "T4"]:
        fig2.add_trace(go.Scatter(x=st.session_state.hist.Time, y=st.session_state.hist[col], name=col))
    fig2.update_layout(height=260, margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig2, use_container_width=True)

st.markdown("**AI 브리핑 (자동)**")
st.info(st.session_state.report)
if st.session_state.decision:
    st.write("**AI 의사결정**", st.session_state.decision)

with st.expander("데이터셋 활용 확인"):
    st.write(f"- 사용 모델: `{model_source}`")
    st.write("- 입력 feature: `T_max, Delta_T, dT/dt, R_est` (Kaggle EV 데이터로 학습한 Isolation Forest)")
    st.write(f"- 최근 feature: `{latest.get('feat')}`")
    st.write(f"- decision_function raw: `{latest.get('raw')}` → AI 점수 `{latest['AI_Score']:.1f}%`")
    if os.path.exists(LOG_PATH):
        st.caption(f"로그 파일: {LOG_PATH}")
        st.dataframe(pd.read_csv(LOG_PATH).tail(8), use_container_width=True)

if run:
    time.sleep(1)
    st.rerun()