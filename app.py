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

C_TH, K_COOL, HEAT_GAIN, DT = 40.0, 0.012, 2.2, 1.0

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
    if power >= 85: return "⚡ 고속 충전"
    if power >= 50: return "🔶 저속 충전"
    if power > 1: return "⚠️ 제한 충전"
    return "⏸️ 일시 정지"

def why_power(score, tmax, r_est, dtdt, power, kind):
    reasons = []
    if kind == "hard":
        reasons.append("열폭주·급상승·고점수 → 완전 차단(래치)")
    elif kind == "soft":
        reasons.append("고온/높은 점수 → 일시 정지(냉각 후 재개)")
    else:
        if tmax > 35:
            reasons.append(f"Tmax {tmax:.1f}°C > 35°C → 온도 감속")
        if score > 15:
            reasons.append(f"학습모델 점수 {score:.1f}% > 15% → 점수 감속")
        if r_est > 0.85:
            reasons.append(f"R {r_est:.2f}Ω 상승 → 추가 감속")
        if not reasons:
            reasons.append("온도·점수·저항 정상 → 고속 유지")
    return " / ".join(reasons) + f" → 목표 전원 {power:.0f}%"

def append_log(row):
    new_df = pd.DataFrame([row])
    if os.path.exists(LOG_PATH):
        try:
            old = pd.read_csv(LOG_PATH, nrows=0)
            if list(old.columns) != list(new_df.columns):
                os.remove(LOG_PATH)
        except Exception:
            os.remove(LOG_PATH)
    new_df.to_csv(
        LOG_PATH, mode="a",
        header=not os.path.exists(LOG_PATH),
        index=False
    )

def get_client():
    k, e = os.getenv("AZURE_OPENAI_API_KEY"), os.getenv("AZURE_OPENAI_ENDPOINT")
    if not k or not e:
        return None
    return AzureOpenAI(api_key=k, api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview"), azure_endpoint=e)

def ask_brief(s, timeline):
    """매 호출은 새 대화. 최근 타임라인을 프롬프트에 넣어서 맥락을 줌."""
    c = get_client()
    if not c:
        return "AI 설정 없음 (.env 확인)"
    recent = timeline[-4:] if timeline else []
    hist_txt = " | ".join(f"{x['time']} {x['status']} {x['reason']}" for x in recent) or "없음"
    p = f"""배터리 안전 관제 브리핑. 반드시 아래 4줄을 지켜라.
1) 현재 위험 요인
2) 데이터셋 학습 모델(Isolation Forest) 점수 해석
3) 그래서 충전 전원을 왜 이렇게 조절했는지
4) 곧 일시정지/차단될 가능성

현재: 전원={s['Power']:.0f}% Tmax={s['T_max']:.1f} ΔT={s['Delta_T']:.2f} R={s['R_est']:.2f} 점수={s['AI_Score']:.1f}% 상태={s['Status']}
조절이유: {s.get('Reason','')}
최근기록: {hist_txt}
참고: 너는 이전 호출을 기억하지 못한다. 위 최근기록만 맥락으로 사용."""
    try:
        r = c.chat.completions.create(
            model=os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini"),
            messages=[{"role": "user", "content": p}],
            temperature=0.3, max_tokens=400
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
        p = f"""JSON만. Tmax={s['T_max']:.1f} ΔT={s['Delta_T']:.2f} R={s['R_est']:.2f} score={s['AI_Score']:.1f}
온도 높으면 100 주지 말 것. {{"recommended_power":0-100,"should_block":false,"reasoning":""}}"""
        r = c.chat.completions.create(
            model=os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini"),
            messages=[{"role": "user", "content": p}],
            temperature=0.2, max_tokens=180,
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
        requests.post(f"https://api.telegram.org/bot{token}/sendMessage", json={"chat_id": cid, "text": msg}, timeout=4)
    except Exception:
        pass

defaults = {
    "hist": pd.DataFrame(columns=["Time","T1","T2","T3","T4","Delta_T","V","I","R","Score","Power","H1","H2","Relay"]),
    "sim": [26.0, 26.1, 26.0, 25.5],
    "report": "대기",
    "decision": None,
    "serA": None, "serB": None,
    "last_ai": 0, "last_brief": 0, "last_status": "", "last_alert": 0,
    "latched": False,
    "timeline": [],
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

st.sidebar.title("제어판")
st.sidebar.caption(f"모델: {model_source}")
mode = st.sidebar.radio("모드", ["시뮬레이션", "하드웨어"])
ports = [p.device for p in serial.tools.list_ports.comports()]
if mode == "하드웨어":
    pa = st.sidebar.selectbox("ESP32 A (히터)", ports or ["COM3"])
    pb = st.sidebar.selectbox("ESP32 B (센서·릴레이)", ports or ["COM4"])
    if st.sidebar.button("A 연결"):
        try:
            if st.session_state.serA: st.session_state.serA.close()
            st.session_state.serA = serial.Serial(pa, 115200, timeout=0.5)
            st.sidebar.success("A 연결")
        except Exception as e:
            st.sidebar.error(str(e))
    if st.sidebar.button("B 연결"):
        try:
            if st.session_state.serB: st.session_state.serB.close()
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
        t3 = update_t(t3, 0.45 if scen.startswith("C") and p_power > 10 else 0.0)
        t4 = t4 + 0.002 * ((t1 + t2 + t3) / 3.0 - t4)
        t1, t2, t3, t4 = [max(20.0, x) for x in (t1, t2, t3, t4)]
        st.session_state.sim = [t1, t2, t3, t4]
        v, i = (18.0 if scen.startswith("D") else 12.0), i_cmd
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

    feat = [tmax, dtv, dtdt, r]
    raw = float(iso.decision_function([feat])[0])
    score = float(np.clip((0.05 - raw) * 180, 0, 100))

    if st.session_state.latched:
        power, status, kind = 0.0, "🛑 차단 (리셋 필요)", "hard"
    else:
        power, kind = power_from_state(score, tmax, r, dtdt) if local_ai else (100.0, "ok")
        if scen.startswith("D") and v >= 17:
            kind = "hard"
        if kind == "hard":
            power, status = 0.0, "🛑 열폭주 차단"
            st.session_state.latched = True
        elif kind == "soft":
            power, status = 0.0, "⏸️ 일시 정지 (냉각 대기)"
        else:
            status = status_from_power(power)
        if ai_ctrl and st.session_state.decision and not st.session_state.latched:
            dec = st.session_state.decision
            if dec.get("should_block"):
                power, status, kind = 0.0, "🛑 AI 차단", "hard"
                st.session_state.latched = True
            else:
                power = min(power, float(dec.get("recommended_power", power)))
                status = status_from_power(power)

    reason = why_power(score, tmax, r, dtdt, power, kind)
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

    if int(time.time()) % 2 == 0:
        append_log({"time": time.strftime("%H:%M:%S"), "mode": mode, "model": model_source,
                    "T_max": round(tmax, 2), "Delta_T": round(dtv, 2), "dT_dt": round(dtdt, 3),
                    "R_est": round(r, 3), "raw_score": round(raw, 4), "AI_Score": round(score, 1),
                    "Power": round(power, 1), "kind": kind, "status": status, "reason": reason})

    st.session_state.hist = pd.concat([st.session_state.hist, pd.DataFrame([{
        "Time": time.strftime("%H:%M:%S"), "T1": t1, "T2": t2, "T3": t3, "T4": t4,
        "Delta_T": dtv, "V": v, "I": i, "R": r, "Score": score, "Power": power,
        "H1": h1, "H2": h2, "Relay": int(not st.session_state.latched)
    }])], ignore_index=True).tail(90)

    return {"T_max": tmax, "Delta_T": dtv, "dT_dt": dtdt, "R_est": r, "AI_Score": score,
            "Status": status, "Power": power, "H1": h1, "H2": h2, "raw": raw, "feat": feat, "Reason": reason}

if run or len(st.session_state.hist) == 0:
    latest = step()
else:
    h = st.session_state.hist.iloc[-1]
    latest = {"T_max": max(h.T1, h.T2, h.T3), "Delta_T": h.Delta_T, "dT_dt": 0.02, "R_est": h.R,
              "AI_Score": h.Score, "Status": "일시정지", "Power": h.Power, "H1": h.H1, "H2": h.H2,
              "raw": 0.0, "feat": [], "Reason": ""}

if run:
    interval = 20 if mode == "시뮬레이션" else 80
    changed = latest["Status"] != st.session_state.last_status
    if changed:
        st.session_state.timeline.append({
            "time": time.strftime("%H:%M:%S"),
            "status": latest["Status"],
            "power": latest["Power"],
            "reason": latest.get("Reason", ""),
        })
        st.session_state.timeline = st.session_state.timeline[-12:]
    if changed or (time.time() - st.session_state.last_brief > interval):
        st.session_state.report = ask_brief(latest, st.session_state.timeline)
        st.session_state.last_brief = time.time()
        st.session_state.last_status = latest["Status"]

if run and ai_auto and not st.session_state.latched and latest["AI_Score"] >= 18 and time.time() - st.session_state.last_ai > 25:
    st.session_state.decision = ask_decision(latest)
    st.session_state.last_ai = time.time()

st.title("🛡️ Battery Sentinel 3.0")
st.caption("학습 모델 점수 + LLM 설명 (호출마다 새 대화, 최근 기록을 프롬프트에 포함)")

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
st.caption(f"제어 이유: {latest.get('Reason','')}")

c1, c2 = st.columns(2)
with c1:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=st.session_state.hist.Time, y=st.session_state.hist.Score, name="위험도"))
    fig.add_trace(go.Scatter(x=st.session_state.hist.Time, y=st.session_state.hist.Power, name="충전%"))
    fig.update_layout(height=250, margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig, use_container_width=True)
with c2:
    fig2 = go.Figure()
    for col in ["T1", "T2", "T3", "T4"]:
        fig2.add_trace(go.Scatter(x=st.session_state.hist.Time, y=st.session_state.hist[col], name=col))
    fig2.update_layout(height=250, margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig2, use_container_width=True)

st.markdown("**AI 브리핑 (자동)**")
st.info(st.session_state.report)

st.markdown("**제어 타임라인**")
if st.session_state.timeline:
    for ev in reversed(st.session_state.timeline[-8:]):
        st.write(f"`{ev['time']}` {ev['status']} · {ev['power']:.0f}% · {ev['reason']}")
else:
    st.caption("상태 변화가 있으면 여기에 쌓입니다.")

if st.session_state.decision:
    st.write("**AI 의사결정**", st.session_state.decision)

with st.expander("데이터셋 활용 확인"):
    st.write(f"모델: `{model_source}`")
    st.write("feature: T_max, Delta_T, dT/dt, R_est")
    st.write(f"최근: {latest.get('feat')} → raw {latest.get('raw')} → {latest['AI_Score']:.1f}%")
    if os.path.exists(LOG_PATH):
        try:
            log_df = pd.read_csv(LOG_PATH, on_bad_lines="skip")
            st.dataframe(log_df.tail(6), use_container_width=True)
        except Exception as e:
            st.warning(f"로그 읽기 실패 (파일 삭제 후 재실행): {e}")
if run:
    time.sleep(1)
    st.rerun()