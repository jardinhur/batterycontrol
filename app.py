import os, json, time, serial, serial.tools.list_ports
import numpy as np, pandas as pd, streamlit as st, plotly.graph_objects as go
from sklearn.ensemble import IsolationForest
from openai import AzureOpenAI
from dotenv import load_dotenv
import requests, joblib

load_dotenv()
st.set_page_config(page_title="Battery Sentinel 3.0", layout="wide")
LOG_PATH = "control_log.csv"

T_CUTOFF = 45.0
RISE_WINDOW_S = 5
RISE_CUTOFF = 5.0
DEV_WARN = 2.5
DEV_CUTOFF = 6.0
RISE_WARN = 0.4
SENSOR_FAIL = -100

@st.cache_resource
def init_model():
    p = "isolation_forest_ev.pkl"
    if os.path.exists(p):
        return joblib.load(p), p
    rng = np.random.default_rng(42)
    X = np.column_stack([
        rng.normal(28, 2.5, 800), np.abs(rng.normal(0.6, 0.3, 800)),
        rng.normal(0.02, 0.03, 800), rng.normal(0.5, 0.15, 800)
    ])
    m = IsolationForest(n_estimators=100, contamination=0.03, random_state=42).fit(X)
    return m, "fallback"

iso, model_source = init_model()
C_TH, K_COOL, HEAT_GAIN, DT = 40.0, 0.012, 2.2, 1.0

def status_from_power(power, warn=False):
    base = "⚡ 고속 충전" if power >= 85 else "🔶 저속 충전" if power >= 50 else "⚠️ 제한 충전" if power > 1 else "⏸️ 대기"
    return ("⚠️ " + base) if warn and power > 1 else base

def decide(score, tmax, dtv, dtdt, r, i_now, i_prev, t_prev5, sensor_ok):
    reasons = []
    if not sensor_ok:
        return 0.0, "hard", ["센서 읽기 실패(-127)"]
    if tmax >= T_CUTOFF:
        return 0.0, "hard", [f"절대온도 {tmax:.1f}°C ≥ {T_CUTOFF:.0f}°C"]
    if t_prev5 is not None and (tmax - t_prev5) >= RISE_CUTOFF:
        return 0.0, "hard", [f"{RISE_WINDOW_S}초에 {tmax - t_prev5:.1f}°C 상승"]
    if dtv >= DEV_CUTOFF:
        return 0.0, "hard", [f"센서 편차 {dtv:.1f}°C"]
    if i_prev is not None and (i_now - i_prev) <= 0.02 and dtdt >= 0.5 and tmax >= 35:
        return 0.0, "hard", ["전류 정체/감소 + 온도 상승"]
    kind = "ok"
    if dtdt >= RISE_WARN:
        kind = "warn"
        reasons.append(f"상승속도 {dtdt:.2f}°C/s 경고")
    if dtv >= DEV_WARN:
        kind = "warn"
        reasons.append(f"센서 편차 {dtv:.1f}°C 경고")
    tf = 1.0 if tmax <= 35 else max(0.15, 1.0 - (tmax - 35) / (T_CUTOFF - 35))
    sf = 1.0 if score <= 15 else 0.25 if score >= 50 else 1.0 - (score - 15) / 35 * 0.6
    rf = 1.0 if r <= 0.85 else max(0.3, 1.0 - (r - 0.85) * 0.5)
    power = float(np.clip(100.0 * min(tf, sf, rf), 0, 100))
    if not reasons:
        reasons.append("정상" if power >= 85 else "온도/점수/저항 감속")
    return power, kind, reasons

def append_log(row):
    df = pd.DataFrame([row])
    if os.path.exists(LOG_PATH):
        try:
            old = pd.read_csv(LOG_PATH, nrows=0)
            if list(old.columns) != list(df.columns):
                os.remove(LOG_PATH)
        except Exception:
            os.remove(LOG_PATH)
    df.to_csv(LOG_PATH, mode="a", header=not os.path.exists(LOG_PATH), index=False)

def get_client():
    k, e = os.getenv("AZURE_OPENAI_API_KEY"), os.getenv("AZURE_OPENAI_ENDPOINT")
    return AzureOpenAI(api_key=k, api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview"), azure_endpoint=e) if k and e else None

def ask_brief(s, timeline):
    c = get_client()
    if not c:
        return "AI 설정 없음"
    hist = " | ".join(f"{x['time']} {x['status']} {x['reason']}" for x in timeline[-4:]) or "없음"
    p = f"""4줄 브리핑. 1)위험 요인 2)Isolation Forest 해석 3)전원 조절 이유 4)차단 가능성
전원={s['Power']:.0f}% Tmax={s['T_max']:.1f} ΔT={s['Delta_T']:.2f} R={s['R_est']:.2f} 점수={s['AI_Score']:.1f} 상태={s['Status']}
이유={s.get('Reason','')} 최근={hist}
이전 호출은 기억하지 않는다."""
    try:
        r = c.chat.completions.create(model=os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini"),
            messages=[{"role": "user", "content": p}], temperature=0.3, max_tokens=380)
        return r.choices[0].message.content
    except Exception as ex:
        return str(ex)

def ask_decision(s):
    c = get_client()
    d = {"recommended_power": 100, "should_block": False, "reasoning": ""}
    if not c:
        return d
    try:
        p = f"""JSON만. Tmax={s['T_max']:.1f} ΔT={s['Delta_T']:.2f} score={s['AI_Score']:.1f}
45°C 이상이면 should_block=true. {{"recommended_power":0-100,"should_block":false,"reasoning":""}}"""
        r = c.chat.completions.create(model=os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini"),
            messages=[{"role": "user", "content": p}], temperature=0.2, max_tokens=160,
            response_format={"type": "json_object"})
        return json.loads(r.choices[0].message.content)
    except Exception:
        return d

def send_tg(msg):
    token, cid = os.getenv("TELEGRAM_BOT_TOKEN"), os.getenv("TELEGRAM_CHAT_ID")
    if token and cid:
        try:
            requests.post(f"https://api.telegram.org/bot{token}/sendMessage", json={"chat_id": cid, "text": msg}, timeout=4)
        except Exception:
            pass

for k, v in {
    "hist": pd.DataFrame(columns=["Time","T1","T2","T3","Delta_T","V","I","R","Score","Power","H1","H2","Relay"]),
    "sim": [26.0, 26.1, 26.0], "report": "대기", "decision": None,
    "serA": None, "serB": None, "last_ai": 0, "last_brief": 0, "last_status": "",
    "last_alert": 0, "latched": False, "timeline": [],
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

st.sidebar.title("제어판")
st.sidebar.caption(f"모델: {model_source} | T1~T3 | 차단 {T_CUTOFF:.0f}°C")
mode = st.sidebar.radio("모드", ["시뮬레이션", "하드웨어"])
ports = [p.device for p in serial.tools.list_ports.comports()]
if mode == "하드웨어":
    pa = st.sidebar.selectbox("ESP32 A (히터)", ports or ["COM3"])
    pb = st.sidebar.selectbox("ESP32 B (센서·LCD)", ports or ["COM4"])
    if st.sidebar.button("A 연결"):
        try:
            if st.session_state.serA: st.session_state.serA.close()
            st.session_state.serA = serial.Serial(pa, 115200, timeout=0.5)
        except Exception as e:
            st.sidebar.error(str(e))
    if st.sidebar.button("B 연결"):
        try:
            if st.session_state.serB: st.session_state.serB.close()
            st.session_state.serB = serial.Serial(pb, 115200, timeout=0.5)
        except Exception as e:
            st.sidebar.error(str(e))
scen = st.sidebar.selectbox("시나리오", ["A.정상", "B.저항↑", "C.HotSpot", "D.센서실패"])
local_ai = st.sidebar.toggle("로컬 AI 감속", True)
ai_auto = st.sidebar.toggle("AI 자동분석", True)
ai_ctrl = st.sidebar.toggle("AI 제어반영", False)
run = st.sidebar.checkbox("실시간", False)
if st.sidebar.button("🔓 차단 해제"):
    st.session_state.latched = False

def send_heaters(h1, h2):
    if st.session_state.serA and st.session_state.serA.is_open:
        try: st.session_state.serA.write(f"CMD:{int(h1*2.55)},{int(h2*2.55)}\n".encode())
        except Exception: pass

def send_relay(on: bool):
    if st.session_state.serB and st.session_state.serB.is_open:
        try: st.session_state.serB.write(f"RELAY:{1 if on else 0}\n".encode())
        except Exception: pass

def step():
    prev = st.session_state.hist.iloc[-1] if len(st.session_state.hist) else None
    p_power = float(prev["Power"]) if prev is not None else 0.0
    i_prev = float(prev["I"]) if prev is not None else None

    if mode == "시뮬레이션":
        t1, t2, t3 = st.session_state.sim
        amb = 25.5
        i_cmd = 1.5 * (p_power / 100.0)
        r = 1.20 if scen.startswith("B") else 0.65 if scen.startswith("C") else 0.50
        p_heat = (i_cmd ** 2) * r * HEAT_GAIN
        def ut(t, extra=0.0):
            return t + ((p_heat + extra) / C_TH - K_COOL * (t - amb)) * DT
        t1, t2 = ut(t1), ut(t2)
        t3 = ut(t3, 0.5 if scen.startswith("C") and p_power > 10 else 0.0)
        if scen.startswith("D"):
            t3 = -127
        else:
            t1, t2, t3 = [max(20.0, x) for x in (t1, t2, t3)]
            st.session_state.sim = [t1, t2, t3]
        v, i = 12.0, i_cmd
    else:
        t1 = t2 = t3 = 25.0
        v, i, r = 12.0, 0.0, 0.5
        if st.session_state.serB and st.session_state.serB.is_open:
            try:
                line = st.session_state.serB.readline().decode(errors="ignore").strip()
                if line.startswith("{"):
                    d = json.loads(line)
                    t1, t2, t3 = d.get("T1", 25), d.get("T2", 25), d.get("T3", 25)
                    v, i = d.get("V", 12), d.get("I", 0) / 1000.0
                    r = (v / i) if i > 0.05 else 0.5
            except Exception:
                pass

    temps = [t1, t2, t3]
    sensor_ok = all(t > SENSOR_FAIL for t in temps)
    tmax = max(temps) if sensor_ok else -127
    dtv = (max(temps) - min(temps)) if sensor_ok else 0.0
    dtdt, t_prev5 = 0.02, None
    if sensor_ok and len(st.session_state.hist) >= RISE_WINDOW_S:
        t_prev5 = max(st.session_state.hist.iloc[-RISE_WINDOW_S][["T1", "T2", "T3"]])
        dtdt = (tmax - t_prev5) / RISE_WINDOW_S
    elif sensor_ok and len(st.session_state.hist) >= 3:
        dtdt = (tmax - max(st.session_state.hist.iloc[-3][["T1", "T2", "T3"]])) / 3.0

    raw = float(iso.decision_function([[25 if not sensor_ok else tmax, dtv, dtdt, r]])[0])
    score = float(np.clip((0.05 - raw) * 180, 0, 100))

    if st.session_state.latched:
        power, kind, reasons = 0.0, "hard", ["차단 유지 (리셋 필요)"]
        status = "🛑 차단 (리셋 필요)"
    else:
        power, kind, reasons = decide(score, tmax, dtv, dtdt, r, i, i_prev, t_prev5, sensor_ok)
        if kind == "hard":
            power = 0.0
            st.session_state.latched = True
            status = "🛑 차단"
        else:
            if ai_ctrl and st.session_state.decision:
                dec = st.session_state.decision
                if dec.get("should_block"):
                    power, kind = 0.0, "hard"
                    st.session_state.latched = True
                    status = "🛑 AI 차단"
                    reasons.append("AI should_block")
                else:
                    power = min(power, float(dec.get("recommended_power", power)))
            if kind != "hard":
                status = status_from_power(power, warn=(kind == "warn"))

    reason = " / ".join(reasons)
    alpha = 0.25
    power = 0.0 if st.session_state.latched else float(np.clip(p_power * (1 - alpha) + power * alpha, 0, 100))
    h1 = 0.0 if st.session_state.latched or power <= 1 else power * 0.85
    h2 = 40.0 if (scen.startswith("C") and not st.session_state.latched and power > 15) else 0.0
    send_heaters(h1, h2)
    send_relay(not st.session_state.latched)

    if st.session_state.latched and time.time() - st.session_state.last_alert > 25:
        send_tg(f"차단 {reason}")
        st.session_state.last_alert = time.time()
    if int(time.time()) % 2 == 0:
        append_log({"time": time.strftime("%H:%M:%S"), "T_max": round(float(tmax), 2), "Delta_T": round(float(dtv), 2),
                    "dT_dt": round(float(dtdt), 3), "R": round(float(r), 3), "Score": round(score, 1),
                    "Power": round(power, 1), "kind": kind, "status": status, "reason": reason})

    st.session_state.hist = pd.concat([st.session_state.hist, pd.DataFrame([{
        "Time": time.strftime("%H:%M:%S"), "T1": t1, "T2": t2, "T3": t3,
        "Delta_T": dtv, "V": v, "I": i, "R": r, "Score": score, "Power": power,
        "H1": h1, "H2": h2, "Relay": int(not st.session_state.latched)
    }])], ignore_index=True).tail(90)
    return {"T_max": tmax, "Delta_T": dtv, "R_est": r, "AI_Score": score, "Status": status,
            "Power": power, "H1": h1, "H2": h2, "Reason": reason}

if run or len(st.session_state.hist) == 0:
    latest = step()
else:
    h = st.session_state.hist.iloc[-1]
    latest = {"T_max": max(h.T1, h.T2, h.T3), "Delta_T": h.Delta_T, "R_est": h.R,
              "AI_Score": h.Score, "Status": "일시정지", "Power": h.Power, "H1": h.H1, "H2": h.H2, "Reason": ""}

if run:
    interval = 20 if mode == "시뮬레이션" else 80
    changed = latest["Status"] != st.session_state.last_status
    if changed:
        st.session_state.timeline.append({"time": time.strftime("%H:%M:%S"), "status": latest["Status"],
                                          "power": latest["Power"], "reason": latest.get("Reason", "")})
        st.session_state.timeline = st.session_state.timeline[-12:]
    if changed or time.time() - st.session_state.last_brief > interval:
        st.session_state.report = ask_brief(latest, st.session_state.timeline)
        st.session_state.last_brief = time.time()
        st.session_state.last_status = latest["Status"]
if run and ai_auto and not st.session_state.latched and latest["AI_Score"] >= 18 and time.time() - st.session_state.last_ai > 25:
    st.session_state.decision = ask_decision(latest)
    st.session_state.last_ai = time.time()

st.title("🛡️ Battery Sentinel 3.0")
st.caption("센서 T1~T3 · LCD/LED/부저는 ESP32 B · 차단 45°C / 5초 5°C")
if "차단" in latest["Status"]:
    st.error(latest["Status"])
elif "⚠️" in latest["Status"] or "저속" in latest["Status"] or "제한" in latest["Status"]:
    st.warning(latest["Status"])
else:
    st.success(latest["Status"])

c = st.columns(6)
c[0].metric("위험도", f"{latest['AI_Score']:.1f}%")
c[1].metric("충전", f"{latest['Power']:.0f}%")
c[2].metric("Tmax", f"{latest['T_max']:.1f}°C")
c[3].metric("ΔT", f"{latest['Delta_T']:.2f}°C")
c[4].metric("H1", f"{latest['H1']:.0f}%")
c[5].metric("H2", f"{latest['H2']:.0f}%")
st.caption(f"판정: {latest.get('Reason','')}")

g1, g2 = st.columns(2)
with g1:
    f = go.Figure()
    f.add_trace(go.Scatter(x=st.session_state.hist.Time, y=st.session_state.hist.Score, name="위험도"))
    f.add_trace(go.Scatter(x=st.session_state.hist.Time, y=st.session_state.hist.Power, name="충전%"))
    f.update_layout(height=240, margin=dict(l=0, r=0, t=8, b=0))
    st.plotly_chart(f, use_container_width=True)
with g2:
    f2 = go.Figure()
    for col in ["T1", "T2", "T3"]:
        f2.add_trace(go.Scatter(x=st.session_state.hist.Time, y=st.session_state.hist[col], name=col))
    f2.update_layout(height=240, margin=dict(l=0, r=0, t=8, b=0))
    st.plotly_chart(f2, use_container_width=True)

st.markdown("**AI 브리핑 (자동)**")
st.info(st.session_state.report)
st.markdown("**제어 타임라인**")
for ev in reversed(st.session_state.timeline[-8:]):
    st.write(f"`{ev['time']}` {ev['status']} · {ev['power']:.0f}% · {ev['reason']}")
if st.session_state.decision:
    st.write("**AI 의사결정**", st.session_state.decision)
with st.expander("데이터셋 활용"):
    st.write(f"모델 `{model_source}` · T1~T3")
    if os.path.exists(LOG_PATH):
        try:
            st.dataframe(pd.read_csv(LOG_PATH, on_bad_lines="skip").tail(6), use_container_width=True)
        except Exception as e:
            st.warning(str(e))
if run:
    time.sleep(1)
    st.rerun()