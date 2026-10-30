import os, json, time, serial, serial.tools.list_ports
import numpy as np, pandas as pd, streamlit as st, plotly.graph_objects as go
from sklearn.ensemble import IsolationForest
from openai import AzureOpenAI
from dotenv import load_dotenv
import requests, joblib

load_dotenv()
try:
    for k, v in st.secrets.items():
        if isinstance(v, str):
            os.environ.setdefault(k, v)
except Exception:
    pass

st.set_page_config(page_title="Battery Sentinel 3.0", layout="wide")
LOG_PATH = "control_log.csv"

I_FULL = 1.50
SOC_PER_A = 0.08
HW_MAX_PWM = 30.0
T_CAUTION, T_STOP = 35.0, 45.0
SCORE_CAUTION, SCORE_STOP = 25.0, 60.0
CAM_TH = 0.60          # swelling 확률 이 이상이면 차단
SENSOR_FAIL = -100

def to_hw_pwm(speed):
    return float(np.clip(speed / 100.0 * HW_MAX_PWM, 0, HW_MAX_PWM))

def pct_cmd(pct):
    pct = int(round(pct))
    return "f" if pct >= 95 else str(max(0, min(9, pct // 10)))

def led_level(speed):
    if speed >= 85: return 4
    if speed >= 60: return 3
    if speed >= 30: return 2
    if speed > 1: return 1
    return 0

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
C_TH, K_COOL, HEAT_GAIN = 40.0, 0.012, 2.2

def ai_charge(score, tmax, dtv, rise5, sensor_ok, b_state, cam_bad):
    if cam_bad:
        return 0.0, "blocked", ["영상 팽창"]
    if b_state == "BLOCKED" or not sensor_ok:
        return 0.0, "blocked", ["B 차단" if b_state == "BLOCKED" else "센서 실패"]
    if tmax >= T_STOP or score >= SCORE_STOP or rise5 >= 5.0:
        why = []
        if tmax >= T_STOP: why.append(f"온도 {tmax:.1f}≥{T_STOP:.0f}")
        if score >= SCORE_STOP: why.append(f"AI {score:.0f}%")
        if rise5 >= 5.0: why.append(f"급상승 {rise5:.1f}")
        return 0.0, "blocked", why
    caution = (b_state == "WARNING") or tmax >= T_CAUTION or score >= SCORE_CAUTION or dtv >= 2.0
    if caution:
        t_fac = 1.0 if tmax <= T_CAUTION else max(0.0, 1.0 - (tmax - T_CAUTION) / (T_STOP - T_CAUTION))
        s_fac = 1.0 if score <= SCORE_CAUTION else max(0.15, 1.0 - (score - SCORE_CAUTION) / (SCORE_STOP - SCORE_CAUTION))
        reasons = []
        if tmax >= T_CAUTION: reasons.append(f"온도 {tmax:.1f}°C → 감속")
        if score >= SCORE_CAUTION: reasons.append(f"AI 점수 {score:.0f}% → 감속")
        if not reasons: reasons.append("B CAUTION")
        return float(np.clip(100.0 * min(t_fac, s_fac), 15, 70)), "caution", reasons
    return 100.0, "normal", ["AI·온도·영상 정상 → 고속 충전"]

def label(mode_name, speed):
    if mode_name == "blocked" or speed <= 1:
        return "🛑 충전 차단"
    if mode_name == "caution":
        return "⚠️ CAUTION · 저속 충전"
    return "⚡ NORMAL · 고속 충전"

def append_log(row):
    df = pd.DataFrame([row])
    if os.path.exists(LOG_PATH):
        try:
            if list(pd.read_csv(LOG_PATH, nrows=0).columns) != list(df.columns):
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
    p = f"""4줄. 1)위험 2)Isolation Forest 해석 3)충전 전류를 왜 이렇게 맞췄는지 4)차단 가능성
충전량={s['SoC']:.1f}% 전류={s['I_ch']:.2f}A 속도={s['Speed']:.0f}% Tmax={s['T_max']:.1f} 점수={s['AI_Score']:.1f} 영상={s.get('Cam','-')} 상태={s['Status']} 이유={s.get('Reason','')} 최근={hist}
이전 호출은 기억하지 않는다."""
    try:
        r = c.chat.completions.create(model=os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini"),
            messages=[{"role": "user", "content": p}], temperature=0.3, max_tokens=380)
        return r.choices[0].message.content
    except Exception as ex:
        return str(ex)

def send_tg(msg):
    token, cid = os.getenv("TELEGRAM_BOT_TOKEN"), os.getenv("TELEGRAM_CHAT_ID")
    if token and cid:
        try:
            requests.post(f"https://api.telegram.org/bot{token}/sendMessage", json={"chat_id": cid, "text": msg}, timeout=4)
        except Exception:
            pass

for k, v in {
    "hist": pd.DataFrame(columns=["Time","T1","T2","T3","Delta_T","Score","Speed","SoC","Ich","Cam","State"]),
    "sim": [26.0, 26.1, 26.0], "soc": 20.0, "report": "대기",
    "serA": None, "serB": None, "serC": None,
    "cam": {"time": "-", "tag": "대기", "prob": 0.0},
    "last_brief": 0, "last_status": "", "last_alert": 0,
    "latched": False, "timeline": [], "lastA": "", "lastB": "",
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

st.sidebar.title("제어판")
st.sidebar.caption(f"학습 모델: {model_source}")
mode = st.sidebar.radio("모드", ["시뮬레이션", "하드웨어"])
ports = [p.device for p in serial.tools.list_ports.comports()]
if mode == "하드웨어":
    pa = st.sidebar.selectbox("ESP32 A", ports or ["COM3"])
    pb = st.sidebar.selectbox("ESP32 B", ports or ["COM4"])
    pc = st.sidebar.selectbox("ESP32-CAM", ["없음"] + (ports or []))
    if st.sidebar.button("A 연결"):
        try:
            if st.session_state.serA: st.session_state.serA.close()
            st.session_state.serA = serial.Serial(pa, 115200, timeout=0.3)
        except Exception as e:
            st.sidebar.error(str(e))
    if st.sidebar.button("B 연결"):
        try:
            if st.session_state.serB: st.session_state.serB.close()
            st.session_state.serB = serial.Serial(pb, 115200, timeout=0.3)
        except Exception as e:
            st.sidebar.error(str(e))
    if st.sidebar.button("CAM 연결") and pc != "없음":
        try:
            if st.session_state.serC: st.session_state.serC.close()
            st.session_state.serC = serial.Serial(pc, 115200, timeout=0.2)
            st.sidebar.success("CAM 연결")
        except Exception as e:
            st.sidebar.error(str(e))
scen = st.sidebar.selectbox("시나리오", ["A.정상", "B.저항↑", "C.HotSpot", "D.센서실패"])
cam_demo = st.sidebar.selectbox("영상 시연", ["자동(CAM)", "normal", "swelling"])
run = st.sidebar.checkbox("실시간", False)
if st.sidebar.button("충전량 20%로"):
    st.session_state.soc = 20.0
if st.sidebar.button("🔓 차단 해제"):
    st.session_state.latched = False
    if st.session_state.serB and st.session_state.serB.is_open:
        try: st.session_state.serB.write(b"r")
        except Exception: pass

def sendA(speed, off=False):
    if not (st.session_state.serA and st.session_state.serA.is_open):
        return
    hw = 0 if off else to_hw_pwm(speed)
    cmd = "x" if hw <= 1 else "h" + pct_cmd(hw)
    if cmd != st.session_state.lastA:
        try:
            st.session_state.serA.write(cmd.encode())
            st.session_state.lastA = cmd
        except Exception:
            pass

def sendB(speed, mode_name, swelling=False):
    if not (st.session_state.serB and st.session_state.serB.is_open):
        return
    if swelling:
        cmd = "k"
    elif mode_name == "blocked":
        cmd = "pxa"
    else:
        lv = led_level(speed)
        cmd = "p" if lv == 0 else f"s{lv}c"
        if mode_name == "caution":
            cmd += "w"
    if cmd != st.session_state.lastB:
        try:
            st.session_state.serB.write(cmd.encode())
            st.session_state.lastB = cmd
        except Exception:
            pass

def parse_b(line):
    if not line or line.startswith("#") or line.startswith("time_s"):
        return None
    p = line.split(",")
    if len(p) < 9:
        return None
    def f(x):
        try: return float(x)
        except Exception: return float("nan")
    return {"T1": f(p[1]), "T2": f(p[2]), "T3": f(p[3]),
            "rise": f(p[5]) if len(p) > 5 else 0,
            "state": p[8].strip() if len(p) > 8 else ""}

def parse_cam_line(line):
    """CAM이 보내는 형식: CAM,swelling,0.91  또는  CAM,normal,0.12"""
    line = line.strip()
    if not line.upper().startswith("CAM"):
        return None
    p = line.split(",")
    if len(p) < 3:
        return None
    tag = p[1].strip().lower()
    try:
        prob = float(p[2])
    except Exception:
        prob = 0.0
    return {"time": time.strftime("%H:%M:%S"), "tag": tag, "prob": prob}

def read_cam():
    if cam_demo == "swelling":
        return {"time": time.strftime("%H:%M:%S"), "tag": "swelling", "prob": 0.92}
    if cam_demo == "normal":
        return {"time": time.strftime("%H:%M:%S"), "tag": "normal", "prob": 0.88}
    if st.session_state.serC and st.session_state.serC.is_open:
        try:
            raw = st.session_state.serC.readline().decode(errors="ignore").strip()
            got = parse_cam_line(raw)
            if got:
                return got
        except Exception:
            pass
    return st.session_state.cam

def step():
    prev = st.session_state.hist.iloc[-1] if len(st.session_state.hist) else None
    p_speed = float(prev["Speed"]) if prev is not None else 0.0
    b_state, rise5 = "", 0.0

    if mode == "시뮬레이션":
        t1, t2, t3 = st.session_state.sim
        amb = 25.5
        i_heat = 1.5 * (to_hw_pwm(p_speed) / HW_MAX_PWM)
        r = 1.2 if scen.startswith("B") else 0.65 if scen.startswith("C") else 0.5
        heat = (i_heat ** 2) * r * HEAT_GAIN
        def ut(t, extra=0.0):
            return t + (heat + extra) / C_TH - K_COOL * (t - amb)
        t1, t2 = ut(t1), ut(t2)
        t3 = -127 if scen.startswith("D") else ut(t3, 0.35 if scen.startswith("C") and p_speed > 10 else 0)
        if not scen.startswith("D"):
            t1, t2, t3 = [max(20.0, x) for x in (t1, t2, t3)]
            st.session_state.sim = [t1, t2, t3]
        if len(st.session_state.hist) >= 5:
            rise5 = max(t1, t2, t3) - max(st.session_state.hist.iloc[-5][["T1", "T2", "T3"]])
    else:
        t1 = t2 = t3 = 25.0
        r = 0.5
        if st.session_state.serB and st.session_state.serB.is_open:
            try:
                d = parse_b(st.session_state.serB.readline().decode(errors="ignore").strip())
                if d:
                    t1, t2, t3, rise5, b_state = d["T1"], d["T2"], d["T3"], d["rise"], d["state"]
            except Exception:
                pass

    cam = read_cam()
    st.session_state.cam = cam
    cam_bad = cam["tag"] == "swelling" and cam["prob"] >= CAM_TH

    temps = [t1, t2, t3]
    ok = all(np.isfinite(t) and t > SENSOR_FAIL for t in temps)
    tmax = max(temps) if ok else -127
    dtv = (max(temps) - min(temps)) if ok else 0.0
    raw = float(iso.decision_function([[25 if not ok else tmax, dtv, rise5 / 5.0, r]])[0])
    score = float(np.clip((0.05 - raw) * 180, 0, 100))

    if st.session_state.latched:
        speed, mode_name, reasons = 0.0, "blocked", ["차단 유지"]
    else:
        speed, mode_name, reasons = ai_charge(score, tmax, dtv, rise5, ok, b_state, cam_bad)
        if mode_name == "blocked":
            st.session_state.latched = True
    status = label(mode_name, speed)
    reason = " / ".join(reasons)

    speed = 0.0 if st.session_state.latched else float(np.clip(p_speed * 0.7 + speed * 0.3, 0, 100))
    i_ch = 0.0 if speed <= 1 else I_FULL * (speed / 100.0)
    if i_ch > 0 and st.session_state.soc < 100:
        st.session_state.soc = min(100.0, st.session_state.soc + i_ch * SOC_PER_A)
    soc = st.session_state.soc

    sendA(speed, off=st.session_state.latched)
    sendB(speed, "blocked" if st.session_state.latched else mode_name, swelling=cam_bad)

    if st.session_state.latched and time.time() - st.session_state.last_alert > 25:
        send_tg(f"충전 차단 {reason}")
        st.session_state.last_alert = time.time()
    if int(time.time()) % 2 == 0:
        append_log({"time": time.strftime("%H:%M:%S"), "T_max": round(float(tmax), 2),
                    "score": round(score, 1), "soc": round(soc, 1), "i": round(i_ch, 2),
                    "cam": cam["tag"], "reason": reason})

    st.session_state.hist = pd.concat([st.session_state.hist, pd.DataFrame([{
        "Time": time.strftime("%H:%M:%S"), "T1": t1, "T2": t2, "T3": t3, "Delta_T": dtv,
        "Score": score, "Speed": speed, "SoC": soc, "Ich": i_ch,
        "Cam": f"{cam['tag']} {cam['prob']:.0%}", "State": b_state
    }])], ignore_index=True).tail(90)
    return {"T_max": tmax, "Delta_T": dtv, "AI_Score": score, "Status": status,
            "Speed": speed, "SoC": soc, "I_ch": i_ch, "Reason": reason,
            "Cam": f"{cam['tag']} {cam['prob']:.0%}"}

if run or len(st.session_state.hist) == 0:
    latest = step()
else:
    h = st.session_state.hist.iloc[-1]
    cam = st.session_state.cam
    latest = {"T_max": max(h.T1, h.T2, h.T3), "Delta_T": h.Delta_T, "AI_Score": h.Score,
              "Status": "일시정지", "Speed": h.Speed, "SoC": h.SoC, "I_ch": h.Ich,
              "Reason": "", "Cam": f"{cam['tag']} {cam['prob']:.0%}"}

if run:
    changed = latest["Status"] != st.session_state.last_status
    if changed:
        st.session_state.timeline.append({"time": time.strftime("%H:%M:%S"), "status": latest["Status"],
                                          "power": latest["Speed"], "reason": latest.get("Reason", "")})
        st.session_state.timeline = st.session_state.timeline[-12:]
    if changed or time.time() - st.session_state.last_brief > (20 if mode == "시뮬레이션" else 80):
        st.session_state.report = ask_brief(latest, st.session_state.timeline)
        st.session_state.last_brief = time.time()
        st.session_state.last_status = latest["Status"]

st.title("🛡️ Battery Sentinel")
st.caption("AI 점수·온도·영상 판정으로 충전 전류를 정하고, 그 전류만큼 가상 충전량이 오릅니다.")
if "차단" in latest["Status"]:
    st.error(latest["Status"])
elif "CAUTION" in latest["Status"]:
    st.warning(latest["Status"])
else:
    st.success(latest["Status"])

c = st.columns(6)
c[0].metric("충전량", f"{latest['SoC']:.1f}%")
c[1].metric("충전 전류", f"{latest['I_ch']:.2f} A")
c[2].metric("충전 속도", f"{latest['Speed']:.0f}%")
c[3].metric("최고 온도", f"{latest['T_max']:.1f}°C")
c[4].metric("AI 위험도", f"{latest['AI_Score']:.1f}%")
c[5].metric("영상 판정", latest.get("Cam", "-"))
st.caption(f"판단: {latest.get('Reason','')}")
st.progress(min(1.0, latest["SoC"] / 100.0))

cam = st.session_state.cam
if cam["tag"] == "swelling" and cam["prob"] >= CAM_TH:
    st.error(f"ESP32-CAM 팽창 감지 ({cam['prob']:.0%}) → 충전 차단, B에 k 전송")
elif cam["tag"] == "normal":
    st.success(f"ESP32-CAM 정상 ({cam['prob']:.0%})")

g1, g2 = st.columns(2)
with g1:
    f = go.Figure()
    f.add_trace(go.Scatter(x=st.session_state.hist.Time, y=st.session_state.hist.SoC, name="충전량 %"))
    f.add_trace(go.Scatter(x=st.session_state.hist.Time, y=st.session_state.hist.Speed, name="충전 속도 %"))
    f.add_trace(go.Scatter(x=st.session_state.hist.Time, y=st.session_state.hist.Score, name="AI 위험도"))
    f.update_layout(height=240, margin=dict(l=0, r=0, t=8, b=0))
    st.plotly_chart(f, use_container_width=True)
with g2:
    f2 = go.Figure()
    for col in ["T1", "T2", "T3"]:
        f2.add_trace(go.Scatter(x=st.session_state.hist.Time, y=st.session_state.hist[col], name=col))
    f2.update_layout(height=240, margin=dict(l=0, r=0, t=8, b=0))
    st.plotly_chart(f2, use_container_width=True)

st.markdown("**AI 브리핑**")
st.info(st.session_state.report)
for ev in reversed(st.session_state.timeline[-8:]):
    st.write(f"`{ev['time']}` {ev['status']} · {ev['power']:.0f}% · {ev['reason']}")
with st.expander("CAM이 보낼 형식"):
    st.code("CAM,normal,0.88\nCAM,swelling,0.91")
    st.write("115200. 팽창 확률 60% 이상이면 차단하고 B에 `k`.")
if run:
    time.sleep(1)
    st.rerun()