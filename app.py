import os, re, json, time, serial, serial.tools.list_ports
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

HW_MAX_PWM = 30.0
MARGIN_CUTOFF = 3.0
MARGIN_SLOW = 1.5
RISE_CUTOFF = 1.0
RISE_WARN = 0.5
DEV_WARN, DEV_CUTOFF = 2.0, 5.0
SENSOR_FAIL = -100
SIM_AMBIENT = 25.5
STALE_LIMIT = 10
I_FULL = 1.50
SOC_PER_A = 0.08
SCORE_ALERT = 80.0
CAM_TH = 0.60

def to_hw_pwm(logical_pct):
    return float(np.clip(logical_pct / 100.0 * HW_MAX_PWM, 0, HW_MAX_PWM))

def pct_cmd(pct):
    pct = int(round(pct / 10.0) * 10)
    if pct >= 100:
        return "f"
    return str(max(0, min(9, pct // 10)))

def charge_level(power):
    if power >= 85: return 4
    if power >= 60: return 3
    if power >= 30: return 2
    if power > 1: return 1
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
C_TH, K_COOL, HEAT_GAIN, DT = 40.0, 0.012, 2.2, 1.0

def status_from_power(power, warn=False):
    base = "⚡ 고속 충전" if power >= 85 else "🔶 저속 충전" if power >= 50 else "⚠️ 제한 충전" if power > 1 else "⏸️ 대기"
    return ("⚠️ " + base) if warn and power > 1 else base

def decide(score, margin, dtv, rise10, r, sensor_ok, b_state, cam_bad):
    if cam_bad:
        return 0.0, "hard", ["영상 팽창"]
    if b_state == "BLOCKED":
        return 0.0, "hard", ["B보드 BLOCKED"]
    if not sensor_ok:
        return 0.0, "hard", ["센서 읽기 실패"]
    if margin >= MARGIN_CUTOFF:
        return 0.0, "hard", [f"주변 대비 +{margin:.1f}°C"]
    if rise10 >= RISE_CUTOFF:
        return 0.0, "hard", [f"10초 상승 {rise10:.1f}°C"]
    if dtv >= DEV_CUTOFF:
        return 0.0, "hard", [f"편차 {dtv:.1f}°C"]
    kind, reasons = "ok", []
    if rise10 >= RISE_WARN:
        kind = "warn"
        reasons.append(f"상승 경고 {rise10:.1f}°C/10s")
    if dtv >= DEV_WARN:
        kind = "warn"
        reasons.append(f"편차 경고 {dtv:.1f}°C")
    tf = 1.0 if margin <= MARGIN_SLOW else max(0.15, 1.0 - (margin - MARGIN_SLOW) / (MARGIN_CUTOFF - MARGIN_SLOW))
    sf = 1.0 if score <= 15 else 0.25 if score >= 50 else 1.0 - (score - 15) / 35 * 0.6
    rf = 1.0 if r <= 0.85 else max(0.3, 1.0 - (r - 0.85) * 0.5)
    power = float(np.clip(100 * min(tf, sf, rf), 0, 100))
    if not reasons:
        reasons.append("정상" if power >= 85 else "온도/점수 감속")
    return power, kind, reasons

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
    p = f"""4줄. 1)위험 2)Isolation Forest 해석 3)충전을 왜 조절했는지 4)차단 가능성
충전량={s.get('SoC',0):.1f}% 속도={s['Power']:.0f}% 전류={s.get('I_meas',0):.2f}A Tmax={s['T_max']:.1f} 주변대비={s.get('Margin',0):+.1f} 점수={s['AI_Score']:.1f} 영상={s.get('Cam','-')} 상태={s['Status']} 이유={s.get('Reason','')} 최근={hist}
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
        p = f"""JSON만. Tmax={s['T_max']:.1f} 주변대비={s.get('Margin',0):+.1f} score={s['AI_Score']:.1f}
{{"recommended_power":0-100,"should_block":false,"reasoning":""}}"""
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
    "hist": pd.DataFrame(columns=["Time","T1","T2","T3","Delta_T","Score","Power","SoC","Ich","Cam","State","Amb","Rise10"]),
    "sim": [26.0, 26.1, 26.0], "soc": 20.0, "report": "대기", "decision": None,
    "serA": None, "serB": None, "serC": None,
    "cam": {"time": "-", "tag": "대기", "prob": 0.0},
    "last_ai": 0, "last_brief": 0, "last_status": "",
    "last_alert": 0, "last_score_alert": 0, "latched": False, "timeline": [],
    "last_cmdA": "", "last_cmdB": "", "ambient": None, "last_b": None, "stale": 0,
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

st.sidebar.title("제어판")
st.sidebar.caption(f"모델: {model_source}")
st.sidebar.caption("필수: B  |  선택: A, CAM")
mode = st.sidebar.radio("모드", ["시뮬레이션", "하드웨어"])
ports = [p.device for p in serial.tools.list_ports.comports()]
amb_manual = 0.0
if mode == "하드웨어":
    st.sidebar.caption("USB는 이 PC에서만 동작합니다.")
    pa = st.sidebar.selectbox("ESP32 A (히터, 선택)", ["연결 안 함"] + (ports or []))
    pb = st.sidebar.selectbox("ESP32 B (센서·LCD)", ports or ["COM4"])
    pc = st.sidebar.selectbox("ESP32-CAM", ["연결 안 함"] + (ports or []))
    if st.sidebar.button("A 연결") and pa != "연결 안 함":
        try:
            if st.session_state.serA: st.session_state.serA.close()
            st.session_state.serA = serial.Serial(pa, 115200, timeout=0.3)
            st.sidebar.success("A 연결")
        except Exception as e:
            st.sidebar.error(str(e))
    if st.sidebar.button("B 연결"):
        try:
            if st.session_state.serB: st.session_state.serB.close()
            st.session_state.serB = serial.Serial(pb, 115200, timeout=0.3)
            st.session_state.ambient = None
            st.session_state.last_b = None
            st.session_state.stale = 0
            st.sidebar.success("B 연결")
        except Exception as e:
            st.sidebar.error(str(e))
    if st.sidebar.button("CAM 연결") and pc != "연결 안 함":
        try:
            if st.session_state.serC: st.session_state.serC.close()
            st.session_state.serC = serial.Serial(pc, 115200, timeout=0.2)
            st.sidebar.success("CAM 연결")
        except Exception as e:
            st.sidebar.error(str(e))
    if st.sidebar.button("B 기준선 다시 (z)"):
        st.session_state.ambient = None
        if st.session_state.serB and st.session_state.serB.is_open:
            try: st.session_state.serB.write(b"z")
            except Exception: pass
    amb_manual = st.sidebar.number_input("주변온도 수동 (0=자동)", 0.0, 50.0, 0.0, 0.5)

scen = st.sidebar.selectbox("시나리오", ["A.정상", "B.저항↑", "C.HotSpot", "D.센서실패"])
cam_demo = st.sidebar.selectbox("영상 시연", ["자동(CAM)", "normal", "swelling"])
local_ai = st.sidebar.toggle("로컬 AI 감속", True)
ai_auto = st.sidebar.toggle("AI 자동분석", True)
ai_ctrl = st.sidebar.toggle("AI 제어반영", False)
run = st.sidebar.checkbox("실시간", False)
fault_pct = st.sidebar.slider("고장 주입 히터 (%)", 0, 30, 0, 10, help="10% 단위. A 연결 시에만 전달.")
if st.sidebar.button("충전량 20%로"):
    st.session_state.soc = 20.0
if st.sidebar.button("🔓 차단 해제 (B에 r)"):
    st.session_state.latched = False
    if st.session_state.serB and st.session_state.serB.is_open:
        try: st.session_state.serB.write(b"r")
        except Exception: pass

def sendA(h1, h2, force_off=False):
    if not (st.session_state.serA and st.session_state.serA.is_open):
        return
    try:
        cmd = "x" if force_off or (h1 <= 1 and h2 <= 1) else "h" + pct_cmd(h1) + "j" + pct_cmd(h2)
        if cmd != st.session_state.last_cmdA:
            st.session_state.serA.write(cmd.encode())
            st.session_state.last_cmdA = cmd
    except Exception:
        pass

def sendB(power, blocked=False, warn=False, ai_block=False, swelling=False):
    if not (st.session_state.serB and st.session_state.serB.is_open):
        return
    try:
        if swelling:
            cmd = "k"
        elif blocked:
            cmd = "pxa" if ai_block else "px"
        else:
            lv = charge_level(power)
            cmd = "p" if lv == 0 else f"s{lv}c"
            if warn:
                cmd += "w"
        if cmd != st.session_state.last_cmdB:
            st.session_state.serB.write(cmd.encode())
            st.session_state.last_cmdB = cmd
    except Exception:
        pass

def handle_comment(s):
    if not s.startswith("#"):
        return
    if "기준선 측정 중" in s:
        st.session_state.ambient = None
    m = re.search(r"주변(?:온도)?\s*(-?\d+(?:\.\d+)?)", s)
    if m:
        st.session_state.ambient = float(m.group(1))

def parse_b_line(line):
    if not line or line.startswith("#") or line.startswith("time_s"):
        return None
    p = line.split(",")
    if len(p) < 13:
        return None
    def fnum(x, default=float("nan")):
        try: return float(x)
        except Exception: return default
    return {
        "T1": fnum(p[1]), "T2": fnum(p[2]), "T3": fnum(p[3]),
        "rise10": fnum(p[5], 0), "dev": fnum(p[6], 0),
        "state": p[8].strip() if len(p) > 8 else "",
        "V": fnum(p[11], 12), "I": fnum(p[12], 0),
    }

def read_b(ser):
    lines = []
    try:
        if ser.in_waiting == 0:
            lines.append(ser.readline())
        while ser.in_waiting:
            lines.append(ser.readline())
    except Exception:
        return None
    last = None
    for raw in lines:
        s = raw.decode(errors="ignore").strip()
        handle_comment(s)
        d = parse_b_line(s)
        if d:
            last = d
    return last

def parse_cam_line(line):
    line = line.strip()
    if not line.upper().startswith("CAM"):
        return None
    p = line.split(",")
    if len(p) < 3:
        return None
    try:
        prob = float(p[2])
    except Exception:
        prob = 0.0
    return {"time": time.strftime("%H:%M:%S"), "tag": p[1].strip().lower(), "prob": prob}

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

def no_data(status, reason):
    sendA(0, 0, force_off=True)
    sendB(0.0)
    cam = st.session_state.cam
    return {"T_max": float("nan"), "Delta_T": 0.0, "R_est": 0.5, "AI_Score": 0.0,
            "Status": status, "Power": 0.0, "SoC": st.session_state.soc, "I_meas": 0.0,
            "Reason": reason, "Margin": float("nan"), "Rise10": 0.0, "Ambient": float("nan"),
            "Cam": f"{cam['tag']} {cam['prob']:.0%}", "NoData": True}

def step():
    prev = st.session_state.hist.iloc[-1] if len(st.session_state.hist) else None
    p_power = float(prev["Power"]) if prev is not None else 0.0
    b_state, rise10 = "", 0.0
    ambient = SIM_AMBIENT
    i_meas = 0.0

    if mode == "시뮬레이션":
        t1, t2, t3 = st.session_state.sim
        amb = SIM_AMBIENT
        i_heat = 1.5 * (fault_pct / HW_MAX_PWM) if HW_MAX_PWM else 0
        r = 1.20 if scen.startswith("B") else 0.65 if scen.startswith("C") else 0.50
        p_heat = (i_heat ** 2) * r * HEAT_GAIN
        def ut(t, extra=0.0):
            return t + ((p_heat + extra) / C_TH - K_COOL * (t - amb)) * DT
        t1, t2 = ut(t1), ut(t2)
        t3 = -127 if scen.startswith("D") else ut(t3, 0.35 if scen.startswith("C") and fault_pct > 0 else 0.0)
        if not scen.startswith("D"):
            t1, t2, t3 = [max(20.0, x) for x in (t1, t2, t3)]
            st.session_state.sim = [t1, t2, t3]
        v = 12.0
        i_meas = i_heat
        if len(st.session_state.hist) >= 10:
            rise10 = max(t1, t2, t3) - float(st.session_state.hist.iloc[-10][["T1", "T2", "T3"]].max())
    else:
        ser = st.session_state.serB
        if not (ser and ser.is_open):
            return no_data("🔌 B 연결 필요", "사이드바에서 B 연결")
        parsed = read_b(ser)
        if parsed:
            st.session_state.last_b = parsed
            st.session_state.stale = 0
        else:
            st.session_state.stale += 1
        data = st.session_state.last_b
        if data is None:
            return no_data("⏳ B 데이터 대기", "센서 줄 대기")
        stale = st.session_state.stale >= STALE_LIMIT
        ambient = amb_manual if amb_manual > 0 else st.session_state.ambient
        if not stale and ambient is None:
            return no_data("⏳ 기준선 측정 중", "약 10초, 또는 z / 수동 주변온도")
        ambient = float("nan") if ambient is None else float(ambient)
        if stale:
            t1 = t2 = t3 = float("nan")
            v, r = 12.0, 0.5
        else:
            t1, t2, t3 = data["T1"], data["T2"], data["T3"]
            v = data["V"] if np.isfinite(data["V"]) else 12.0
            i_meas = data["I"] if np.isfinite(data["I"]) else 0.0
            rise10 = data["rise10"] if np.isfinite(data["rise10"]) else 0.0
            b_state = data["state"]
            r = (v / i_meas) if i_meas > 0.05 else 0.5

    cam = read_cam()
    st.session_state.cam = cam
    cam_bad = cam["tag"] == "swelling" and cam["prob"] >= CAM_TH

    temps = [t1, t2, t3]
    sensor_ok = all(np.isfinite(t) and t > SENSOR_FAIL for t in temps)
    tmax = max(temps) if sensor_ok else -127
    margin = (tmax - ambient) if (sensor_ok and np.isfinite(ambient)) else 0.0
    dtv = (max(temps) - min(temps)) if sensor_ok else 0.0
    dtdt = (max(rise10, 0.0) / 10.0) or 0.02
    raw = float(iso.decision_function([[25 if not sensor_ok else tmax, dtv, dtdt, r]])[0])
    score = float(np.clip((0.05 - raw) * 180, 0, 100))

    if st.session_state.latched or b_state == "BLOCKED" or cam_bad:
        if b_state == "BLOCKED" or cam_bad:
            st.session_state.latched = True
        power, kind, reasons = 0.0, "hard", ["영상 팽창"] if cam_bad else ["차단 유지"]
        status = "🛑 충전 차단"
    else:
        power, kind, reasons = decide(score, margin, dtv, rise10, r, sensor_ok, b_state, cam_bad) if local_ai else (100.0, "ok", ["수동"])
        if kind == "hard":
            power = 0.0
            st.session_state.latched = True
            status = "🛑 충전 차단"
        else:
            if ai_ctrl and st.session_state.decision:
                dec = st.session_state.decision
                if dec.get("should_block"):
                    power, kind = 0.0, "hard"
                    st.session_state.latched = True
                    status = "🛑 AI 차단"
                    reasons.append("AI block")
                else:
                    power = min(power, float(dec.get("recommended_power", power)))
            if kind != "hard":
                status = status_from_power(power, warn=(kind == "warn"))

    reason = " / ".join(reasons)
    alpha = 0.25
    power = 0.0 if st.session_state.latched else float(np.clip(p_power * (1 - alpha) + power * alpha, 0, 100))

    i_ch = 0.0 if power <= 1 else I_FULL * (power / 100.0)
    if i_ch > 0 and st.session_state.soc < 100:
        st.session_state.soc = min(100.0, st.session_state.soc + i_ch * SOC_PER_A)
    soc = st.session_state.soc

    h1 = 0.0 if st.session_state.latched else float(fault_pct)
    h2 = 0.5 * h1 if scen.startswith("C") else 0.0
    sendA(h1, h2, force_off=st.session_state.latched)
    sendB(power, blocked=st.session_state.latched, warn=(kind == "warn"),
           ai_block=("AI" in reason), swelling=cam_bad)

    now = time.time()
    if st.session_state.latched and now - st.session_state.last_alert > 25:
        send_tg(f"충전 차단 {reason}")
        st.session_state.last_alert = now
    if score >= SCORE_ALERT and now - st.session_state.last_score_alert > 25:
        send_tg(f"AI 위험도 {score:.0f}% (기준 {SCORE_ALERT:.0f}%) Tmax={tmax:.1f}")
        st.session_state.last_score_alert = now

    if int(time.time()) % 2 == 0:
        append_log({"time": time.strftime("%H:%M:%S"), "T_max": round(float(tmax), 2),
                    "soc": round(soc, 1), "i": round(i_meas, 3), "speed": round(power, 1),
                    "score": round(score, 1), "cam": cam["tag"], "status": status, "reason": reason})

    st.session_state.hist = pd.concat([st.session_state.hist, pd.DataFrame([{
        "Time": time.strftime("%H:%M:%S"), "T1": t1, "T2": t2, "T3": t3, "Delta_T": dtv,
        "Score": score, "Power": power, "SoC": soc, "Ich": i_meas,
        "Cam": f"{cam['tag']} {cam['prob']:.0%}", "State": b_state,
        "Amb": ambient, "Rise10": rise10
    }])], ignore_index=True).tail(90)
    return {"T_max": tmax, "Delta_T": dtv, "R_est": r, "AI_Score": score, "Status": status,
            "Power": power, "SoC": soc, "I_meas": i_meas, "Reason": reason,
            "Margin": margin, "Rise10": rise10, "Ambient": ambient,
            "Cam": f"{cam['tag']} {cam['prob']:.0%}"}

def fmt_c(x, signed=False):
    if x is None or not np.isfinite(x):
        return "—"
    return f"{x:+.1f}°C" if signed else f"{x:.1f}°C"

if run or len(st.session_state.hist) == 0:
    latest = step()
else:
    h = st.session_state.hist.iloc[-1]
    tm = max(h.T1, h.T2, h.T3)
    cam = st.session_state.cam
    latest = {"T_max": tm, "Delta_T": h.Delta_T, "R_est": 0.5, "AI_Score": h.Score,
              "Status": "일시정지", "Power": h.Power, "SoC": h.SoC, "I_meas": h.Ich,
              "Reason": "", "Margin": tm - h.Amb if np.isfinite(h.Amb) else float("nan"),
              "Rise10": h.Rise10, "Ambient": h.Amb, "Cam": f"{cam['tag']} {cam['prob']:.0%}"}

if run:
    interval = 20 if mode == "시뮬레이션" else 80
    changed = latest["Status"] != st.session_state.last_status
    if changed:
        st.session_state.timeline.append({"time": time.strftime("%H:%M:%S"), "status": latest["Status"],
                                          "power": latest["Power"], "reason": latest.get("Reason", "")})
        st.session_state.timeline = st.session_state.timeline[-12:]
    if not latest.get("NoData") and (changed or time.time() - st.session_state.last_brief > interval):
        st.session_state.report = ask_brief(latest, st.session_state.timeline)
        st.session_state.last_brief = time.time()
        st.session_state.last_status = latest["Status"]
    elif changed:
        st.session_state.last_status = latest["Status"]
if run and ai_auto and not latest.get("NoData") and not st.session_state.latched and latest["AI_Score"] >= 18 and time.time() - st.session_state.last_ai > 25:
    st.session_state.decision = ask_decision(latest)
    st.session_state.last_ai = time.time()

st.title("🛡️ Battery Sentinel")
st.caption("AI·온도·영상 팽창으로 충전을 조절하거나 차단")
stt = latest["Status"]
if "차단" in stt:
    st.error(stt)
elif latest.get("NoData"):
    st.info(stt)
elif "⚠️" in stt or "저속" in stt or "제한" in stt:
    st.warning(stt)
else:
    st.success(stt)

c = st.columns(6)
c[0].metric("충전량", f"{latest.get('SoC', 0):.1f}%")
c[1].metric("충전 속도", f"{latest['Power']:.0f}%")
c[2].metric("전류 (INA219)", f"{latest.get('I_meas', 0):.2f} A")
c[3].metric("최고 온도", fmt_c(latest["T_max"]), delta=f"주변 {fmt_c(latest.get('Margin'), True)}", delta_color="off")
c[4].metric("AI 위험도", f"{latest['AI_Score']:.1f}%")
c[5].metric("영상 판정", latest.get("Cam", "-"))
st.caption(f"판단: {latest.get('Reason','')}  |  10초 상승 {latest.get('Rise10', 0):.2f}°C  |  주변 {fmt_c(latest.get('Ambient'))}")
st.progress(min(1.0, latest.get("SoC", 0) / 100.0))

g1, g2 = st.columns(2)
with g1:
    f = go.Figure()
    f.add_trace(go.Scatter(x=st.session_state.hist.Time, y=st.session_state.hist.SoC, name="충전량 %"))
    f.add_trace(go.Scatter(x=st.session_state.hist.Time, y=st.session_state.hist.Power, name="충전 속도 %"))
    f.add_trace(go.Scatter(x=st.session_state.hist.Time, y=st.session_state.hist.Score, name="AI 위험도"))
    f.update_layout(height=240, margin=dict(l=0, r=0, t=8, b=0))
    st.plotly_chart(f, use_container_width=True)
with g2:
    f2 = go.Figure()
    for coln in ["T1", "T2", "T3"]:
        f2.add_trace(go.Scatter(x=st.session_state.hist.Time, y=st.session_state.hist[coln], name=coln))
    f2.update_layout(height=240, margin=dict(l=0, r=0, t=8, b=0))
    st.plotly_chart(f2, use_container_width=True)

st.markdown("**AI 브리핑**")
st.info(st.session_state.report)
for ev in reversed(st.session_state.timeline[-8:]):
    st.write(f"`{ev['time']}` {ev['status']} · {ev['power']:.0f}% · {ev['reason']}")
if st.session_state.decision:
    st.write("**AI 의사결정**", st.session_state.decision)
if run:
    time.sleep(1)
    st.rerun()