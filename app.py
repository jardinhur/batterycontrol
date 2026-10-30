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

HW_MAX_PWM = 30.0          # 보드로만 보냄. 화면에는 안 씀
I_FULL = 1.50              # 가상 만충 전류 (A)
SOC_RATE = 0.12            # 100% 속도일 때 초당 충전량 (%)
T_CUTOFF = 45.0
RISE_CUTOFF = 5.0
DEV_WARN, DEV_CUTOFF = 2.0, 5.0
RISE_WARN = 0.3
SENSOR_FAIL = -100

def to_hw_pwm(logical_pct):
    return float(np.clip(logical_pct / 100.0 * HW_MAX_PWM, 0, HW_MAX_PWM))

def pct_cmd(pct):
    pct = int(round(pct))
    if pct >= 95:
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

def decide(score, tmax, dtv, dtdt, r, rise5, sensor_ok, b_state):
    if b_state == "BLOCKED":
        return 0.0, "hard", ["B보드 BLOCKED"]
    if not sensor_ok:
        return 0.0, "hard", ["센서 읽기 실패"]
    if tmax >= T_CUTOFF:
        return 0.0, "hard", [f"절대온도 {tmax:.1f}≥{T_CUTOFF:.0f}"]
    if rise5 >= RISE_CUTOFF:
        return 0.0, "hard", [f"5초 상승 {rise5:.1f}°C"]
    if dtv >= DEV_CUTOFF:
        return 0.0, "hard", [f"편차 {dtv:.1f}°C"]
    kind, reasons = "ok", []
    if rise5 >= 1.5 or dtdt >= RISE_WARN:
        kind = "warn"
        reasons.append(f"상승 경고 {rise5:.1f}°C/5s")
    if dtv >= DEV_WARN:
        kind = "warn"
        reasons.append(f"편차 경고 {dtv:.1f}°C")
    tf = 1.0 if tmax <= 35 else max(0.15, 1.0 - (tmax - 35) / max(1, T_CUTOFF - 35))
    sf = 1.0 if score <= 15 else 0.25 if score >= 50 else 1.0 - (score - 15) / 35 * 0.6
    rf = 1.0 if r <= 0.85 else max(0.3, 1.0 - (r - 0.85) * 0.5)
    power = float(np.clip(100 * min(tf, sf, rf), 0, 100))
    if not reasons:
        reasons.append("정상" if power >= 85 else "온도/점수/저항 감속")
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
충전량={s.get('SoC',0):.1f}% 속도={s['Power']:.0f}% 전류={s.get('I_ch',0):.2f}A Tmax={s['T_max']:.1f} ΔT={s['Delta_T']:.2f} 점수={s['AI_Score']:.1f} 상태={s['Status']} 이유={s.get('Reason','')} 최근={hist}
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
        p = f"""JSON만. Tmax={s['T_max']:.1f} score={s['AI_Score']:.1f}
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
    "hist": pd.DataFrame(columns=["Time","T1","T2","T3","Delta_T","V","I","R","Score","Power","SoC","Ich","Relay","State"]),
    "sim": [26.0, 26.1, 26.0], "soc": 20.0, "report": "대기", "decision": None,
    "serA": None, "serB": None, "last_ai": 0, "last_brief": 0, "last_status": "",
    "last_alert": 0, "latched": False, "timeline": [], "last_cmdA": "", "last_cmdB": "",
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

st.sidebar.title("제어판")
st.sidebar.caption(f"모델: {model_source}")
mode = st.sidebar.radio("모드", ["시뮬레이션", "하드웨어"])
ports = [p.device for p in serial.tools.list_ports.comports()]
if mode == "하드웨어":
    pa = st.sidebar.selectbox("ESP32 A (히터)", ports or ["COM3"])
    pb = st.sidebar.selectbox("ESP32 B (센서·표시)", ports or ["COM4"])
    if st.sidebar.button("A 연결"):
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
            st.sidebar.success("B 연결")
        except Exception as e:
            st.sidebar.error(str(e))
scen = st.sidebar.selectbox("시나리오", ["A.정상", "B.저항↑", "C.HotSpot", "D.센서실패"])
local_ai = st.sidebar.toggle("로컬 AI 감속", True)
ai_auto = st.sidebar.toggle("AI 자동분석", True)
ai_ctrl = st.sidebar.toggle("AI 제어반영", False)
run = st.sidebar.checkbox("실시간", False)
if st.sidebar.button("가상 충전량 20%로"):
    st.session_state.soc = 20.0
if st.sidebar.button("🔓 차단 해제"):
    st.session_state.latched = False
    if st.session_state.serB and st.session_state.serB.is_open:
        try: st.session_state.serB.write(b"r")
        except Exception: pass

def sendA(hw1, hw2, force_off=False):
    if not (st.session_state.serA and st.session_state.serA.is_open):
        return
    try:
        cmd = "x" if force_off or (hw1 <= 1 and hw2 <= 1) else "h" + pct_cmd(hw1) + "j" + pct_cmd(hw2)
        if cmd != st.session_state.last_cmdA:
            st.session_state.serA.write(cmd.encode())
            st.session_state.last_cmdA = cmd
    except Exception:
        pass

def sendB(power, blocked=False, warn=False, ai_block=False):
    if not (st.session_state.serB and st.session_state.serB.is_open):
        return
    try:
        if blocked:
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
        "rise5": fnum(p[5], 0), "state": p[8].strip() if len(p) > 8 else "",
        "V": fnum(p[11], 12), "I": fnum(p[12], 0),
    }

def step():
    prev = st.session_state.hist.iloc[-1] if len(st.session_state.hist) else None
    p_power = float(prev["Power"]) if prev is not None else 0.0
    b_state, rise5 = "", 0.0

    if mode == "시뮬레이션":
        t1, t2, t3 = st.session_state.sim
        amb = 25.5
        hw_now = to_hw_pwm(p_power)
        i_heat = 1.5 * (hw_now / HW_MAX_PWM) if HW_MAX_PWM else 0
        r = 1.20 if scen.startswith("B") else 0.65 if scen.startswith("C") else 0.50
        p_heat = (i_heat ** 2) * r * HEAT_GAIN
        def ut(t, extra=0.0):
            return t + ((p_heat + extra) / C_TH - K_COOL * (t - amb)) * DT
        t1, t2 = ut(t1), ut(t2)
        t3 = -127 if scen.startswith("D") else ut(t3, 0.35 if scen.startswith("C") and p_power > 10 else 0.0)
        if not scen.startswith("D"):
            t1, t2, t3 = [max(20.0, x) for x in (t1, t2, t3)]
            st.session_state.sim = [t1, t2, t3]
        v = 12.0
        if len(st.session_state.hist) >= 5:
            rise5 = max(t1, t2, t3) - max(st.session_state.hist.iloc[-5][["T1", "T2", "T3"]])
    else:
        t1 = t2 = t3 = 25.0
        v, r = 12.0, 0.5
        if st.session_state.serB and st.session_state.serB.is_open:
            try:
                raw = st.session_state.serB.readline().decode(errors="ignore").strip()
                parsed = parse_b_line(raw)
                if parsed:
                    t1, t2, t3 = parsed["T1"], parsed["T2"], parsed["T3"]
                    v, rise5 = parsed["V"], parsed["rise5"]
                    b_state = parsed["state"]
                    r = 0.5
            except Exception:
                pass

    temps = [t1, t2, t3]
    sensor_ok = all(np.isfinite(t) and t > SENSOR_FAIL for t in temps)
    tmax = max(temps) if sensor_ok else -127
    dtv = (max(temps) - min(temps)) if sensor_ok else 0.0
    dtdt = rise5 / 5.0 if rise5 else 0.02
    raw = float(iso.decision_function([[25 if not sensor_ok else tmax, dtv, dtdt, r]])[0])
    score = float(np.clip((0.05 - raw) * 180, 0, 100))

    if st.session_state.latched or b_state == "BLOCKED":
        if b_state == "BLOCKED":
            st.session_state.latched = True
        power, kind, reasons = 0.0, "hard", ["차단 유지"]
        status = "🛑 충전 차단"
    else:
        power, kind, reasons = decide(score, tmax, dtv, dtdt, r, rise5, sensor_ok, b_state) if local_ai else (100.0, "ok", ["수동"])
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

    # 가상 충전 전류·충전량 (실제 충전 아님)
    i_ch = 0.0 if st.session_state.latched or power <= 1 else I_FULL * (power / 100.0)
    if i_ch > 0 and st.session_state.soc < 100:
        st.session_state.soc = min(100.0, st.session_state.soc + SOC_RATE * (power / 100.0))
    soc = st.session_state.soc

    hw = 0.0 if st.session_state.latched else to_hw_pwm(power)
    h1 = hw * 0.85
    h2 = min(HW_MAX_PWM, hw * 0.5) if (scen.startswith("C") and not st.session_state.latched and power > 15) else 0.0
    sendA(h1, h2, force_off=st.session_state.latched)
    sendB(power, blocked=st.session_state.latched, warn=(kind == "warn"), ai_block=("AI" in reason))

    if st.session_state.latched and time.time() - st.session_state.last_alert > 25:
        send_tg(f"충전 차단 {reason}")
        st.session_state.last_alert = time.time()
    if int(time.time()) % 2 == 0:
        append_log({"time": time.strftime("%H:%M:%S"), "T_max": round(float(tmax), 2),
                    "SoC": round(soc, 1), "Ich": round(i_ch, 2), "Power": round(power, 1),
                    "status": status, "reason": reason})

    st.session_state.hist = pd.concat([st.session_state.hist, pd.DataFrame([{
        "Time": time.strftime("%H:%M:%S"), "T1": t1, "T2": t2, "T3": t3, "Delta_T": dtv,
        "V": v, "I": i_ch, "R": r, "Score": score, "Power": power, "SoC": soc, "Ich": i_ch,
        "Relay": int(not st.session_state.latched), "State": b_state
    }])], ignore_index=True).tail(90)
    return {"T_max": tmax, "Delta_T": dtv, "R_est": r, "AI_Score": score, "Status": status,
            "Power": power, "SoC": soc, "I_ch": i_ch, "Reason": reason}

if run or len(st.session_state.hist) == 0:
    latest = step()
else:
    h = st.session_state.hist.iloc[-1]
    latest = {"T_max": max(h.T1, h.T2, h.T3), "Delta_T": h.Delta_T, "R_est": h.R,
              "AI_Score": h.Score, "Status": "일시정지", "Power": h.Power,
              "SoC": h.SoC, "I_ch": h.Ich, "Reason": ""}

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
st.caption("가상 충전 · 온도로 속도를 줄이거나 차단")
if "차단" in latest["Status"]:
    st.error(latest["Status"])
elif "⚠️" in latest["Status"] or "저속" in latest["Status"] or "제한" in latest["Status"]:
    st.warning(latest["Status"])
else:
    st.success(latest["Status"])

c = st.columns(5)
c[0].metric("충전량", f"{latest.get('SoC', 0):.1f}%")
c[1].metric("충전 전류", f"{latest.get('I_ch', 0):.2f} A")
c[2].metric("충전 속도", f"{latest['Power']:.0f}%")
c[3].metric("최고 온도", f"{latest['T_max']:.1f}°C")
c[4].metric("위험도", f"{latest['AI_Score']:.1f}%")
st.caption(f"판정: {latest.get('Reason','')}")
st.progress(min(1.0, latest.get("SoC", 0) / 100.0))

g1, g2 = st.columns(2)
with g1:
    f = go.Figure()
    f.add_trace(go.Scatter(x=st.session_state.hist.Time, y=st.session_state.hist.SoC, name="충전량 %"))
    f.add_trace(go.Scatter(x=st.session_state.hist.Time, y=st.session_state.hist.Power, name="충전 속도 %"))
    f.add_trace(go.Scatter(x=st.session_state.hist.Time, y=st.session_state.hist.Ich * 50, name="전류×50"))
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
st.markdown("**기록**")
for ev in reversed(st.session_state.timeline[-8:]):
    st.write(f"`{ev['time']}` {ev['status']} · 속도 {ev['power']:.0f}% · {ev['reason']}")
if st.session_state.decision:
    st.write("**AI 의사결정**", st.session_state.decision)
if run:
    time.sleep(1)
    st.rerun()