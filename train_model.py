import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import joblib
import os

# ======================
# 설정
# ======================
CSV_PATH = "EV_Battery_Charging_TR_Dataset_with_Notes.csv"  # 파일 경로 맞게 수정
MODEL_PATH = "isolation_forest_ev.pkl"

# ======================
# 1. 데이터 로드
# ======================
print("CSV 로드 중...")
df = pd.read_csv(CSV_PATH)
print(f"전체 행 수: {len(df)}")
print("\nEventFlag 분포:")
print(df["EventFlag"].value_counts())

# ======================
# 2. Feature 생성
# ======================
df["T_max"] = df["MaxTemp_C"]
df["Delta_T"] = df["MaxTemp_C"] - df["MinTemp_C"]
df["R_est"] = df["InternalResistance_mOhm"] / 1000.0  # mΩ → Ω

# dT/dt는 이 데이터가 셀별 연속 시계열이 아니므로 기본값 사용
# (원하면 나중에 Timestamp + CellID로 그룹해서 계산 가능)
df["dT_dt"] = 0.02

# ======================
# 3. 정상 데이터 필터링
# ======================
# EventFlag가 Normal 또는 Watch인 경우를 정상으로 간주
# (데이터에 따라 조정 가능)
normal_mask = df["EventFlag"].isin(["Normal", "Watch"])
# 만약 Normal이 거의 없으면 아래처럼 TR_Probability 기준으로 바꿔도 됨
# normal_mask = df["TR_Probability"] < 0.4

normal_df = df[normal_mask].copy()
print(f"\n정상으로 사용할 샘플 수: {len(normal_df)}")

if len(normal_df) < 50:
    print("경고: 정상 샘플이 너무 적습니다. EventFlag 조건을 완화하세요.")
    # fallback: TR_Probability가 낮은 것
    normal_df = df[df["TR_Probability"] < 0.5].copy()
    print(f"Fallback 정상 샘플 수: {len(normal_df)}")

# ======================
# 4. 학습용 Feature 추출 (첫 번째 데이터)
# ======================
features = ["T_max", "Delta_T", "dT_dt", "R_est"]
X1 = normal_df[features].dropna()
print(f"첫 번째 데이터 정상 샘플: {len(X1)}")

# ======================
# 4-2. 두 번째 데이터셋 추가
# ======================
print("\n두 번째 CSV 로드 중...")
df2 = pd.read_csv("ev_battery_failure_dataset.csv")

# 정상만 (battery_failure == 0)
normal2 = df2[df2["battery_failure"] == 0].copy()

normal2["T_max"] = normal2["cell_temperature_max"]
normal2["Delta_T"] = normal2["temperature_variance"].fillna(
    normal2["cell_temperature_max"] - normal2["cell_temperature_avg"]
)
normal2["R_est"] = normal2["internal_resistance"]   # 스케일이 작으면 나중에 조정
normal2["dT_dt"] = 0.02

X2 = normal2[features].dropna()
print(f"두 번째 데이터 정상 샘플: {len(X2)}")

# 합치기
X = pd.concat([X1, X2], ignore_index=True)
print(f"\n최종 합친 학습 샘플 수: {len(X)}")
print(X.describe())

# ======================
# 5. Isolation Forest 학습
# ======================
print("\n모델 학습 중...")
model = IsolationForest(
    n_estimators=150,
    contamination=0.05,      # 정상 데이터 안에 섞인 이상 비율 가정
    random_state=42,
    n_jobs=-1
)
model.fit(X)

# ======================
# 6. 저장
# ======================
joblib.dump(model, MODEL_PATH)
print(f"\n모델 저장 완료: {MODEL_PATH}")

# ======================
# 7. 간단 검증
# ======================
scores = model.decision_function(X)
print(f"정상 데이터 decision_function 평균: {scores.mean():.3f}")
print(f"정상 데이터 decision_function 최소: {scores.min():.3f}")