import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. 데이터 로드 및 전처리
url = "https://raw.githubusercontent.com/greatsong/modudata/bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"
df = pd.read_csv(url, encoding="utf-8")

# 컬럼명 공백 제거 및 날짜 형변환
df.columns = df.columns.str.strip()
df["날짜"] = pd.to_datetime(df["날짜"])
df["연도"] = df["날짜"].dt.year

# 2025년 이하 데이터 및 결측치 제거
df = df.dropna(subset=["평균기온"])
df = df[df["연도"] <= 2025]

# 관측일수 300일 이상 연도만 추출
day_counts = df.groupby("연도")["평균기온"].count()
valid_years = day_counts[day_counts >= 300].index
df_valid = df[df["연도"].isin(valid_years)]

# 연도별 평균기온 계산
annual_df = (
    df_valid.groupby("연도")["평균기온"].mean().reset_index()
)

# 2. Dataset 분할
# 공통 테스트 데이터: 최근 20년 (2006~2025)
test_df = annual_df[
    (annual_df["연도"] >= 2006) & (annual_df["연도"] <= 2025)
]

# 학습 데이터 1: 최근 50년 (1956~2005)
train_50 = annual_df[
    (annual_df["연도"] >= 1956) & (annual_df["연도"] <= 2005)
]

# 학습 데이터 2: 최근 100년 (1906~2005)
train_100 = annual_df[
    (annual_df["연도"] >= 1906) & (annual_df["연도"] <= 2005)
]

# 전체 데이터 (1908~2025)
train_all = annual_df.copy()


# 3. 모델 학습 및 평가 함수 정의
def evaluate_model(train_data, test_data, label="Model"):
    # 독립변수 X: 연도 - 1908, 종속변수 y: 평균기온
    X_train = train_data[["연도"]].values - 1908
    y_train = train_data["평균기온"].values

    X_test = test_data[["연도"]].values - 1908
    y_test = test_data["평균기온"].values

    # 선형회귀 모델적합
    model = LinearRegression()
    model.fit(X_train, y_train)

    # 예측 및 성능 평가
    y_pred = model.predict(X_test)

    slope = model.coef_[0]  # 1년당 기온 변화량
    slope_100yr = slope * 100  # 100년당 기온 상승 폭
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return {
        "모델": label,
        "학습 데이터 기간": f"{train_data['연도'].min()}~{train_data['연도'].max()} ({len(train_data)}개 해)",
        "기울기 (100년당)": f"+{slope_100yr:.3f} °C",
        "MAE": f"{mae:.4f}",
        "MSE": f"{mse:.4f}",
        "R² Score": f"{r2:.4f}",
    }


# 4. 각 모델 평가 수행 (공통 테스트 데이터 2006~2025 평가)
res_50 = evaluate_model(
    train_50, test_df, label="최근 50년 학습 모델 (1956~2005)"
)
res_100 = evaluate_model(
    train_100, test_df, label="최근 100년 학습 모델 (1906~2005)"
)
res_all = evaluate_model(
    train_all, train_all, label="전체 데이터 모델 (1908~2025 전체 적합)"
)

# 결과 출력
results_df = pd.DataFrame([res_50, res_100, res_all])
print(results_df.to_string(index=False))
