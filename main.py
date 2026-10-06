import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

st.set_page_config(
    page_title="서울 기온 예측기 (다항 회귀)", page_icon="📈", layout="wide"
)

st.title("📈 서울 연평균 기온 다항 회귀 분석 (1차 vs 3차 vs 9차)")
st.write(
    "2005년 이전 데이터로 모델을 학습하고, 2005년 이후 테스트 데이터로 평균 오차(MAE)와 2050년 예측값을 평가합니다."
)


# 1. 데이터 로드 및 전처리
@st.cache_data
def load_and_preprocess_data():
    url = "https://raw.githubusercontent.com/greatsong/modudata/bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"
    df = pd.read_csv(url, encoding="utf-8")

    # 컬럼 공백 제거 및 날짜 형변환
    df.columns = df.columns.str.strip()
    df["날짜"] = pd.to_datetime(df["날짜"])
    df["연도"] = df["날짜"].dt.year

    # 결측치 제거 및 2025년 이하 데이터만 추출
    df = df.dropna(subset=["평균기온"])
    df = df[df["연도"] <= 2025]

    # 관측일 300일 이상 연도 필터링
    day_counts = df.groupby("연도")["평균기온"].count()
    valid_years = day_counts[day_counts >= 300].index
    df_valid = df[df["연도"].isin(valid_years)]

    # 연도별 평균기온 계산
    annual_df = (
        df_valid.groupby("연도")["평균기온"].mean().reset_index()
    )
    return annual_df.sort_values("연도").reset_index(drop=True)


try:
    annual_df = load_and_preprocess_data()
except Exception as e:
    st.error(f"데이터를 불러오는 중 오류가 발생했습니다: {e}")
    st.stop()

# 2. 훈련용 / 테스트용 데이터 분할
# 2005년 이전(< 2005): 훈련용 / 2005년부터(>= 2005): 테스트용
train_df = annual_df[annual_df["연도"] < 2005]
test_df = annual_df[annual_df["연도"] >= 2005]

num_train = len(train_df)
num_test = len(test_df)

X_train = train_df[["연도"]].values
y_train = train_df["평균기온"].values

X_test = test_df[["연도"]].values
y_test = test_df["평균기온"].values

# 3. 데이터 개수 안내
st.info(
    f"📌 **데이터 분할 현황**: "
    f"훈련용 데이터 (**2005년 이전**): **{num_train}개 해** ({train_df['연도'].min()}년 ~ {train_df['연도'].max()}년) | "
    f"테스트용 데이터 (**2005년부터**): **{num_test}개 해** ({test_df['연도'].min()}년 ~ {test_df['연도'].max()}년)"
)

# 4. 차수별 모델 학습 및 평가 (1차, 3차, 9차)
degrees = [1, 3, 9]
results = []
models = {}

X_2050 = np.array([[2050]])

for d in degrees:
    # 연도 수치 안정성을 위해 StandardScaler + PolynomialFeatures 적용
    model = make_pipeline(
        StandardScaler(),
        PolynomialFeatures(degree=d, include_bias=False),
        LinearRegression(),
    )
    # 훈련용 데이터로만 학습
    model.fit(X_train, y_train)

    # 테스트 데이터 예측 및 채점 (MAE: 평균 절대 오차)
    y_pred_test = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred_test)

    # 2050년 기온 예측
    pred_2050 = model.predict(X_2050)[0]

    models[d] = model
    results.append(
        {
            "차수": f"{d}차 곡선" if d > 1 else "1차 (직선)",
            "테스트 데이터 평균 오차 (MAE)": f"{mae:.2f} °C",
            "2050년 예상 기온": f"{pred_2050:.2f} °C",
        }
    )

# 5. 평가 결과 표 출력
st.subheader("📊 모델별 테스트 데이터 평가 및 2050년 예측 비교")
st.table(pd.DataFrame(results))

# 6. Plotly 시각화
st.subheader("📉 회귀 곡선 추세 비교 그래프")

# 그래프 표현 범위 (1908 ~ 2050년)
plot_years = np.linspace(1900, 2050, 300).reshape(-1, 1)

fig = go.Figure()

# 훈련용 관측 데이터 산점도
fig.add_trace(
    go.Scatter(
        x=train_df["연도"],
        y=train_df["평균기온"],
        mode="markers",
        name="훈련용 데이터 (< 2005)",
        marker=dict(color="#1f77b4", size=7, opacity=0.7),
        hovertemplate="훈련 %{x}년: %{y:.2f}°C<extra></extra>",
    )
)

# 테스트용 관측 데이터 산점도
fig.add_trace(
    go.Scatter(
        x=test_df["연도"],
        y=test_df["평균기온"],
        mode="markers",
        name="테스트용 데이터 (>= 2005)",
        marker=dict(color="#2ca02c", size=9, symbol="diamond"),
        hovertemplate="테스트 %{x}년: %{y:.2f}°C<extra></extra>",
    )
)

# 차수별 곡선 그리기
colors = {1: "#ff7f0e", 3: "#9467bd", 9: "#d62728"}
dash_styles = {1: "solid", 3: "dash", 9: "dot"}

for d in degrees:
    y_plot = models[d].predict(plot_years)
    fig.add_trace(
        go.Scatter(
            x=plot_years.flatten(),
            y=y_plot,
            mode="lines",
            name=f"{d}차 곡선 예측",
            line=dict(color=colors[d], width=2.5, dash=dash_styles[d]),
            hovertemplate=f"%{{x:.0f}}년 ({d}차): %{{y:.2f}}°C<extra></extra>",
        )
    )

fig.update_layout(
    xaxis_title="연도 (Year)",
    yaxis_title="평균기온 (°C)",
    xaxis=dict(tickformat="d", range=[1895, 2055]),
    yaxis=dict(range=[8, 25]),  # 9차 곡선의 극단적 이탈을 고려해 Y축 범위 고정
    hovermode="closest",
    legend=dict(
        yanchor="top", y=0.99, xanchor="left", x=0.01, bgcolor="rgba(255,255,255,0.8)"
    ),
    height=550,
)

st.plotly_chart(fig, use_container_width=True)
