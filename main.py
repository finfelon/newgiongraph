import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(
    page_title="서울 기온 예측기 & 모델 비교", page_icon="🌡️", layout="wide"
)

st.title("🌡️ 서울 연평균 기온 예측기 및 모델 평가")
st.write(
    "서울 기온 데이터를 활용하여 최근 50년 학습 모델과 최근 100년 학습 모델의 예측 성능을 비교합니다."
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

    # 관측일이 300일 이상인 해만 필터링
    day_counts = df.groupby("연도")["평균기온"].count()
    valid_years = day_counts[day_counts >= 300].index
    df_valid = df[df["연도"].isin(valid_years)]

    # 연도별 평균기온 계산 및 정렬
    annual_df = (
        df_valid.groupby("연도")["평균기온"].mean().reset_index()
    )
    annual_df = annual_df.sort_values("연도").reset_index(drop=True)

    # 1908년 기준 경과 연수
    annual_df["X_1908"] = annual_df["연도"] - 1908

    return annual_df


try:
    annual_df = load_and_preprocess_data()
except Exception as e:
    st.error(f"데이터를 불러오는 중 오류가 발생했습니다: {e}")
    st.stop()

# 2. 데이터 분할
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


# 3. 모델 학습 및 평가 함수
def train_and_eval(train_data, test_data):
    X_train = train_data[["X_1908"]].values
    y_train = train_data["평균기온"].values

    X_test = test_data[["X_1908"]].values
    y_test = test_data["평균기온"].values

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    slope = model.coef_[0]
    intercept = model.intercept_
    rate_100yr = slope * 100

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return model, slope, intercept, rate_100yr, mae, mse, r2


# 각 모델 학습 및 공통 테스트 데이터(2006~2025) 평가
model_50, slope_50, intercept_50, rate_50, mae_50, mse_50, r2_50 = (
    train_and_eval(train_50, test_df)
)
model_100, slope_100, intercept_100, rate_100, mae_100, mse_100, r2_100 = (
    train_and_eval(train_100, test_df)
)
model_all, slope_all, intercept_all, rate_all, mae_all, mse_all, r2_all = (
    train_and_eval(train_all, train_all)
)

start_year = int(annual_df["연도"].min())
end_year = int(annual_df["연도"].max())
num_years = len(annual_df)
corr_all = np.corrcoef(annual_df["연도"], annual_df["평균기온"])[0, 1]

# 4. 상단 데이터 요약
st.info(
    f"📌 **분석 기준**: 총 **{num_years}개 해**의 데이터 ({start_year}년 ~ {end_year}년) | "
    f"공통 테스트 데이터: **최근 20년 (2006년 ~ 2025년)** | 전체 상관계수(r): **{corr_all:.4f}**"
)

# 5. 모델 비교 카드 (기울기 및 성능 평가)
st.subheader("🔥 학습 데이터 기간별 기울기 및 테스트 데이터(2006~2025) 평가")

col1, col2 = st.columns(2)

with col1:
    st.markdown(
        f"""
        <div style="background-color: #e8f4f8; padding: 20px; border-radius: 12px; border-left: 6px solid #1f77b4;">
            <h4 style="margin:0; color: #2c3e50;">📘 최근 50년 학습 모델 (1956~2005)</h4>
            <h2 style="margin:10px 0; color: #1f77b4;">+{rate_50:.2f} °C <span style="font-size: 1rem;">/ 100년</span></h2>
            <p style="margin:2px 0;"><b>MAE:</b> {mae_50:.4f} °C</p>
            <p style="margin:2px 0;"><b>MSE:</b> {mse_50:.4f}</p>
            <p style="margin:2px 0;"><b>R² Score:</b> {r2_50:.4f}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col2:
    st.markdown(
        f"""
        <div style="background-color: #fde8e8; padding: 20px; border-radius: 12px; border-left: 6px solid #d62728;">
            <h4 style="margin:0; color: #2c3e50;">📕 최근 100년 학습 모델 (1906~2005)</h4>
            <h2 style="margin:10px 0; color: #d62728;">+{rate_100:.2f} °C <span style="font-size: 1rem;">/ 100년</span></h2>
            <p style="margin:2px 0;"><b>MAE:</b> {mae_100:.4f} °C</p>
            <p style="margin:2px 0;"><b>MSE:</b> {mse_100:.4f}</p>
            <p style="margin:2px 0;"><b>R² Score:</b> {r2_100:.4f}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("---")

# 6. 연도 선택 슬라이더 및 예측 결과
st.subheader("🔮 연도별 예상 기온 예측")
selected_year = st.slider(
    "예측할 연도를 선택하세요",
    min_value=1900,
    max_value=2100,
    value=2026,
    step=1,
)

pred_x = np.array([[selected_year - 1908]])
pred_50 = model_50.predict(pred_x)[0]
pred_100 = model_100.predict(pred_x)[0]
pred_all = model_all.predict(pred_x)[0]

st.markdown(
    f"""
    <div style="background-color: #f8f9fa; padding: 20px; border-radius: 10px; border: 1px solid #e9ecef; text-align: center;">
        <h3 style="margin:0; color: #495057;">{selected_year}년 서울 예상 평균기온</h3>
        <h1 style="margin:10px 0; color: #ff4b4b; font-size: 3rem;">{pred_50:.2f} °C <span style="font-size: 1.2rem; color: #6c757d;">(최근 50년 모델 기준)</span></h1>
        <p style="margin:0; color: #6c757d;">최근 100년 모델 예측값: <strong>{pred_100:.2f} °C</strong> | 전체 기간 모델 예측값: <strong>{pred_all:.2f} °C</strong></p>
    </div>
    """,
    unsafe_allow_html=True,
)

# 7. Plotly 시각화
st.subheader("📊 연도별 기온 변화 및 회귀선 비교")

plot_years = np.arange(1900, 2101)
plot_x = (plot_years - 1908).reshape(-1, 1)

y_pred_50 = model_50.predict(plot_x)
y_pred_100 = model_100.predict(plot_x)
y_pred_all = model_all.predict(plot_x)

fig = go.Figure()

# 1) 전체 관측 데이터 산점도
fig.add_trace(
    go.Scatter(
        x=annual_df["연도"],
        y=annual_df["평균기온"],
        mode="markers",
        name="연평균 관측 기온",
        marker=dict(color="#7f7f7f", size=7, opacity=0.6),
        hovertemplate="%{x}년: %{y:.2f}°C<extra></extra>",
    )
)

# 2) 테스트 데이터 강조
fig.add_trace(
    go.Scatter(
        x=test_df["연도"],
        y=test_df["평균기온"],
        mode="markers",
        name="테스트 데이터 (2006~2025)",
        marker=dict(color="#2ca02c", size=9, symbol="circle"),
        hovertemplate="테스트 %{x}년: %{y:.2f}°C<extra></extra>",
    )
)

# 3) 최근 50년 회귀선
fig.add_trace(
    go.Scatter(
        x=plot_years,
        y=y_pred_50,
        mode="lines",
        name=f"최근 50년 학습 회귀선 (+{rate_50:.2f}°C/100년)",
        line=dict(color="#1f77b4", width=2.5),
        hovertemplate="%{x}년 (50년 모델): %{y:.2f}°C<extra></extra>",
    )
)

# 4) 최근 100년 회귀선
fig.add_trace(
    go.Scatter(
        x=plot_years,
        y=y_pred_100,
        mode="lines",
        name=f"최근 100년 학습 회귀선 (+{rate_100:.2f}°C/100년)",
        line=dict(color="#d62728", width=2.5, dash="dash"),
        hovertemplate="%{x}년 (100년 모델): %{y:.2f}°C<extra></extra>",
    )
)

# 5) 선택 연도 점 표시
fig.add_trace(
    go.Scatter(
        x=[selected_year],
        y=[pred_50],
        mode="markers",
        name=f"선택 연도 ({selected_year}년)",
        marker=dict(color="#ff7f0e", size=14, symbol="star"),
        hovertemplate=f"선택 연도: {selected_year}년<br>예상 기온: {pred_50:.2f}°C<extra></extra>",
    )
)

fig.update_layout(
    xaxis_title="연도 (Year)",
    yaxis_title="평균기온 (°C)",
    xaxis=dict(tickformat="d", range=[1895, 2105]),
    hovermode="closest",
    legend=dict(
        yanchor="top", y=0.99, xanchor="left", x=0.01, bgcolor="rgba(255,255,255,0.8)"
    ),
    height=550,
)

st.plotly_chart(fig, use_container_width=True)
