
import re
from io import BytesIO

import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


# =====================================================
# 1. 기본 설정
# =====================================================

st.set_page_config(
    page_title="지하수 전기전도도 예측기",
    page_icon="💧",
    layout="wide"
)

DATA_URL = (
    "https://www.data.go.kr/cmm/cmm/fileDownload.do"
    "?atchFileId=FILE_000000003114451"
    "&fileDetailSn=1"
    "&insertDataPrcus=N"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Gowun+Dodum&family=Jua&display=swap');

html, body, [class*="css"] {
    font-family: 'Gowun Dodum', sans-serif;
}

.stApp {
    background: linear-gradient(180deg, #effaff 0%, #f7fffc 48%, #fffaf0 100%);
}

.block-container {
    max-width: 1250px;
    padding-top: 2rem;
    padding-bottom: 4rem;
}

h1, h2, h3 {
    color: #467f91 !important;
    font-family: 'Jua', sans-serif !important;
}

p, label, li {
    color: #526d77;
}

[data-testid="stMetric"] {
    background: rgba(255,255,255,0.85);
    border: 1px solid #d9edf0;
    border-radius: 20px;
    padding: 20px;
    box-shadow: 0 5px 18px rgba(115, 175, 190, 0.10);
}

[data-testid="stMetricLabel"] {
    color: #7399a4;
}

[data-testid="stMetricValue"] {
    color: #43899a;
}

div[data-testid="stVerticalBlockBorderWrapper"] {
    border-radius: 20px;
}

.stTabs [data-baseweb="tab"] {
    border-radius: 12px;
    color: #598d9a;
}

.stTabs [aria-selected="true"] {
    background-color: #dff4f5 !important;
}

div.stButton > button {
    border-radius: 15px;
    border: 1px solid #b9dfe5;
    background-color: #e6f7f7;
    color: #467f91;
}

div.stButton > button:hover {
    background-color: #cceef0;
    color: #467f91;
}

div[data-testid="stDataFrame"] {
    border-radius: 15px;
    overflow: hidden;
}

hr {
    border-color: #d9edf0;
}

.mascot {
    background: linear-gradient(120deg, #dff5fa, #e8faf1);
    border: 1px solid #ccebef;
    border-radius: 24px;
    padding: 25px;
    margin: 15px 0 25px 0;
}

.mascot-title {
    font-family: 'Jua', sans-serif;
    font-size: 23px;
    color: #508b9b;
}

.mascot-text {
    color: #6b8e98;
    font-size: 15px;
    line-height: 1.8;
}

.section-note {
    background: #fffdf5;
    border-left: 4px solid #b8ded7;
    border-radius: 12px;
    padding: 13px 17px;
    color: #66877f;
    margin: 10px 0 20px 0;
}

</style>
""", unsafe_allow_html=True)


# =====================================================
# 2. 제목 및 소개
# =====================================================

st.markdown("""
<div class="mascot">
    <div style="font-size:38px;">💧 🫧 ✨ 🐚</div>
    <div class="mascot-title">지하수 전기전도도 예측기</div>
    <div class="mascot-text">
        물속에 녹아 있는 다양한 수질 성분을 살펴보고,<br>
        여러 성분이 전기전도도와 어떤 관계를 가지는지 분석해 봅니다.<br>
        데이터를 바탕으로 전기전도도를 예측하는 회귀 모델도 만들어 봅니다.
    </div>
    <div style="text-align:right; font-size:25px;">🫧 💧</div>
</div>
""", unsafe_allow_html=True)


# =====================================================
# 3. 실제 데이터 항목 이름과 의미
# =====================================================

MEANINGS = {
    "측정소": "지하수 수질을 측정한 장소",
    "구분": "측정 지점의 구분 정보",
    "채수일": "지하수 시료를 채취한 날짜",
    "ph": "물의 산성 또는 알칼리성 정도",
    "염소이온": "물에 녹아 있는 염소 이온의 농도",
    "질산성질소": "물에 포함된 질산성 질소의 농도",
    "시안": "물에 포함된 시안 성분의 농도",
    "비소": "물에 포함된 비소의 농도",
    "카드뮴": "물에 포함된 카드뮴의 농도",
    "수은": "물에 포함된 수은의 농도",
    "납": "물에 포함된 납의 농도",
    "크롬": "물에 포함된 크롬의 농도",
    "총대장균군": "물에 포함된 총대장균군의 수",
    "페놀": "물에 포함된 페놀의 농도",
    "벤젠": "물에 포함된 벤젠의 농도",
    "톨루엔": "물에 포함된 톨루엔의 농도",
    "에틸벤젠": "물에 포함된 에틸벤젠의 농도",
    "크실렌": "물에 포함된 크실렌의 농도",
    "111트리클로로에탄": "물에 포함된 1,1,1-트리클로로에탄의 농도",
    "테트라클로로에틸렌": "물에 포함된 테트라클로로에틸렌의 농도",
    "트리클로로에틸렌": "물에 포함된 트리클로로에틸렌의 농도",
    "다이아지논": "물에 포함된 다이아지논의 농도",
    "파라티온": "물에 포함된 파라티온의 농도",
    "전기전도도": "물속에 녹아 있는 이온에 의해 전기가 통하는 정도"
}

ALIASES = {
    "pH": ["ph", "수소이온농도"],
    "염소이온": ["염소이온"],
    "질산성질소": ["질산성질소"],
    "전기전도도": ["전기전도도"],
    "시안": ["시안"],
    "비소": ["비소"],
    "카드뮴": ["카드뮴"],
    "수은": ["수은"],
    "납": ["납"],
    "크롬": ["크롬"],
    "총대장균군": ["총대장균군"],
    "페놀": ["페놀"],
    "벤젠": ["벤젠"],
    "톨루엔": ["톨루엔"],
    "에틸벤젠": ["에틸벤젠"],
    "크실렌": ["크실렌"],
    "1,1,1-트리클로로에탄": ["111트리클로로에탄"],
    "테트라클로로에틸렌": ["테트라클로로에틸렌"],
    "트리클로로에틸렌": ["트리클로로에틸렌"],
    "다이아지논": ["다이아지논"],
    "파라티온": ["파라티온"]
}

DISPLAY_NAMES = {
    "pH": "pH",
    "염소이온": "염소이온",
    "질산성질소": "질산성질소",
    "전기전도도": "전기전도도"
}


def normalize_name(name):
    return re.sub(
        r"[^가-힣a-z0-9]",
        "",
        str(name).lower().strip()
    )


def find_column(columns, aliases):
    for column in columns:
        normalized = normalize_name(column)

        for alias in aliases:
            target = normalize_name(alias)

            if normalized == target or normalized.startswith(target):
                return column

    return None


def get_meaning(column):
    normalized = normalize_name(column)

    for key, meaning in MEANINGS.items():
        if normalize_name(key) in normalized:
            return meaning

    return "공식 데이터의 항목명 확인 필요"


# =====================================================
# 4. 결측값 및 불검출 처리
# =====================================================

MISSING_VALUES = {
    "", "nan", "na", "n/a", "null",
    "none", "자료없음", "미측정"
}


def is_missing(value):
    return str(value).strip().lower() in MISSING_VALUES


def is_non_detect(value):
    value = str(value).strip().lower()

    if any(x in value for x in [
        "불검출", "검출한계미만", "정량한계미만"
    ]):
        return True

    if value in {"nd", "n.d."}:
        return True

    if re.match(r"^<\s*\d", value):
        return True

    return False


def numeric_series(series):
    cleaned = series.astype(str).str.strip()

    invalid = (
        cleaned.apply(is_missing)
        | cleaned.apply(is_non_detect)
    )

    cleaned = cleaned.str.replace(",", "", regex=False)

    result = pd.to_numeric(cleaned, errors="coerce")
    result[invalid] = np.nan

    return result


# =====================================================
# 5. 데이터 불러오기
# =====================================================

@st.cache_data(ttl=3600)
def load_data():
    response = requests.get(DATA_URL, timeout=60)
    response.raise_for_status()

    content = response.content

    if not content:
        raise ValueError("다운로드한 파일이 비어 있습니다.")

    if content.lstrip().startswith(b"<"):
        raise ValueError("CSV 대신 웹페이지가 반환되었습니다.")

    errors = []

    for encoding in ["utf-8-sig", "cp949", "euc-kr", "utf-8"]:
        try:
            df = pd.read_csv(
                BytesIO(content),
                encoding=encoding,
                dtype=str,
                keep_default_na=False,
                sep=None,
                engine="python"
            )
            break
        except Exception as e:
            errors.append(str(e))
    else:
        raise ValueError("\n".join(errors))

    df.columns = [str(c).strip() for c in df.columns]

    df = df.dropna(how="all")

    for column in df.columns:
        df[column] = df[column].astype(str).str.strip()

    return df


try:
    with st.spinner("맑은 지하수 데이터를 불러오고 있어요..."):
        df = load_data()

except Exception as e:
    st.error("데이터를 불러오지 못했습니다.")
    st.exception(e)
    st.stop()


if df.empty:
    st.warning("데이터가 없습니다.")
    st.stop()


# =====================================================
# 6. 실제 데이터 열 확인
# =====================================================

COLUMN_MAP = {}

for name, aliases in ALIASES.items():
    found = find_column(df.columns, aliases)

    if found is not None:
        COLUMN_MAP[name] = found


ec_column = COLUMN_MAP.get("전기전도도")

if ec_column is None:
    st.error(
        "실제 데이터에서 전기전도도 열을 찾지 못했습니다. "
        "아래 실제 열 이름을 확인해 주세요."
    )
    st.write(list(df.columns))
    st.stop()


# 수질 성분 열 중 실제로 존재하는 열만 사용
analyte_columns = []

for name, column in COLUMN_MAP.items():
    if name != "전기전도도":
        analyte_columns.append(column)


# 각 수질 성분의 수치형 데이터 생성
numeric_data = pd.DataFrame(index=df.index)

for column in analyte_columns + [ec_column]:
    numeric_data[column] = numeric_series(df[column])


# =====================================================
# 7. 결측값 및 불검출 정보
# =====================================================

def get_missing_count(column):
    return int(df[column].apply(is_missing).sum())


def get_non_detect_count(column):
    return int(
        (
            df[column].apply(is_non_detect)
            & ~df[column].apply(is_missing)
        ).sum()
    )


def get_numeric_count(column):
    return int(numeric_data[column].notna().sum())


# =====================================================
# 8. 상단 메뉴
# =====================================================

tabs = st.tabs([
    "💧 데이터 개요",
    "🔎 데이터 탐색",
    "🫧 상관관계 분석",
    "📈 전기전도도 예측",
    "📊 모델 성능 비교",
    "✨ 예측 결과 분석"
])


# =====================================================
# TAB 1. 데이터 개요
# =====================================================

with tabs[0]:

    st.header("💧 지하수 데이터 개요")

    st.write(
        "국가지하수 관리측정망의 수질 자료를 바탕으로 "
        "전체 데이터의 구조와 항목을 확인합니다."
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("전체 측정 자료", f"{len(df):,}개")

    with col2:
        st.metric("전체 항목 수", f"{len(df.columns):,}개")

    with col3:
        st.metric(
            "전기전도도 측정값",
            f"{get_numeric_count(ec_column):,}개"
        )

    with col4:
        st.metric(
            "전기전도도 결측값",
            f"{get_missing_count(ec_column):,}개"
        )

    st.markdown("### 🐚 데이터 항목과 의미")

    information = []

    for column in df.columns:
        information.append({
            "실제 데이터 열 이름": column,
            "항목의 의미": get_meaning(column),
            "값의 유형": (
                "수치형"
                if column in numeric_data.columns
                else "문자형 또는 날짜형"
            ),
            "결측값": get_missing_count(column),
            "불검출값": get_non_detect_count(column)
        })

    st.dataframe(
        pd.DataFrame(information),
        use_container_width=True,
        hide_index=True
    )

    st.markdown("### 🫧 원본 데이터 미리 보기")

    st.dataframe(
        df.head(10),
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        "출처: 한국수자원공사 국가지하수 관리측정망 수질정보"
    )


# =====================================================
# TAB 2. 데이터 탐색
# =====================================================

with tabs[1]:

    st.header("🔎 수질 항목별 데이터 탐색")

    st.write(
        "각 수질 항목의 값이 어떤 범위에 분포하는지 "
        "히스토그램을 통해 살펴봅니다."
    )

    histogram_columns = [
        ec_column
    ] + analyte_columns

    histogram_columns = list(dict.fromkeys(histogram_columns))

    default_hist = []

    for preferred in ["pH", "염소이온", "질산성질소", "전기전도도"]:
        actual = COLUMN_MAP.get(preferred)

        if actual in histogram_columns:
            default_hist.append(actual)

    selected_hist = st.multiselect(
        "히스토그램을 확인할 수질 항목을 선택하세요.",
        options=histogram_columns,
        default=default_hist
    )

    if selected_hist:

        for i in range(0, len(selected_hist), 2):

            cols = st.columns(2)

            for j, col in enumerate(cols):

                if i + j >= len(selected_hist):
                    continue

                column = selected_hist[i + j]

                values = numeric_data[column].dropna()

                with col:

                    st.markdown(f"#### 🫧 {column}")

                    if len(values) == 0:
                        st.warning("분석 가능한 수치형 자료가 없습니다.")
                        continue

                    fig = px.histogram(
                        x=values,
                        nbins=30,
                        labels={"x": column, "count": "자료 수"},
                        color_discrete_sequence=["#8ccfce"]
                    )

                    fig.update_layout(
                        height=300,
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="#ffffff",
                        margin=dict(l=20, r=20, t=20, b=30),
                        showlegend=False
                    )

                    fig.update_traces(
                        marker_line_color="#ffffff",
                        marker_line_width=1
                    )

                    st.plotly_chart(
                        fig,
                        use_container_width=True
                    )

                    st.markdown("**이 그래프로 알 수 있는 것**")

                    if len(values) > 1:
                        mean_value = values.mean()
                        median_value = values.median()

                        if mean_value > median_value * 1.2:
                            explanation = (
                                "평균이 중앙값보다 높아 큰 측정값이 "
                                "분포의 오른쪽에 영향을 주고 있을 가능성이 있습니다."
                            )
                        elif mean_value < median_value * 0.8:
                            explanation = (
                                "평균이 중앙값보다 낮아 작은 측정값이 "
                                "분포의 왼쪽에 영향을 주고 있을 가능성이 있습니다."
                            )
                        else:
                            explanation = (
                                "평균과 중앙값이 비교적 가까워 "
                                "분포의 중심이 크게 치우치지 않은 모습을 보입니다."
                            )
                    else:
                        explanation = "측정 자료가 충분하지 않아 분포 특징을 판단하기 어렵습니다."

                    st.markdown(
                        f'<div class="section-note">{explanation}</div>',
                        unsafe_allow_html=True
                    )

    else:
        st.info("확인할 항목을 선택해 주세요.")

    st.divider()

    st.subheader("📋 항목별 기초 통계")

    statistics = []

    for column in histogram_columns:

        values = numeric_data[column].dropna()

        statistics.append({
            "항목": column,
            "평균": values.mean() if len(values) else np.nan,
            "최솟값": values.min() if len(values) else np.nan,
            "최댓값": values.max() if len(values) else np.nan,
            "측정값 개수": len(values),
            "결측값": get_missing_count(column),
            "불검출값": get_non_detect_count(column)
        })

    stats_df = pd.DataFrame(statistics)

    st.dataframe(
        stats_df.style.format({
            "평균": "{:.4f}",
            "최솟값": "{:.4f}",
            "최댓값": "{:.4f}"
        }, na_rep="-"),
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        "평균, 최솟값, 최댓값은 수치로 변환할 수 있는 측정값만 사용합니다. "
        "불검출값은 0으로 바꾸지 않고 통계 계산에서 제외합니다."
    )


# =====================================================
# TAB 3. 상관관계 분석
# =====================================================

with tabs[2]:

    st.header("🫧 수질 항목과 전기전도도의 관계")

    st.write(
        "각 수질 성분의 값이 변할 때 전기전도도가 "
        "어떤 경향을 보이는지 살펴봅니다."
    )

    relationship_columns = [
        column
        for column in analyte_columns
        if numeric_data[column].notna().sum() >= 3
    ]

    if relationship_columns:

        selected_relationship = st.selectbox(
            "분석할 수질 항목을 선택하세요.",
            relationship_columns,
            index=(
                relationship_columns.index(COLUMN_MAP["pH"])
                if COLUMN_MAP.get("pH") in relationship_columns
                else 0
            )
        )

        pair = pd.DataFrame({
            "수질 항목": numeric_data[selected_relationship],
            "전기전도도": numeric_data[ec_column]
        }).dropna()

        st.markdown(f"### {selected_relationship}와 전기전도도")

        if len(pair) >= 3:

            correlation = pair["수질 항목"].corr(
                pair["전기전도도"]
            )

            fig = px.scatter(
                pair,
                x="수질 항목",
                y="전기전도도",
                opacity=0.65,
                color_discrete_sequence=["#72bfc0"]
            )

            if pair["수질 항목"].nunique() >= 2:

                x = pair["수질 항목"].to_numpy()
                y = pair["전기전도도"].to_numpy()

                slope, intercept = np.polyfit(x, y, 1)

                x_line = np.linspace(x.min(), x.max(), 100)
                y_line = slope * x_line + intercept

                fig.add_trace(
                    go.Scatter(
                        x=x_line,
                        y=y_line,
                        mode="lines",
                        name="선형 추세선",
                        line=dict(color="#e8a7a3", width=3)
                    )
                )

            fig.update_layout(
                height=430,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="#ffffff",
                margin=dict(l=20, r=20, t=30, b=30)
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

            col1, col2 = st.columns(2)

            with col1:
                st.metric(
                    "상관계수",
                    f"{correlation:.4f}"
                )

            with col2:
                st.metric(
                    "분석에 사용된 자료",
                    f"{len(pair):,}개"
                )

            st.markdown("**이 그래프로 알 수 있는 것**")

            if correlation >= 0.7:
                explanation = (
                    "두 항목 사이에 비교적 강한 양의 선형 상관관계가 나타나며, "
                    "수질 항목의 값이 커질수록 전기전도도도 커지는 경향을 보입니다."
                )
            elif correlation >= 0.3:
                explanation = (
                    "두 항목 사이에 어느 정도 양의 상관관계가 나타나며, "
                    "수질 항목의 값이 커질수록 전기전도도가 증가하는 경향이 있습니다."
                )
            elif correlation > -0.3:
                explanation = (
                    "두 항목 사이에 뚜렷한 선형 상관관계가 나타나지 않습니다."
                )
            elif correlation > -0.7:
                explanation = (
                    "두 항목 사이에 어느 정도 음의 상관관계가 나타나며, "
                    "수질 항목의 값이 커질수록 전기전도도가 감소하는 경향이 있습니다."
                )
            else:
                explanation = (
                    "두 항목 사이에 비교적 강한 음의 선형 상관관계가 나타납니다."
                )

            st.markdown(
                f'<div class="section-note">{explanation}</div>',
                unsafe_allow_html=True
            )

        else:
            st.warning("두 항목을 함께 측정한 자료가 부족합니다.")

    else:
        st.warning("분석 가능한 수질 항목이 없습니다.")

    st.divider()

    st.subheader("📋 수질 항목별 상관계수 비교")

    correlation_results = []

    for column in analyte_columns:

        pair = pd.DataFrame({
            "x": numeric_data[column],
            "y": numeric_data[ec_column]
        }).dropna()

        if len(pair) >= 3 and pair["x"].nunique() > 1:
            corr = pair["x"].corr(pair["y"])
        else:
            corr = np.nan

        correlation_results.append({
            "수질 항목": column,
            "상관계수": corr,
            "분석 자료 수": len(pair)
        })

    corr_df = pd.DataFrame(correlation_results)

    corr_df = corr_df.sort_values(
        "상관계수",
        ascending=False,
        key=lambda x: x.abs(),
        na_position="last"
    )

    st.dataframe(
        corr_df.style.format({
            "상관계수": "{:.4f}"
        }, na_rep="-"),
        use_container_width=True,
        hide_index=True
    )

    st.info(
        "상관계수의 절댓값이 1에 가까울수록 선형 관계가 강하고, "
        "0에 가까울수록 선형 관계가 약합니다. "
        "양수는 같은 방향, 음수는 반대 방향의 관계를 의미합니다. "
        "상관관계만으로 원인과 결과를 단정할 수는 없습니다."
    )


# =====================================================
# TAB 4. 전기전도도 예측
# =====================================================

with tabs[3]:

    st.header("📈 다중 선형회귀 전기전도도 예측")

    st.write(
        "사용자가 선택한 수질 항목을 독립 변수로 설정하고 "
        "전기전도도를 예측하는 회귀 모델을 학습합니다."
    )

    st.markdown("### 🫧 독립 변수 선택")

    default_features = []

    for preferred in ["pH", "염소이온", "질산성질소"]:
        actual = COLUMN_MAP.get(preferred)

        if actual in analyte_columns:
            default_features.append(actual)

    selected_features = st.multiselect(
        "예측에 사용할 수질 항목을 선택하세요.",
        options=analyte_columns,
        default=default_features
    )

    st.markdown(
        '<div class="section-note">'
        '선택한 수질 항목은 독립 변수(X), 전기전도도는 종속 변수(y)로 사용합니다.'
        '</div>',
        unsafe_allow_html=True
    )

    test_size = st.slider(
        "테스트 데이터 비율",
        min_value=0.1,
        max_value=0.4,
        value=0.2,
        step=0.05
    )

    if len(selected_features) == 0:

        st.warning("독립 변수를 하나 이상 선택해 주세요.")

    else:

        # 목표값이 실제 수치로 확인되는 자료만 사용
        model_df = pd.DataFrame(index=df.index)

        for feature in selected_features:
            model_df[feature] = numeric_data[feature]

        model_df["전기전도도"] = numeric_data[ec_column]

        model_df = model_df.dropna(subset=["전기전도도"])

        if len(model_df) < 10:

            st.warning("모델을 학습하기 위한 자료가 부족합니다.")

        else:

            train_idx, test_idx = train_test_split(
                np.arange(len(model_df)),
                test_size=test_size,
                random_state=42
            )

            X_train_all = model_df.iloc[train_idx][selected_features]
            X_test_all = model_df.iloc[test_idx][selected_features]

            y_train = model_df.iloc[train_idx]["전기전도도"]
            y_test = model_df.iloc[test_idx]["전기전도도"]

            # 학습 데이터에서 계산한 중앙값만 사용
            usable_features = []

            for feature in selected_features:
                median_value = X_train_all[feature].median()

                if pd.isna(median_value):
                    st.warning(
                        f"{feature}: 학습 데이터에 수치형 자료가 없어 "
                        "이번 모델에서 제외합니다."
                    )
                else:
                    usable_features.append(feature)

            if len(usable_features) == 0:

                st.error("학습할 수 있는 독립 변수가 없습니다.")

            else:

                X_train = X_train_all[usable_features].copy()
                X_test = X_test_all[usable_features].copy()

                medians = X_train.median()

                X_train = X_train.fillna(medians)
                X_test = X_test.fillna(medians)

                model = LinearRegression()

                model.fit(X_train, y_train)

                predictions = model.predict(X_test)

                r2 = r2_score(y_test, predictions)
                mae = mean_absolute_error(y_test, predictions)
                mse = mean_squared_error(y_test, predictions)

                st.markdown("### ✨ 모델 학습 결과")

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("결정계수 R²", f"{r2:.4f}")

                with col2:
                    st.metric("평균 절대 오차 MAE", f"{mae:.4f}")

                with col3:
                    st.metric("평균 제곱 오차 MSE", f"{mse:.4f}")

                st.markdown("### 🧪 회귀식 계수")

                coefficient_df = pd.DataFrame({
                    "독립 변수": usable_features,
                    "회귀 계수": model.coef_
                })

                st.dataframe(
                    coefficient_df.style.format({
                        "회귀 계수": "{:.6f}"
                    }),
                    use_container_width=True,
                    hide_index=True
                )

                st.write(
                    f"절편: **{model.intercept_:.6f}**"
                )

                st.markdown("### 📊 예측 결과")

                result_df = pd.DataFrame({
                    "실제 전기전도도": y_test.to_numpy(),
                    "예측 전기전도도": predictions
                })

                st.dataframe(
                    result_df.head(20).style.format("{:.4f}"),
                    use_container_width=True
                )

                st.caption(
                    f"전체 {len(model_df):,}개 자료 중 "
                    f"학습 {len(train_idx):,}개, "
                    f"테스트 {len(test_idx):,}개를 사용했습니다."
                )

                st.session_state["model_result"] = {
                    "features": usable_features,
                    "model": model,
                    "medians": medians,
                    "model_df": model_df,
                    "train_idx": train_idx,
                    "test_idx": test_idx,
                    "y_test": y_test,
                    "predictions": predictions,
                    "metrics": (r2, mae, mse)
                }

                st.success(
                    "모델 학습과 테스트 데이터 예측이 완료되었습니다."
                )


# =====================================================
# TAB 5. 모델 성능 비교
# =====================================================

with tabs[4]:

    st.header("📊 여러 변수 조합의 예측 성능 비교")

    st.write(
        "각 수질 항목을 단독으로 사용한 모델과 "
        "여러 항목을 함께 사용한 모델을 비교합니다."
    )

    st.info(
        "모든 모델은 동일한 데이터 분할을 사용하며, "
        "각 모델의 결측값 대체 기준은 학습 데이터에서만 계산합니다."
    )

    comparison_features = []

    for preferred in ["pH", "염소이온", "질산성질소"]:

        actual = COLUMN_MAP.get(preferred)

        if actual is not None and actual in analyte_columns:
            comparison_features.append(actual)

    if not comparison_features:
        st.warning("기본 비교 항목이 실제 데이터에서 발견되지 않았습니다.")

    else:

        comparison_df = pd.DataFrame({
            column: numeric_data[column]
            for column in comparison_features
        })

        comparison_df["전기전도도"] = numeric_data[ec_column]

        comparison_df = comparison_df.dropna(
            subset=["전기전도도"]
        )

        if len(comparison_df) < 10:

            st.warning("모델 비교를 위한 자료가 부족합니다.")

        else:

            train_idx, test_idx = train_test_split(
                np.arange(len(comparison_df)),
                test_size=0.2,
                random_state=42
            )

            y_train = comparison_df.iloc[train_idx]["전기전도도"]
            y_test = comparison_df.iloc[test_idx]["전기전도도"]

            comparison_models = []

            for feature in comparison_features:
                comparison_models.append({
                    "이름": f"{feature} 단독 모델",
                    "변수": [feature]
                })

            comparison_models.append({
                "이름": "여러 항목 통합 모델",
                "변수": comparison_features
            })

            results = []

            for item in comparison_models:

                features = item["변수"]

                X_train = comparison_df.iloc[train_idx][features].copy()
                X_test = comparison_df.iloc[test_idx][features].copy()

                medians = X_train.median()

                usable = [
                    column
                    for column in features
                    if pd.notna(medians[column])
                ]

                if not usable:
                    continue

                X_train = X_train[usable].fillna(medians[usable])
                X_test = X_test[usable].fillna(medians[usable])

                comparison_model = LinearRegression()

                comparison_model.fit(X_train, y_train)

                pred = comparison_model.predict(X_test)

                results.append({
                    "모델": item["이름"],
                    "사용 변수": ", ".join(usable),
                    "R²": r2_score(y_test, pred),
                    "MAE": mean_absolute_error(y_test, pred),
                    "MSE": mean_squared_error(y_test, pred)
                })

            result_table = pd.DataFrame(results)

            if not result_table.empty:

                st.markdown("### 📋 모델 성능 비교표")

                st.dataframe(
                    result_table.style.format({
                        "R²": "{:.4f}",
                        "MAE": "{:.4f}",
                        "MSE": "{:.4f}"
                    }),
                    use_container_width=True,
                    hide_index=True
                )

                st.markdown("### 📈 결정계수 비교")

                fig = px.bar(
                    result_table,
                    x="모델",
                    y="R²",
                    color="모델",
                    text_auto=".3f",
                    color_discrete_sequence=[
                        "#9ed8d5",
                        "#a9cde8",
                        "#b9dfcf",
                        "#e9d8ad"
                    ]
                )

                fig.update_layout(
                    showlegend=False,
                    height=380,
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="#ffffff"
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

                st.markdown("### 📉 오차 지표 비교")

                error_long = result_table.melt(
                    id_vars="모델",
                    value_vars=["MAE", "MSE"],
                    var_name="평가 지표",
                    value_name="오차"
                )

                fig = px.bar(
                    error_long,
                    x="모델",
                    y="오차",
                    color="평가 지표",
                    barmode="group",
                    color_discrete_sequence=["#e9b6aa", "#a8cde1"]
                )

                fig.update_layout(
                    height=380,
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="#ffffff"
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

                best_r2 = result_table.loc[
                    result_table["R²"].idxmax()
                ]

                best_mae = result_table.loc[
                    result_table["MAE"].idxmin()
                ]

                st.markdown("### 🐚 결과 해석")

                st.markdown(
                    f'<div class="section-note">'
                    f'결정계수가 가장 높은 모델은 '
                    f'<b>{best_r2["모델"]}</b>이며, '
                    f'R²는 {best_r2["R²"]:.4f}입니다.'
                    f'</div>',
                    unsafe_allow_html=True
                )

                st.markdown(
                    f'<div class="section-note">'
                    f'평균 절대 오차가 가장 작은 모델은 '
                    f'<b>{best_mae["모델"]}</b>이며, '
                    f'MAE는 {best_mae["MAE"]:.4f}입니다.'
                    f'</div>',
                    unsafe_allow_html=True
                )

                st.caption(
                    "R²는 1에 가까울수록 좋고, MAE와 MSE는 작을수록 좋습니다. "
                    "모델의 성능은 동일한 테스트 자료에서 비교해야 합니다."
                )


# =====================================================
# TAB 6. 예측 결과 분석
# =====================================================

with tabs[5]:

    st.header("✨ 실제 전기전도도와 예측값 비교")

    if "model_result" not in st.session_state:

        st.info(
            "먼저 '전기전도도 예측' 메뉴에서 모델을 학습해 주세요."
        )

    else:

        result = st.session_state["model_result"]

        actual = result["y_test"].to_numpy()
        predicted = result["predictions"]

        prediction_df = pd.DataFrame({
            "자료 번호": result["model_df"].index[result["test_idx"]] + 1,
            "실제 전기전도도": actual,
            "예측 전기전도도": predicted
        })

        prediction_df["오차"] = (
            prediction_df["실제 전기전도도"]
            - prediction_df["예측 전기전도도"]
        )

        prediction_df["절대 오차"] = (
            prediction_df["오차"].abs()
        )

        st.markdown("### 🫧 실제값과 예측값 산점도")

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=actual,
                y=predicted,
                mode="markers",
                name="예측 자료",
                marker=dict(
                    color="#7fc7c9",
                    size=9,
                    opacity=0.7
                )
            )
        )

        min_value = min(actual.min(), predicted.min())
        max_value = max(actual.max(), predicted.max())

        fig.add_trace(
            go.Scatter(
                x=[min_value, max_value],
                y=[min_value, max_value],
                mode="lines",
                name="실제값 = 예측값",
                line=dict(
                    color="#e6a9a2",
                    dash="dash",
                    width=3
                )
            )
        )

        fig.update_layout(
            xaxis_title="실제 전기전도도",
            yaxis_title="예측 전기전도도",
            height=480,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="#ffffff"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.markdown("**이 그래프로 알 수 있는 것**")

        st.markdown(
            '<div class="section-note">'
            '점들이 기준선에 가까울수록 실제 전기전도도와 '
            '모델의 예측값이 비슷하다는 의미입니다. '
            '기준선에서 멀리 떨어진 점은 예측 오차가 큰 자료입니다.'
            '</div>',
            unsafe_allow_html=True
        )

        st.divider()

        st.markdown("### 📋 예측 오차가 큰 자료")

        worst = prediction_df.sort_values(
            "절대 오차",
            ascending=False
        ).head(20)

        st.dataframe(
            worst.style.format({
                "실제 전기전도도": "{:.4f}",
                "예측 전기전도도": "{:.4f}",
                "오차": "{:.4f}",
                "절대 오차": "{:.4f}"
            }),
            use_container_width=True,
            hide_index=True
        )

        st.caption(
            "오차 = 실제 전기전도도 - 예측 전기전도도"
        )

        st.divider()

        st.markdown("### 🔎 예측이 잘 맞지 않는 이유 살펴보기")

        st.write(
            "오차가 큰 자료의 특징을 살펴보고, "
            "데이터의 상태가 예측 결과와 관련이 있을 가능성을 확인합니다."
        )

        st.markdown("#### ① 입력 변수의 결측값과 불검출")

        features = result["features"]
        model_df = result["model_df"]

        original_test = df.loc[
            model_df.index[result["test_idx"]]
        ]

        quality_rows = []

        for feature in features:

            original_column = feature

            missing_count = original_test[
                original_column
            ].apply(is_missing).sum()

            nd_count = (
                original_test[original_column].apply(is_non_detect)
                & ~original_test[original_column].apply(is_missing)
            ).sum()

            quality_rows.append({
                "입력 변수": feature,
                "테스트 자료 결측값": int(missing_count),
                "테스트 자료 불검출값": int(nd_count)
            })

        st.dataframe(
            pd.DataFrame(quality_rows),
            use_container_width=True,
            hide_index=True
        )

        st.markdown("#### ② 변수별 값의 분포")

        for feature in features:

            fig = px.histogram(
                x=model_df[feature].dropna(),
                nbins=25,
                labels={"x": feature, "count": "자료 수"},
                color_discrete_sequence=["#a8d8c8"]
            )

            fig.update_layout(
                title=f"{feature}의 분포",
                height=280,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="#ffffff"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        st.markdown("#### ③ 오차 분석 시 유의할 점")

        st.markdown("""
        - 일부 입력 변수에 결측값이나 불검출값이 많으면 실제 측정값 대신 학습 자료의 중앙값으로 대체한 자료가 많아질 수 있습니다.
        - 수질 항목의 분포가 한쪽으로 치우쳐 있으면 일부 큰 값에서 예측 오차가 커질 수 있습니다.
        - 선택한 변수만으로 전기전도도에 영향을 주는 여러 요인을 충분히 설명하지 못할 가능성도 있습니다.
        - 측정 지점이나 측정 시기에 따른 차이가 예측에 영향을 주었을 가능성도 검토할 수 있습니다.
        - 이러한 특징만으로 특정 원인이 예측 오차를 발생시켰다고 단정할 수는 없습니다.
        """)

        st.divider()

        st.markdown("### 📥 예측 결과 다운로드")

        csv = prediction_df.to_csv(
            index=False,
            encoding="utf-8-sig"
        )

        st.download_button(
            label="예측 결과 CSV 다운로드",
            data=csv,
            file_name="groundwater_prediction.csv",
            mime="text/csv"
        )


# =====================================================
# 10. 하단 안내
# =====================================================

st.divider()

st.markdown("""
<div style="text-align:center; padding:20px; color:#83aab0;">
    💧 지하수 전기전도도 예측기 🫧<br>
    <span style="font-size:13px;">
        실제 측정 자료를 바탕으로 살펴보는 지하수 수질 분석
    </span><br>
    ✨ 🐚 ✨
</div>
""", unsafe_allow_html=True)
