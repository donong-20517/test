        )

        st.divider()

        st.markdown(
            "### 🔎 예측 오차의 원인 살펴보기"
        )

        st.write(
            "예측 오차가 큰 자료의 특징을 살펴보고 "
            "데이터의 상태와 변수의 분포를 확인합니다."
        )

        st.markdown(
            "#### ① 입력 변수의 결측값과 불검출"
        )

        original_test = df.loc[
            result["model_df"].index[
                result["test_idx"]
            ]
        ]

        quality_rows = []

        for feature in result["features"]:

            missing_count = original_test[
                feature
            ].apply(is_missing).sum()

            nd_count = (

                original_test[
                    feature
                ].apply(is_non_detect)

                & ~

                original_test[
                    feature
                ].apply(is_missing)

            ).sum()

            quality_rows.append({

                "입력 변수": feature,

                "테스트 자료 결측값": int(
                    missing_count
                ),

                "테스트 자료 불검출값": int(
                    nd_count
                )

            })

        st.dataframe(
            pd.DataFrame(quality_rows),
            use_container_width=True,
            hide_index=True
        )

        st.markdown(
            "#### ② 변수별 값의 분포"
        )

        for feature in result["features"]:

            fig = px.histogram(
                x=result["model_df"][
                    feature
                ].dropna(),
                nbins=25,
                labels={
                    "x": feature,
                    "count": "자료 수"
                },
                color_discrete_sequence=[
                    "#a8d8c8"
                ]
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

        st.markdown(
            "#### ③ 예측 오차 해석 시 유의 사항"
        )

        st.markdown("""
        - 결측값이나 불검출값이 많으면 실제 측정값 대신 학습 자료의 중앙값으로 대체된 자료가 많아질 수 있습니다.
        - 일부 수질 항목의 값이 특정 범위에 집중되거나 큰 값이 드물게 나타나면 예측 오차에 영향을 줄 수 있습니다.
        - 선택한 변수만으로 전기전도도와 관련된 모든 특성을 설명하지 못할 가능성이 있습니다.
        - 측정 지점이나 측정 시기에 따른 차이도 검토할 필요가 있습니다.
        - 위 특징들은 오차와 관련될 가능성을 보여 줄 뿐, 특정 원인이 오차를 발생시켰다고 확정할 수는 없습니다.
        """)

        st.divider()

        st.markdown(
            "### 📥 예측 결과 다운로드"
        )

        csv = prediction_df.to_csv(
            index=False,
            encoding="utf-8-sig"
        )

        st.download_button(
            "예측 결과 CSV 다운로드",
            data=csv,
            file_name="groundwater_prediction.csv",
            mime="text/csv"
        )


# ==================================================
# 11. 하단 안내
# ==================================================

st.divider()

st.markdown("""
<div style="
    text-align:center;
    padding:20px;
    color:#83aab0;
">
    💧 지하수 전기전도도 예측기 🫧<br>
    <span style="font-size:13px;">
        실제 측정 자료를 바탕으로 살펴보는 지하수 수질 분석
    </span><br>
    ✨ 🐚 ✨
</div>
""", unsafe_allow_html=True)
