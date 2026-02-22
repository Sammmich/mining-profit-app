# -*- coding: utf-8 -*-
"""
Streamlit-сервис для предсказания Profit по признакам блока.
Готов к публикации на Streamlit Cloud: при отсутствии модели обучает её из встроенной выборки.
"""

import tempfile
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

from model_trainer import train_and_save

SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_PATH = SCRIPT_DIR / "model_artifacts" / "profit_model.joblib"

st.set_page_config(
    page_title="Прогноз Profit — блоковая модель",
    page_icon="⛏️",
    layout="centered",
)

st.title("⛏️ Прогноз прибыли по блоку (линейная регрессия)")
st.markdown("Введите параметры блока для предсказания **Profit (USD)**.")

# Загрузка модели (если нет — обучаем из встроенной выборки и сохраняем во временную папку)
@st.cache_resource
def load_model():
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)
    with st.spinner("Модель не найдена. Обучаем на встроенной выборке…"):
        temp_dir = Path(tempfile.gettempdir())
        temp_model = temp_dir / "profit_model.joblib"
        train_and_save(temp_model)
        return joblib.load(temp_model)

artifacts = load_model()

pipeline = artifacts["pipeline"]
numeric_features = artifacts["numeric_features"]
cat_feature = artifacts["cat_feature"]
target_col = artifacts["target_col"]

# Форма ввода
with st.form("block_params"):
    st.subheader("Координаты блока")
    c1, c2, c3 = st.columns(3)
    with c1:
        x = st.number_input("X", min_value=0, max_value=500, value=200, step=1)
    with c2:
        y = st.number_input("Y", min_value=0, max_value=500, value=250, step=1)
    with c3:
        z = st.number_input("Z", min_value=0, max_value=100, value=50, step=1)

    st.subheader("Тип породы и руда")
    rock_type = st.selectbox(
        "Rock_Type",
        options=["Magnetite", "Hematite", "Waste"],
        index=0,
    )
    ore_grade = st.slider("Ore_Grade (%)", 0.0, 100.0, 55.0, 0.5)
    tonnage = st.number_input("Tonnage", min_value=100, max_value=5000, value=2000, step=50)
    ore_value = st.number_input("Ore_Value (USD/tonne)", 0.0, 5.0, 1.9, 0.01)

    st.subheader("Затраты")
    mining_cost = st.number_input("Mining_Cost (USD)", 0.0, 1.0, 0.35, 0.01)
    processing_cost = st.number_input("Processing_Cost (USD)", 0.0, 1.0, 0.22, 0.01)

    waste_flag = 1 if rock_type == "Waste" else 0
    if rock_type == "Waste":
        ore_grade_effective = 0.0
        ore_value_effective = 0.0
    else:
        ore_grade_effective = ore_grade
        ore_value_effective = ore_value

    if rock_type == "Waste":
        st.caption("Для Waste Ore_Grade и Ore_Value при расчёте берутся как 0.")

    submitted = st.form_submit_button("Рассчитать Profit")

if submitted:
    row = pd.DataFrame([{
        "X": x,
        "Y": y,
        "Z": z,
        "Ore_Grade (%)": ore_grade_effective,
        "Tonnage": tonnage,
        "Ore_Value (USD/tonne)": ore_value_effective,
        "Mining_Cost (USD)": mining_cost,
        "Processing_Cost (USD)": processing_cost,
        "Waste_Flag": waste_flag,
        "Rock_Type": rock_type,
    }])
    pred = pipeline.predict(row)[0]
    st.success(f"**Предсказанный {target_col}:** {pred:,.2f} USD")
    if pred >= 0:
        st.info("Блок целесообразен к добыче (ожидаемая прибыль положительная).")
    else:
        st.warning("Блок убыточен (waste или низкая ценность руды).")

st.divider()
st.caption("Модель: линейная регрессия по признакам X, Y, Z, Rock_Type, Ore_Grade, Tonnage, Ore_Value, Mining_Cost, Processing_Cost, Waste_Flag.")
