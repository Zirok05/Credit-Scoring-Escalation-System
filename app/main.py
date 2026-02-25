import streamlit as st
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

st.set_page_config(
    page_title="GiveMeSomeCredit",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="collapsed"  # ← сворачивает сайдбар по умолчанию
)


st.title("🏦 GiveMeSomeCredit - Кредитный скоринг")
st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    st.subheader("📝 Анкета")
    if st.button("Перейти к анкете"):
        st.switch_page("pages/application.py")  # ← вызовет main()

with col2:
    st.subheader("📊 Симуляция")
    if st.button("Перейти к симуляции"):
        st.switch_page("pages/simulation.py")  # ← вызовет main()

st.markdown("---")

# streamlit run app/main.py

