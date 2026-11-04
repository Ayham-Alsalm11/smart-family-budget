"""
💰 ميزان — مساعد الميزانية العائلي الذكي
streamlit run app.py
"""
import streamlit as st
import plotly.graph_objects as go
import os
from datetime import date
from src.chat.engine import ChatEngine
from src.rag.retriever import FinancialRetriever
from src.analysis.spending import SpendingAnalyzer
from src.database.models import init_db
from src.database.crud import get_family, update_income
from config import GROQ_API_KEY

st.set_page_config(page_title="ميزان — مساعد الميزانية", page_icon="💰", layout="wide")


def init_session():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "engine" not in st.session_state:
        api_key = os.getenv("GROQ_API_KEY", GROQ_API_KEY)
        if api_key and api_key != "YOUR_API_KEY_HERE":
            st.session_state.engine = ChatEngine(api_key=api_key)
            st.session_state.api_configured = True
        else:
            st.session_state.api_configured = False
    if "db_initialized" not in st.session_state:
        init_db()
        st.session_state.db_initialized = True


def render_sidebar():
    with st.sidebar:
        st.markdown("## ⚙️ الإعدادات")
        api_key = st.text_input("🔑 مفتاح Groq API", type="password",
                                value=os.getenv("GROQ_API_KEY", ""))
        if api_key and api_key != "YOUR_API_KEY_HERE":
            st.session_state.engine = ChatEngine(api_key=api_key)
            st.session_state.api_configured = True

        st.markdown("---")
        st.markdown("## 👨‍👩‍👧‍👦 العائلة")
        family = get_family(1)
        if family:
            new_income = st.number_input("الدخل الشهري (ل.س)",
                                         value=int(family.monthly_income), step=50000)
            if new_income != family.monthly_income:
                update_income(1, new_income)

        st.markdown("---")
        st.markdown("## 📊 ملخص سريع")
        if family:
            analyzer = SpendingAnalyzer(1)
            report = analyzer.get_monthly_report()
            col1, col2 = st.columns(2)
            with col1:
                st.metric("المصروفات", f"{report['total_spending']:,.0f}")
            with col2:
                st.metric("المتبقي", f"{report['remaining']:,.0f}")
            if report['income'] > 0:
                st.progress(min(report['total_spending'] / report['income'], 1.0))

        st.markdown("---")
        st.markdown("## 💡 أمثلة")
        for ex in ["دفعت 5000 بنزين", "كم صرفت هالشهر؟", "كيف وفّر؟", "اشتريت خضرا بـ 3000"]:
            if st.button(ex, use_container_width=True):
                st.session_state.pending_message = ex


def render_chart(data):
    if not data.get("labels"):
        return
    fig = go.Figure(data=[go.Pie(
        labels=data["labels"], values=data["values"],
        marker_colors=data["colors"], textinfo="label+percent", hole=0.4
    )])
    fig.update_layout(title="توزيع المصاريف", height=400,
                      legend=dict(orientation="h", y=-0.3))
    st.plotly_chart(fig, use_container_width=True)


def main():
    init_session()
    st.markdown("<h1 style='text-align:center'>💰 ميزان</h1>"
                "<p style='text-align:center;color:gray'>مساعد الميزانية العائلي الذكي</p>",
                unsafe_allow_html=True)
    render_sidebar()

    if not st.session_state.get("api_configured"):
        st.warning("⚠️ أدخل مفتاح Groq API في الشريط الجانبي.\nhttps://console.groq.com/keys")
        return

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"], avatar="👤" if msg["role"] == "user" else "💰"):
            st.markdown(msg["content"])
            if msg.get("chart_data"):
                render_chart(msg["chart_data"])

    pending = st.session_state.pop("pending_message", None)
    message = pending or st.chat_input("اكتب مصاريفك أو اسأل عنها...")

    if message:
        with st.chat_message("user", avatar="👤"):
            st.markdown(message)
        st.session_state.messages.append({"role": "user", "content": message})

        with st.chat_message("assistant", avatar="💰"):
            with st.spinner("🤔 جاري التحليل..."):
                result = st.session_state.engine.process_message(message)
            st.markdown(result["response"])
            if result.get("show_chart") and result.get("data"):
                chart_data = result["data"] if "labels" in result["data"] else SpendingAnalyzer(1).get_chart_data()
                if chart_data.get("labels"):
                    render_chart(chart_data)

        st.session_state.messages.append({
            "role": "assistant", "content": result["response"],
            "chart_data": result.get("data") if result.get("show_chart") else None
        })
        st.rerun()


if __name__ == "__main__":
    main()
