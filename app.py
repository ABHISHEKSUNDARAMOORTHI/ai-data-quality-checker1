import streamlit as st
import pandas as pd
import os
from data_processor import load_data, generate_profile_report, detect_anomalies
from gemini_explainer import generate_explanation # Ensure this is correctly imported
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv() # Load environment variables from .env

# Configure Gemini API
try:
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        st.error("GEMINI_API_KEY not found in .env file. Please set it up.")
    else:
        genai.configure(api_key=gemini_api_key)
except Exception as e:
    st.error(f"Error configuring Gemini API: {e}. Check your API key and internet connection.")


st.set_page_config(layout="wide", page_title="AI-Powered Data Quality & Anomaly Explainer")

st.title("📊 AI-Powered Data Quality & Anomaly Explainer")
st.markdown("Upload a CSV file, and this tool will analyze it for data quality issues and anomalies, then use Google Gemini to provide explanations and remediation steps.")

uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type="csv")

if uploaded_file is not None:
    df = load_data(uploaded_file)
    if df is None:
        st.error("Failed to load data. Please ensure it's a valid CSV.")
        st.stop()

    st.subheader("Raw Data Preview")
    st.dataframe(df.head())

    tab1, tab2, tab3 = st.tabs(["📊 Data Profile", "🚨 Detected Anomalies", "🧠 AI Explanations"])

    with tab1:
        st.subheader("Data Profile Report (ydata-profiling)")
        if st.button("Generate Full Profile Report"):
            with st.spinner("Generating detailed data profile... This may take a moment."):
                profile = generate_profile_report(df)
                st.components.v1.html(profile.to_html(), height=800, scrolling=True)
            st.success("Profile report generated!")
        st.info("A full profile report provides detailed statistics, distributions, and interactions for your data.")

    with tab2:
        st.subheader("Detected Data Quality Issues & Anomalies")
        with st.spinner("Detecting anomalies..."):
            anomalies = detect_anomalies(df)
        
        if not anomalies:
            st.success("🎉 No significant data quality issues or anomalies detected!")
        else:
            st.write(f"Found **{len(anomalies)}** potential issues:")
            anomaly_options = [f"{i+1}. {a['type']} in '{a['column']}'" for i, a in enumerate(anomalies)]
            
            selected_anomaly_idx = st.selectbox(
                "Select an anomaly for more details and AI explanation:",
                range(len(anomaly_options)),
                format_func=lambda x: anomaly_options[x]
            )

            selected_anomaly = anomalies[selected_anomaly_idx]
            st.json(selected_anomaly) # Display raw anomaly data for debugging

            # Prepare context for Gemini
            column_stats = {}
            if selected_anomaly['column'] in df.columns:
                col_data = df[selected_anomaly['column']].dropna()
                if pd.api.types.is_numeric_dtype(col_data):
                    column_stats = {
                        "min": col_data.min(),
                        "max": col_data.max(),
                        "mean": col_data.mean(),
                        "std": col_data.std(),
                        "unique": col_data.nunique()
                    }
                elif pd.api.types.is_string_dtype(col_data) or pd.api.types.is_categorical_dtype(col_data):
                     column_stats = {
                        "unique": col_data.nunique(),
                        "top_5_values": col_data.value_counts().head(5).to_dict()
                    }

            df_head_str = df.head().to_markdown(index=False)

            with tab3:
                st.subheader("AI Explanation & Remediation Suggestion")
                if st.button(f"Get AI Explanation for selected anomaly"):
                    with st.spinner("Asking Gemini for explanation..."):
                        explanation = generate_explanation(selected_anomaly, column_stats, df_head_str)
                        st.markdown(explanation)
                    st.success("Explanation generated!")
else:
    st.info("Please upload a CSV file to begin data quality analysis.")

st.sidebar.markdown("---")
st.sidebar.markdown("Developed by Your Name/Organization")
st.sidebar.markdown("[GitHub Repo Link (Coming Soon!)](#)") # Placeholder, add your actual repo link
