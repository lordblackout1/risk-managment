import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Now you can use your environment variables anywhere
ACLED_API_KEY = os.getenv("ACLED_API_KEY")
WORLD_BANK_API_URL = os.getenv("WORLD_BANK_API_URL")
DEBUG_MODE = os.getenv("DEBUG", "False") == "True"


import streamlit as st
import pandas as pd
import numpy as np
import wbdata
import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

st.set_page_config(page_title="Somalia Risk Analysis Model", layout="centered")
st.title("📊 Somalia Risk Analysis Predictive Model")

uploaded_file = st.file_uploader("Upload ACLED Somalia CSV file", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # --- Preprocess ACLED data ---
    if 'event_date' not in df.columns:
        st.error("❗ 'event_date' column missing from your ACLED data.")
        st.stop()

    df['event_date'] = pd.to_datetime(df['event_date'], errors='coerce')
    df = df.dropna(subset=['event_date'])
    df['month'] = df['event_date'].dt.to_period('M').dt.to_timestamp()

    events_monthly = df.groupby('month').size().reset_index(name='event_count')

    # --- Economic indicators from World Bank ---
    st.info("Fetching World Bank economic data...")
    indicators = {
        "NY.GDP.PCAP.KD": "gdp_per_capita",
        "FP.CPI.TOTL.ZG": "inflation"
    }
    start_date = datetime.datetime(2000, 1, 1)
    end_date = datetime.datetime(datetime.datetime.now().year, 1, 1)
    wb_df = wbdata.get_dataframe(indicators, country="SO", data_date=(start_date, end_date), convert_date=True)
    wb_df = wb_df.reset_index().rename(columns={"date": "year"})
    wb_df['month'] = pd.to_datetime(wb_df['year'].astype(str) + '-01-01')
    wb_df = wb_df.drop(columns=['region', 'year'])

    # --- Governance Indicators from WGI ---
    st.info("Fetching World Bank Governance Indicators...")
    wgi_indicators = {
        "RL.EST": "rule_of_law",
        "CC.EST": "control_of_corruption",
        "GE.EST": "government_effectiveness"
    }
    governance_df = wbdata.get_dataframe(wgi_indicators, country="SO", data_date=(start_date, end_date), convert_date=True)
    governance_df = governance_df.reset_index().rename(columns={"date": "year"})
    governance_df['month'] = pd.to_datetime(governance_df['year'].astype(str) + '-01-01')
    governance_df = governance_df.drop(columns=['region', 'year'])

    # --- Merge datasets ---
    merged_df = pd.merge(events_monthly, wb_df, on="month", how="left")
    merged_df = pd.merge(merged_df, governance_df, on="month", how="left")
    merged_df = merged_df.sort_values("month")

    merged_df[['gdp_per_capita', 'inflation',
            'rule_of_law', 'control_of_corruption', 'government_effectiveness']] = merged_df[
        ['gdp_per_capita', 'inflation',
        'rule_of_law', 'control_of_corruption', 'government_effectiveness']
    ].fillna(method='ffill')

    st.subheader("📅 Merged Data Preview")
    st.dataframe(merged_df.tail(12))

    # --- Define risk levels ---
    merged_df['risk_level'] = pd.cut(merged_df['event_count'],
                                    bins=[-1, 10, 50, np.inf],
                                    labels=['Low', 'Medium', 'High'])

    # --- Train model ---
    X = merged_df[['event_count', 'gdp_per_capita', 'inflation',
                'rule_of_law', 'control_of_corruption', 'government_effectiveness']]
    y = merged_df['risk_level']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    report = classification_report(y_test, predictions, output_dict=True)

    st.subheader("📈 Model Performance")
    st.json(report)

    # --- Prediction Interface ---
    st.subheader("🎯 Predict Risk for New Scenario")
    event_count = st.number_input("Number of conflict events this month", min_value=0, value=5)
    gdp_per_capita = st.number_input("GDP per capita (constant 2010 US$)", min_value=0.0, value=200.0)
    inflation = st.number_input("Inflation (annual %)", value=5.0)
    rule_of_law = st.number_input("Rule of Law Estimate (approx -2.5 to 2.5)", value=-1.0)
    corruption = st.number_input("Control of Corruption Estimate (approx -2.5 to 2.5)", value=-1.0)
    effectiveness = st.number_input("Government Effectiveness Estimate (approx -2.5 to 2.5)", value=-1.0)

    if st.button("Predict Risk Level"):
        input_features = pd.DataFrame([[event_count, gdp_per_capita, inflation,
                                        rule_of_law, corruption, effectiveness]],
                                    columns=['event_count', 'gdp_per_capita', 'inflation',
                                    'rule_of_law', 'control_of_corruption', 'government_effectiveness'])
        prediction = model.predict(input_features)[0]
        st.success(f"✅ **Predicted Risk Level:** {prediction}")

else:
    st.warning("📁 Please upload the ACLED Somalia CSV to continue.")

import streamlit as st

def login():
    st.title("Login")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username == "admin" and password == "secret":
            st.session_state["logged_in"] = True
            st.success("Login successful!")
            st.experimental_rerun()
        else:
            st.error("Invalid credentials")

if "logged_in" not in st.session_state:
    login()
    st.stop()
st.sidebar.title("Navigation")
st.sidebar.write("Welcome to the Somalia Risk Analysis App!")