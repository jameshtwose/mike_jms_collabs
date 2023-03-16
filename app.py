# from dotenv import load_dotenv, find_dotenv
import pandas as pd
import plotly.express as px
from plotly.colors import n_colors
import streamlit as st
from utils import run_analysis, check_password, run_forecast

# CONFIG
st.set_page_config(
    page_title="Business Analytiq Pricing Dashboard", page_icon=":sparkles:", layout="wide"
)

if check_password():
    # CREATE CACHE DATA FUNCTION
    @st.cache_data(ttl=3600)
    def get_data(data_source="original"):
        if data_source == "original":
            return (
                pd.read_csv("data/EPDM benchmark vs EPDM rubber Carbon black and Energy.csv")
                .assign(**{"Date": lambda x: pd.to_datetime(x["Date"]).dt.date})
                .dropna()
                .set_index("Date")
            )
        elif data_source == "preprocessed":
            return (
                pd.read_csv("data/Revised input.csv")
                .assign(**{"Date": lambda x: pd.to_datetime(x["Date"]).dt.date})
                .dropna()
                .set_index("Date")
            )
        elif data_source == "ESP EPDM":
            return (
                pd.read_excel("data/ESP data.xlsx", sheet_name="EPDM")
                .assign(**{"Date": lambda x: pd.to_datetime(x["date"]).dt.date})
                .drop(columns=["date"])
                .dropna()
                .set_index("Date")
            )
        elif data_source == "ESP FKM":
            return (
                pd.read_excel("data/ESP data.xlsx", sheet_name="FKM")
                .assign(**{"Date": lambda x: pd.to_datetime(x["date"]).dt.date})
                .drop(columns=["date"])
                .dropna()
                .set_index("Date")
            )
        elif data_source == "ESP HNBR":
            return (
                pd.read_excel("data/ESP data.xlsx", sheet_name="HNBR")
                .assign(**{"Date": lambda x: pd.to_datetime(x["date"]).dt.date})
                .drop(columns=["date"])
                .dropna()
                .set_index("Date")
            )

        elif data_source == "ESP NBR":
            return (
                pd.read_excel("data/ESP data.xlsx", sheet_name="NBR")
                .assign(**{"Date": lambda x: pd.to_datetime(x["date"]).dt.date})
                .drop(columns=["date"])
                .dropna()
                .set_index("Date")
            )
        elif data_source == "ESP VMQ":
            return (
                pd.read_excel("data/ESP data.xlsx", sheet_name="VMQ")
                .assign(**{"Date": lambda x: pd.to_datetime(x["date"]).dt.date})
                .drop(columns=["date"])
                .dropna()
                .set_index("Date")
            )

        else:
            return (
                pd.read_csv("data/EPDM.csv")
                .assign(**{"Date": lambda x: pd.to_datetime(x["date"]).dt.date})
                .drop(columns=["date"])
                .dropna()
                .set_index("Date")
            )

    # CREATE ANALYSIS CACHE FUNCTION
    @st.cache_data(ttl=3600)
    def get_analysis_output(df, outcome, feature_list):
        df_pred_test, shap_df, gini_df = run_analysis(
            data=df, outcome=outcome, feature_list=feature_list
        )
        return df_pred_test, shap_df, gini_df

    # CREATE FORECAST CACHE FUNCTION
    @st.cache_data(ttl=3600)
    def get_forecast_output(df, outcome, feature_list):
        plot_df = run_forecast(data=df, outcome=outcome, feature_list=feature_list)
        return plot_df

    # SIDEBAR - TITLE AND DATA SOURCE
    st.sidebar.header("Please filter here:")
    data_source = st.sidebar.selectbox(
        "Select a data source:",
        options=[
            "ESP EPDM",
            "ESP FKM",
            "ESP HNBR",
            "ESP NBR",
            "ESP VMQ",
            "previous ESP EPDM",
            "original",
            "preprocessed",
        ],
    )

    # READ DATA
    df = get_data(data_source=data_source)

    # SIDEBAR - OUTCOME AND FEATURE LIST
    outcome = st.sidebar.selectbox(
        "Select an outcome measure of interest:", options=df.columns.tolist()
    )
    feature_list = df.drop(columns=[outcome]).columns.tolist()

    # SIDEBAR - LOGO AND CREDITS
    st.sidebar.markdown("---")
    st.sidebar.markdown("<br><br><br>", unsafe_allow_html=True)
    st.sidebar.markdown(
        """
        <div style="text-align: center; padding-right: 10px;">
            <img alt="logo" src="https://services.jms.rocks/img/logo.png" width="100">
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.sidebar.markdown(
        """
        <div style="text-align: center; color: #E8C003; margin-top: 40px; margin-bottom: 40px;">
            <a href="https://services.jms.rocks" style="color: #E8C003;">Created by James Twose</a>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # RUN MAIN ANALYSIS
    df_pred_test, shap_df, gini_df = get_analysis_output(
        df=df, outcome=outcome, feature_list=feature_list
    )

    # RUN FORECAST ANALYSIS
    # top 3 important features based on SHAP variance
    top_3_features = (
        shap_df.groupby("variable")
        .var()["shap_data"]
        .sort_values(ascending=False)
        .head(3)
        .index.tolist()
    )
    forecast_plot_df = get_forecast_output(df=df, outcome=outcome, feature_list=top_3_features)
    # forecast_plot_df = get_forecast_output(df=df, outcome=outcome, feature_list=feature_list)

    pred_test_corr = df_pred_test.corr().iloc[0, 1].round(3)
    likelihood_overfit = "Yes" if pred_test_corr > 0.8 else "No"
    shap_best_feature = (
        shap_df.groupby("variable")
        .var()["shap_data"]
        .sort_values(ascending=False)
        .head(1)
        .index.tolist()[0]
    )
    gini_best_feature = gini_df.head(1)["feature"].tolist()[0]

    # MAINPAGE
    st.markdown("<h1>Business Analytiq Pricing Dashboard</h1>", unsafe_allow_html=True)
    left_column, middle_column, right_column = st.columns(3)
    with left_column:
        st.subheader("Chosen Outcome")
        st.subheader(outcome)
    with middle_column:
        st.subheader("Mean of chosen outcome")
        st.subheader(f"{df[outcome].mean():,.3f}")
    with right_column:
        st.subheader("Variance of chosen outcome")
        st.subheader(f"{df[outcome].var():,.3f}")
    st.markdown("---")

    st.header("Main Report")
    if likelihood_overfit == "Yes":
        st.markdown(
            f"""There is a likelihood that the model will not generalize (i.e.
            it is overfitting the current data). This is based on the Pearson
            Correlation between the predicted and actual :green[{outcome}] being so
            high (r-value=:green[{pred_test_corr}])."""
        )
    st.markdown(
        f"""Based on the variance in SHAP values, the following feature
        is the most important: :green[{shap_best_feature}]"""
    )
    st.markdown(
        f"""Based on the gini importance, the following feature
        is the most important: :green[{gini_best_feature}]"""
    )
    st.markdown("---")

    st.header("Selected Dataframe")
    st.dataframe(df)
    st.markdown("---")
    st.header("Descriptive Statistics")
    st.dataframe(df.describe())
    st.markdown("---")

    st.header("Pearson Correlations between all columns")
    plot_df = df.corr().round(3)
    # correlation plot
    corr_heat = px.imshow(plot_df, text_auto=True, color_continuous_scale="RdBu")
    st.plotly_chart(corr_heat)
    st.markdown("---")

    # SHOW MAIN ANALYSIS OUTPUT
    st.header("Model Creation and Feature Importance Calculation")

    # SHAP PLOT
    st.markdown(
        """
        <div>
            <p>For more information on SHAP Values, please see the following: 
                <a href="https://christophm.github.io/interpretable-ml-book/shap.html"
                style="color: #E8C003;">SHAP Values Explanation</a>
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    shap_fig = px.strip(
        shap_df,
        x="shap_data",
        y="variable",
        color="actual_data",
        color_discrete_sequence=n_colors(
            "rgb(143, 15, 212)", "rgb(252, 221, 20)", df.shape[0], colortype="rgb"
        ),
        title="Feature Importance Based on SHAP Values",
    )
    shap_fig.update_layout(showlegend=False, coloraxis_showscale=True)
    st.plotly_chart(shap_fig)

    # GINI PLOT
    st.markdown(
        """
        <div>
            <p>For more information on Gini Gain, please see the following:
                <a href="https://www.codecademy.com/article/fe-feature-importance-final"
                style="color: #E8C003;">Gini Gain Explanation</a>
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    gini_fig = px.bar(
        gini_df,
        x="feature_importance",
        y="feature",
        color="feature",
        title="Feature Importance Based on Gini Gain",
    )
    st.plotly_chart(gini_fig)
    st.markdown("---")

    # SHOW MAIN FORECAST OUTPUT
    st.header("Forecasting time series")
    # TIME SERIES PLOT
    df_melt = df.reset_index().melt(id_vars="Date")
    time_series_fig = px.line(
        df_melt,
        x="Date",
        y="value",
        color="variable",
        title="Time Series Plot of all variables",
    )
    st.plotly_chart(time_series_fig)
    # FORECAST TIME SERIES PLOT
    st.markdown(
        f"""The forecast is based on the top 3 important features as defined
        by the variance in SHAP Values. These are: :green[{top_3_features}]"""
    )
    forecast_fig = px.line(
        forecast_plot_df,
        x="Date",
        y=outcome,
        color="type",
        markers=True,
        title=f"Forecasted outcome == {outcome}",
    )
    st.plotly_chart(forecast_fig)
    st.markdown("---")

    # HIDE STREAMLIT STYLE
    hide_streamlit_style = """
                            <style>
                            #MainMenu {visibility: hidden;}
                            footer {visibility: hidden;}
                            header {visibility: hidden;}
                            </style>
                            """
    # st.markdown(hide_streamlit_style, unsafe_allow_html=True)
