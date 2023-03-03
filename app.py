# from dotenv import load_dotenv, find_dotenv
import pandas as pd
import plotly.express as px
from plotly.colors import n_colors
import streamlit as st
from utils import run_analysis, check_password

# CONFIG
st.set_page_config(
    page_title="Business Analytiq Pricing Dashboard", page_icon=":sparkles:", layout="wide"
)

if check_password():
    # CREATE CACHE DATA FUNCTION
    @st.cache_data(ttl=3600)
    def get_data(df_choice="unclear_result"):
        if df_choice == "clear_result":
            df = (
                pd.read_excel("data/Input for statistics v1.xlsx", sheet_name=0)
                .assign(**{"Date": lambda x: pd.to_datetime(x["Date"]).dt.date})
                .dropna()
                .set_index("Date")
            )
        else:
            df = (
                pd.read_excel("data/Input for statistics v1.xlsx", sheet_name=1)
                .assign(**{"Date": lambda x: pd.to_datetime(x["Date"]).dt.date})
                .dropna()
                .set_index("Date")
            )
        return df

    # CREATE ANALYSIS CACHE FUNCTION
    @st.cache_data(ttl=3600)
    def get_analysis_output(df, outcome, feature_list):
        df_pred_test, shap_df, gini_df = run_analysis(
            data=df, outcome=outcome, feature_list=feature_list
        )
        return df_pred_test, shap_df, gini_df

    df = get_data()

    # SIDEBAR
    st.sidebar.header("Please filter here:")
    outcome = st.sidebar.selectbox(
        "Select an outcome measure of interest:", options=df.columns.tolist()
    )

    # READ DATA
    feature_list = df.drop(columns=[outcome]).columns.tolist()

    # RUN MAIN ANALYSIS
    df_pred_test, shap_df, gini_df = get_analysis_output(
        df=df, outcome=outcome, feature_list=feature_list
    )

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
    corr_heat = px.imshow(plot_df, text_auto=True)
    st.plotly_chart(corr_heat)
    st.markdown("---")

    # SHOW MAIN ANALYSIS OUTPUT
    st.header("Model Creation and Feature Importance Calculation")

    # SHAP PLOT
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
    gini_fig = px.bar(
        gini_df,
        x="feature_importance",
        y="feature",
        color="feature",
        title="Feature Importance Based on Gini Values",
    )
    st.plotly_chart(gini_fig)
    st.markdown("---")

    # HIDE STREAMLIT STYLE
    hide_streamlit_style = """
                            <style>
                            #MainMenu {visibility: hidden;}
                            footer {visibility: hidden;}
                            header {visibility: hidden;}
                            </style>
                            """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)
