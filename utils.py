import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import shap
import streamlit as st

shap.initjs()


def check_password():
    """Returns `True` if the user had a correct password."""

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if (
            st.session_state["username"] in st.secrets["passwords"]
            and st.session_state["password"]
            == st.secrets["passwords"][st.session_state["username"]]
        ):
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # don't store username + password
            del st.session_state["username"]
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        # First run, show inputs for username + password.
        st.text_input("Username", on_change=password_entered, key="username")
        st.text_input("Password", type="password", on_change=password_entered, key="password")
        return False
    elif not st.session_state["password_correct"]:
        # Password not correct, show input + error.
        st.text_input("Username", on_change=password_entered, key="username")
        st.text_input("Password", type="password", on_change=password_entered, key="password")
        st.error("😕 User not known or password incorrect")
        return False
    else:
        # Password correct.
        return True


def run_analysis(data, outcome, feature_list, random_state=42):
    """
    Run a random forest regression on the data.
    BE AWARE: This function is not optimized for speed. There is no hyperparameter tuning so it is
    likely overfitting. It is only meant to be a quick and dirty way to get a sense of the data.
    """
    X = data[feature_list]
    y = data[outcome]
    model = RandomForestRegressor(random_state=random_state)
    model.fit(X, y)
    y_pred = model.predict(X)
    df_test = pd.DataFrame({"y_pred": y_pred, outcome: y})

    explainer = shap.Explainer(model)
    shap_values = explainer(X)
    shap_actual_df = pd.concat(
        [
            pd.DataFrame(shap_values.values, columns=feature_list).melt(value_name="shap_data"),
            pd.DataFrame(X.values, columns=feature_list)
            .melt(value_name="actual_data")
            .drop("variable", axis=1),
        ],
        axis=1,
    )
    # shap.plots.beeswarm(shap_values)

    feature_importance = pd.Series(dict(zip(X.columns, model.feature_importances_.round(2))))
    gini_importance_df = (
        pd.DataFrame(feature_importance.sort_values(ascending=False))
        .reset_index()
        .rename(columns={"index": "feature", 0: "feature_importance"})
    )

    return (df_test, shap_actual_df, gini_importance_df)
