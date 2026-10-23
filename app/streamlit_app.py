from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import requests
import streamlit as st


st.set_page_config(
    page_title="Bayesian Stroke Model Comparison",
    page_icon="🧠",
    layout="wide",
)

ROOT = Path(__file__).resolve().parents[1]
API_URL = st.sidebar.text_input(
    "FastAPI base URL",
    "http://127.0.0.1:8000",
).rstrip("/")

st.title("Bayesian Stroke Model Comparison")
st.write(
    "Compare four educational Bayesian approaches for one selected profile. "
    "The results are not a medical diagnosis or a clinically validated risk score."
)

st.sidebar.header("Patient profile")
with st.sidebar.form("prediction_form"):
    age = st.slider("Age", 1, 100, 58)
    glucose = st.number_input(
        "Average glucose level",
        min_value=30.0,
        max_value=300.0,
        value=110.0,
        step=1.0,
    )
    bmi = st.number_input(
        "BMI",
        min_value=10.0,
        max_value=70.0,
        value=27.0,
        step=0.5,
    )
    hypertension = st.selectbox(
        "Hypertension",
        [0, 1],
        format_func=lambda value: "Yes" if value else "No",
    )
    heart_disease = st.selectbox(
        "Heart disease",
        [0, 1],
        format_func=lambda value: "Yes" if value else "No",
    )
    gender = st.selectbox("Gender", ["Female", "Male", "Other"])
    smoking = st.selectbox(
        "Smoking status",
        ["never smoked", "formerly smoked", "smokes", "Unknown"],
    )
    submitted = st.form_submit_button("Predict and Compare", type="primary")


def patient_payload():
    return {
        "age": age,
        "avg_glucose_level": glucose,
        "bmi": bmi,
        "hypertension": hypertension,
        "heart_disease": heart_disease,
        "gender": gender,
        "smoking_status": smoking,
    }


def api_error(response, fallback):
    try:
        return response.json().get("detail", fallback)
    except ValueError:
        return fallback


if submitted:
    current_payload = patient_payload()
    st.session_state.pop("prediction_result", None)
    st.session_state.pop("explanation", None)
    st.session_state.pop("explanation_error", None)
    st.session_state.pop("evidence_report", None)
    st.session_state["submitted_payload"] = current_payload

    with st.spinner("Calculating the four Bayesian predictions..."):
        try:
            prediction_response = requests.post(
                f"{API_URL}/predict",
                json=current_payload,
                timeout=30,
            )
            prediction_response.raise_for_status()
            st.session_state["prediction_result"] = prediction_response.json()
        except requests.HTTPError as exc:
            response = exc.response
            detail = api_error(response, str(exc)) if response is not None else str(exc)
            st.error(f"The prediction request was rejected: {detail}")
        except requests.RequestException as exc:
            st.error(
                f"Could not reach the API at {API_URL}. "
                f"Start FastAPI first. Details: {exc}"
            )

    if st.session_state.get("prediction_result"):
        with st.spinner("Preparing the evidence report and AI observation..."):
            try:
                explanation_response = requests.post(
                    f"{API_URL}/explain",
                    json=current_payload,
                    timeout=40,
                )
                explanation_response.raise_for_status()
                explanation_result = explanation_response.json()
                st.session_state["explanation"] = explanation_result["explanation"]
                st.session_state["evidence_report"] = explanation_result.get(
                    "evidence_report"
                )
            except requests.HTTPError as exc:
                response = exc.response
                st.session_state["explanation_error"] = api_error(
                    response,
                    str(exc),
                ) if response is not None else str(exc)
            except requests.RequestException as exc:
                st.session_state["explanation_error"] = (
                    f"The AI observation is unavailable because the explanation "
                    f"request failed: {exc}"
                )
            except (KeyError, TypeError, ValueError) as exc:
                st.session_state["explanation_error"] = (
                    f"The AI observation response was incomplete: {exc}"
                )


result = st.session_state.get("prediction_result")
if result:
    st.subheader("AI-generated observation")
    if st.session_state.get("explanation"):
        st.markdown(st.session_state["explanation"])
    else:
        st.warning(
            "The Bayesian predictions are available, but the AI observation is "
            f"unavailable: {st.session_state.get('explanation_error', 'Unknown error')}"
        )

    st.subheader("Four-model probability comparison")
    probabilities = result["stroke_probability_percent"]
    chart_df = pd.DataFrame(
        {
            "Model": list(probabilities.keys()),
            "Estimated probability (%)": list(probabilities.values()),
        }
    )
    fig, ax = plt.subplots(figsize=(10, 4.5))
    bars = ax.bar(chart_df["Model"], chart_df["Estimated probability (%)"], color="#386cb0")
    ax.bar_label(bars, fmt="%.2f%%", padding=3, fontsize=9)
    ax.set_ylim(0, max(10, chart_df["Estimated probability (%)"].max() * 1.25))
    ax.set_ylabel("Estimated probability (%)")
    ax.set_title("Estimated stroke probability for the selected profile")
    ax.tick_params(axis="x", rotation=18)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    evidence_report = st.session_state.get("evidence_report")
    if evidence_report:
        input_summary = evidence_report["input_summary"]
        categories = input_summary["discretized_evidence"]
        st.caption(
            "Categorized inputs: "
            + ", ".join(f"{name.replace('_', ' ')} = {value}" for name, value in categories.items())
        )
        agreement = evidence_report["agreement"]
        st.info(
            f"{agreement['description']} "
            f"The model estimates span {agreement['range_percentage_points']:.3f} percentage points."
        )

st.caption(
    "Educational classroom demonstration. Do not use these estimates for medical decisions."
)
