
from pathlib import Path
import os
import json

import pandas as pd
from fastapi import FastAPI, HTTPException
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from sklearn.model_selection import train_test_split
from bayes_core import (
    FEATURES,
    STATE_VALUES,
    TARGET,
    evidence_from_patient,
    load_and_prepare,
    load_bundle,
    predict_model,
)

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")
MODEL_PATH = ROOT / "models" / "bayesian_models.joblib"
DATA_PATH = ROOT / "data" / "healthcare-dataset-stroke-data.csv"
METRICS_PATH = ROOT / "models" / "evaluation_metrics.csv"
app = FastAPI(
    title="Bayesian Stroke Model Comparison API",
    description="Educational comparison of four Bayesian approaches. Not for medical diagnosis.",
    version="1.0.0",
)

bundle = None
EXPLANATION_TIMEOUT_SECONDS = 30

@app.on_event("startup")
def load_saved_models():
    global bundle
    if MODEL_PATH.exists():
        bundle = load_bundle(MODEL_PATH)

@app.get("/health")
def health():
    return {"status": "ok", "models_loaded": bundle is not None}

class PatientInput(BaseModel):
    age: float = Field(ge=0, le=120)
    avg_glucose_level: float = Field(gt=0, le=1000)
    bmi: float = Field(gt=0, le=100)
    hypertension: int = Field(ge=0, le=1)
    heart_disease: int = Field(ge=0, le=1)
    gender: str
    smoking_status: str


class ExplanationRequest(PatientInput):
    pass


def _training_data():
    data = load_and_prepare(DATA_PATH)
    train_df, _ = train_test_split(
        data,
        test_size=0.25,
        random_state=42,
        stratify=data[TARGET],
    )
    return train_df


def _probabilities(bundle, evidence):
    return {
        name: round(100 * predict_model(bundle, name, evidence), 3)
        for name in bundle["models"]
    }


def _category_statistics(train_df, evidence):
    statistics = {}
    for feature in FEATURES:
        value = str(evidence[feature])
        subset = train_df[train_df[feature].astype(str) == value]
        statistics[feature] = {
            "category": value,
            "count": int(len(subset)),
            "observed_stroke_rate_percent": (
                round(100 * float(subset[TARGET].mean()), 3)
                if len(subset)
                else None
            ),
            "note": (
                "Observed training-data association, not individual causation."
            ),
        }
    return statistics


def _sensitivity(bundle, evidence):
    baseline = _probabilities(bundle, evidence)
    results = {}
    for feature in FEATURES:
        alternatives = [value for value in STATE_VALUES[feature] if str(value) != str(evidence[feature])]
        if not alternatives:
            results[feature] = {"status": "No valid alternative category available."}
            continue
        alternative = alternatives[0]
        changed = dict(evidence)
        changed[feature] = alternative
        alternative_probabilities = _probabilities(bundle, changed)
        results[feature] = {
            "from_category": evidence[feature],
            "to_category": alternative,
            "absolute_probability_change_percentage_points": {
                name: round(abs(alternative_probabilities[name] - baseline[name]), 3)
                for name in baseline
            },
            "alternative_probabilities_percent": alternative_probabilities,
            "interpretation": (
                "Model sensitivity only; this controlled comparison does not prove "
                "that the feature causes or prevents stroke."
            ),
        }
    return results


def _build_evidence_report(request, bundle):
    patient = request.model_dump()
    evidence = evidence_from_patient(patient)
    train_df = _training_data()
    predictions = _probabilities(bundle, evidence)
    metrics = pd.read_csv(METRICS_PATH).to_dict(orient="records") if METRICS_PATH.exists() else []
    spread = round(max(predictions.values()) - min(predictions.values()), 3)
    report = {
        "input_summary": {
            "entered_values": patient,
            "discretized_evidence": evidence,
        },
        "model_predictions_percent": predictions,
        "agreement": {
            "range_percentage_points": spread,
            "description": (
                "Models are relatively close." if spread <= 5
                else "Models show noticeable disagreement."
            ),
        },
        "dataset_context": {
            "training_rows": int(len(train_df)),
            "overall_observed_stroke_prevalence_percent": round(
                100 * float(train_df[TARGET].mean()), 3
            ),
            "observed_rates_for_selected_categories": _category_statistics(
                train_df, evidence
            ),
        },
        "held_out_evaluation_metrics": metrics,
        "feature_sensitivity": _sensitivity(bundle, evidence),
        "limitations": [
            "Sensitivity changes are controlled model comparisons, not causal evidence.",
            "Observed category stroke rates describe the training data and do not predict an individual outcome.",
            "The dataset is imbalanced; some category counts may be small.",
            "The project is educational and not clinically validated.",
        ],
    }
    return report


def _explanation_prompt(report) -> str:
    """Build a grounded prompt from calculated evidence only."""
    return (
        "Write a concise, natural-English interpretation of this educational Bayesian "
        "stroke comparison using only the calculated evidence report below. The main goal "
        "is to explain WHY these estimates were produced from the selected inputs and HOW "
        "those inputs affect the model outputs, not to dump the report or repeat every "
        "number visible in the chart.\n\n"
        "Use exactly these six headings, in this order: A. Input Summary, B. Model "
        "Predictions, C. Why These Estimates?, D. Dataset Context, E. Model Reliability, "
        "F. Final Interpretation. Write in the style of a short project presentation: "
        "each section should contain one or two connected paragraphs, with an occasional "
        "short bullet list only when it improves clarity. Do not begin with phrases such "
        "as 'Illustrative example'.\n\n"
        "A should summarize the actual entered values and explain their derived categories "
        "in plain English. B should explain the prediction range and whether the models "
        "agree; mention the highest and lowest estimate only if useful, rather than listing "
        "all four values. C should explain how the selected categories relate to the "
        "largest two or three calculated sensitivity changes, including the direction and "
        "model affected. Explain that sensitivity is a controlled model response, not "
        "causation. Also explain the four model assumptions briefly: conditional "
        "independence, different network structures, and stronger prior smoothing. "
        "D should compare the overall training prevalence with the most relevant selected "
        "category rates and explain that they are dataset-level associations, not individual "
        "causation. E should summarize reliability in practical language: high accuracy "
        "can be misleading for this imbalanced outcome, low recall means many observed "
        "positive cases are missed, and PR-AUC/ROC-AUC describe ranking performance. Do "
        "not produce a metric-by-metric list. F should summarize what the results suggest "
        "about this comparison and what they cannot establish.\n\n"
        "Do not use Markdown tables, JSON, code blocks, or raw report dumps. Do not "
        "calculate new results, invent values, or override supplied predictions. Never say "
        "a feature caused stroke, that a higher probability means stroke will occur, or "
        "that a lower probability means someone is safe. State that the project is "
        "educational and not clinically validated. Target approximately 450 to 650 words, "
        "and complete every section before ending.\n\n"
        f"CALCULATED EVIDENCE REPORT:\n{json.dumps(report, indent=2, default=str)}"
    )


REQUIRED_EXPLANATION_SECTIONS = (
    "A. Input Summary",
    "B. Model Predictions",
    "C. Why These Estimates?",
    "D. Dataset Context",
    "E. Model Reliability",
    "F. Final Interpretation",
)


def _generate_explanation(report) -> str:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key or api_key.startswith("replace-with-"):
        raise HTTPException(
            status_code=503,
            detail=(
                "GROQ_API_KEY is missing or still a placeholder. "
                "Set a valid key in the project .env file, then restart FastAPI."
            ),
        )
    model = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
    try:
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a careful results explainer for a college project. "
                    "Follow the user's grounding and formatting instructions exactly.",
                ),
                ("human", "{results}"),
            ]
        )
        chain = prompt | ChatGroq(
            model=model,
            api_key=api_key,
            temperature=0.2,
            reasoning_format="hidden",
            max_tokens=1800,
            timeout=EXPLANATION_TIMEOUT_SECONDS,
        )
        response = chain.invoke(
            {"results": _explanation_prompt(report)},
            config={"run_name": "bayesian-results-explanation"},
        )
        explanation = response.content
        if isinstance(explanation, list):
            explanation = "\n".join(
                item.get("text", "") if isinstance(item, dict) else str(item)
                for item in explanation
            )
        if not explanation:
            raise ValueError("Groq returned an empty explanation.")
        explanation = explanation.strip()
        missing_sections = [
            section for section in REQUIRED_EXPLANATION_SECTIONS
            if section not in explanation
        ]
        if missing_sections:
            raise ValueError(
                "Groq returned an incomplete explanation. Missing sections: "
                + ", ".join(missing_sections)
            )
        return explanation
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Groq explanation request failed: {exc}",
        ) from exc

@app.get("/models")
def list_models():
    if bundle is None:
        raise HTTPException(status_code=503, detail="Models not trained. Run the notebook first.")
    return {"models": list(bundle["models"].keys())}

@app.post("/predict")
def predict(patient: PatientInput):
    if bundle is None:
        raise HTTPException(status_code=503, detail="Models not trained. Run the notebook first.")
    try:
        evidence = evidence_from_patient(patient.model_dump())
        outputs = {
            name: round(100 * predict_model(bundle, name, evidence), 3)
            for name in bundle["models"]
        }
        return {
            "educational_notice": "Experimental estimates only; not a medical diagnosis or clinical risk score.",
            "evidence": evidence,
            "stroke_probability_percent": outputs,
        }
    except (ValueError, KeyError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/explain")
def explain_results(request: ExplanationRequest):
    if bundle is None:
        raise HTTPException(status_code=503, detail="Models not trained. Run the notebook first.")
    try:
        report = _build_evidence_report(request, bundle)
    except (ValueError, KeyError, FileNotFoundError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {
        "explanation": _generate_explanation(report),
        "evidence_report": report,
        "model": os.getenv("GROQ_MODEL", "openai/gpt-oss-20b"),
    }
