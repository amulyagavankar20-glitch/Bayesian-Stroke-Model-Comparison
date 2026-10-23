
"""Shared Bayesian modelling functions for the project.

The Bayesian networks are discrete, so continuous values are binned before fitting.
Exact inference is performed by enumerating the small set of possible states.
This keeps the implementation transparent and avoids depending on a version-specific
Bayesian-network API.
"""
from itertools import product
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.naive_bayes import GaussianNB, CategoricalNB
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix
)

FEATURES = [
    "age_group", "glucose_group", "bmi_group",
    "hypertension", "heart_disease", "gender", "smoking_status"
]
TARGET = "stroke"
STATE_VALUES = {
    "age_group": ["Young", "Adult", "Senior"],
    "glucose_group": ["Normal", "Elevated", "High"],
    "bmi_group": ["Low", "Normal", "High"],
    "hypertension": [0, 1],
    "heart_disease": [0, 1],
    "gender": ["Female", "Male", "Other"],
    "smoking_status": ["never smoked", "formerly smoked", "smokes", "Unknown"],
    "stroke": [0, 1],
}

# A transparent, domain-informed graph. Edges represent modelling assumptions,
# not proof of causal relationships.
EXPERT_EDGES = [
    ("age_group", "hypertension"),
    ("age_group", "heart_disease"),
    ("age_group", "glucose_group"),
    ("age_group", "bmi_group"),
    ("gender", "smoking_status"),
    ("smoking_status", "stroke"),
    ("hypertension", "stroke"),
    ("heart_disease", "stroke"),
    ("glucose_group", "stroke"),
    ("bmi_group", "stroke"),
]
COMPACT_EDGES = [
    ("age_group", "stroke"),
    ("hypertension", "stroke"),
    ("heart_disease", "stroke"),
    ("glucose_group", "stroke"),
    ("bmi_group", "stroke"),
    ("smoking_status", "stroke"),
    ("gender", "stroke"),
]

def load_and_prepare(csv_path):
    """Load data, remove identifier, impute BMI, and create discrete categories."""
    df = pd.read_csv(csv_path)
    df = df.drop(columns=["id"], errors="ignore").copy()
    df["bmi"] = pd.to_numeric(df["bmi"], errors="coerce")
    df["bmi"] = df["bmi"].fillna(df["bmi"].median())
    df["gender"] = df["gender"].replace({"Other": "Other"}).fillna("Other")
    df["smoking_status"] = df["smoking_status"].fillna("Unknown")
    df["age_group"] = pd.cut(
        df["age"], bins=[-0.01, 18, 45, 120],
        labels=["Young", "Adult", "Senior"], include_lowest=True
    ).astype(str)
    df["glucose_group"] = pd.cut(
        df["avg_glucose_level"], bins=[-0.01, 100, 160, 1000],
        labels=["Normal", "Elevated", "High"], include_lowest=True
    ).astype(str)
    df["bmi_group"] = pd.cut(
        df["bmi"], bins=[-0.01, 18.5, 25, 100],
        labels=["Low", "Normal", "High"], include_lowest=True
    ).astype(str)
    df["hypertension"] = df["hypertension"].astype(int)
    df["heart_disease"] = df["heart_disease"].astype(int)
    df["stroke"] = df["stroke"].astype(int)
    # Only columns used by the models are retained.
    return df[FEATURES + [TARGET]].reset_index(drop=True)

def clean_evidence(evidence):
    """Normalize and validate evidence received from the API or UI."""
    result = {}
    for key in FEATURES:
        if key not in evidence or evidence[key] in (None, ""):
            continue
        value = evidence[key]
        if key in ("hypertension", "heart_disease"):
            value = int(value)
            if value not in (0, 1):
                raise ValueError(f"{key} must be 0 or 1")
        else:
            value = str(value)
            if value not in STATE_VALUES[key]:
                raise ValueError(f"Invalid value for {key}: {value}")
        result[key] = value
    return result

def evidence_from_patient(patient):
    """Convert familiar patient inputs into the same categories used for training."""
    age = float(patient["age"])
    glucose = float(patient["avg_glucose_level"])
    bmi = float(patient["bmi"])
    evidence = {
        "age_group": "Young" if age <= 18 else ("Adult" if age <= 45 else "Senior"),
        "glucose_group": "Normal" if glucose <= 100 else ("Elevated" if glucose <= 160 else "High"),
        "bmi_group": "Low" if bmi < 18.5 else ("Normal" if bmi < 25 else "High"),
        "hypertension": int(patient["hypertension"]),
        "heart_disease": int(patient["heart_disease"]),
        "gender": str(patient["gender"]),
        "smoking_status": str(patient["smoking_status"]),
    }
    return clean_evidence(evidence)

class DiscreteBayesNetwork:
    """Small discrete Bayesian network with Dirichlet-smoothed conditional tables."""
    def __init__(self, edges, alpha=1.0):
        self.edges = list(edges)
        self.alpha = float(alpha)
        self.nodes = list(FEATURES) + [TARGET]
        self.parents = {node: [] for node in self.nodes}
        graph = nx.DiGraph()
        graph.add_nodes_from(self.nodes)
        graph.add_edges_from(self.edges)
        if not nx.is_directed_acyclic_graph(graph):
            raise ValueError("Network edges must form a directed acyclic graph.")
        self.order = list(nx.topological_sort(graph))
        for parent, child in self.edges:
            self.parents[child].append(parent)
        self.cpts = {}

    def fit(self, data):
        self.cpts = {}
        for node in self.nodes:
            parents = self.parents[node]
            node_states = STATE_VALUES[node]
            parent_states = [STATE_VALUES[p] for p in parents]
            rows = []
            for parent_values in product(*parent_states) if parents else [()]:
                mask = np.ones(len(data), dtype=bool)
                for parent, value in zip(parents, parent_values):
                    mask &= data[parent].astype(str).to_numpy() == str(value)
                subset = data.loc[mask, node].astype(str)
                counts = subset.value_counts()
                smoothed = np.array(
                    [counts.get(str(state), 0) + self.alpha for state in node_states],
                    dtype=float
                )
                probs = smoothed / smoothed.sum()
                rows.append((tuple(str(v) for v in parent_values), probs))
            self.cpts[node] = dict(rows)
        return self

    def _probability(self, node, assignment):
        parent_key = tuple(str(assignment[p]) for p in self.parents[node])
        probabilities = self.cpts[node][parent_key]
        state_index = [str(x) for x in STATE_VALUES[node]].index(str(assignment[node]))
        return float(probabilities[state_index])

    def predict_proba(self, evidence):
        """Return P(stroke=1 | evidence) using exact enumeration."""
        evidence = clean_evidence(evidence)
        unknown = [k for k in evidence if k not in self.nodes or evidence[k] not in STATE_VALUES[k]]
        if unknown:
            raise ValueError(f"Invalid evidence: {unknown}")
        free_nodes = [node for node in self.nodes if node not in evidence and node != TARGET]
        total_by_stroke = {0: 0.0, 1: 0.0}
        for stroke_value in STATE_VALUES[TARGET]:
            for free_values in product(*(STATE_VALUES[node] for node in free_nodes)):
                assignment = dict(evidence)
                assignment[TARGET] = stroke_value
                assignment.update(dict(zip(free_nodes, free_values)))
                joint = 1.0
                for node in self.order:
                    joint *= self._probability(node, assignment)
                total_by_stroke[stroke_value] += joint
        denominator = total_by_stroke[0] + total_by_stroke[1]
        if denominator <= 0:
            return 0.5
        return total_by_stroke[1] / denominator

    def to_dict(self):
        return {"edges": self.edges, "alpha": self.alpha, "cpts": self.cpts,
                "parents": self.parents, "order": self.order}

    @classmethod
    def from_dict(cls, payload):
        model = cls(payload["edges"], payload["alpha"])
        model.cpts = payload["cpts"]
        model.parents = payload["parents"]
        model.order = payload["order"]
        return model

def build_models(train_df):
    """Fit four complementary Bayesian approaches using training data only."""
    # Model 1: Gaussian Naive Bayes on raw numeric + one-hot categorical features.
    raw = pd.read_csv(Path(__file__).parent / "data" / "healthcare-dataset-stroke-data.csv") if False else None
    # For a consistent API, use the discrete table for NB too. CategoricalNB is
    # appropriate after encoding each feature as an integer category.
    encoded = pd.DataFrame()
    category_maps = {}
    for col in FEATURES:
        states = STATE_VALUES[col]
        category_maps[col] = {str(v): i for i, v in enumerate(states)}
        encoded[col] = train_df[col].astype(str).map(category_maps[col]).fillna(0).astype(int)
    model1 = CategoricalNB(alpha=1.0)
    model1.fit(encoded[FEATURES], train_df[TARGET])

    # Model 2: expert-designed network with Dirichlet smoothing.
    model2 = DiscreteBayesNetwork(EXPERT_EDGES, alpha=1.0).fit(train_df)

    # Model 3: compact star network, useful as a simpler structural comparison.
    model3 = DiscreteBayesNetwork(COMPACT_EDGES, alpha=1.0).fit(train_df)

    # Model 4: same expert graph but a stronger equivalent pseudo-count prior.
    model4 = DiscreteBayesNetwork(EXPERT_EDGES, alpha=5.0).fit(train_df)

    return {
        "models": {
            "Categorical Naive Bayes": model1,
            "Expert Bayesian Network": model2,
            "Compact Bayesian Network": model3,
            "Strong-Prior Bayesian Network": model4,
        },
        "category_maps": category_maps,
        "feature_order": FEATURES,
        "target": TARGET,
        "edges": {"Expert Bayesian Network": EXPERT_EDGES,
                  "Compact Bayesian Network": COMPACT_EDGES},
    }

def predict_model(bundle, model_name, evidence):
    evidence = clean_evidence(evidence)
    model = bundle["models"][model_name]
    if model_name == "Categorical Naive Bayes":
        row = {}
        for col in FEATURES:
            state_map = bundle["category_maps"][col]
            row[col] = state_map.get(str(evidence.get(col, STATE_VALUES[col][0])), 0)
        # CategoricalNB doesn't perform missing-evidence marginalization. For the
        # app we therefore require a complete patient profile.
        X = pd.DataFrame([row], columns=FEATURES)
        return float(model.predict_proba(X)[0][list(model.classes_).index(1)])
    return float(model.predict_proba(evidence))

def evaluate_models(bundle, test_df):
    """Evaluate all approaches on identical held-out observations."""
    y_true = test_df[TARGET].astype(int).to_numpy()
    rows = []
    probabilities = {}
    for name, model in bundle["models"].items():
        preds = []
        for _, row in test_df.iterrows():
            evidence = {feature: row[feature] for feature in FEATURES}
            if name == "Categorical Naive Bayes":
                encoded_row = {
                    col: bundle["category_maps"][col].get(str(evidence[col]), 0)
                    for col in FEATURES
                }
                X = pd.DataFrame([encoded_row], columns=FEATURES)
                p = float(model.predict_proba(X)[0][list(model.classes_).index(1)])
            else:
                p = float(model.predict_proba(evidence))
            preds.append(p)
        p_arr = np.array(preds)
        probabilities[name] = p_arr
        y_pred = (p_arr >= 0.5).astype(int)
        rows.append({
            "Model": name,
            "Accuracy": accuracy_score(y_true, y_pred),
            "Precision": precision_score(y_true, y_pred, zero_division=0),
            "Recall": recall_score(y_true, y_pred, zero_division=0),
            "F1": f1_score(y_true, y_pred, zero_division=0),
            "ROC-AUC": roc_auc_score(y_true, p_arr),
            "PR-AUC": average_precision_score(y_true, p_arr),
            "Confusion Matrix": confusion_matrix(y_true, y_pred).tolist(),
        })
    return pd.DataFrame(rows), probabilities

def save_bundle(bundle, path):
    """Save sklearn models and custom-network parameters to a portable joblib file."""
    serializable = dict(bundle)
    serializable["models"] = {
        name: (model.to_dict() if isinstance(model, DiscreteBayesNetwork) else model)
        for name, model in bundle["models"].items()
    }
    serializable["custom_model_names"] = [
        name for name, model in bundle["models"].items()
        if isinstance(model, DiscreteBayesNetwork)
    ]
    joblib.dump(serializable, path)

def load_bundle(path):
    serializable = joblib.load(path)
    for name in serializable.get("custom_model_names", []):
        serializable["models"][name] = DiscreteBayesNetwork.from_dict(serializable["models"][name])
    return serializable

def train_and_save(csv_path, model_path, test_size=0.25, random_state=42):
    from sklearn.model_selection import train_test_split
    data = load_and_prepare(csv_path)
    train_df, test_df = train_test_split(
        data, test_size=test_size, random_state=random_state, stratify=data[TARGET]
    )
    bundle = build_models(train_df)
    metrics, probabilities = evaluate_models(bundle, test_df)
    save_bundle(bundle, model_path)
    return data, train_df, test_df, bundle, metrics, probabilities
