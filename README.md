# Comparative Bayesian Models for Stroke Dataset Analysis

A small, presentable Intelligent Systems-II project comparing four Bayesian approaches on the same public stroke dataset. It includes a Jupyter notebook, saved model bundle, FastAPI service, and Streamlit dashboard.

## Four approaches

1. **Categorical Naive Bayes** — baseline with conditional independence.
2. **Expert Bayesian Network** — explicit graph with smoothed conditional probability tables.
3. **Compact Bayesian Network** — simpler alternative graph.
4. **Strong-Prior Bayesian Network** — expert graph with stronger Dirichlet smoothing, used for prior-sensitivity comparison.

Approach 4 is a sensitivity variant, not a wholly separate model family. The graph edges are modelling assumptions and do not prove causal relationships.

## Setup (Windows PowerShell)

Open a terminal in this project folder and run:

Using the requested Conda environment:

```powershell
conda activate ai
python -m pip install -r requirements.txt
```

If Conda is not initialized in PowerShell, open an Anaconda/Miniconda
PowerShell prompt first, or run `conda init powershell` once and restart
PowerShell.

The project can also be installed in a regular virtual environment:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 1. Run the notebook first

```powershell
jupyter notebook
```

Open `notebooks/Bayesian_Analysis.ipynb` and choose **Kernel → Restart Kernel and Run All Cells**. The notebook trains and saves the model bundle to `models/bayesian_models.joblib` and evaluation metrics to `models/evaluation_metrics.csv`.

## 2. Start the API

From the project root, in terminal 1:

```powershell
uvicorn api.main:app --reload
```

Check `http://127.0.0.1:8000/health` and API documentation at `http://127.0.0.1:8000/docs`.

## 3. Start Streamlit

In terminal 2, from the project root:

```powershell
streamlit run app/streamlit_app.py
```

The main dashboard uses a submitted patient form to call `/predict`, then
automatically calls `/explain` with the same current inputs. Held-out metrics
and confusion matrices are available on the separate **Model Evaluation** page
in the Streamlit navigation.

## 4. Configure the Groq results explainer chatbot

The explainer is a small LangChain `ChatGroq` chain. It sends only the current
form inputs and calculated aggregate evidence to Groq. The API key remains on
the FastAPI backend; it is never placed in Streamlit code or sent by the
browser. After **Predict and Compare** is clicked, the evidence report and LLM
interpretation are requested automatically.

In the same Windows PowerShell terminal used to start FastAPI, set the key for
that terminal session:

```powershell
$env:GROQ_API_KEY = "your-groq-api-key"
$env:GROQ_MODEL = "openai/gpt-oss-20b"
uvicorn api.main:app --reload
```

Alternatively, edit the project `.env` file using this format:

```env
GROQ_API_KEY=your-real-groq-api-key
GROQ_MODEL=openai/gpt-oss-20b
```

Restart FastAPI after changing `.env`; environment files are read when the API
process starts.

`GROQ_MODEL` is optional and defaults to `openai/gpt-oss-20b`. Do not commit
the key or put it in the repository. If a real key was previously exposed,
revoke it in the Groq console and create a replacement. To remove the variables from the current
PowerShell session:

```powershell
Remove-Item Env:GROQ_API_KEY -ErrorAction SilentlyContinue
Remove-Item Env:GROQ_MODEL -ErrorAction SilentlyContinue
```

## API endpoints

- `GET /health` — reports whether saved models loaded.
- `GET /models` — lists the model names.
- `POST /predict` — accepts age, glucose, BMI, hypertension, heart disease, gender and smoking status; returns probability estimates from all four approaches.
- `POST /explain` — accepts the same patient inputs as `/predict`, calculates the input categories, four predictions, agreement, training-data context, feature sensitivity, and held-out metrics, then returns that evidence report with a grounded Groq explanation. It returns a helpful error if the backend key is missing or Groq is unavailable.

Example JSON for `POST /predict`:

```json
{
  "age": 67,
  "avg_glucose_level": 145,
  "bmi": 29,
  "hypertension": 1,
  "heart_disease": 0,
  "gender": "Female",
  "smoking_status": "formerly smoked"
}
```

`POST /explain` uses the same patient-input JSON shape as `/predict`:

```json
{
  "age": 67,
  "avg_glucose_level": 145,
  "bmi": 29,
  "hypertension": 1,
  "heart_disease": 0,
  "gender": "Female",
  "smoking_status": "formerly smoked"
}
```

The response contains the calculated `evidence_report` and the LLM
`explanation`. The report includes the entered values, discretized evidence,
all four predictions, agreement range, training-data category statistics,
feature sensitivity comparisons, and all saved held-out metrics.

## Team contribution split

- **Member 1:** dataset review, preprocessing, prior/conditional probability analysis and EDA plots.
- **Member 2:** expert Bayesian Network graph, conditional probability tables and inference examples.
- **Member 3:** compact-network comparison, evaluation metrics, confusion matrices and prior-sensitivity experiment.
- **Member 4:** model saving, FastAPI endpoints, Streamlit dashboard, integration testing and run instructions.

All members should understand the four model assumptions and the evaluation results before the viva.

## Important limitation

This is a classroom demonstration using an imbalanced public dataset. It is not a diagnostic tool, and outputs must not be used to make medical decisions. Performance should not be judged by accuracy alone.
