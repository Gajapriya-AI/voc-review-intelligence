
# AI-Powered Customer VoC Review Intelligence Platform

## 🚀 Live Demo

**Try the application here:** [AI-Powered Customer VoC Review Intelligence](https://voc-review-intelligence-xexqrcqbzh4o3ulmfo9h7h.streamlit.app/)

A local-first portfolio project that analyzes customer reviews, identifies aspect-level feedback and pain points, and turns evidence into concise business insights. It supports single-review analysis and optional batch CSV analysis.

## What is included

- Streamlit app with Single Review, Bulk CSV Analysis, Demo Dashboard, and Methodology tabs.
- A trainable word/character TF-IDF + Logistic Regression sentiment baseline.
- An optional BiLSTM Deep Learning experiment (`train_bilstm.py`) with separate metrics and saved artifacts; TensorFlow is not installed by the default requirements.
- Transparent aspect-level phrase rules designed to preserve mixed feedback (for example, good performance but poor battery life).
- Optional local Small LLM summaries through Ollama. The app works without Ollama and clearly labels its fallback.
- Training and demo CSV files, training/evaluation script, automated tests, and an evaluation report.

## Important honesty note

The bundled 330-row training set and demo reviews are **synthetic starter data**, not real customer research. The included classifier is genuinely trained by `train_model.py`, but its metrics on this small synthetic dataset do **not** establish real-world accuracy. For a portfolio-grade evaluation, replace or augment the starter data with a properly licensed, human-labelled public review dataset, prevent duplicate/customer leakage between splits, and re-run evaluation. Do not claim 100% accuracy.

The aspect analyzer is an interpretable keyword/phrase baseline, not a fully trained aspect-based sentiment model. It can miss paraphrases, sarcasm, misspellings, slang, and Tamil-English code-mixing. Confidence is a model estimate, not a guarantee.

## Requirements

- Python 3.10–3.12 recommended.
- Internet is needed only once to install the Python packages. The sentiment model itself trains locally; no large model is downloaded.
- Optional: Ollama installed separately and a small local model downloaded to enable generated LLM summaries. This is optional and can require substantial RAM/storage depending on model choice.

## Setup on Windows PowerShell

1. Extract the ZIP.
2. Open the extracted `voc_review_intelligence` folder in VS Code.
3. Create a virtual environment:

   ```powershell
   py -m venv .venv
   .\.venv\Scripts\Activate.ps1
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. Train the local baseline and create metrics:

   ```powershell
   python train_model.py
   ```

   This creates `models/sentiment_model.joblib` and `reports/metrics.json`.

5. Run tests:

   ```powershell
   python -m unittest discover -s tests -v
   ```

6. Start the app:

   ```powershell
   streamlit run app.py
   ```

   Streamlit will show a local URL, usually `http://localhost:8501`.

## Optional BiLSTM Deep Learning experiment

To run the separate sequence-model experiment, install `requirements-dl.txt` (TensorFlow is a larger dependency), then run `python train_bilstm.py`. This creates `models/bilstm/` with the model, tokenizer, label map, and held-out metrics. The default Streamlit app continues to use the compact baseline model unless you explicitly extend it to load the BiLSTM artifact. The bundled synthetic dataset is for pipeline demonstration only.

## Optional Small LLM (Ollama)

The app can call an Ollama server running on the same computer. Install Ollama separately, download a model appropriate for your computer, start Ollama, then set environment variables in the same PowerShell session. Example (replace the model name with one you actually installed):

```powershell
$env:OLLAMA_MODEL="your-installed-model-name"
$env:OLLAMA_HOST="http://localhost:11434"
streamlit run app.py
```

If the environment variable is not set or the server is unavailable, the app uses a rule-based insight and labels the source. No external API key is required for local Ollama use.

## Dataset formats

### Training data
`data/training_reviews.csv` requires these columns:

- `review`: review text
- `sentiment`: `Positive`, `Negative`, or `Neutral`
- `category`: product category (metadata; optional to the baseline trainer)

Update the data carefully and rerun `python train_model.py`. For a robust evaluation, use real labelled data and a separate test set.

### Prediction CSV
Upload any CSV with a review text column. Select the correct text column in the app. Sentiment labels are not required for prediction. A category column can be selected if present. The uploaded file is not used to retrain the model.

## Project structure

```text
voc_review_intelligence/
├── app.py
├── train_model.py
├── make_dataset.py
├── requirements.txt
├── requirements-llm.txt
├── src/
│   ├── analysis.py
│   ├── model.py
│   └── llm_insights.py
├── data/
│   ├── training_reviews.csv
│   └── demo_reviews.csv
├── models/                 # trained model artifact
├── reports/                # generated evaluation metrics
├── tests/
└── notebooks/
```

## Evaluation and responsible use

- Report accuracy, macro-F1, precision, recall and the confusion matrix from the held-out test split.
- Test mixed reviews separately; a single overall sentiment label can hide different opinions about different aspects.
- Review predictions manually before using them for business decisions.
- Check the licence and terms for any public dataset you add. Do not bundle customer personal information.
- The app is a prototype and not a substitute for human review or a validated production system.
=======
# voc-review-intelligence
AI-powered Voice of the Customer (VoC) platform that analyzes customer reviews using NLP and Machine Learning, predicts sentiment, identifies customer pain points, and visualizes insights through an interactive Streamlit dashboard.
