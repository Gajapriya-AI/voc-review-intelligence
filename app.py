from pathlib import Path
import sys
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from src.analysis import analyze_review
from src.model import load_model, train_and_evaluate
from src.llm_insights import generate_insight

st.set_page_config(page_title="VoC Review Intelligence", page_icon="🧠", layout="wide")

@st.cache_resource
def get_model():
    model = load_model()
    if model is None:
        try:
            train_and_evaluate()
            model = load_model()
        except Exception as exc:
            st.error(f"Could not prepare the local baseline model: {exc}")
    return model

model = get_model()

with st.sidebar:
    st.title("VoC Intelligence")
    st.caption("Customer review analysis platform")
    st.markdown("---")
    category = st.selectbox("Product category", ["Electronics", "Home Appliances", "Fashion & Apparel", "Beauty & Personal Care", "Food & Grocery", "Furniture & Home", "Other Products"])
    st.markdown("**Analysis modules**")
    st.caption("Sentiment · Aspect signals · Pain points · Business insight")
    st.markdown("---")
    st.caption("Local-first demo. Review predictions are estimates, not guarantees.")

st.title("AI-Powered Customer VoC Review Intelligence")
st.write("Turn customer feedback into structured sentiment, aspect-level signals, and practical product insights.")
st.info("This app combines a locally trained text-classification baseline with transparent aspect rules. An optional local Small LLM can generate summaries when configured through Ollama.")

tab_single, tab_batch, tab_dashboard, tab_about = st.tabs(["Single Review", "Bulk CSV Analysis", "Demo Dashboard", "Methodology & Limits"])


def show_result(result, key_prefix="single"):
    a,b,c = st.columns(3)
    a.metric("Overall sentiment", result["sentiment"])
    b.metric("Aspects detected", len(result["aspects"]))
    c.metric("Pain points", len(result["pain_points"]))
    if result.get("confidence") is not None:
        st.caption(f"Classifier confidence estimate: {result['confidence']:.1%}. This is not a guarantee of correctness.")
    left,right=st.columns(2)
    with left:
        st.subheader("Aspect-level findings")
        if result["aspects"]:
            st.dataframe(pd.DataFrame(result["aspects"]), use_container_width=True, hide_index=True)
        else:
            st.write("No known aspect terms were detected. Human review may be useful.")
        st.subheader("Positive feedback")
        st.write(", ".join(result["positive_feedback"]) if result["positive_feedback"] else "No clear positive evidence detected.")
    with right:
        st.subheader("Pain points")
        st.write(", ".join(result["pain_points"]) if result["pain_points"] else "No clear pain-point evidence detected.")
        st.subheader("Business insight")
        insight, source = generate_insight(result)
        st.write(insight)
        st.caption(f"Insight source: {source}")

with tab_single:
    st.subheader("Analyze one customer review")
    review = st.text_area("Paste review text", height=150, placeholder="Example: The laptop is fast and lightweight, but the battery drains quickly.")
    if st.button("Analyze review", type="primary", key="analyze_single"):
        if not review.strip():
            st.warning("Enter a review before analyzing.")
        else:
            result=analyze_review(review, category, model)
            show_result(result)
            st.download_button("Download this result as JSON", data=pd.Series(result).to_json(force_ascii=False, indent=2), file_name="review_analysis.json", mime="application/json")

with tab_batch:
    st.subheader("Analyze multiple reviews")
    st.write("Upload a CSV containing a review-text column. Sentiment labels are not required for prediction.")
    uploaded=st.file_uploader("Choose CSV file", type=["csv"], key="csv_upload")
    if uploaded is not None:
        try:
            df=pd.read_csv(uploaded)
            st.write("Preview", df.head(5))
            if df.empty:
                st.error("The uploaded CSV has no rows.")
            else:
                review_col=st.selectbox("Select review text column", list(df.columns), key="review_col")
                cat_col=st.selectbox("Product category column (optional)", ["None"]+list(df.columns), key="category_col")
                if st.button("Analyze CSV", type="primary"):
                    output=[]
                    progress=st.progress(0)
                    for i, row in df.iterrows():
                        review_text="" if pd.isna(row[review_col]) else str(row[review_col])
                        row_category=category if cat_col=="None" or pd.isna(row[cat_col]) else str(row[cat_col])
                        result=analyze_review(review_text,row_category,model)
                        output.append({"review":result["review"],"category":result["category"],"sentiment":result["sentiment"],"confidence":result["confidence"],"aspects":"; ".join(a["aspect"]+":"+a["sentiment"] for a in result["aspects"]),"positive_feedback":"; ".join(result["positive_feedback"]),"pain_points":"; ".join(result["pain_points"]),"business_insight":result["business_insight"]})
                        progress.progress((i+1)/len(df))
                    result_df=pd.DataFrame(output)
                    st.success(f"Analyzed {len(result_df)} rows.")
                    st.dataframe(result_df.head(100),use_container_width=True,hide_index=True)
                    st.download_button("Download analyzed CSV", data=result_df.to_csv(index=False).encode("utf-8-sig"), file_name="voc_analyzed_reviews.csv", mime="text/csv")
        except Exception as exc:
            st.error(f"Could not read this CSV. Check that it is a valid UTF-8/standard CSV. Details: {exc}")

with tab_dashboard:
    st.subheader("Demo review insights")
    demo_path=ROOT/"data"/"demo_reviews.csv"
    if demo_path.exists():
        demo=pd.read_csv(demo_path)
        if st.button("Run demo analysis", key="run_demo"):
            analyzed=[analyze_review(r, c, model) for r,c in zip(demo.review.astype(str),demo.category.astype(str))]
            results=pd.DataFrame([{"category":x["category"],"sentiment":x["sentiment"],"pain_points":", ".join(x["pain_points"])} for x in analyzed])
            counts=results.sentiment.value_counts()
            st.bar_chart(counts)
            st.dataframe(results,use_container_width=True,hide_index=True)
            st.caption("Demo reviews are illustrative examples, not real customer research.")
    else:
        st.warning("Demo dataset is missing.")

with tab_about:
    st.subheader("How this project works")
    st.markdown("""
- **Training:** `python train_model.py` trains a word/character TF-IDF + Logistic Regression baseline using the bundled starter dataset.
- **Prediction:** the app uses the saved model and transparent aspect phrase rules.
- **Optional Small LLM:** set up Ollama locally and set `OLLAMA_MODEL` to an installed model name. Without this setting, the app uses a rule-based insight fallback.
- **Data:** bundled training and demo reviews are synthetic starter examples. Replace/augment them with appropriately licensed, labelled real data and re-evaluate before making real-world claims.
- **Evaluation:** run `python train_model.py` to save held-out metrics to `reports/metrics.json`.
""")
    st.warning("Limitations: phrase-based aspect detection may miss paraphrases, sarcasm, spelling errors, and mixed-language reviews. The starter dataset is synthetic and small; evaluation metrics on it do not establish real-world accuracy. Validate against human-labelled reviews before using this for business decisions.")
