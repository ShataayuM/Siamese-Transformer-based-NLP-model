import streamlit as st
import requests
import datetime
import json
import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModel
import numpy as np
import re

# --- 0. Page Configuration ---
st.set_page_config(
    page_title="Ethical Lens",
    page_icon="📰",
    layout="centered"
)

# --- 1. Constants and Mappings ---
COUNTRY_MAP = {
    'Argentina': 'ar', 'Australia': 'au', 'Austria': 'at', 'Belgium': 'be',
    'Brazil': 'br', 'Bulgaria': 'bg', 'Canada': 'ca', 'China': 'cn',
    'Colombia': 'co', 'Cuba': 'cu', 'Czech Republic': 'cz', 'Egypt': 'eg',
    'France': 'fr', 'Germany': 'de', 'Greece': 'gr', 'Hong Kong': 'hk',
    'Hungary': 'hu', 'India': 'in', 'Indonesia': 'id', 'Ireland': 'ie',
    'Israel': 'il', 'Italy': 'it', 'Japan': 'jp', 'Latvia': 'lv',
    'Lithuania': 'lt', 'Malaysia': 'my', 'Mexico': 'mx', 'Morocco': 'ma',
    'Netherlands': 'nl', 'New Zealand': 'nz', 'Nigeria': 'ng', 'Norway': 'no',
    'Philippines': 'ph', 'Poland': 'pl', 'Portugal': 'pt', 'Romania': 'ro',
    'Russia': 'ru', 'Saudi Arabia': 'sa', 'Serbia': 'rs', 'Singapore': 'sg',
    'Slovakia': 'sk', 'Slovenia': 'si', 'South Africa': 'za', 'South Korea': 'kr',
    'Sweden': 'se', 'Switzerland': 'ch', 'Taiwan': 'tw', 'Thailand': 'th',
    'Turkey': 'tr', 'UAE': 'ae', 'Ukraine': 'ua', 'United Arab Emirates': 'ae',
    'United Kingdom': 'gb', 'USA': 'us', 'Venezuela': 've'
}
NEWS_CATEGORIES = ['business', 'entertainment', 'general', 'health', 'science', 'sports', 'technology']
STOP_WORDS = {'a', 'an', 'the', 'is', 'in', 'on', 'of', 'for', 'to', 'with', 'and', 'vs', 'highlights', 'final', 'live'}

# --- 2. Model Loading and Architecture (Using Keras Subclassing) ---
@st.cache_resource
def load_model_and_tokenizer():
    class SiameseModel(tf.keras.Model):
        def __init__(self, model_name="distilbert-base-uncased", **kwargs):
            super().__init__(**kwargs)
            self.encoder = TFAutoModel.from_pretrained(model_name, from_pt=True)
            self.dot = tf.keras.layers.Dot(axes=1, normalize=True)
        def call(self, inputs):
            input_ids1, attention_mask1, input_ids2, attention_mask2 = inputs
            outputs1 = self.encoder({'input_ids': input_ids1, 'attention_mask': attention_mask1})
            embedding1 = outputs1.last_hidden_state[:, 0, :]
            outputs2 = self.encoder({'input_ids': input_ids2, 'attention_mask': attention_mask2})
            embedding2 = outputs2.last_hidden_state[:, 0, :]
            return self.dot([embedding1, embedding2])
    try:
        model = SiameseModel()
        tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
        return model, tokenizer
    except Exception as e:
        st.error(f"❌ Error building pre-trained model: {e}")
        return None, None

# --- 3. Helper Functions ---
def preprocess(texts, tokenizer):
    return tokenizer(texts, padding="max_length", truncation=True, max_length=128, return_tensors="tf")

def build_response(original, compared, score, article_meta=None):
    timestamp = datetime.datetime.utcnow().isoformat() + "Z"
    is_verified = score > 0.9 
    return {
        "verdict": "VERIFIED" if is_verified else "MISINFORMATION",
        "status": "✅ Verified" if is_verified else "🚫 Likely False / Misinformation",
        "confidence_score": float(round(score, 2)), "similarity_score": float(round(score, 2)),
        "evidence": [{"source": article_meta.get("source", {}).get("name") if is_verified and article_meta else "No reliable match",
                      "url": article_meta.get("url") if is_verified and article_meta else None,
                      "snippet": (article_meta.get("description") or "")[:150] if is_verified and article_meta else "No corroborating reports found."}],
        "original_headline": original, "compared_headline": compared if is_verified else "N/A",
        "processed_at": timestamp
    }

def extract_keywords(headline):
    """Extracts important keywords from a headline for a better search query."""
    # Use regex to find words, ignoring punctuation
    words = re.findall(r'\b\w+\b', headline.lower())
    keywords = [word for word in words if word not in STOP_WORDS]
    # Join the most important 5-6 keywords
    return " AND ".join(keywords[:6])

# --- 4. Main Application UI ---
st.title("📰 Ethical Lens")
st.markdown("Enter a news headline to check its authenticity against recent articles from trusted sources.")
st.info("ℹ️ Please note: This tool can only verify news published within the last 30 days.", icon="ℹ️")

siamese_model, tokenizer = load_model_and_tokenizer()

if siamese_model and tokenizer:
    with st.form("verify_form"):
        headline = st.text_area("Headline to Verify", "", placeholder="e.g., India vs Korea, Hockey Asia Cup 2025 Final Highlights: India beat Korea 4-1 to win Asia Cup, qualify for World Cup")
        col1, col2, col3 = st.columns(3)
        with col1:
            country = st.selectbox("Country", options=sorted(COUNTRY_MAP.keys()), index=list(sorted(COUNTRY_MAP.keys())).index('India'))
        with col2:
            category = st.selectbox("Category", options=NEWS_CATEGORIES, index=list(NEWS_CATEGORIES).index('sports'))
        with col3:
            page_size = st.number_input("Articles to check (per search)", min_value=10, max_value=50, value=25, step=5)
        
        submitted = st.form_submit_button("Analyze Headline")

    if submitted:
        if not headline.strip():
            st.warning("Please enter a headline to verify.")
        else:
            with st.spinner('🔎 Analyzing... Searching news sources...'):
                API_KEY = "21d6501d58264ca79e8490881db2ed61"
                search_keywords = extract_keywords(headline)
                
                # --- *** THE FIX IS HERE: Cascading Search Logic *** ---
                all_articles = []
                try:
                    # 1. Broad search on /everything with keywords
                    URL_everything = "https://newsapi.org/v2/everything"
                    params_everything = {"q": search_keywords, "searchIn": "title", "pageSize": page_size, "language": "en", "sortBy": "relevancy", "apiKey": API_KEY}
                    response_everything = requests.get(URL_everything, params=params_everything)
                    response_everything.raise_for_status()
                    all_articles.extend(response_everything.json().get("articles", []))

                    # 2. Targeted search on /top-headlines for the specific country/category
                    URL_top = "https://newsapi.org/v2/top-headlines"
                    country_code = COUNTRY_MAP[country]
                    params_top = {"country": country_code, "category": category, "q": search_keywords, "pageSize": page_size, "apiKey": API_KEY}
                    response_top = requests.get(URL_top, params=params_top)
                    response_top.raise_for_status()
                    all_articles.extend(response_top.json().get("articles", []))
                
                except requests.exceptions.RequestException as e:
                    st.error(f"Could not connect to NewsAPI: {e}")

                # Remove duplicates based on URL
                if all_articles:
                    unique_articles = {article['url']: article for article in all_articles}.values()
                    articles = list(unique_articles)
                else:
                    articles = []

                if not articles:
                    st.warning("Could not find any matching articles in trusted sources.")
                    result = build_response(headline, "", 0.0)
                else:
                    fetched_headlines = [article['title'] for article in articles]
                    ref_headline_list = [headline] * len(fetched_headlines)
                    inputs_ref = preprocess(ref_headline_list, tokenizer)
                    inputs_test = preprocess(fetched_headlines, tokenizer)
                    
                    predictions = siamese_model.predict([
                        inputs_ref['input_ids'], inputs_ref['attention_mask'],
                        inputs_test['input_ids'], inputs_test['attention_mask']
                    ], verbose=0)
                    
                    scores = predictions.flatten()
                    best_score_index = np.argmax(scores)
                    best_score = scores[best_score_index]
                    best_match_headline = fetched_headlines[best_score_index]
                    best_match_article_meta = articles[best_score_index]
                    result = build_response(headline, best_match_headline, best_score, best_match_article_meta)

                st.divider()
                st.subheader("Verification Result")
                if result['verdict'] == "VERIFIED":
                    st.success(f"**Status:** {result['status']}")
                else:
                    st.error(f"**Status:** {result['status']}")
                c1, c2 = st.columns(2)
                c1.metric("Confidence Score", f"{result['confidence_score'] * 100:.0f}%")
                c2.metric("Similarity Score", f"{result['similarity_score'] * 100:.0f}%")
                with st.expander("Show Evidence and Details"):
                    st.json(result)
else:
    st.error("Model could not be loaded. The application cannot start.")
