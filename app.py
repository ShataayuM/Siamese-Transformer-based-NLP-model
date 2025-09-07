import streamlit as st
import requests
import datetime
import json
import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModel
import numpy as np

# --- 0. Page Configuration ---
st.set_page_config(
    page_title="Headline Verifier",
    page_icon="📰",
    layout="centered"
)

# --- 1. Country Name to Code Mapping ---
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

# --- 2. Model Loading and Architecture ---
@st.cache_resource
def load_model_and_tokenizer():
    """
    Loads the saved model into a new, deployment-friendly architecture.
    """
    # -- A more robust model architecture for deployment --
    def build_deployment_model():
        input_ids1 = tf.keras.layers.Input(shape=(None,), dtype=tf.int32, name='input_ids_1')
        attention_mask1 = tf.keras.layers.Input(shape=(None,), dtype=tf.int32, name='attention_mask_1')
        input_ids2 = tf.keras.layers.Input(shape=(None,), dtype=tf.int32, name='input_ids_2')
        attention_mask2 = tf.keras.layers.Input(shape=(None,), dtype=tf.int32, name='attention_mask_2')

        # Create a single, reusable encoder model
        base_model = TFAutoModel.from_pretrained("distilbert-base-uncased", from_pt=True)
        encoder_input_ids = tf.keras.layers.Input(shape=(None,), dtype=tf.int32)
        encoder_attention_mask = tf.keras.layers.Input(shape=(None,), dtype=tf.int32)
        encoder_output = base_model({'input_ids': encoder_input_ids, 'attention_mask': encoder_attention_mask}).last_hidden_state
        # Use the CLS token for the sentence embedding
        embedding_output = encoder_output[:, 0, :]
        encoder = tf.keras.Model(inputs=[encoder_input_ids, encoder_attention_mask], outputs=embedding_output)
        
        # Get embeddings for both inputs
        embedding1 = encoder([input_ids1, attention_mask1])
        embedding2 = encoder([input_ids2, attention_mask2])

        # Calculate similarity
        similarity = tf.keras.layers.Dot(axes=1, normalize=True)([embedding1, embedding2])
        
        # Create the final model
        model = tf.keras.Model(inputs=[input_ids1, attention_mask1, input_ids2, attention_mask2], outputs=similarity)
        return model

    # -- Build, Load Weights, and Return --
    try:
        model = build_deployment_model()
        # This will now only load the weights that match the new, simpler architecture
        model.load_weights("siamese_model.weights.h5")
        tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
        return model, tokenizer
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        st.error("Please ensure 'siamese_model.weights.h5' is in the correct directory.")
        return None, None

# --- 3. Helper Functions (Unchanged) ---
def preprocess(texts, tokenizer):
    return tokenizer(texts, padding="max_length", truncation=True, max_length=128, return_tensors="tf")

def build_response(original, compared, score, article_meta=None):
    timestamp = datetime.datetime.utcnow().isoformat() + "Z"
    is_verified = score > 0.5
    return {
        "verdict": "VERIFIED" if is_verified else "MISINFORMATION",
        "status": "✅ Verified" if is_verified else "🚫 Likely False / Misinformation",
        "confidence_score": float(round(score, 2)),
        "similarity_score": float(round(score, 2)),
        "evidence": [{
            "source": article_meta.get("source", {}).get("name") if is_verified and article_meta else "No reliable match",
            "url": article_meta.get("url") if is_verified and article_meta else None,
            "snippet": (article_meta.get("description") or "")[:150] if is_verified and article_meta else "No corroborating reports found."
        }],
        "original_headline": original,
        "compared_headline": compared if is_verified else "N/A",
        "processed_at": timestamp
    }

# --- 4. Main Application UI (Unchanged) ---
st.title("📰 Headline Verification System")
st.markdown("Enter a news headline to check its authenticity against recent articles from trusted sources.")

siamese_model, tokenizer = load_model_and_tokenizer()

if siamese_model and tokenizer:
    with st.form("verify_form"):
        headline = st.text_area("Headline to Verify", "", placeholder="e.g., New study shows coffee cures all diseases.")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            country = st.selectbox("Country", options=sorted(COUNTRY_MAP.keys()), index=list(sorted(COUNTRY_MAP.keys())).index('USA'))
        with col2:
            category = st.selectbox("Category", options=NEWS_CATEGORIES, index=2)
        with col3:
            page_size = st.number_input("Articles to check", min_value=5, max_value=100, value=20, step=5)
        
        submitted = st.form_submit_button("Analyze Headline")

    if submitted:
        if not headline.strip():
            st.warning("Please enter a headline to verify.")
        else:
            with st.spinner('🔎 Analyzing... Fetching news, running model, and comparing results...'):
                country_code = COUNTRY_MAP[country]
                API_KEY = "21d6501d58264ca79e8490881db2ed61" # Consider hiding this in st.secrets
                params = {"country": country_code, "category": category, "pageSize": page_size, "apiKey": API_KEY}
                try:
                    response = requests.get("https://newsapi.org/v2/top-headlines", params=params)
                    response.raise_for_status()
                    articles = response.json().get("articles", [])
                except requests.exceptions.RequestException as e:
                    st.error(f"Could not connect to NewsAPI: {e}")
                    articles = []

                if not articles:
                    st.warning("Could not fetch any articles for the given criteria. Please try again.")
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
