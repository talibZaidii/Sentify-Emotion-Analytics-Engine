import streamlit as tf
import streamlit as st
import string
import nltk
from nltk.corpus import stopwords
import joblib

# Ensure NLTK resources are loaded smoothly
try:
    stop_words = set(stopwords.words('english'))
except LookupError:
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))

# ------------------------------------------------------------------
# 1. PAGE CONFIGURATION & THEME STYLING
# ------------------------------------------------------------------
st.set_page_config(
    page_title="Sentify - Emotion Analytics Studio",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Clean CSS Injection for UI adjustments
st.markdown("""
    <style>
    .main { background-color: #fcfcfd; }
    .stButton>button {
        background-color: #4F46E5;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 0.5rem 2rem;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #4338CA;
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(79, 70, 229, 0.2);
    }
    .metric-card {
        background-color: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        border: 1px solid #E5E7EB;
    }
    </style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------
# 2. DATA CLEANING PIPELINE (Mirrored from Notebook)
# ------------------------------------------------------------------


def clean_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Remove digits
    text = "".join([char for char in text if not char.isdigit()])
    # Remove emojis / non-ascii symbols
    text = "".join([char for char in text if char.isascii()])
    # Remove stopwords
    words = text.split()
    cleaned_words = [w for w in words if w not in stop_words]
    return " ".join(cleaned_words)

# ------------------------------------------------------------------
# 3. RESOURCE CACHING FOR LOSSLESS SPEED
# ------------------------------------------------------------------


@st.cache_resource
def load_pipeline_components():
    try:
        model = joblib.load('emotion_model.pkl')
        vectorizer = joblib.load('vectorizer.pkl')
        mapping = joblib.load('label_mapping.pkl')
        return model, vectorizer, mapping
    except FileNotFoundError:
        return None, None, None


model, vectorizer, label_mapping = load_pipeline_components()

# ------------------------------------------------------------------
# 4. SIDEBAR LOGOUT & METRICS INFO
# ------------------------------------------------------------------
with st.sidebar:
    st.image("https://images.unsplash.com/photo-1618005182384-a83a8bd57fbe?auto=format&fit=crop&w=300&q=80",
             use_container_width=True)
    st.markdown("### 📊 Model Diagnostic Specs")
    st.markdown("---")
    st.markdown("<div class='metric-card'><strong>Selected Classifier:</strong>< f'Logistic Regression' </div>",
                unsafe_allow_html=True)
    st.write("")
    st.metric(label="Validation Accuracy Accuracy",
              value="86.28%", delta="Baseline Model (+14.1%)")
    st.markdown("---")
    st.caption("Sentify UI Engine v1.0.0 • Built for clean analytics")

# ------------------------------------------------------------------
# 5. CORE BODY APPLICATION VIEW
# ------------------------------------------------------------------
st.title("🎭 Sentify: Emotion Analytics Engine")
st.markdown(
    "Analyze raw human expression through a text-vectorized classification pipeline.")
st.markdown("---")

if model is None:
    st.warning("⚠️ **Pipeline Files Not Found:** Please make sure `emotion_model.pkl`, `vectorizer.pkl`, and `label_mapping.pkl` are exported and placed in the root directory alongside this app.")
else:
    # Split interface horizontally into interactive input vs dynamic charts
    col1, col2 = st.columns([1.1, 0.9], gap="large")

    with col1:
        st.subheader("📝 Text Analysis Input")
        user_input = st.text_area(
            "What's on your mind? Paste individual journaling, tweets, or feedback records below:",
            height=180,
            placeholder="Type or paste text here (e.g., 'i can go from feeling so hopeless to so damned hopeful...')"
        )

        analyze_btn = st.button("Analyze Sentiment Engine")

    with col2:
        st.subheader("🎯 Classification Insights")

        if analyze_btn and user_input.strip():
            with st.spinner("Processing text and mapping vectors..."):
                # Run the text through the exact same transformations from your notebook
                cleaned = clean_text(user_input)
                transformed_vector = vectorizer.transform([cleaned])

                # Make prediction
                pred_id = model.predict(transformed_vector)[0]
                pred_label = label_mapping.get(pred_id, "Unknown").upper()

                # Calculate probability metrics safely if supported
                if hasattr(model, "predict_proba"):
                    probabilities = model.predict_proba(transformed_vector)[0]
                    classes = model.classes_
                else:
                    probabilities = None

                # Clean contextual styling map for different emotions
                emotion_themes = {
                    "JOY": ("#D1FAE5", "#065F46", "😃"),
                    "LOVE": ("#FCE7F3", "#9D174D", "💖"),
                    "SURPRISE": ("#FEF3C7", "#92400E", "😲"),
                    "SADNESS": ("#DBEAFE", "#1E40AF", "😢"),
                    "ANGER": ("#FEE2E2", "#991B1B", "😡"),
                    "FEAR": ("#F3E8FF", "#6B21A8", "😨"),
                }

                bg_color, text_color, emoji = emotion_themes.get(
                    pred_label, ("#F3F4F6", "#374151", "📊"))

                # Big clean banner showing core prediction
                st.markdown(f"""
                    <div style="background-color: {bg_color}; padding: 2rem; border-radius: 12px; text-align: center; border: 1px solid opacity 0.2;">
                        <span style="font-size: 3.5rem;">{emoji}</span>
                        <h3 style="color: {text_color}; margin-top: 0.5rem; letter-spacing: 1px;">PREDICTED EMOTION: {pred_label}</h3>
                    </div>
                """, unsafe_allow_html=True)

                # Build beautiful probability bar breakdown
                if probabilities is not None:
                    st.write("")
                    st.markdown("**Confidence Breakdown Accross Classes:**")
                    for cls_id, prob in zip(classes, probabilities):
                        class_name = label_mapping.get(
                            cls_id, f"Class {cls_id}").capitalize()
                        st.text(f"{class_name} ({prob*100:.1f}%)")
                        st.progress(float(prob))

        elif analyze_btn and not user_input.strip():
            st.error("Please insert a valid text block to parse classification.")
        else:
            st.info("💡 Write or paste some thoughts inside the text input workspace and click 'Analyze Sentiment Engine' to witness predictive UI metrics instantly.")
