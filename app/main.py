import streamlit as st
import numpy as np
from PIL import Image
import os
import sys
import base64

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from model import SkinDiseaseClassifier
from ar_visualizer import ARVisualizer

st.set_page_config(
    page_title="DermAI — Skin Disease Detector",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── Load images as base64 ──────────────────────────────────────
def img_to_b64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

base_dir   = os.path.join(os.path.dirname(__file__), '..', 'imag')
bg_path    = os.path.abspath(os.path.join(base_dir, '1 for baground.webp'))
acne_path  = os.path.abspath(os.path.join(base_dir, 'faceacne img.jpg'))

bg_b64   = img_to_b64(bg_path)
acne_b64 = img_to_b64(acne_path)

st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    * {{ font-family: 'Inter', sans-serif; }}

    #MainMenu, footer, header {{ visibility: hidden; }}
    .block-container {{ padding: 0 2rem 2rem 2rem !important; max-width: 1200px; }}

    /* Light background with image */
    .stApp {{
        background-image: url("data:image/webp;base64,{bg_b64}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
        background-repeat: no-repeat;
    }}
    .stApp::before {{
        content: '';
        position: fixed;
        inset: 0;
        background: rgba(255,255,255,0.88);
        z-index: 0;
    }}
    .block-container {{ position: relative; z-index: 1; }}

    /* Navbar */
    .navbar {{
        display: flex; align-items: center; justify-content: space-between;
        padding: 1.2rem 0; border-bottom: 2px solid #e2e8f0;
        margin-bottom: 2.5rem; background: rgba(255,255,255,0.95);
        border-radius: 0 0 12px 12px;
    }}
    .navbar-brand {{
        font-size: 1.5rem; font-weight: 700; color: #6d28d9 !important;
        display: flex; align-items: center; gap: 0.5rem;
    }}
    .navbar-links {{ display: flex; gap: 2rem; }}
    .navbar-links a {{
        color: #475569; font-size: 0.92rem; text-decoration: none;
        font-weight: 500; padding: 0.3rem 0.6rem; border-radius: 6px;
        transition: all 0.2s;
    }}
    .navbar-links a:hover {{ color: #6d28d9; background: #f3f0ff; }}

    /* Hero */
    .hero {{ text-align: center; padding: 2rem 0 2.5rem 0; }}
    .hero-badge {{
        display: inline-block; background: #f3f0ff;
        color: #6d28d9 !important; border: 1px solid #c4b5fd;
        padding: 0.3rem 1rem; border-radius: 999px;
        font-size: 0.8rem; font-weight: 600; margin-bottom: 1.2rem;
    }}
    .hero h1 {{
        font-size: 2.8rem !important; font-weight: 700 !important; color: #1e293b !important;
        line-height: 1.2; margin: 0 0 1rem 0;
    }}
    .hero h1 span {{ color: #6d28d9 !important; }}
    .hero p {{ color: #64748b !important; font-size: 1.05rem; max-width: 560px; margin: 0 auto; }}

    /* Stats */
    .stats-row {{
        display: flex; justify-content: center; gap: 3rem;
        margin: 2rem 0 3rem 0;
    }}
    .stat {{ text-align: center; }}
    .stat-value {{ font-size: 1.8rem !important; font-weight: 700 !important; color: #6d28d9 !important; }}
    .stat-label {{ font-size: 0.8rem !important; color: #64748b !important; margin-top: 0.2rem; }}

    /* Cards */
    .result-card {{
        background: rgba(255,255,255,0.95); border-radius: 16px;
        padding: 1.4rem; margin-bottom: 1rem;
        border: 1px solid #e2e8f0;
        box-shadow: 0 2px 12px rgba(0,0,0,0.06);
    }}
    .result-label {{
        font-size: 0.72rem !important; color: #94a3b8 !important; text-transform: uppercase;
        letter-spacing: 0.08em; margin-bottom: 0.3rem;
    }}
    .result-value {{ font-size: 1.15rem !important; font-weight: 600 !important; color: #1e293b !important; }}

    /* Confidence bar */
    .conf-bar-bg {{
        background: #e2e8f0; border-radius: 999px;
        height: 8px; margin-top: 0.8rem; overflow: hidden;
    }}
    .conf-bar-fill {{
        height: 100%; border-radius: 999px;
        background: linear-gradient(90deg, #6d28d9, #a78bfa);
    }}

    /* Badges */
    .badge {{ display: inline-block; padding: 0.25rem 0.75rem; border-radius: 999px; font-size: 0.78rem; font-weight: 600; }}
    .badge-high {{ background: #d1fae5; color: #065f46; border: 1px solid #6ee7b7; }}
    .badge-med  {{ background: #fef3c7; color: #92400e; border: 1px solid #fcd34d; }}
    .badge-low  {{ background: #fee2e2; color: #991b1b; border: 1px solid #fca5a5; }}

    /* Info cards */
    .info-card {{
        background: rgba(255,255,255,0.95); border-radius: 16px;
        padding: 1.4rem; border: 1px solid #e2e8f0; margin-bottom: 1rem;
        box-shadow: 0 2px 12px rgba(0,0,0,0.06);
    }}
    .info-card h4 {{
        color: #6d28d9 !important; font-size: 0.82rem !important; text-transform: uppercase;
        letter-spacing: 0.08em; margin: 0 0 0.5rem 0;
    }}
    .info-card p {{ color: #475569 !important; font-size: 0.9rem !important; margin: 0; line-height: 1.6; }}

    /* Force all text visible on light bg */
    .stApp, .stApp * {{ color: #1e293b; }}
    p, li, span, div {{ color: #475569; }}
    h1, h2, h3, h4, h5, h6 {{ color: #1e293b !important; }}

    /* Section panels (About, Diseases, Disclaimer) */
    .section-panel {{
        background: rgba(255,255,255,0.97) !important; border-radius: 20px;
        padding: 2rem; border: 1px solid #e2e8f0;
        box-shadow: 0 4px 24px rgba(0,0,0,0.07); margin-bottom: 2rem;
    }}
    .section-panel h2 {{ color: #1e293b !important; font-size: 1.6rem !important; margin: 0 0 0.5rem 0 !important; font-weight: 700 !important; }}
    .section-panel h3 {{ color: #6d28d9 !important; font-size: 1.1rem !important; margin: 1.2rem 0 0.4rem 0 !important; font-weight: 600 !important; }}
    .section-panel p  {{ color: #475569 !important; line-height: 1.7 !important; font-size: 0.95rem !important; }}
    .section-panel ul {{ color: #475569 !important; line-height: 2 !important; font-size: 0.95rem !important; padding-left: 1.2rem !important; }}
    .section-panel li {{ color: #475569 !important; }}
    .section-panel strong {{ color: #1e293b !important; }}

    /* Disease grid */
    .disease-grid {{
        display: grid; grid-template-columns: repeat(auto-fill, minmax(260px, 1fr));
        gap: 1rem; margin-top: 1rem;
    }}
    .disease-item {{
        background: #f8f5ff; border-radius: 12px; padding: 1rem 1.2rem;
        border-left: 4px solid #6d28d9;
    }}
    .disease-item strong {{ color: #1e293b !important; font-size: 0.92rem; }}
    .disease-item p {{ color: #64748b !important; font-size: 0.82rem; margin: 0.3rem 0 0 0; }}

    /* Upload placeholder with acne image */
    .upload-placeholder {{
        background: rgba(255,255,255,0.95); border: 2px dashed #c4b5fd;
        border-radius: 16px; padding: 0; overflow: hidden; text-align: center;
        box-shadow: 0 2px 12px rgba(0,0,0,0.06);
    }}
    .upload-placeholder img {{
        width: 100%; max-height: 220px; object-fit: cover; opacity: 0.7;
        border-radius: 14px 14px 0 0;
    }}
    .upload-placeholder-text {{ padding: 1rem; }}
    .upload-placeholder-text p {{ color: #94a3b8; font-size: 0.85rem; margin: 0; }}
    .upload-placeholder-text strong {{ color: #6d28d9; font-size: 1rem; }}

    /* Disclaimer box */
    .disclaimer {{
        background: #fef3c7; border: 1px solid #fcd34d;
        border-radius: 12px; padding: 1rem 1.2rem;
        color: #78350f; font-size: 0.85rem; line-height: 1.6; margin-top: 1rem;
    }}

    /* Section title */
    .section-title {{
        font-size: 1rem !important; font-weight: 600 !important; color: #1e293b !important;
        margin-bottom: 1rem;
    }}

    /* File uploader */
    [data-testid="stFileUploader"] section {{
        background: rgba(255,255,255,0.95) !important;
        border: 2px dashed #c4b5fd !important;
        border-radius: 16px !important;
    }}

    /* Radio */
    .stRadio > div {{ flex-direction: row !important; gap: 0.5rem; }}
    .stRadio label {{
        background: white !important; border: 1px solid #e2e8f0 !important;
        border-radius: 8px !important; padding: 0.4rem 1rem !important;
        color: #475569 !important; font-size: 0.85rem !important;
    }}

    hr {{ border-color: #e2e8f0 !important; margin: 2rem 0 !important; }}

    [data-testid="stImage"] img {{
        border-radius: 12px !important;
        border: 1px solid #e2e8f0 !important;
        box-shadow: 0 2px 12px rgba(0,0,0,0.08) !important;
    }}

    /* Nav section anchors */
    .nav-section {{ scroll-margin-top: 80px; }}
</style>
""", unsafe_allow_html=True)

# ── Session state ──────────────────────────────────────────────
if 'model'             not in st.session_state: st.session_state.model = None
if 'ar_viz'            not in st.session_state: st.session_state.ar_viz = ARVisualizer()
if 'prediction_result' not in st.session_state: st.session_state.prediction_result = None

# ── Load model ─────────────────────────────────────────────────
def load_model():
    model_path   = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", "skin_disease_model.h5"))
    encoder_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", "label_encoder.pkl"))
    classifier   = SkinDiseaseClassifier()
    if os.path.exists(model_path) and os.path.exists(encoder_path):
        try:
            classifier.load_model(model_path, encoder_path)
            return classifier
        except: pass
    with st.spinner("Training model… this may take a few minutes."):
        train_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "dataset", "train"))
        if os.path.exists(train_dir):
            classifier.train(train_dir, epochs=5)
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            classifier.save_model(model_path, encoder_path)
        else:
            st.error(f"Training data not found: {train_dir}")
            return None
    return classifier

# ── Navbar ─────────────────────────────────────────────────────
st.markdown("""
<div class="navbar">
    <div class="navbar-brand">🩺 DermAI</div>
    <div class="navbar-links">
        <a href="#home">Home</a>
        <a href="#about">About</a>
        <a href="#diseases">Diseases</a>
        <a href="#disclaimer">Disclaimer</a>
    </div>
</div>
""", unsafe_allow_html=True)

# ── HOME section ───────────────────────────────────────────────
st.markdown('<div class="nav-section" id="home"></div>', unsafe_allow_html=True)

st.markdown(f"""
<div class="hero">
    <div class="hero-badge">✦ AI-Powered Dermatology</div>
    <h1>Detect Skin Diseases<br>with <span>Confidence</span></h1>
    <p>Upload a skin image and get instant AI analysis with AR visualization across 7 disease categories.</p>
</div>
<div class="stats-row">
    <div class="stat"><div class="stat-value">7</div><div class="stat-label">Disease Categories</div></div>
    <div class="stat"><div class="stat-value">CNN</div><div class="stat-label">Model Architecture</div></div>
    <div class="stat"><div class="stat-value">AR</div><div class="stat-label">Visualization</div></div>
    <div class="stat"><div class="stat-value">Real-time</div><div class="stat-label">Analysis</div></div>
</div>
""", unsafe_allow_html=True)

# ── Detector columns ───────────────────────────────────────────
left, right = st.columns([1, 1], gap="large")

with left:
    st.markdown('<div class="section-title">📤 Upload Skin Image</div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Upload", type=['jpg', 'jpeg', 'png'], label_visibility="collapsed"
    )

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, use_column_width=True)

        if st.session_state.model is None:
            with st.spinner("Loading AI model…"):
                st.session_state.model = load_model()

        if st.session_state.model:
            with st.spinner("Analyzing image…"):
                try:
                    disease_name, confidence = st.session_state.model.predict(image)
                    st.session_state.prediction_result = {
                        'disease_name': disease_name,
                        'confidence': confidence,
                        'image': image
                    }
                except Exception as e:
                    st.error(f"Prediction error: {e}")
                    st.session_state.prediction_result = None

        if st.session_state.prediction_result:
            r        = st.session_state.prediction_result
            conf_pct = r['confidence'] * 100
            badge    = "badge-high" if r['confidence'] >= 0.8 else "badge-med" if r['confidence'] >= 0.6 else "badge-low"
            badge_text = "High Confidence" if r['confidence'] >= 0.8 else "Medium Confidence" if r['confidence'] >= 0.6 else "Low Confidence"

            st.markdown(f"""
            <div class="result-card">
                <div class="result-label">Detected Condition</div>
                <div class="result-value">{r['disease_name']}</div>
            </div>
            <div class="result-card">
                <div class="result-label">Confidence Score</div>
                <div class="result-value">{conf_pct:.1f}% &nbsp;<span class="badge {badge}">{badge_text}</span></div>
                <div class="conf-bar-bg"><div class="conf-bar-fill" style="width:{conf_pct:.1f}%"></div></div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="upload-placeholder">
            <img src="data:image/jpeg;base64,{acne_b64}" alt="Sample skin image"/>
            <div class="upload-placeholder-text">
                <strong>Upload a skin image to begin analysis</strong>
                <p>Supports JPG, JPEG, PNG &nbsp;·&nbsp; Example: Acne condition shown above</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

with right:
    st.markdown('<div class="section-title">🔬 AR Visualization</div>', unsafe_allow_html=True)
    ar_mode = st.radio("View mode", ["AR Overlay", "3D Marker", "Both"], horizontal=True, label_visibility="collapsed")

    if st.session_state.prediction_result:
        r            = st.session_state.prediction_result
        disease_name = r['disease_name']
        confidence   = r['confidence']
        image        = r['image']
        ar_viz       = st.session_state.ar_viz

        try:
            if ar_mode in ["AR Overlay", "Both"]:
                st.markdown('<div class="section-title" style="font-size:0.82rem;color:#94a3b8;">AR Overlay</div>', unsafe_allow_html=True)
                st.image(ar_viz.create_ar_overlay(image, disease_name, confidence), use_column_width=True)
            if ar_mode in ["3D Marker", "Both"]:
                st.markdown('<div class="section-title" style="font-size:0.82rem;color:#94a3b8;">3D Confidence Marker</div>', unsafe_allow_html=True)
                st.image(ar_viz.create_3d_marker(image, disease_name, confidence), use_column_width=True)
        except Exception as e:
            st.error(f"AR error: {e}")

        info = ar_viz.disease_info.get(disease_name, {})
        if info:
            st.markdown(f"""
            <div class="info-card">
                <h4>📋 Description</h4>
                <p>{info.get('description', 'N/A')}</p>
            </div>
            <div class="info-card">
                <h4>💊 Recommended Treatment</h4>
                <p>{info.get('treatment', 'N/A')}</p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("""
        <div class="disclaimer">
            ⚠️ <strong>Medical Disclaimer:</strong> This tool is for educational purposes only.
            Always consult a qualified dermatologist for proper diagnosis and treatment.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="upload-placeholder">
            <img src="data:image/jpeg;base64,{acne_b64}" alt="AR preview"/>
            <div class="upload-placeholder-text">
                <strong>AR visualization will appear here</strong>
                <p>Upload an image on the left to see AR overlays and 3D markers</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)

# ── ABOUT section ──────────────────────────────────────────────
st.markdown('<div class="nav-section" id="about"></div>', unsafe_allow_html=True)
st.markdown('<div class="section-panel">', unsafe_allow_html=True)
st.markdown('<h2 style="color:#1e293b;font-size:1.6rem;font-weight:700;margin:0 0 0.5rem 0;">🩺 About DermAI</h2>', unsafe_allow_html=True)
st.markdown('<p style="color:#475569;line-height:1.7;font-size:0.95rem;">DermAI is an AI-powered skin disease detection system built to assist users in identifying common dermatological conditions using deep learning and augmented reality visualization.</p>', unsafe_allow_html=True)
st.markdown('<h3 style="color:#6d28d9;font-size:1.1rem;font-weight:600;margin:1.2rem 0 0.4rem 0;">🎯 Our Mission</h3>', unsafe_allow_html=True)
st.markdown('<p style="color:#475569;line-height:1.7;font-size:0.95rem;">To make early skin disease awareness accessible to everyone through cutting-edge AI technology. DermAI bridges the gap between technology and healthcare by providing instant, visual, and informative analysis of skin conditions.</p>', unsafe_allow_html=True)
st.markdown('<h3 style="color:#6d28d9;font-size:1.1rem;font-weight:600;margin:1.2rem 0 0.4rem 0;">🧠 How It Works</h3>', unsafe_allow_html=True)
st.markdown('<ul style="color:#475569;line-height:2;font-size:0.95rem;padding-left:1.2rem;"><li>Upload a clear photo of the affected skin area</li><li>Our CNN model analyzes the image across 7 disease categories</li><li>Results are displayed with a confidence score and AR visualization</li><li>Treatment recommendations and disease descriptions are provided instantly</li></ul>', unsafe_allow_html=True)
st.markdown('<h3 style="color:#6d28d9;font-size:1.1rem;font-weight:600;margin:1.2rem 0 0.4rem 0;">⚙️ Technology Stack</h3>', unsafe_allow_html=True)
st.markdown('<ul style="color:#475569;line-height:2;font-size:0.95rem;padding-left:1.2rem;"><li><strong style="color:#1e293b;">Model:</strong> Convolutional Neural Network (CNN) with 3 Conv layers</li><li><strong style="color:#1e293b;">Framework:</strong> TensorFlow / Keras</li><li><strong style="color:#1e293b;">AR Visualization:</strong> OpenCV + PIL overlays</li><li><strong style="color:#1e293b;">Dataset:</strong> DermNet skin disease image dataset</li><li><strong style="color:#1e293b;">Interface:</strong> Streamlit web application</li></ul>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# ── DISEASES section ───────────────────────────────────────────
st.markdown('<div class="nav-section" id="diseases"></div>', unsafe_allow_html=True)
st.markdown('<div class="section-panel">', unsafe_allow_html=True)
st.markdown('<h2 style="color:#1e293b;font-size:1.6rem;font-weight:700;margin:0 0 0.5rem 0;">🔬 Detectable Diseases</h2>', unsafe_allow_html=True)
st.markdown('<p style="color:#475569;line-height:1.7;font-size:0.95rem;">DermAI is trained to detect and classify the following 7 skin disease categories from the DermNet dataset:</p>', unsafe_allow_html=True)
st.markdown("""
<div class="disease-grid">
    <div class="disease-item"><strong style="color:#1e293b;">🔴 Acne and Rosacea</strong><p style="color:#64748b;font-size:0.82rem;margin:0.3rem 0 0 0;">Inflammatory condition affecting hair follicles and oil glands, causing pimples, blackheads, and facial redness.</p></div>
    <div class="disease-item"><strong style="color:#1e293b;">🟠 Atopic Dermatitis</strong><p style="color:#64748b;font-size:0.82rem;margin:0.3rem 0 0 0;">Chronic inflammatory skin condition causing dry, itchy, and inflamed skin. Often linked to allergies and asthma.</p></div>
    <div class="disease-item"><strong style="color:#1e293b;">🟣 Bacterial Infections</strong><p style="color:#64748b;font-size:0.82rem;margin:0.3rem 0 0 0;">Includes Cellulitis and Impetigo — bacterial skin infections that cause redness, swelling, and sores.</p></div>
    <div class="disease-item"><strong style="color:#1e293b;">🔵 Eczema</strong><p style="color:#64748b;font-size:0.82rem;margin:0.3rem 0 0 0;">Skin condition causing itchy, red, cracked patches. Triggered by environmental factors and stress.</p></div>
    <div class="disease-item"><strong style="color:#1e293b;">🟡 Pigmentation Disorders</strong><p style="color:#64748b;font-size:0.82rem;margin:0.3rem 0 0 0;">Conditions affecting skin color including vitiligo, melasma, and hyperpigmentation caused by light exposure.</p></div>
    <div class="disease-item"><strong style="color:#1e293b;">🟢 Psoriasis &amp; Lichen Planus</strong><p style="color:#64748b;font-size:0.82rem;margin:0.3rem 0 0 0;">Autoimmune conditions causing scaly, thick patches of skin. Can affect nails and joints as well.</p></div>
    <div class="disease-item"><strong style="color:#1e293b;">🟤 Seborrheic Keratoses</strong><p style="color:#64748b;font-size:0.82rem;margin:0.3rem 0 0 0;">Benign skin growths that appear as waxy, scaly patches. Common in older adults and usually harmless.</p></div>
</div>
""", unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# ── DISCLAIMER section ─────────────────────────────────────────
st.markdown('<div class="nav-section" id="disclaimer"></div>', unsafe_allow_html=True)
st.markdown('<div class="section-panel">', unsafe_allow_html=True)
st.markdown('<h2 style="color:#1e293b;font-size:1.6rem;font-weight:700;margin:0 0 0.5rem 0;">⚠️ Medical Disclaimer</h2>', unsafe_allow_html=True)
st.markdown('<p style="color:#475569;line-height:1.7;font-size:0.95rem;"><strong style="color:#1e293b;">DermAI is strictly for educational and informational purposes only.</strong></p>', unsafe_allow_html=True)
st.markdown('<h3 style="color:#6d28d9;font-size:1.1rem;font-weight:600;margin:1.2rem 0 0.4rem 0;">Important Notice</h3>', unsafe_allow_html=True)
st.markdown('<p style="color:#475569;line-height:1.7;font-size:0.95rem;">The predictions made by this tool are generated by an AI model and should <strong style="color:#1e293b;">not</strong> be used as a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of a qualified dermatologist or healthcare provider.</p>', unsafe_allow_html=True)
st.markdown('<h3 style="color:#6d28d9;font-size:1.1rem;font-weight:600;margin:1.2rem 0 0.4rem 0;">Limitations</h3>', unsafe_allow_html=True)
st.markdown('<ul style="color:#475569;line-height:2;font-size:0.95rem;padding-left:1.2rem;"><li>The model is trained on a limited dataset and may not cover all skin conditions</li><li>Image quality, lighting, and angle can significantly affect prediction accuracy</li><li>Low confidence scores indicate unreliable predictions — always consult a doctor</li><li>This tool does not store or transmit any uploaded images</li></ul>', unsafe_allow_html=True)
st.markdown('<h3 style="color:#6d28d9;font-size:1.1rem;font-weight:600;margin:1.2rem 0 0.4rem 0;">Usage Agreement</h3>', unsafe_allow_html=True)
st.markdown('<p style="color:#475569;line-height:1.7;font-size:0.95rem;">By using DermAI, you acknowledge that this tool is not a medical device and the developers are not liable for any decisions made based on its output. Always consult a licensed medical professional for proper diagnosis and treatment.</p>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# ── Footer ─────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center; color:#94a3b8; font-size:0.8rem; padding: 1rem 0 2rem 0;">
    DermAI &nbsp;·&nbsp; AI-Powered Skin Disease Detection &nbsp;·&nbsp; For educational use only
</div>
""", unsafe_allow_html=True)
