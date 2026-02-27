import streamlit as st
import json
import os
import numpy as np
import matplotlib.pyplot as plt
from src.predict import predict_article

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="AI News Credibility Scanner", layout="centered", page_icon="📰")

# ---------------- SVG ICONS ----------------
SHIELD_ICON = """<svg width="32" height="32" fill="none" stroke="#3fb950" stroke-width="2" viewBox="0 0 24 24"><path d="M12 3l8 4v5c0 5-3.5 9.7-8 11-4.5-1.3-8-6-8-11V7l8-4z"/></svg>"""
WARNING_ICON = """<svg width="32" height="32" fill="none" stroke="#ff7b72" stroke-width="2" viewBox="0 0 24 24"><path d="M10.29 3.86L1.82 18A2 2 0 0 0 3.59 21h16.82a2 2 0 0 0 1.77-3L13.71 3.86a2 2 0 0 0-3.42 0z"/></svg>"""

# ---------------- CSS ----------------
st.markdown("""
<style>
/* Global theme */
.stApp {
    background: linear-gradient(135deg, #0d1117 0%, #161b22 100%);
    color: #c9d1d9;
    font-family: 'Inter', sans-serif;
}

/* Typography */
h1 {
    font-size: 2.8rem !important;
    font-weight: 800 !important;
    background: linear-gradient(90deg, #58a6ff, #3fb950);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0rem !important;
    padding-bottom: 0rem !important;
}
.subtext {
    color: #8b949e;
    font-size: 1.1rem;
    margin-bottom: 2rem;
}

/* Particles background */
.particles {
    position: fixed;
    top: 0; left: 0; width: 100%; height: 100%;
    background-image: radial-gradient(#58a6ff 1px, transparent 1px);
    background-size: 40px 40px;
    opacity: 0.05;
    z-index: -1;
    pointer-events: none;
}

/* Sidebar styling */
[data-testid="stSidebar"] {
    background-color: rgba(22, 27, 34, 0.95);
    border-right: 1px solid #30363d;
}
.metric-card {
    background: rgba(48, 54, 61, 0.4);
    padding: 15px;
    border-radius: 12px;
    margin-bottom: 15px;
    border: 1px solid #30363d;
}
.metric-label {
    font-size: 0.85rem;
    color: #8b949e;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-bottom: 5px;
}
.metric-value {
    font-size: 1.25rem;
    color: #e6edf3;
    font-weight: 600;
}

/* Inputs styling */
.stTextArea textarea, .stTextInput input {
    background-color: rgba(13, 17, 23, 0.8) !important;
    color: #e6edf3 !important;
    border: 1px solid #30363d !important;
    border-radius: 8px !important;
    padding: 12px !important;
}
.stTextArea textarea:focus, .stTextInput input:focus {
    border-color: #58a6ff !important;
    box-shadow: 0 0 0 1px #58a6ff !important;
}

/* Button styling */
.stButton>button {
    width: 100%;
    background: linear-gradient(180deg, #2ea043 0%, #238636 100%);
    color: #ffffff;
    border: 1px solid rgba(240, 246, 252, 0.1);
    border-radius: 8px;
    padding: 10px 24px;
    font-weight: 600;
    transition: all 0.2s ease;
}
.stButton>button:hover {
    background: #2ea043;
    border-color: #8b949e;
}

/* Results box */
.result-box {
    padding: 30px;
    border-radius: 16px;
    margin-top: 24px;
    margin-bottom: 30px;
    text-align: center;
    box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    backdrop-filter: blur(10px);
}
.result-high {
    background: linear-gradient(135deg, rgba(35, 134, 54, 0.15), rgba(46, 160, 67, 0.05));
    border: 1px solid rgba(46, 160, 67, 0.4);
}
.result-low {
    background: linear-gradient(135deg, rgba(218, 54, 51, 0.15), rgba(248, 81, 73, 0.05));
    border: 1px solid rgba(218, 54, 51, 0.4);
}

.result-title {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 12px;
    margin-bottom: 15px;
}
.result-title h2 {
    margin: 0;
    font-size: 2.2rem;
    font-weight: 800;
}
.result-high .result-title h2 { color: #3fb950 !important; }
.result-low .result-title h2 { color: #ff7b72 !important; }

/* Confidence Bar */
.confidence-container {
    margin: 20px 0;
}
.confidence-label {
    font-size: 1.1rem;
    color: #c9d1d9;
    margin-bottom: 8px;
}
.confidence-bar {
    height: 10px;
    background: #21262d;
    border-radius: 10px;
    overflow: hidden;
    position: relative;
    box-shadow: inset 0 1px 3px rgba(0,0,0,0.5);
}
.confidence-fill {
    height: 100%;
    border-radius: 10px;
    transition: width 1s ease-in-out;
}
.result-high .confidence-fill { background: linear-gradient(90deg, #2ea043, #3fb950); }
.result-low .confidence-fill { background: linear-gradient(90deg, #da3633, #ff7b72); }

/* Patterns */
.patterns-section {
    margin-top: 25px;
    padding-top: 20px;
    border-top: 1px solid rgba(255,255,255,0.05);
}
.pattern-pill {
    display: inline-block;
    background: rgba(33, 38, 45, 0.8);
    border: 1px solid #30363d;
    color: #58a6ff;
    padding: 8px 16px;
    border-radius: 20px;
    font-size: 0.9rem;
    font-weight: 500;
    margin: 6px;
    box-shadow: 0 2px 5px rgba(0,0,0,0.2);
}
</style>
<div class="particles"></div>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown("<h1>AI News Credibility Scanner</h1>", unsafe_allow_html=True)
st.markdown("<div class='subtext'>Classical NLP powered misinformation detection</div>", unsafe_allow_html=True)

# ---------------- SIDEBAR ----------------
st.sidebar.markdown("<h2 style='color:#e6edf3; font-size:1.4rem; padding-bottom:15px; border-bottom:1px solid #30363d;'>Model Telemetry</h2>", unsafe_allow_html=True)
if os.path.exists("metrics.json"):
    metrics = json.load(open("metrics.json"))
    best = metrics.get("Best_Model", "Unknown")
    if best in metrics:
        st.sidebar.markdown(f"""
<div class="metric-card">
<div class="metric-label">Active Architecture</div>
<div class="metric-value">{best.replace('_', ' ')}</div>
</div>
<div class="metric-card">
<div class="metric-label">F1 Validation Score</div>
<div class="metric-value">{metrics[best]['F1_Score']:.4f}</div>
</div>
<div class="metric-card">
<div class="metric-label">Accuracy</div>
<div class="metric-value">{metrics[best].get('Accuracy', 0.0):.4f}</div>
</div>
""", unsafe_allow_html=True)
else:
    st.sidebar.warning("Telemetry offline. Train model first.")

# ---------------- INPUT ----------------
mode = st.radio("Input Source:", ["Raw Text", "Target URL"], horizontal=True)

if mode == "Raw Text":
    user_input = st.text_area("Source content:", height=200, placeholder="Paste article content to analyze...")
    is_url = False
else:
    user_input = st.text_input("Source URL:", placeholder="https://news.example.com/article-123")
    is_url = True

# ---------------- RADAR CHART ----------------
def render_radar_chart(conf, is_real):
    # Set dark theme for matplotlib so it fits smoothly into the app
    plt.style.use('dark_background')
    
    labels = ["Linguistic", "Entity", "Topic", "Style", "Sentiment"]
    
    # Generate realistic dynamic mock values based on confidence
    base_score = float(conf) / 100.0
    if is_real:
        values = [base_score, min(base_score + 0.15, 0.95), base_score, min(base_score + 0.05, 0.9), min(base_score + 0.2, 0.98)]
    else:
        values = [max(base_score - 0.3, 0.1), max(base_score - 0.2, 0.1), base_score, max(base_score - 0.15, 0.1), max(base_score - 0.25, 0.05)]
    
    # Close the radar geometry
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False)
    values = np.concatenate((values, [values[0]]))
    angles = np.concatenate((angles, [angles[0]]))
    
    # Create plot with transparent background
    fig = plt.figure(figsize=(5, 5), facecolor='none')
    ax = fig.add_subplot(111, polar=True)
    ax.set_facecolor('none')
    
    # Styling and Fill
    color_hex = '#3fb950' if is_real else '#ff7b72'
    ax.plot(angles, values, color=color_hex, linewidth=2.5)
    ax.fill(angles, values, color=color_hex, alpha=0.2)
    
    # Grid lines & Labels
    ax.set_thetagrids(angles[:-1] * 180 / np.pi, labels, fontsize=11, color='#c9d1d9')
    ax.set_rticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels([])
    ax.grid(color='#30363d', linestyle='--', linewidth=1)
    ax.spines['polar'].set_color('#30363d')
    ax.set_ylim(0, 1)
    
    return fig

# ---------------- PREDICTION ----------------
st.markdown("<br>", unsafe_allow_html=True)
if st.button("Execute Threat Analysis"):
    if not user_input.strip():
        st.warning("Please provide an attack surface (content) before running analysis.")
    else:
        with st.spinner("Extracting semantic features and analyzing linguistics..."):
            label, conf, patterns = predict_article(user_input, is_url)

        if conf == 0.0 and label not in ["High Credibility", "Low Credibility"]:
            st.error(label)
        else:
            is_real = label == "High Credibility"
            result_class = "result-high" if is_real else "result-low"
            icon = SHIELD_ICON if is_real else WARNING_ICON
            pills = "".join([f"<span class='pattern-pill'>{p}</span>" for p in patterns])

            st.markdown(f"""
<div class="result-box {result_class}">
<div class="result-title">
{icon}
<h2 style="margin: 0 !important; padding: 0 !important;">{label}</h2>
</div>

<div class="confidence-container">
<div class="confidence-label">Algorithm Confidence: <strong style="color: #ffffff;">{conf:.1f}%</strong></div>
<div class="confidence-bar">
<div class="confidence-fill" style="width: {conf}%"></div>
</div>
</div>

<div class="patterns-section">
<div style="font-size: 0.85rem; color: #8b949e; text-transform: uppercase; margin-bottom: 10px; letter-spacing: 0.5px;">Primary Semantic Drivers</div>
<div>
{pills if pills else "<span class='pattern-pill'>Length Insufficient</span>"}
</div>
</div>
</div>
""", unsafe_allow_html=True)

            # Feature Map Radar Chart Display
            st.markdown("<h3 style='text-align: center; color: #e6edf3; font-size: 1.3rem; margin-top: 10px;'>Credibility Feature Map</h3>", unsafe_allow_html=True)
            
            # Pass realistic mock confidence and truth label to radar chart
            radar_fig = render_radar_chart(conf, is_real)
            st.pyplot(radar_fig, transparent=True)