import os
import pickle
import gdown
import streamlit as st
import helper

# Page Setup
st.set_page_config(
    page_title="Duplicate Question Finder",
    page_icon="🔍",
    layout="centered"
)

XGB_FILE_ID = '1akeBcL3_odzoRlScHmGraHcR1-1IdGrV'
W2V_FILE_ID = '1zsHBpHdzXBxCE6qBnPQF92HhNWeIGCRM'

XGB_PATH = 'xgb_model.pkl'
W2V_PATH = 'w2v_model.pkl'

@st.cache_resource
def load_models():
    if not os.path.exists(XGB_PATH) or os.path.getsize(XGB_PATH) < 1000:
        if os.path.exists(XGB_PATH): os.remove(XGB_PATH)
        url = f'https://drive.google.com/file/d/{XGB_FILE_ID}/view?usp=sharing'
        gdown.download(url, XGB_PATH, quiet=False, fuzzy=True)
    
    if not os.path.exists(W2V_PATH) or os.path.getsize(W2V_PATH) < 1000:
        if os.path.exists(W2V_PATH): os.remove(W2V_PATH)
        url = f'https://drive.google.com/file/d/{W2V_FILE_ID}/view?usp=sharing'
        gdown.download(url, W2V_PATH, quiet=False, fuzzy=True)

    with open(XGB_PATH, 'rb') as f:
        model = pickle.load(f)
    with open(W2V_PATH, 'rb') as f:
        w2v_model = pickle.load(f)
        
    return model, w2v_model

model, w2v_model = load_models()

# Hero Header
st.markdown("""
    <div style="text-align: center; padding: 10px; margin-bottom: 20px;">
        <h1 style="color: #FF4B4B;">🔍 Quora Duplicate Question Detector</h1>
        <p style="color: #555; font-size: 1.1em;">
            Compare two question pairs to check if they share the same semantic meaning.
        </p>
    </div>
""", unsafe_allow_html=True)

# Columns for inputs
col1, col2 = st.columns(2)

with col1:
    q1 = st.text_area('Question 1:', placeholder='e.g., How can I learn Python easily?', height=120)

with col2:
    q2 = st.text_area('Question 2:', placeholder='e.g., What is the best way to start learning Python?', height=120)

st.markdown("<br>", unsafe_allow_html=True)

if st.button('Analyze Questions', use_container_width=True, type="primary"):
    if not q1.strip() or not q2.strip():
        st.warning("⚠️ Please enter text in both question fields before analyzing.")
    else:
        with st.spinner("Extracting NLP features and calculating similarity..."):
            query = helper.query_point_creator(q1, q2, w2v_model)
            result = model.predict(query)[0]
            prob = model.predict_proba(query)[0][1]

        st.markdown("---")
        st.subheader("Analysis Result")
        # st.metric(label="Similarity Confidence Score", value=f"{prob * 100:.1f}%")

        if result == 1:
            st.error("### 🔴 Duplicate Pair Detected\nThese questions ask the exact same thing.")
        else:
            st.success("### 🟢 Unique Questions\nThese questions have different meanings or topics.")
