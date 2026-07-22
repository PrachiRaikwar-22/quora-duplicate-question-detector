import os
import pickle
import gdown
import streamlit as st
import helper

XGB_FILE_ID = '1akeBcL3_odzoRlScHmGraHcR1-1IdGrV'
W2V_FILE_ID = '1zsHBpHdzXBxCE6qBnPQF92HhNWeIGCRM'

XGB_PATH = 'xgb_model.pkl'
W2V_PATH = 'w2v_model.pkl'

# Function to download and load models safely
@st.cache_resource
def load_models():
    # Download XGBoost Model if missing
    if not os.path.exists(XGB_PATH):
        st.info("Downloading XGBoost model from Google Drive...")
        gdown.download(f'https://drive.google.com/uc?id={XGB_FILE_ID}', XGB_PATH, quiet=False)
    
    # Download Word2Vec Model if missing
    if not os.path.exists(W2V_PATH):
        st.info("Downloading Word2Vec model from Google Drive...")
        gdown.download(f'https://drive.google.com/uc?id={W2V_FILE_ID}', W2V_PATH, quiet=False)

    # Load the pickle files
    with open(XGB_PATH, 'rb') as f:
        model = pickle.load(f)
    with open(W2V_PATH, 'rb') as f:
        w2v_model = pickle.load(f)
        
    return model, w2v_model

# Load both models
model, w2v_model = load_models()

st.title('Duplicate Question Pairs Finder')

q1 = st.text_input('Enter Question 1:')
q2 = st.text_input('Enter Question 2:')

if st.button('Find'):
    if q1.strip() == "" or q2.strip() == "":
        st.warning("Please enter both questions!")
    else:
        query = helper.query_point_creator(q1, q2, w2v_model)
        result = model.predict(query)[0]
        prob = model.predict_proba(query)[0][1]

        if result == 1:
            st.error('Duplicate Questions!')
        else:
            st.success('Not Duplicate Questions.')
