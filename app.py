import streamlit as st
import helper
import pickle

XGB_FILE_ID = '1akeBcL3_odzoRlScHmGraHcR1-1IdGrV'
# Paste your w2v_model.pkl Google Drive File ID below:
W2V_FILE_ID = '1zsHBpHdzXBxCE6qBnPQF92HhNWeIGCRM'

XGB_PATH = 'xgb_model.pkl'
W2V_PATH = 'w2v_model.pkl'

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
            st.error(f'Duplicate Questions!')
           
        else:
            st.success(f'Not Duplicate Questions.')
