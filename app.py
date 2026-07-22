import streamlit as st
import helper
import pickle

model = pickle.load(open('xgb_model.pkl', 'rb'))
w2v_model = pickle.load(open('w2v_model.pkl', 'rb'))

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
