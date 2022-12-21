import streamlit as st
import pandas as pd
import numpy as np
import pickle
import numpy as np

label_encoder_state = pickle.load(open("label_encoding.pkl", "rb"))

model = pickle.load(open("model.pkl", "rb"))

states = ('ANDHRA PRADESH', 'ARUNACHAL PRADESH', 'ASSAM', 'BIHAR',
       'CHHATTISGARH', 'GOA', 'GUJARAT', 'HARYANA', 'HIMACHAL PRADESH',
       'JHARKHAND', 'KARNATAKA', 'KERALA', 'MADHYA PRADESH',
       'MAHARASHTRA', 'MANIPUR', 'MEGHALAYA', 'MIZORAM', 'NAGALAND',
       'PUNJAB', 'RAJASTHAN', 'SIKKIM', 'TAMIL NADU', 'TRIPURA',
       'UTTAR PRADESH', 'WEST BENGAL', 'CHANDIGARH', 'LAKSHADWEEP')
state = st.selectbox("State",states)
state_le = label_encoder_state.transform([state])
pop = st.number_input("Population: ")
fiirank = st.number_input("FII rank: ") 
crime = st.number_input("CRIME(2015): ")
crime1 = st.number_input("CRIME(2016): ")
corrup1 = st.number_input("CORRUPTION CASE (2015): ")
corrup2= st.number_input("CORRUPTION CASE (2016): ")
Rul = st.number_input("Rural 2011-12 Poverty Expenditure Per Capita: ")
Urb = st.number_input("Urban 2011-12 Poverty Expenditure Per Capita: ")

ok = st.button("Predict Crime in 2014")

if ok:
    crime14 = model.predict([pop,fiirank,state_le,crime,crime1,corrup1,corrup2,Rul,Urb])
    st.write("The Estimated Crime in 2014 is ",crime14)



