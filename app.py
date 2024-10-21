import streamlit as st
import pandas as pd
from datetime import datetime
import time

ts = time.time()
date = datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
df = pd.read_csv("Attendance/Attendance_" + date + ".csv")


st.dataframe(df.style.highlight_max(axis=0))


#run -> streamlit run .\app.py